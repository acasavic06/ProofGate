"""Demo stranica za pitch: jedan samostalan HTML (bez interneta, bez Colab-a).

  python demo.py --baseline models/baseline.pt --candidate models/hardened.pt \
                 --rb results/baseline --rc results/hardened --out demo.html [--lang sr|en] [--synthetic]

Tri ekrana (strelice levo/desno ili dugmad):
  1. udica: original i adversarial snimak koji izgledaju isto, a dijagnoza je druga
  2. klizaci za kontrast i osvetljenje + mapa opsega za verziju 1 i verziju 2 + rezultat dokaza
  3. brojevi iz results.json i link na izvestaj
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import core

GRID = 41
DEPTH = 5


def load_model(path, dev):
    m = core.make_model().to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval()


@torch.no_grad()
def proof_map(model, x, y, c_rng, b_rng, dev, depth=DEPTH):
    """Branch-and-bound nad kutijom parametara jedne slike; vraca listove sa statusom.
    Za razliku od Verifier-a, ne staje na prvom kontraprimeru, da bi mapa bila potpuna."""
    leaves, boxes = [], [(c_rng[0], c_rng[1], b_rng[0], b_rng[1], 0)]
    yy = torch.tensor([y], device=dev)
    while boxes:
        pl = torch.tensor([[b[0], b[2]] for b in boxes], device=dev)
        pu = torch.tensor([[b[1], b[3]] for b in boxes], device=dev)
        lb = core.forward_linear_margin(model, x.expand(len(boxes), -1, -1, -1), yy.expand(len(boxes)), pl, pu)
        nxt = []
        for bx, v in zip(boxes, lb.tolist()):
            cl, cu, bl, bu, d = bx
            if v > 0:
                leaves.append([cl, cu, bl, bu, "proven"])
                continue
            cm, bm = (cl + cu) / 2, (bl + bu) / 2
            pred = model(core.acquire(x, torch.tensor([cm], device=dev), torch.tensor([bm], device=dev))).argmax(1).item()
            if d >= depth:
                leaves.append([cl, cu, bl, bu, "counterexample" if pred != y else "open"])
            else:
                nxt += [(cl, cm, bl, bm, d + 1), (cm, cu, bl, bm, d + 1), (cl, cm, bm, bu, d + 1), (cm, cu, bm, bu, d + 1)]
        boxes = nxt
    area = (c_rng[1] - c_rng[0]) * (b_rng[1] - b_rng[0])
    proven = sum((l[1] - l[0]) * (l[3] - l[2]) for l in leaves if l[4] == "proven") / area
    return [[round(v, 5) if isinstance(v, float) else v for v in l] for l in leaves], proven


@torch.no_grad()
def grid_preds(model, x, cs, bs):
    C, B = torch.meshgrid(cs, bs, indexing="ij")
    xs = core.acquire(x.expand(C.numel(), -1, -1, -1), C.reshape(-1), B.reshape(-1))
    p = F.softmax(model(xs), 1)[:, 1]  # verovatnoca pneumonije
    return p.reshape(len(cs), len(bs))


def pick_images(m1, m2, X, y, c_rng, b_rng, dev, n=3):
    """Snimci koje oba modela tacno klasifikuju, a v1 ima kontraprimer u opsegu.
    Prednost: pneumonija i kontraprimer blizu identiteta (najubedljivije)."""
    x = torch.from_numpy(X).to(dev)
    t = torch.from_numpy(y).to(dev)
    with torch.no_grad():
        ok = (m1(x).argmax(1) == t) & (m2(x).argmax(1) == t)
    cand = ok.nonzero().flatten()[:600]
    fooled, cx, bx, _ = core.acquisition_attack(m1, x[cand], t[cand], c_rng, b_rng, grid=21)
    fooled2, *_ = core.acquisition_attack(m2, x[cand], t[cand], c_rng, b_rng, grid=21)
    good = [(int(cand[i]), float(abs(cx[i] - 1) + abs(bx[i]))) for i in range(len(cand)) if fooled[i] and not fooled2[i]]
    good.sort(key=lambda g: -g[1])  # "veca" crvena oblast prvo
    pn = [g[0] for g in good if y[g[0]] == 1]
    no = [g[0] for g in good if y[g[0]] == 0]
    # mesavina: bar jedan normalan snimak (ako postoji), ostalo pneumonija
    n_no = min(len(no), max(1, n // 3)) if no else 0
    picked = pn[:n - n_no] + no[:n_no]
    for g in pn[n - n_no:] + no[n_no:]:          # dopuni ako neke klase nema dovoljno
        if len(picked) >= n:
            break
        picked.append(g)
    return picked


def pick_hook(m1, X, y, dev):
    x = torch.from_numpy(X[:800]).to(dev)
    t = torch.from_numpy(y[:800]).to(dev)
    with torch.no_grad():
        ok = m1(x).argmax(1) == t
    for eps in [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]:
        adv = core.pgd_linf(m1, x, t, eps, steps=60, restarts=3)
        with torch.no_grad():
            flip = (m1(adv).argmax(1) != t) & ok
        idx = flip.nonzero().flatten().tolist()
        if idx:
            idx.sort(key=lambda i: -int(y[i] == 1))
            i = idx[0]
            with torch.no_grad():
                p0 = F.softmax(m1(x[i:i + 1]), 1)[0].tolist()
                p1 = F.softmax(m1(adv[i:i + 1]), 1)[0].tolist()
            return {"eps255": round(eps * 255, 1), "label": int(y[i]),
                    "x": np.round(X[i, 0], 4).tolist(), "adv": np.round(adv[i, 0].cpu().numpy(), 4).tolist(),
                    "p_orig": p0, "p_adv": p1}
    return None


def summary(d):
    if not d or not os.path.exists(os.path.join(d, "results.json")):
        return None
    r = json.load(open(os.path.join(d, "results.json")))
    c = r["acquisition"]["counts"]
    correct = c["certified"] + c["falsified"] + c["unknown"]
    return {"acc": r["model"]["clean_accuracy"], "certified": c["certified"], "correct": correct,
            "falsified": c["falsified"], "unknown": c["unknown"], "n": r["verifier"]["n_samples"],
            "linf": r["linf"]}


def build_data(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = yaml.safe_load(open(args.spec))["verify"]
    c_rng, b_rng = spec["contrast"], spec["brightness"]
    X, y = core.load_data(synthetic=args.synthetic)["test"]
    m1, m2 = load_model(args.baseline, dev), load_model(args.candidate, dev)
    cs = torch.linspace(*c_rng, GRID, device=dev)
    bs = torch.linspace(*b_rng, GRID, device=dev)
    imgs = []
    for i in pick_images(m1, m2, X, y, c_rng, b_rng, dev, n=args.n):
        x = torch.from_numpy(X[i:i + 1]).to(dev)
        item = {"index": i, "label": int(y[i]), "x": np.round(X[i, 0], 4).tolist(), "models": []}
        for m in (m1, m2):
            leaves, proven = proof_map(m, x, int(y[i]), c_rng, b_rng, dev)
            item["models"].append({"p": np.round(grid_preds(m, x, cs, bs).cpu().numpy(), 4).tolist(),
                                   "leaves": leaves, "proven_area": proven})
        imgs.append(item)
        print(f"  snimak {i}: v1 dokazano {item['models'][0]['proven_area']:.0%} oblasti, "
              f"v2 dokazano {item['models'][1]['proven_area']:.0%} oblasti")
    return {"contrast": c_rng, "brightness": b_rng, "grid": GRID, "classes": core.CLASSES,
            "hook": pick_hook(m1, X, y, dev), "images": imgs,
            "summary": [summary(args.rb), summary(args.rc)], "report": args.report_link}


TEXT = {
    "sr": {"classes": ["Normalno", "Pneumonija"], "s1t": "Isti snimak. Druga dijagnoza.",
           "s1d": "Promena od najviše {eps}/255 po pikselu. {vis}", "vis_lo": "Oko je ne vidi, a model menja dijagnozu.", "vis_hi": "Jedva primetna, a model menja dijagnozu.",
           "orig": "Original", "adv": "Izmenjen", "diff": "Razlika (pojačana)", "conf": "sigurnost",
           "s2t": "Test proverava tačke. Mi dokazujemo celu oblast.",
           "s2d": "Svako podešavanje skenera u opsegu kontrasta {cr} i osvetljenja {br}.",
           "contrast": "Kontrast", "brightness": "Osvetljenje", "v1": "Verzija 1", "v2": "Verzija 2",
           "proven_all": "DOKAZANO za ceo opseg", "not_proven": "NIJE DOKAZANO",
           "cex": "kontraprimer", "proven_area": "dokazano {p} oblasti",
           "legend": ["test prošao, nije dokazano", "dijagnoza promenjena (kontraprimer)", "dokazano: dijagnoza ne može da se promeni"],
           "s3t": "Dokaz za auditora, za svaku verziju modela.",
           "certified": "dokazano robusno", "cexs": "kontraprimera", "acc": "tačnost",
           "of": "od", "report": "Otvori izveštaj za auditora →", "image": "Snimak",
           "foot": "PneumoniaMNIST · 200 test snimaka · formalna verifikacija (linearne granice + branch-and-bound)"},
    "en": {"classes": ["Normal", "Pneumonia"], "s1t": "Same scan. Different diagnosis.",
           "s1d": "A change of at most {eps}/255 per pixel. {vis}", "vis_lo": "Invisible to the eye, yet it flips the diagnosis.", "vis_hi": "Barely visible, yet it flips the diagnosis.",
           "orig": "Original", "adv": "Perturbed", "diff": "Difference (amplified)", "conf": "confidence",
           "s2t": "Testing checks points. We prove the whole region.",
           "s2d": "Every scanner setting with contrast in {cr} and brightness in {br}.",
           "contrast": "Contrast", "brightness": "Brightness", "v1": "Version 1", "v2": "Version 2",
           "proven_all": "PROVEN for the whole range", "not_proven": "NOT PROVEN",
           "cex": "counterexample", "proven_area": "{p} of the region proven",
           "legend": ["test passed, not proven", "diagnosis changed (counterexample)", "proven: diagnosis cannot change"],
           "s3t": "Evidence for the auditor, for every model version.",
           "certified": "proven robust", "cexs": "counterexamples", "acc": "accuracy",
           "of": "of", "report": "Open the auditor report →", "image": "Scan",
           "foot": "PneumoniaMNIST · 200 test scans · formal verification (linear bounds + branch-and-bound)"},
}

PAGE = r"""<!DOCTYPE html><html lang="__LANG__"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robustness demo</title><style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--ink:#f2f3f5;--quiet:#9aa3b2;--good:#3fbf7f;--bad:#ef5350;--amber:#f0b23e;--accent:#7aa7ff}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:16px/1.5 Inter,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.slide{display:none;min-height:100vh;padding:4vh 5vw 4vh;flex-direction:column;justify-content:center}.slide.on{display:flex}
h1{font-size:clamp(26px,3vw,48px);line-height:1.1;margin:0 0 .3em;letter-spacing:-.02em}.lead{color:var(--quiet);font-size:clamp(14px,1.2vw,20px);margin:0 0 3vh}
.row{display:flex;gap:3vw;align-items:flex-start;justify-content:center}.fig{text-align:center}
canvas.img{width:min(24vw,40vh);height:min(24vw,40vh);image-rendering:pixelated;border-radius:10px;display:block;background:#000}
.tag{display:inline-block;margin-top:14px;padding:8px 18px;border-radius:999px;font-weight:700;font-size:clamp(16px,1.5vw,24px)}
.tag.pn{background:rgba(239,83,80,.16);color:#ff8a85}.tag.no{background:rgba(63,191,127,.16);color:#7fe0ad}
.cap{color:var(--quiet);font-size:14px;margin-top:6px}.arrow{align-self:center;font-size:40px;color:var(--quiet);margin-top:-8vh}
.s2{display:grid;grid-template-columns:min(20vw,32vh) auto auto;gap:2.4vw;align-items:start;justify-content:center}
.left{display:flex;flex-direction:column;gap:14px;align-items:center}.left canvas.img{width:min(17vw,26vh);height:min(17vw,26vh)}
.ctl{width:100%}.ctl label{display:flex;justify-content:space-between;color:var(--quiet);font-size:14px}.ctl b{color:var(--ink);font-variant-numeric:tabular-nums}
input[type=range]{width:100%;accent-color:var(--accent)}
.thumbs{display:flex;gap:8px}.thumbs canvas{width:46px;height:46px;image-rendering:pixelated;border-radius:6px;cursor:pointer;opacity:.5;border:2px solid transparent}.thumbs canvas.sel{opacity:1;border-color:var(--accent)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
.card h2{margin:0 0 10px;font-size:clamp(16px,1.4vw,22px);display:flex;justify-content:space-between;align-items:center}
.card canvas.map{width:min(25vw,40vh);height:min(25vw,40vh);border-radius:8px;display:block;cursor:crosshair}.s2 .card{width:calc(min(25vw,40vh) + 32px);padding:14px 16px}
.dx{margin-top:8px;font-size:clamp(15px,1.3vw,20px)}.verdict{margin-top:10px;padding:10px 14px;border-radius:10px;font-weight:800;font-size:clamp(15px,1.35vw,21px)}
.verdict.ok{background:rgba(63,191,127,.15);color:#7fe0ad}.verdict.bad{background:rgba(239,83,80,.15);color:#ff8a85}.verdict small{display:block;font-weight:500;font-size:13px;opacity:.9}
.axes{display:flex;justify-content:space-between;color:var(--quiet);font-size:12px;margin-top:4px}
.legend{display:flex;gap:18px;color:var(--quiet);font-size:13px;margin-top:1.6vh;justify-content:center}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.big{display:grid;grid-template-columns:1fr 1fr;gap:3vw;max-width:1100px;margin:0 auto;width:100%}
.big .card{padding:3vh 2.4vw}.num{font-size:clamp(48px,6vw,96px);font-weight:800;line-height:1;letter-spacing:-.03em}
.num.ok{color:var(--good)}.num.bad{color:var(--amber)}.sub{color:var(--quiet);margin-top:8px;font-size:clamp(14px,1.2vw,19px)}
.stat{display:flex;gap:2vw;margin-top:2.4vh;font-size:clamp(14px,1.2vw,18px)}.stat b{display:block;font-size:clamp(22px,2.2vw,34px)}
a.btn{display:inline-block;margin:4vh auto 0;padding:14px 26px;border-radius:12px;background:var(--accent);color:#0b1020;font-weight:800;text-decoration:none;font-size:clamp(15px,1.3vw,20px)}
.nav{position:fixed;bottom:2vh;right:2vw;display:flex;gap:8px;opacity:.55}.nav:hover{opacity:1}.nav button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:999px;padding:8px 16px;font:inherit;cursor:pointer}
.dots{display:flex;gap:8px;align-items:center}.dots span{width:8px;height:8px;border-radius:50%;background:var(--line)}.dots span.on{background:var(--accent)}
.foot{position:fixed;bottom:1vh;right:2vw;color:#5b6475;font-size:12px}
</style></head><body>
<section class="slide" id="s1"></section><section class="slide" id="s2"></section><section class="slide" id="s3"></section>
<div class="nav"><button id="prev">←</button><div class="dots"><span></span><span></span><span></span></div><button id="next">→</button></div>

<script>
const D = __DATA__, T = __TEXT__;
const $ = s => document.querySelector(s);
const fmtPct = v => (Math.round(v * 1000) / 10).toString().replace('.', T.classes[0] === 'Normalno' ? ',' : '.') + '%';
const fmtR = (r, sign) => '[' + r.map(v => (sign && v > 0 ? '+' : '') + v).join(', ') + ']';
function drawImg(cv, px, c = 1, b = 0, scale = 1) {
  const n = px.length, ctx = cv.getContext('2d'); cv.width = n; cv.height = n;
  let m = 0; px.forEach(r => r.forEach(v => m += v)); m /= n * n;
  const im = ctx.createImageData(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let v = c * (px[i][j] - m) + m + b; v = Math.max(0, Math.min(1, v * scale));
    const k = 4 * (i * n + j); im.data[k] = im.data[k + 1] = im.data[k + 2] = Math.round(v * 255); im.data[k + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
function tag(cls) { return `<span class="tag ${cls ? 'pn' : 'no'}">${T.classes[cls]}</span>`; }

// ---------- slide 1
(function () {
  const h = D.hook, s = $('#s1');
  if (!h) { s.innerHTML = `<h1>${T.s1t}</h1>`; return; }
  const c0 = h.p_orig[1] > .5 ? 1 : 0, c1 = h.p_adv[1] > .5 ? 1 : 0;
  s.innerHTML = `<h1>${T.s1t}</h1><p class="lead">${T.s1d.replace('{eps}', h.eps255).replace('{vis}', h.eps255 <= 2 ? T.vis_lo : T.vis_hi)}</p>
  <div class="row"><div class="fig"><canvas class="img" id="h0"></canvas>${tag(c0)}<div class="cap">${T.orig} · ${T.conf} ${fmtPct(Math.max(...h.p_orig))}</div></div>
  <div class="arrow">→</div>
  <div class="fig"><canvas class="img" id="h1"></canvas>${tag(c1)}<div class="cap">${T.adv} · ${T.conf} ${fmtPct(Math.max(...h.p_adv))}</div></div>
  <div class="fig"><canvas class="img" id="h2"></canvas><div class="cap" style="margin-top:18px">${T.diff}</div></div></div>`;
  drawImg($('#h0'), h.x); drawImg($('#h1'), h.adv);
  const k = 0.45 / (h.eps255 / 255); const diff = h.x.map((r, i) => r.map((v, j) => 0.5 + k * (h.adv[i][j] - v)));
  drawImg($('#h2'), diff);
})();

// ---------- slide 2
let cur = 0, ci = (D.grid - 1) / 2, bi = (D.grid - 1) / 2;
const cAt = i => D.contrast[0] + (D.contrast[1] - D.contrast[0]) * i / (D.grid - 1);
const bAt = i => D.brightness[0] + (D.brightness[1] - D.brightness[0]) * i / (D.grid - 1);
(function () {
  const s = $('#s2');
  const thumbs = D.images.map((_, k) => `<canvas data-k="${k}"></canvas>`).join('');
  const card = v => `<div class="card"><h2><span>${v ? T.v2 : T.v1}</span></h2><canvas class="map" id="map${v}"></canvas>
    <div class="axes"><span>${T.contrast} ${D.contrast[0]}</span><span>${T.contrast} ${D.contrast[1]} →</span></div>
    <div class="dx" id="dx${v}"></div><div class="verdict" id="vd${v}"></div></div>`;
  s.innerHTML = `<h1>${T.s2t}</h1><p class="lead">${T.s2d.replace('{cr}', fmtR(D.contrast)).replace('{br}', fmtR(D.brightness, true))}</p>
  <div class="s2"><div class="left"><canvas class="img" id="live"></canvas><div class="cap" id="lbl"></div>
    <div class="ctl"><label>${T.contrast}<b id="cv"></b></label><input type="range" id="cs" min="0" max="${D.grid - 1}" value="${ci}"></div>
    <div class="ctl"><label>${T.brightness}<b id="bv"></b></label><input type="range" id="bs" min="0" max="${D.grid - 1}" value="${bi}"></div>
    <div class="thumbs">${thumbs}</div></div>${card(0)}${card(1)}</div>
  <div class="legend"><span><i style="background:rgb(46,160,103)"></i>${T.legend[2]}</span><span><i style="background:#3a4150"></i>${T.legend[0]}</span><span><i style="background:var(--bad)"></i>${T.legend[1]}</span></div>`;
  document.querySelectorAll('.thumbs canvas').forEach(cv => { drawImg(cv, D.images[+cv.dataset.k].x); cv.onclick = () => { cur = +cv.dataset.k; render(); }; });
  $('#cs').oninput = e => { ci = +e.target.value; render(); };
  $('#bs').oninput = e => { bi = +e.target.value; render(); };
  [0, 1].forEach(v => {
    const cv = $('#map' + v);
    const pick = e => { const r = cv.getBoundingClientRect(); ci = Math.round((e.clientX - r.left) / r.width * (D.grid - 1)); bi = Math.round((1 - (e.clientY - r.top) / r.height) * (D.grid - 1)); ci = Math.max(0, Math.min(D.grid - 1, ci)); bi = Math.max(0, Math.min(D.grid - 1, bi)); $('#cs').value = ci; $('#bs').value = bi; render(); };
    cv.onmousedown = e => { pick(e); cv.onmousemove = pick; }; window.addEventListener('mouseup', () => cv.onmousemove = null);
  });
})();
function drawMap(v) {
  const im = D.images[cur], m = im.models[v], cv = $('#map' + v), S = 600, ctx = cv.getContext('2d');
  cv.width = cv.height = S; const g = D.grid, cell = S / g;
  // test: predikcija modela u svakoj tacki mreze (x = kontrast, y = osvetljenje, gore = svetlije)
  for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) {
    const pred = m.p[i][j] > .5 ? 1 : 0;
    ctx.fillStyle = pred === im.label ? '#3a4150' : '#ef5350';
    ctx.fillRect(i * cell, S - (j + 1) * cell, cell + .5, cell + .5);
  }
  // dokaz: listovi branch-and-bound pretrage
  const X = c => (c - D.contrast[0]) / (D.contrast[1] - D.contrast[0]) * S;
  const Y = b => S - (b - D.brightness[0]) / (D.brightness[1] - D.brightness[0]) * S;
  m.leaves.forEach(([cl, cu, bl, bu, st]) => {
    if (st !== 'proven') return;
    ctx.fillStyle = 'rgba(46,160,103,.92)'; ctx.fillRect(X(cl), Y(bu), X(cu) - X(cl), Y(bl) - Y(bu));
    ctx.strokeStyle = 'rgba(15,17,21,.35)'; ctx.lineWidth = 1; ctx.strokeRect(X(cl) + .5, Y(bu) + .5, X(cu) - X(cl) - 1, Y(bl) - Y(bu) - 1);
  });
  // trenutna tacka
  const px = (ci + .5) * cell, py = S - (bi + .5) * cell;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 11, 0, 7); ctx.stroke();
  ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px - 20, py); ctx.lineTo(px - 13, py); ctx.moveTo(px + 13, py); ctx.lineTo(px + 20, py); ctx.moveTo(px, py - 20); ctx.lineTo(px, py - 13); ctx.moveTo(px, py + 13); ctx.lineTo(px, py + 20); ctx.stroke();
  // dijagnoza i presuda
  const p = m.p[ci][bi], pred = p > .5 ? 1 : 0;
  $('#dx' + v).innerHTML = `${tag(pred)} <span class="cap">${T.conf} ${fmtPct(Math.max(p, 1 - p))}</span>`;
  const vd = $('#vd' + v), all = m.proven_area > 0.9999;
  let cex = null;
  let best = 1e9; const sc = D.contrast[1] - D.contrast[0], sb = D.brightness[1] - D.brightness[0];
  for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) if ((m.p[i][j] > .5 ? 1 : 0) !== im.label) { const d = Math.hypot((cAt(i) - 1) / sc, bAt(j) / sb); if (d < best) { best = d; cex = [cAt(i), bAt(j)]; } }
  vd.className = 'verdict ' + (all ? 'ok' : 'bad');
  vd.innerHTML = all ? `✓ ${T.proven_all}<small>${T.proven_area.replace('{p}', '100%')}</small>`
    : `✗ ${T.not_proven}<small>${cex ? T.cex + ': c=' + cex[0].toFixed(2) + ', b=' + (cex[1] >= 0 ? '+' : '') + cex[1].toFixed(2) + ' · ' : ''}${T.proven_area.replace('{p}', fmtPct(m.proven_area))}</small>`;
}
function render() {
  const im = D.images[cur]; if (!im) return;
  drawImg($('#live'), im.x, cAt(ci), bAt(bi));
  $('#cv').textContent = cAt(ci).toFixed(2); $('#bv').textContent = (bAt(bi) >= 0 ? '+' : '') + bAt(bi).toFixed(2);
  $('#lbl').innerHTML = `${T.image} #${im.index} · ${T.classes[im.label]}`;
  document.querySelectorAll('.thumbs canvas').forEach(cv => cv.classList.toggle('sel', +cv.dataset.k === cur));
  drawMap(0); drawMap(1);
}

// ---------- slide 3
(function () {
  const s = $('#s3'), [a, b] = D.summary;
  const card = (r, v) => !r ? '' : `<div class="card"><div class="cap" style="font-size:16px;margin:0 0 1.4vh">${v ? T.v2 : T.v1}</div>
    <div class="num ${v ? 'ok' : 'bad'}">${fmtPct(r.certified / Math.max(r.correct, 1))}</div><div class="sub">${T.certified} · ${r.certified} ${T.of} ${r.correct}</div>
    <div class="stat"><div><b>${r.falsified}</b>${T.cexs}</div><div><b>${fmtPct(r.acc)}</b>${T.acc}</div></div></div>`;
  s.innerHTML = `<h1 style="text-align:center">${T.s3t}</h1><p class="lead" style="text-align:center">${T.s2d.replace('{cr}', fmtR(D.contrast)).replace('{br}', fmtR(D.brightness, true))}</p>
  <div class="big">${card(a, 0)}${card(b, 1)}</div><p class="cap" style="text-align:center;margin-top:2vh">${T.foot}</p>${D.report ? `<div style="text-align:center"><a class="btn" href="${D.report}" target="_blank">${T.report}</a></div>` : ''}`;
})();

// ---------- navigacija
let slide = 0; const slides = [...document.querySelectorAll('.slide')];
function go(k) { slide = Math.max(0, Math.min(slides.length - 1, k)); slides.forEach((s, i) => s.classList.toggle('on', i === slide)); document.querySelectorAll('.dots span').forEach((d, i) => d.classList.toggle('on', i === slide)); if (slide === 1) render(); }
$('#prev').onclick = () => go(slide - 1); $('#next').onclick = () => go(slide + 1);
document.addEventListener('keydown', e => { if (e.target.tagName === 'INPUT') return; if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(slide + 1); } if (['ArrowLeft', 'PageUp'].includes(e.key)) go(slide - 1); });

go(0);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--rb", help="results folder verzije 1")
    ap.add_argument("--rc", help="results folder verzije 2")
    ap.add_argument("--spec", default="spec.yaml")
    ap.add_argument("--out", default="demo.html")
    ap.add_argument("--lang", default="en", choices=["sr", "en"])
    ap.add_argument("--n", type=int, default=3, help="broj snimaka na ekranu 2")
    ap.add_argument("--report-link", default=None, help="podrazumevano report_<lang>.html")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    args.report_link = args.report_link or f"report_{args.lang}.html"
    data = build_data(args)
    if not data["images"]:
        print("UPOZORENJE: nema snimka gde v1 ima kontraprimer, a v2 nema. Probaj siri opseg u spec.yaml.")
    html = (PAGE.replace("__LANG__", args.lang)
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__TEXT__", json.dumps(TEXT[args.lang], ensure_ascii=False)))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Sacuvano: {args.out} ({len(html) / 1e6:.1f} MB), snimaka: {len(data['images'])}")


if __name__ == "__main__":
    main()
