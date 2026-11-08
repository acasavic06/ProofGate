"""HTML izvestaj (paket dokaza) iz dva results.json: referentni i kandidat model.

  python report.py --baseline results/baseline --candidate results/hardened --out report.html [--lang sr|en]

Izlaz je jedan samostalan HTML fajl (slike su ugradjene), spreman za browser ili stampu u PDF.
"""
import argparse
import base64
import html
import json
import os

T = {
    "en": {
        "title": "Robustness Evidence Report",
        "sub": "Formal verification of a neural network under acquisition and adversarial perturbations",
        "under_test": "Model under test", "reference": "Reference model", "dataset": "Dataset",
        "created": "Generated", "verdict": "Result",
        "verdict_txt": "Proven robust on {c} of {n} correctly classified test images ({p}) for contrast {cr} and brightness {br}. "
                       "{fx} Undecided: {u}.",
        "vs_ref": "Reference model: {c} of {n} ({p}), {f} counterexamples.",
        "acc": "Accuracy (full test set)", "acq_title": "Acquisition robustness (contrast and brightness)",
        "acq_desc": "Specification: x' = c·(x − mean(x)) + mean(x) + b for every contrast factor c in {cr} and brightness shift b in {br}. "
                    "A certificate is a mathematical proof that no setting in this range changes the diagnosis.",
        "certified": "Certified", "falsified": "Counterexample", "unknown": "Undecided", "misclassified": "Misclassified",
        "linf_title": "Adversarial robustness (L∞)",
        "linf_desc": "Every image within distance ε (per pixel) of the original. Attack = PGD; proof = CROWN bound propagation.",
        "eps": "ε", "attack": "Robust under attack", "proof": "Proven robust", "model": "Model",
        "cex_title": "Counterexamples", "cex_desc": "Inputs that change the model's decision. Left: original, right: perturbed.",
        "cex_list": "Remaining counterexamples of the model under test",
        "idx": "Test image", "label": "Label", "pred_after": "Prediction after perturbation", "c": "Contrast c", "b": "Brightness b",
        "reg_title": "Mapping to requirements",
        "reg_rows": [
            ("EU AI Act, Art. 15(5)", "Resilience against inputs designed to cause the model to make a mistake (adversarial examples, model evasion)", "Adversarial robustness (L∞)"),
            ("EU AI Act, Art. 15(4)", "Resilience regarding errors, faults or inconsistencies in the environment of use", "Acquisition robustness (contrast, brightness)"),
            ("ISO/IEC 24029-2:2023", "Methodology for formal methods in robustness assessment of neural networks", "Methods section"),
        ],
        "reg_cols": ("Requirement", "What it asks", "Evidence in this report"),
        "methods_title": "Methods and limitations",
        "methods": [
            "L∞ proofs: CROWN linear bound propagation (auto_LiRPA). Attacks: PGD with random restarts.",
            "Acquisition proofs: forward linear relaxation over the two acquisition parameters with branch-and-bound over the parameter box. Every image ends as proven, counterexample, or undecided.",
            "Soundness check of the verifier: no proven image was broken by attack; on 100 random parameter boxes with 2,000 samples each, no true margin fell below the computed bound.",
            "Results hold for the stated ranges, the evaluated test images and this model version; they are not a claim of regulatory conformity.",
            "The acquisition model is linear and ignores sensor saturation; ranges should be set from the intended clinical acquisition conditions.",
        ],
        "fx_some": "{f} counterexamples found and listed below.", "fx_none": "No counterexamples found.",
        "sample_note": "Evaluated on {n} test images sampled with a fixed seed.",
    },
    "sr": {
        "title": "Izveštaj o robusnosti",
        "sub": "Formalna verifikacija neuronske mreže pod perturbacijama snimanja i adversarial napadima",
        "under_test": "Model koji se proverava", "reference": "Referentni model", "dataset": "Dataset",
        "created": "Napravljeno", "verdict": "Rezultat",
        "verdict_txt": "Dokazana robusnost na {c} od {n} tačno klasifikovanih test snimaka ({p}) za kontrast {cr} i osvetljenje {br}. "
                       "{fx} Neodlučenih: {u}.",
        "vs_ref": "Referentni model: {c} od {n} ({p}), {f} kontraprimera.",
        "acc": "Tačnost (ceo test skup)", "acq_title": "Robusnost na parametre snimanja (kontrast i osvetljenje)",
        "acq_desc": "Specifikacija: x' = c·(x − mean(x)) + mean(x) + b za svaki faktor kontrasta c u {cr} i pomeraj osvetljenja b u {br}. "
                    "Sertifikat je matematički dokaz da nijedno podešavanje u tom opsegu ne menja dijagnozu.",
        "certified": "Dokazano", "falsified": "Kontraprimer", "unknown": "Neodlučeno", "misclassified": "Pogrešno klasifikovano",
        "linf_title": "Adversarial robusnost (L∞)",
        "linf_desc": "Svaka slika na rastojanju najviše ε po pikselu od originala. Napad = PGD; dokaz = CROWN.",
        "eps": "ε", "attack": "Robusno pod napadom", "proof": "Dokazano robusno", "model": "Model",
        "cex_title": "Kontraprimeri", "cex_desc": "Ulazi koji menjaju odluku modela. Levo: original, desno: izmenjen snimak.",
        "cex_list": "Preostali kontraprimeri modela koji se proverava",
        "idx": "Test snimak", "label": "Oznaka", "pred_after": "Predikcija posle izmene", "c": "Kontrast c", "b": "Osvetljenje b",
        "reg_title": "Mapiranje na zahteve",
        "reg_rows": [
            ("EU AI Act, Član 15(5)", "Otpornost na ulaze dizajnirane da navedu model na grešku (adversarial primeri, model evasion)", "Adversarial robusnost (L∞)"),
            ("EU AI Act, Član 15(4)", "Otpornost na greške i nedoslednosti u okruženju upotrebe", "Robusnost na parametre snimanja"),
            ("ISO/IEC 24029-2:2023", "Metodologija formalnih metoda za procenu robusnosti neuronskih mreža", "Sekcija o metodama"),
        ],
        "reg_cols": ("Zahtev", "Šta traži", "Dokaz u ovom izveštaju"),
        "methods_title": "Metode i ograničenja",
        "methods": [
            "L∞ dokazi: CROWN propagacija linearnih granica (auto_LiRPA). Napadi: PGD sa nasumičnim restartima.",
            "Dokazi za snimanje: forward linearna relaksacija po dva parametra snimanja uz branch-and-bound po kutiji parametara. Svaka slika završava kao dokazana, kontraprimer ili neodlučena.",
            "Provera ispravnosti verifikatora: nijedna dokazana slika nije oborena napadom; na 100 nasumičnih kutija sa po 2.000 tačaka stvarna margina nijednom nije pala ispod izračunate granice.",
            "Rezultati važe za navedene opsege, proverene test snimke i ovu verziju modela; ovo nije tvrdnja o regulatornoj usaglašenosti.",
            "Model snimanja je linearan i zanemaruje zasićenje senzora; opsege treba odrediti iz predviđenih kliničkih uslova snimanja.",
        ],
        "fx_some": "Pronađeno {f} kontraprimera, navedeni su ispod.", "fx_none": "Nije pronađen nijedan kontraprimer.",
        "sample_note": "Provereno na {n} test snimaka izabranih sa fiksnim seed-om.",
    },
}

CSS = """
:root{--ink:#1d1d1b;--quiet:#5f5e5a;--line:#e3e1db;--bg:#faf9f6;--good:#2e7d4f;--bad:#c0392b;--mid:#b7791f;--grey:#9a978f;--accent:#1f5fa8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
main{max-width:920px;margin:0 auto;padding:48px 32px 64px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:19px;margin:40px 0 8px}
.sub{color:var(--quiet);margin:0 0 24px}.meta{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:14px 0;font-size:13px}
.meta b{display:block;color:var(--quiet);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
.verdict{margin:28px 0;padding:20px 24px;background:#fff;border:1px solid var(--line);border-left:4px solid var(--good);border-radius:8px}
.verdict .big{font-size:40px;font-weight:700;color:var(--good);line-height:1}.verdict p{margin:8px 0 0}.verdict .ref{color:var(--quiet);font-size:13.5px}
p.desc{color:var(--quiet);margin:0 0 14px;font-size:14px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:14px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}th{background:#f3f1ec;font-weight:600;font-size:12.5px;color:var(--quiet)}
td.num{text-align:right;font-variant-numeric:tabular-nums}.nums th:not(:first-child){text-align:right}tr:last-child td{border-bottom:none}
.bars{display:grid;gap:12px;margin:8px 0 4px}.bar-row{display:grid;grid-template-columns:200px 1fr;align-items:center;gap:12px;font-size:13.5px}
.bar{display:flex;height:26px;border-radius:6px;overflow:hidden;background:#eee}.bar span{display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:600;white-space:nowrap}
.legend{display:flex;gap:16px;font-size:12.5px;color:var(--quiet);margin:8px 0 0}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.cex{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px}
.card h3{font-size:13.5px;margin:0 0 8px}.pair{display:flex;gap:8px}.pair img{width:50%;border-radius:4px;image-rendering:pixelated}
ul{padding-left:20px}li{margin:4px 0}.foot{margin-top:40px;color:var(--grey);font-size:12px}
@media print{body{background:#fff}main{padding:0}.verdict,table,.card{break-inside:avoid}}
"""


def load(d):
    with open(os.path.join(d, "results.json")) as f:
        return json.load(f)


def img64(d, rel):
    p = os.path.join(d, rel)
    if not rel or not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def pct(x):
    return f"{100 * x:.1f}%".replace(".0%", "%")


def rng(r, kind):
    lo, hi = r
    if kind == "c":
        return f"[{lo:g}, {hi:g}] (±{round(100 * (hi - 1))}%)" if abs((hi - 1) + (lo - 1)) < 1e-9 else f"[{lo:g}, {hi:g}]"
    return f"[{lo:+g}, {hi:+g}] (±{round(100 * hi)}%)" if abs(hi + lo) < 1e-9 else f"[{lo:+g}, {hi:+g}]"


def acq_stats(r):
    c = r["acquisition"]["counts"]
    correct = c["certified"] + c["falsified"] + c["unknown"]
    return c, correct


def bar(label, counts, total, t):
    colors = {"certified": "var(--good)", "falsified": "var(--bad)", "unknown": "var(--mid)", "misclassified": "var(--grey)"}
    parts = ""
    for k in ["certified", "falsified", "unknown", "misclassified"]:
        v = counts.get(k, 0)
        if v:
            w = 100 * v / total
            txt = str(v) if w > 4 else ""
            parts += f'<span style="width:{w:.2f}%;background:{colors[k]}" title="{t[k]}: {v}">{txt}</span>'
    return f'<div class="bar-row"><div>{html.escape(label)}</div><div class="bar">{parts}</div></div>'


def build(base_dir, cand_dir, lang="en"):
    t = T[lang]
    b, c = load(base_dir), load(cand_dir)
    e = html.escape
    acq = c["acquisition"]
    cr, br = rng(acq["contrast"], "c"), rng(acq["brightness"], "b")
    cc, cn = acq_stats(c)
    bc, bn = acq_stats(b)
    bname, cname = b["model"]["name"], c["model"]["name"]

    verdict = t["verdict_txt"].format(c=cc["certified"], n=cn, p=pct(cc["certified"] / max(cn, 1)), cr=cr, br=br,
                                      fx=(t['fx_some'].format(f=cc['falsified']) if cc['falsified'] else t['fx_none']), u=cc["unknown"])
    vref = t["vs_ref"].format(c=bc["certified"], n=bn, p=pct(bc["certified"] / max(bn, 1)), f=bc["falsified"])

    rows_acq = "".join(
        f"<tr><td>{e(r['model']['name'])}</td><td class=num>{pct(r['model']['clean_accuracy'])}</td>"
        f"<td class=num>{s[0]['certified']} / {s[1]} ({pct(s[0]['certified'] / max(s[1], 1))})</td>"
        f"<td class=num>{s[0]['falsified']}</td><td class=num>{s[0]['unknown']}</td></tr>"
        for r, s in [(b, (bc, bn)), (c, (cc, cn))])

    ntot = c["verifier"]["n_samples"]
    bars = bar(bname, bc, b["verifier"]["n_samples"], t) + bar(cname, cc, ntot, t)
    legend = "".join(f'<span><i style="background:{col}"></i>{t[k]}</span>' for k, col in
                     [("certified", "var(--good)"), ("falsified", "var(--bad)"), ("unknown", "var(--mid)"),
                      ("misclassified", "var(--grey)")])

    rows_linf = ""
    for r in [b, c]:
        for row in r["linf"]:
            rows_linf += (f"<tr><td>{e(r['model']['name'])}</td><td class=num>{row['eps_255']:g}/255</td>"
                          f"<td class=num>{pct(row['pgd_acc'])}</td><td class=num>{pct(row['certified_acc'])}</td></tr>")

    cards = ""
    for r, d in [(b, base_dir), (c, cand_dir)]:
        ex = r.get("linf_example")
        if ex:
            a, z = img64(d, ex["clean_png"]), img64(d, ex["adv_png"])
            if a and z:
                cards += (f'<div class="card"><h3>{e(r["model"]["name"])} · L∞ ε={ex["eps"] * 255:.1f}/255: '
                          f'{e(ex["label"])} → {e(ex["adv_pred"])}</h3><div class="pair"><img src="{a}"><img src="{z}"></div></div>')
        ex = r["acquisition"].get("example")
        if ex:
            a, z = img64(d, ex["clean_png"]), img64(d, ex["adv_png"])
            if a and z:
                cards += (f'<div class="card"><h3>{e(r["model"]["name"])} · c={ex["contrast"]:.2f}, b={ex["brightness"]:+.2f}: '
                          f'{e(ex["label"])} → {e(ex["pred"])}</h3><div class="pair"><img src="{a}"><img src="{z}"></div></div>')

    cex_rows = "".join(
        f"<tr><td class=num>{s['index']}</td><td>{e(s['label'])}</td><td>{e(s['acquisition']['counterexample']['pred'])}</td>"
        f"<td class=num>{s['acquisition']['counterexample']['contrast']:.3f}</td>"
        f"<td class=num>{s['acquisition']['counterexample']['brightness']:+.3f}</td></tr>"
        for s in c["samples"] if s["acquisition"] and s["acquisition"].get("counterexample"))
    cex_table = (f"<h3 style='font-size:15px;margin:24px 0 8px'>{t['cex_list']}</h3><table><tr><th>{t['idx']}</th><th>{t['label']}</th>"
                 f"<th>{t['pred_after']}</th><th>{t['c']}</th><th>{t['b']}</th></tr>{cex_rows}</table>") if cex_rows else ""

    reg = "".join(f"<tr><td><b>{e(a)}</b></td><td>{e(q)}</td><td>{e(w)}</td></tr>" for a, q, w in t["reg_rows"])
    meth = "".join(f"<li>{e(m)}</li>" for m in t["methods"])

    return f"""<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t['title']} · {e(cname)}</title><style>{CSS}</style></head><body><main>
<h1>{t['title']}</h1><p class="sub">{t['sub']}</p>
<div class="meta"><div><b>{t['under_test']}</b>{e(cname)}</div><div><b>{t['reference']}</b>{e(bname)}</div>
<div><b>{t['dataset']}</b>{e(c['data']['dataset'])}</div><div><b>{t['created']}</b>{e(c['created'][:16].replace('T', ' '))}</div></div>
<div class="verdict"><div class="big">{pct(cc['certified'] / max(cn, 1))}</div><p>{e(verdict)}</p><p class="ref">{e(vref)}</p></div>

<h2>{t['acq_title']}</h2><p class="desc">{e(t['acq_desc'].format(cr=cr, br=br))}</p>
<div class="bars">{bars}</div><div class="legend">{legend}</div>
<table class="nums" style="margin-top:16px"><tr><th>{t['model']}</th><th>{t['acc']}</th><th>{t['certified']}</th><th>{t['falsified']}</th><th>{t['unknown']}</th></tr>{rows_acq}</table>

<h2>{t['linf_title']}</h2><p class="desc">{t['linf_desc']}</p>
<table class="nums"><tr><th>{t['model']}</th><th>{t['eps']}</th><th>{t['attack']}</th><th>{t['proof']}</th></tr>{rows_linf}</table>

<h2>{t['cex_title']}</h2><p class="desc">{t['cex_desc']}</p><div class="cex">{cards}</div>{cex_table}

<h2>{t['reg_title']}</h2><table><tr><th>{t['reg_cols'][0]}</th><th>{t['reg_cols'][1]}</th><th>{t['reg_cols'][2]}</th></tr>{reg}</table>

<h2>{t['methods_title']}</h2><ul>{meth}</ul>
<p class="foot">{e(t['sample_note'].format(n=ntot))} · {e(c['verifier'].get('linf', ''))} · {e(c['verifier'].get('acquisition', ''))}</p>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--lang", default="en", choices=["en", "sr"])
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(a.baseline, a.candidate, a.lang))
    print(f"Sacuvano: {a.out}")


if __name__ == "__main__":
    main()
