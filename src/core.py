"""Jezgro: podaci, model, napadi i formalna verifikacija.

Dve vrste specifikacije:
  * L-inf: ||x' - x||_inf <= eps  (adversarial primeri, Clan 15(5))
  * Akvizicija (parametri snimanja): x' = c * (x - mean(x)) + mean(x) + b,
    c u [c_lo, c_hi] (kontrast), b u [b_lo, b_hi] (osvetljenje).
    Ulaz u verifikator su samo 2 parametra, pa CROWN ostaje precizan.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["normal", "pneumonia"]


# ----------------------------------------------------------------------------
# Podaci
# ----------------------------------------------------------------------------
def load_data(synthetic=False, n_synth=2000, seed=0):
    """Vraca dict split -> (X float32 [N,1,28,28] u [0,1], y int64 [N])."""
    if synthetic:
        g = np.random.default_rng(seed)
        out = {}
        for split, n in [("train", n_synth), ("val", n_synth // 5), ("test", n_synth // 4)]:
            y = g.integers(0, 2, n)
            X = g.normal(0.45, 0.08, (n, 1, 28, 28))
            yy, xx = np.mgrid[0:28, 0:28]
            blob = np.exp(-((yy - 14) ** 2 + (xx - 10) ** 2) / 30.0)
            X += (y[:, None, None, None] * 0.35) * blob  # "zamucenje" za pneumoniju
            out[split] = (np.clip(X, 0, 1).astype(np.float32), y.astype(np.int64))
        return out
    import medmnist  # pip install medmnist (skida sa zenodo.org)
    out = {}
    for split in ["train", "val", "test"]:
        ds = medmnist.PneumoniaMNIST(split=split, download=True, size=28)
        X = (ds.imgs.astype(np.float32) / 255.0)[:, None, :, :]
        y = ds.labels.reshape(-1).astype(np.int64)
        out[split] = (X, y)
    return out


def loader(X, y, batch=128, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)


# ----------------------------------------------------------------------------
# Model (bez BatchNorm-a, da bi verifikacija bila jednostavna i precizna)
# ----------------------------------------------------------------------------
def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 100), nn.ReLU(),
        nn.Linear(100, 2),
    )


def acquire(x, c, b):
    """Model akvizicije: kontrast c oko srednje vrednosti slike + osvetljenje b."""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return c.view(-1, 1, 1, 1) * (x - m) + m + b.view(-1, 1, 1, 1)


@torch.no_grad()
def accuracy(model, X, y, device, batch=512):
    model.eval()
    correct, preds = 0, []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        p = model(xb).argmax(1).cpu()
        preds.append(p)
    preds = torch.cat(preds).numpy()
    return float((preds == y).mean()), preds


# ----------------------------------------------------------------------------
# Napadi
# ----------------------------------------------------------------------------
def pgd_linf(model, x, y, eps, steps=20, alpha=None, restarts=1):
    """PGD u L-inf lopti, slika ostaje u [0,1]. Vraca adversarial slike."""
    if eps <= 0:
        return x.clone()
    alpha = alpha or 2.5 * eps / steps
    model.eval()
    best = x.clone()
    fooled = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for _ in range(restarts):
        delta = (torch.rand_like(x) * 2 - 1) * eps
        for _ in range(steps):
            delta.requires_grad_(True)
            loss = F.cross_entropy(model((x + delta).clamp(0, 1)), y)
            g, = torch.autograd.grad(loss, delta)
            delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        adv = (x + delta).clamp(0, 1).detach()
        with torch.no_grad():
            now = model(adv).argmax(1) != y
        new = now & ~fooled
        best[new] = adv[new]
        fooled |= now
    return best


@torch.no_grad()
def acquisition_attack(model, x, y, c_rng, b_rng, grid=21):
    """Gusta pretraga po (kontrast, osvetljenje). Vraca (fooled, c*, b*, pred*)."""
    model.eval()
    cs = torch.linspace(*c_rng, grid, device=x.device)
    bs = torch.linspace(*b_rng, grid, device=x.device)
    C, B = torch.meshgrid(cs, bs, indexing="ij")
    C, B = C.reshape(-1), B.reshape(-1)
    n = len(x)
    fooled = torch.zeros(n, dtype=torch.bool, device=x.device)
    cx = torch.full((n,), float("nan"), device=x.device)
    bx = cx.clone()
    px = torch.full((n,), -1, dtype=torch.long, device=x.device)
    for i in range(n):
        xi = x[i:i + 1].expand(len(C), -1, -1, -1)
        pred = model(acquire(xi, C, B)).argmax(1)
        bad = (pred != y[i]).nonzero().flatten()
        if len(bad):
            # najblizi kontraprimer identitetu (c=1, b=0)
            d = (C[bad] - 1).abs() + B[bad].abs()
            j = bad[d.argmin()]
            fooled[i], cx[i], bx[i], px[i] = True, C[j], B[j], pred[j]
    return fooled, cx, bx, px


# ----------------------------------------------------------------------------
# Formalna verifikacija
# ----------------------------------------------------------------------------
def _spec_matrix(y, k=2):
    """C tako da je C @ logits = logit_y - logit_j za svako j != y."""
    n = len(y)
    E = F.one_hot(y, k).float()                                   # (n,k)
    full = E[:, None, :] - torch.eye(k, device=y.device)[None]      # (n,k,k), red j: e_y - e_j
    keep = E == 0                                                   # izbaci red j == y
    return full[keep].view(n, k - 1, k)


def _ibp(net, lo, hi):
    """Interval bound propagation (rezervna opcija bez auto_LiRPA)."""
    for layer in net:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            mid, rad = (hi + lo) / 2, (hi - lo) / 2
            W = layer.weight
            if isinstance(layer, nn.Conv2d):
                f = lambda t, w, b=None: F.conv2d(t, w, b, layer.stride, layer.padding)
            else:
                f = lambda t, w, b=None: F.linear(t, w, b)
            mid, rad = f(mid, W, layer.bias), f(rad, W.abs())
            lo, hi = mid - rad, mid + rad
        elif isinstance(layer, nn.ReLU):
            lo, hi = lo.clamp(min=0), hi.clamp(min=0)
        elif isinstance(layer, nn.Flatten):
            lo, hi = lo.flatten(1), hi.flatten(1)
        else:
            raise ValueError(f"IBP ne podrzava sloj {layer}")
    return lo, hi


def _has_lirpa():
    try:
        import auto_LiRPA  # noqa: F401
        return True
    except Exception:
        return False


class Verifier:
    """Formalna verifikacija. method: 'crown' (auto_LiRPA) ili 'ibp' (rezerva)."""

    def __init__(self, net, device, method="crown"):
        self.net = net.eval()
        self.device = device
        self.method = method if (method != "crown" or _has_lirpa()) else "ibp"
        self._lin = None

    # -- L-inf --------------------------------------------------------------
    @torch.no_grad()
    def certify_linf(self, x, y, eps, batch=256):
        """True ako je dokazano da nijedna slika u L-inf lopti ne menja klasu."""
        out = []
        for i in range(0, len(x), batch):
            xb, yb = x[i:i + batch], y[i:i + batch]
            lo, hi = (xb - eps).clamp(0, 1), (xb + eps).clamp(0, 1)
            out.append(self._margin_lb_img(xb, yb, lo, hi) > 0)
        return torch.cat(out)

    def _margin_lb_img(self, x, y, lo, hi):
        C = _spec_matrix(y)
        if self.method == "ibp":
            l, u = _ibp(self.net, lo, hi)
            return torch.einsum("nkc,nc->nk", C.clamp(min=0), l).add(
                torch.einsum("nkc,nc->nk", C.clamp(max=0), u)).min(1).values
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        if self._lin is None:
            self._lin = BoundedModule(self.net, torch.zeros(1, 1, 28, 28, device=self.device),
                                      device=self.device, verbose=0)
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=lo, x_U=hi)
        lb, _ = self._lin.compute_bounds(x=(BoundedTensor(x, ptb),), C=C, method="CROWN", bound_upper=False)
        return lb.min(1).values

    # -- Akvizicija (kontrast, osvetljenje) ----------------------------------
    @torch.no_grad()
    def _margin_lb_acq(self, x0, y, pl, pu):
        """Donja granica margine za kutiju parametara [pl, pu] (N x 2)."""
        C = _spec_matrix(y)
        if self.method == "ibp":
            lo, hi = _acq_interval(x0, pl, pu)
            l, u = _ibp(self.net, lo, hi)
            return torch.einsum("nkc,nc->nk", C.clamp(min=0), l).add(
                torch.einsum("nkc,nc->nk", C.clamp(max=0), u)).min(1).values
        # sopstvena forward linearna relaksacija nad 2 parametra (ne treba auto_LiRPA)
        return forward_linear_margin(self.net, x0, y, pl, pu)

    @torch.no_grad()
    def certify_acquisition(self, x, y, c_rng, b_rng, max_depth=6, chunk=2048):
        """Branch-and-bound po prostoru parametara snimanja.

        Za svaku sliku vraca status: 'certified' (dokaz za ceo opseg),
        'falsified' (nadjen kontraprimer) ili 'unknown' (dokaz nije zavrsen).
        """
        n = len(x)
        status = ["certified"] * n
        boxes = [(i, c_rng[0], c_rng[1], b_rng[0], b_rng[1], 0) for i in range(n)]
        while boxes:
            nxt = []
            for s in range(0, len(boxes), chunk):
                part = [bx for bx in boxes[s:s + chunk] if status[bx[0]] == "certified"]
                if not part:
                    continue
                idx = torch.tensor([bx[0] for bx in part], device=self.device)
                pl = torch.tensor([[bx[1], bx[3]] for bx in part], device=self.device)
                pu = torch.tensor([[bx[2], bx[4]] for bx in part], device=self.device)
                ok = self._margin_lb_acq(x[idx], y[idx], pl, pu) > 0
                for bx, good in zip(part, ok.tolist()):
                    if good:
                        continue
                    i, cl, cu, bl, bu, d = bx
                    cm, bm = (cl + cu) / 2, (bl + bu) / 2
                    # da li je centar kutije pravi kontraprimer?
                    xc = acquire(x[i:i + 1], torch.tensor([cm], device=self.device),
                                 torch.tensor([bm], device=self.device))
                    if self.net(xc).argmax(1).item() != int(y[i]):
                        status[i] = "falsified"
                    elif d >= max_depth:
                        status[i] = "unknown"
                    else:
                        nxt += [(i, cl, cm, bl, bm, d + 1), (i, cm, cu, bl, bm, d + 1),
                                (i, cl, cm, bm, bu, d + 1), (i, cm, cu, bm, bu, d + 1)]
            boxes = nxt
        return status


def _concretize(Lc, pl, pu, lower=True):
    """min (ili max) linearne forme Lc (N, P+1, F) po kutiji p u [pl, pu]."""
    coef, const = Lc[:, :-1], Lc[:, -1]
    lo, hi = pl.unsqueeze(-1), pu.unsqueeze(-1)
    if lower:
        return const + torch.where(coef > 0, coef * lo, coef * hi).sum(1)
    return const + torch.where(coef > 0, coef * hi, coef * lo).sum(1)


def forward_linear_margin(net, x0, y, pl, pu):
    """Sopstveni verifikator za parametre snimanja (forward linearna relaksacija).

    Svaki neuron se drzi kao donja i gornja linearna funkcija parametara p=(c,b):
    tensor oblika (N, 3, ...) gde su [:,0] i [:,1] koeficijenti uz c i b, a [:,2] konstanta.
    Ulaz je tacno afin u p, pa relaksacija uvodi gresku samo na nestabilnim ReLU-ovima.
    Vraca donju granicu min_j (logit_y - logit_j) za svaku kutiju.
    """
    m = x0.mean(dim=(1, 2, 3), keepdim=True)
    d = x0 - m
    form = torch.stack([d, torch.ones_like(x0), m.expand_as(x0)], 1)  # (N,3,1,28,28)
    L, U = form, form.clone()
    N = len(x0)
    for layer in net:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            Wp, Wn = layer.weight.clamp(min=0), layer.weight.clamp(max=0)
            sh = L.shape
            l2, u2 = L.reshape(N * 3, *sh[2:]), U.reshape(N * 3, *sh[2:])
            if isinstance(layer, nn.Conv2d):
                op = lambda t, w: F.conv2d(t, w, None, layer.stride, layer.padding)
            else:
                op = lambda t, w: F.linear(t, w)
            nl = op(l2, Wp) + op(u2, Wn)
            nu = op(u2, Wp) + op(l2, Wn)
            nl = nl.reshape(N, 3, *nl.shape[1:])
            nu = nu.reshape(N, 3, *nu.shape[1:])
            if layer.bias is not None:
                bshape = (1,) + tuple(layer.bias.shape) + (1,) * (nl.dim() - 3)
                nl[:, 2] += layer.bias.view(bshape[1:])
                nu[:, 2] += layer.bias.view(bshape[1:])
            L, U = nl, nu
        elif isinstance(layer, nn.Flatten):
            L, U = L.flatten(2), U.flatten(2)
        elif isinstance(layer, nn.ReLU):
            fl, fu = L.flatten(2), U.flatten(2)
            l = _concretize(fl, pl, pu, lower=True)
            u = _concretize(fu, pl, pu, lower=False)
            active, dead = l >= 0, u <= 0
            unstable = ~active & ~dead
            # gornja: u/(u-l) * (z - l) ; donja: alpha * z, alpha in {0,1}
            s = torch.where(unstable, u / (u - l).clamp(min=1e-12), active.float())
            shift = torch.where(unstable, -s * l, torch.zeros_like(l))
            alpha = torch.where(unstable, (u > -l).float(), active.float())
            nu = fu * s.unsqueeze(1)
            nu[:, 2] += shift
            nl = fl * alpha.unsqueeze(1)
            L, U = nl.reshape(L.shape), nu.reshape(U.shape)
        else:
            raise ValueError(f"Nepodrzan sloj {layer}")
    # margina: logit_y - logit_j
    Cm = _spec_matrix(y, k=L.shape[-1])                       # (N, K-1, K)
    Cp, Cn = Cm.clamp(min=0), Cm.clamp(max=0)
    marg = torch.einsum("nkc,nqc->nqk", Cp, L) + torch.einsum("nkc,nqc->nqk", Cn, U)  # (N,3,K-1)
    return _concretize(marg, pl, pu, lower=True).min(1).values


def _acq_interval(x0, pl, pu):
    """Interval slike za kutiju parametara (koristi ga samo IBP rezerva)."""
    m = x0.mean(dim=(1, 2, 3), keepdim=True)
    d = x0 - m
    cl, cu = pl[:, 0].view(-1, 1, 1, 1), pu[:, 0].view(-1, 1, 1, 1)
    bl, bu = pl[:, 1].view(-1, 1, 1, 1), pu[:, 1].view(-1, 1, 1, 1)
    lo = torch.minimum(cl * d, cu * d) + m + bl
    hi = torch.maximum(cl * d, cu * d) + m + bu
    return lo, hi


# ----------------------------------------------------------------------------
# Trening
# ----------------------------------------------------------------------------
def train(model, X, y, device, epochs=10, lr=1e-3, harden=None, log=print):
    """Standardni trening, ili ojacavanje ako je zadat harden dict:
    {'eps': 2/255, 'contrast': [0.8, 1.2], 'brightness': [-0.1, 0.1], 'pgd_steps': 7,
     'cert_weight': 1.0, 'cert_contrast': [0.7, 1.3], 'cert_brightness': [-0.15, 0.15], 'ramp_epochs': 4}
    cert_weight > 0 ukljucuje trening ka sertifikatu: donja granica margine iz verifikatora
    (forward_linear_margin) ulazi u funkciju greske, pa model uci da bude dokazivo robustan.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # tezine klasa: PneumoniaMNIST je neuravnotezen (~74% pneumonija)
    w = torch.tensor(np.bincount(y, minlength=2), dtype=torch.float32)
    w = (w.sum() / (2 * w)).to(device)
    cw = (harden or {}).get("cert_weight", 0)
    for ep in range(epochs):
        t0, tot, n, ncert = time.time(), 0.0, 0, 0
        k = min(1.0, (ep + 1) / max(1, (harden or {}).get("ramp_epochs", 1)))  # rast kutije
        for xb, yb in loader(X, y):
            xb, yb = xb.to(device), yb.to(device)
            x_clean = xb
            if harden:
                c = torch.empty(len(xb), device=device).uniform_(*harden["contrast"])
                b = torch.empty(len(xb), device=device).uniform_(*harden["brightness"])
                xb = acquire(xb, c, b)
                if harden.get("eps", 0) > 0:
                    model.eval()
                    xb = pgd_linf(model, xb.clamp(0, 1), yb, harden["eps"], steps=harden.get("pgd_steps", 7))
                    model.train()
            loss = F.cross_entropy(model(xb), yb, weight=w)
            if cw > 0:
                cl, cu = harden["cert_contrast"]
                bl, bu = harden["cert_brightness"]
                pl = torch.tensor([[1 - k * (1 - cl), k * bl]], device=device).expand(len(yb), 2)
                pu = torch.tensor([[1 + k * (cu - 1), k * bu]], device=device).expand(len(yb), 2)
                lb = forward_linear_margin(model, x_clean, yb, pl, pu)
                loss = loss + cw * (w[yb] * F.relu(1.0 - lb)).mean()
                ncert += int((lb > 0).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        extra = f"  dokazivo na treningu={ncert / n:.3f} (kutija {k:.0%})" if cw > 0 else ""
        log(f"  epoha {ep + 1}/{epochs}  loss={tot / n:.4f}{extra}  ({time.time() - t0:.1f}s)")
    return model.eval()