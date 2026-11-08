"""Spiegelzelle: Lage der Auflagepunkte (3 / 6 / 9 / 18) und Durchbiegung des Spiegels unter Eigengewicht.
(neu in v9.0.0)

Modell: dünne Kirchhoff-Platte (Kreisscheibe, freier Rand, gleichmäßige Dicke, volle Fläche), Eigengewicht senkrecht
(Spiegel liegt auf dem Rücken, Zenit), Punktlager.  Lösung per Ritz-Verfahren (Chebyshev-Produktbasis, Grad 14).
Die Wippen/Dreiecke sind statisch bestimmt (Drehpunkt in Mitte bzw. Schwerpunkt), jeder Punkt trägt dieselbe Last.

Bewertet wird der Oberflächenfehler nach Abzug von Piston, Kippung und Defokus (Brennweitenänderung ist beim
Newton refokussierbar).  Mit  refokus=False  ohne Defokus-Abzug (klassische Werte 0,645·R / 0,68·R).
Keine Tk-/Streamlit-Abhängigkeit.
"""
import math
import numpy as np
from numpy.polynomial import chebyshev as _C

MATERIALIEN = {   # E [GPa], Querkontraktion, Dichte [kg/m³]
    "Borosilikat (Pyrex/Duran)": (63.0, 0.20, 2230.0),
    "BK7 / Floatglas":           (82.0, 0.206, 2510.0),
    "Quarzglas":                 (72.0, 0.17, 2200.0),
    "Zerodur / Cervit":          (90.0, 0.24, 2530.0),
}
G = 9.81

# Optimierte Geometrie in Bruchteilen des Spiegelradius R (Rechnung mit Grad 20, Defokus abgezogen).
#  3:  Radius des Auflagekreises
#  6:  Radius des Punktkreises (Sechseck, 3 Wippen tangential, Punkte je ±30° um den Wippen-Drehpunkt)
#  9:  3 Dreiecke: Drehpunktradius rp, Umkreisradius rho, Drehung phi (60° = Ecke zur Mitte)
# 18:  3 Wippen (Drehpunktradius rp, halbe Länge a) mit je 2 Dreiecken (Umkreisradius rho, Drehung phi,
#      bezogen auf die eigene Radialrichtung des Dreiecks; 60° = Ecke zur Spiegelmitte)
OPT = {
    3:  dict(r=0.4014),
    6:  dict(r=0.5729),
    9:  dict(rp=0.440, rho=0.331, phi=math.pi / 3),
    18: dict(rp=0.590, a=0.340, rho=0.272, phi=math.pi / 3),
}
OPT_OHNE = {   # klassische Werte (ohne Defokus-Abzug)
    3:  dict(r=0.6489),
    6:  dict(r=0.6817),
    9:  dict(rp=0.500, rho=0.377, phi=math.pi / 3),
    18: dict(rp=0.590, a=0.340, rho=0.272, phi=math.pi / 3),
}


# ───────────────────────── Geometrie ─────────────────────────
def _ring(n, r, phi0):
    a = phi0 + 2 * math.pi * np.arange(n) / n
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def _dreieck(c, rho, ang):
    return np.array([c + rho * np.array([math.cos(ang + 2 * math.pi * i / 3), math.sin(ang + 2 * math.pi * i / 3)])
                     for i in range(3)])


def punkte(n, p):
    """Auflagepunkte (Einheitskreis, R=1) und Struktur für die Zeichnung.
    Rückgabe (pts (m,2), struktur-Liste); Struktur: ('wippe', P1, P2, Drehpunkt), ('dreieck', [P1,P2,P3], Schwerpunkt)."""
    st = []
    if n == 3:
        pts = _ring(3, p["r"], math.pi / 2)
        return pts, st
    if n == 6:
        r = p["r"]; rp = r * math.cos(math.pi / 6); a = r * math.sin(math.pi / 6)
        pts = []
        for k in range(3):
            th = math.pi / 2 + 2 * math.pi * k / 3
            er = np.array([math.cos(th), math.sin(th)]); et = np.array([-math.sin(th), math.cos(th)])
            P1 = rp * er - a * et; P2 = rp * er + a * et
            pts += [P1, P2]; st.append(("wippe", P1, P2, rp * er))
        return np.array(pts), st
    if n == 9:
        pts = []
        for k in range(3):
            th = math.pi / 2 + 2 * math.pi * k / 3
            c = p["rp"] * np.array([math.cos(th), math.sin(th)])
            T = _dreieck(c, p["rho"], th + p["phi"]); pts += list(T); st.append(("dreieck", T, c))
        return np.array(pts), st
    if n == 18:
        pts = []
        for k in range(3):
            th = math.pi / 2 + 2 * math.pi * k / 3
            er = np.array([math.cos(th), math.sin(th)]); et = np.array([-math.sin(th), math.cos(th)])
            cs = []
            for s in (-1, 1):
                c = p["rp"] * er + s * p["a"] * et; cs.append(c)
                T = _dreieck(c, p["rho"], math.atan2(c[1], c[0]) + p["phi"]); pts += list(T); st.append(("dreieck", T, c))
            st.append(("wippe", cs[0], cs[1], p["rp"] * er))
        return np.array(pts), st
    raise ValueError("Variante 3, 6, 9 oder 18")


# ───────────────────────── Plattenmodell (Ritz) ─────────────────────────
class _Platte:
    def __init__(self, P=14, nu=0.2, nr=40):
        self.P = P; self.nu = nu
        self.idx = [(a, b) for a in range(P + 1) for b in range(P + 1 - a)]
        xg, wg = np.polynomial.legendre.leggauss(nr); r = (xg + 1) / 2; wr = wg / 2 * r
        nphi = 4 * P + 8; ph = np.arange(nphi) * 2 * math.pi / nphi
        R, PH = np.meshgrid(r, ph, indexing="ij")
        self.W = (wr[:, None] * np.full(nphi, 2 * math.pi / nphi)[None, :]).ravel()
        self.X = (R * np.cos(PH)).ravel(); self.Y = (R * np.sin(PH)).ravel()
        B = self.basis(self.X, self.Y); self.B0 = B["0"]
        Wt = self.W[:, None]; lap = B["xx"] + B["yy"]
        K = (lap * Wt).T @ lap + 2 * (1 - nu) * ((B["xy"] * Wt).T @ B["xy"]
                                                 - 0.5 * ((B["xx"] * Wt).T @ B["yy"] + (B["yy"] * Wt).T @ B["xx"]))
        self.K = (K + K.T) / 2
        self.f = (B["0"] * Wt).sum(axis=0)

    def basis(self, x, y, nur0=False):
        P = self.P
        def tab(t):
            T = np.zeros((P + 1, len(t))); T1 = np.zeros_like(T); T2 = np.zeros_like(T)
            for k in range(P + 1):
                c = np.zeros(k + 1); c[k] = 1
                T[k] = _C.chebval(t, c)
                if k > 0: T1[k] = _C.chebval(t, _C.chebder(c))
                if k > 1: T2[k] = _C.chebval(t, _C.chebder(c, 2))
            return T, T1, T2
        Tx, Tx1, Tx2 = tab(x); Ty, Ty1, Ty2 = tab(y)
        n = len(self.idx)
        keys = ("0",) if nur0 else ("0", "xx", "yy", "xy")
        out = {k: np.zeros((len(x), n)) for k in keys}
        for j, (a, b) in enumerate(self.idx):
            out["0"][:, j] = Tx[a] * Ty[b]
            if not nur0:
                out["xx"][:, j] = Tx2[a] * Ty[b]; out["yy"][:, j] = Tx[a] * Ty2[b]; out["xy"][:, j] = Tx1[a] * Ty1[b]
        return out

    def loese(self, pts):
        pts = np.asarray(pts, float)
        Bp = self.basis(pts[:, 0], pts[:, 1], True)["0"]
        n = self.K.shape[0]; m = len(pts)
        A = np.zeros((n + m, n + m))
        A[:n, :n] = self.K + 1e-9 * np.eye(n) * np.trace(self.K) / n
        A[:n, n:] = Bp.T; A[n:, :n] = Bp
        sol = np.linalg.lstsq(A, np.concatenate([self.f, np.zeros(m)]), rcond=1e-13)[0]
        return sol[:n], sol[n:]

    def rest(self, c, refokus=True):
        """Durchbiegung minus Piston/Kippung(/Defokus) auf den Quadraturpunkten; (RMS, Rest)."""
        w = self.B0 @ c
        cols = [np.ones_like(self.X), self.X, self.Y] + ([self.X ** 2 + self.Y ** 2] if refokus else [])
        M = np.stack(cols, 1); sw = np.sqrt(self.W)
        k = np.linalg.lstsq(M * sw[:, None], w * sw, rcond=None)[0]
        res = w - M @ k
        return float(math.sqrt((res ** 2 * self.W).sum() / self.W.sum())), k

    def feld(self, c, k, n=121):
        """Rest-Durchbiegung auf einem Raster (für die Grafik)."""
        g = np.linspace(-1, 1, n); X, Y = np.meshgrid(g, g)
        m = (X ** 2 + Y ** 2) <= 1.0
        x, y = X[m], Y[m]
        w = self.basis(x, y, True)["0"] @ c
        cols = [np.ones_like(x), x, y] + ([x ** 2 + y ** 2] if len(k) == 4 else [])
        w = w - np.stack(cols, 1) @ k
        Z = np.full(X.shape, np.nan); Z[m] = w
        return X, Y, Z


_PLATTE = {}
def _platte():
    if "p" not in _PLATTE:
        _PLATTE["p"] = _Platte()
    return _PLATTE["p"]


def _skala(D_mm, t_mm, mat):
    """Umrechnung dimensionslos -> mm:  w = 12(1-nu²)·rho·g·R⁴/(E·t²) · w*   (Querkontraktion des Modells fix 0,2)."""
    E, nu, rho = MATERIALIEN[mat]
    R = D_mm / 2000.0; t = t_mm / 1000.0                 # m
    return 12 * (1 - nu ** 2) * rho * G * R ** 4 / (E * 1e9 * t ** 2) * 1e9   # nm


# ───────────────────────── Rechner ─────────────────────────
def zelle_rechner(D_mm, t_mm, n, mat="Borosilikat (Pyrex/Duran)", refokus=True, lam_nm=550.0, p=None):
    """Auflagepunkte und Durchbiegung für n = 3, 6, 9 oder 18 Punkte.
    p = eigene Parameter (wie OPT[n], in Bruchteilen von R); None = Optimum.
    Liefert Punkte in mm (Radius, Winkel), Wippen-/Dreiecksmaße, RMS (Oberfläche, Wellenfront), Strehl, Rest-Raster."""
    R = D_mm / 2.0
    par = dict(p) if p else dict((OPT if refokus else OPT_OHNE)[n])
    pts, st = punkte(n, par)
    pl = _platte()
    c, lam = pl.loese(pts)
    rms_r, k_r = pl.rest(c, True); rms_o, k_o = pl.rest(c, False)
    s = _skala(D_mm, t_mm, mat)
    rms = (rms_r if refokus else rms_o) * s              # nm Oberfläche
    wf = 2 * rms                                          # Wellenfront (Reflexion)
    strehl = math.exp(-(2 * math.pi * wf / lam_nm) ** 2)
    X, Y, Z = pl.feld(c, k_r if refokus else k_o)
    pm = pts * R
    pr = np.hypot(pm[:, 0], pm[:, 1]); pa = (np.degrees(np.arctan2(pm[:, 1], pm[:, 0])) + 360) % 360
    d = dict(n=n, par=par, pts_mm=pm, radius_mm=pr, winkel=pa, struktur=st, R=R,
             rms_nm=rms, wf_rms_nm=wf, wf_lambda=(lam_nm / wf if wf > 0 else float("inf")), strehl=strehl,
             rms_refokus_nm=rms_r * s, rms_ohne_nm=rms_o * s, refokus=refokus,
             last_pro_punkt_N=(math.pi * (R / 1000) ** 2 * t_mm / 1000 * MATERIALIEN[mat][2] * G / n),
             raster=(X * R, Y * R, Z * s))
    # Maße
    if n == 3:
        d["masse"] = dict(r=par["r"] * R, abstand=par["r"] * R * math.sqrt(3))
    elif n == 6:
        r = par["r"] * R
        d["masse"] = dict(r=r, rp=r * math.cos(math.pi / 6), wippe=r)            # Sechseck: Seite = r
    elif n == 9:
        rho = par["rho"] * R
        d["masse"] = dict(rp=par["rp"] * R, rho=rho, seite=rho * math.sqrt(3), phi=math.degrees(par["phi"]),
                          r_innen=abs(par["rp"] - par["rho"]) * R if abs(par["phi"] - math.pi / 3) < 1e-6 else None,
                          r_aussen=float(pr.max()))
    else:
        rho = par["rho"] * R
        d["masse"] = dict(rp=par["rp"] * R, a=par["a"] * R, wippe=2 * par["a"] * R, rho=rho, seite=rho * math.sqrt(3),
                          phi=math.degrees(par["phi"]), r_aussen=float(pr.max()))
    return d


def zelle_vergleich(D_mm, t_mm, mat="Borosilikat (Pyrex/Duran)", refokus=True, lam_nm=550.0):
    """RMS aller vier Varianten (Optimum) für Tabelle und Balkendiagramm."""
    return [zelle_rechner(D_mm, t_mm, n, mat, refokus, lam_nm) for n in (3, 6, 9, 18)]


def zelle_rms_kurve(D_mm, t_mm, n, mat="Borosilikat (Pyrex/Duran)", refokus=True, r_von=0.2, r_bis=0.9, anz=36):
    """RMS über den Ring-Radius (nur 3 und 6 Punkte): zeigt das Optimum. Rückgabe (r/R, rms_nm)."""
    pl = _platte(); s = _skala(D_mm, t_mm, mat)
    rs = np.linspace(r_von, r_bis, anz); out = []
    for r in rs:
        pts, _ = punkte(n, dict(r=r))
        c, _l = pl.loese(pts)
        out.append(pl.rest(c, refokus)[0] * s)
    return rs, np.array(out)


# ───────────────────────── Grafik ─────────────────────────
def zelle_zeichnen(fig, D_mm, t_mm, n, mat="Borosilikat (Pyrex/Duran)", refokus=True, lam_nm=550.0, p=None):
    """2×2: Aufsicht mit Wippen/Dreiecken · Durchbiegung (Rest) · Vergleich der Varianten · Hinweis/Radiuskurve."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    fig.clf()
    d = zelle_rechner(D_mm, t_mm, n, mat, refokus, lam_nm, p)
    R = d["R"]; pm = d["pts_mm"]
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.38, wspace=0.18)
    # 1) Aufsicht
    ax = fig.add_subplot(gs[:, 0]); ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), R, fc="#e8f1f8", ec="#33566b", lw=1.6))
    ax.add_patch(Circle((0, 0), 0.92 * R, fc="none", ec="#33566b", lw=0.6, ls=":"))
    for kind, *a in d["struktur"]:
        if kind == "dreieck":
            T, c = a
            ax.add_patch(Polygon(T * R, closed=True, fc="#fde9c9", ec="#d98a00", lw=1.2, alpha=0.8))
            ax.plot(*(c * R), "s", color="#d98a00", ms=4)
        else:
            P1, P2, M = a
            ax.plot([P1[0] * R, P2[0] * R], [P1[1] * R, P2[1] * R], "-", color="#2e7d32", lw=3 if n == 6 else 1.8, alpha=0.9)
            ax.plot(*(M * R), "^", color="#2e7d32", ms=7)
    if n == 3:
        ax.add_patch(Polygon(pm, closed=True, fc="none", ec="#d98a00", lw=1.2))
    ax.plot(pm[:, 0], pm[:, 1], "o", color="#c0392b", ms=7, zorder=5)
    for i, (x, y) in enumerate(pm):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, color="#7a1f16")
    ax.plot(0, 0, "+", color="#555", ms=8)
    ax.set_xlim(-R * 1.08, R * 1.08); ax.set_ylim(-R * 1.08, R * 1.08)
    ax.set_xlabel("mm"); ax.set_ylabel("mm"); ax.grid(alpha=0.2)
    ax.set_title(f"{n} Auflagepunkte – Aufsicht auf die Spiegelrückseite (Ø{D_mm:g} mm)", fontsize=10)
    ax.plot([], [], "o", color="#c0392b", label="Auflagepunkt"); 
    if n in (6, 18): ax.plot([], [], "-^", color="#2e7d32", label="Wippe (Drehpunkt ▲)")
    if n in (9, 18): ax.add_patch(Polygon([[0, 0]] * 3, fc="#fde9c9", ec="#d98a00", label="Dreieck"))
    ax.legend(loc="lower left", fontsize=7)
    # 2) Durchbiegungskarte
    ax2 = fig.add_subplot(gs[0, 1]); X, Y, Z = d["raster"]
    lim = float(np.nanmax(np.abs(Z))) or 1.0
    im = ax2.contourf(X, Y, Z, levels=np.linspace(-lim, lim, 15), cmap="RdBu_r")
    ax2.plot(pm[:, 0], pm[:, 1], "k.", ms=3); ax2.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03); cb.set_label("Oberfläche [nm]", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax2.set_title(f"Durchbiegung (Rest nach Abzug von {'Kipp+Defokus' if refokus else 'Kippung'})\n"
                  f"RMS {d['rms_nm']:.1f} nm Oberfl. · Wellenfront λ/{d['wf_lambda']:.0f} · Strehl {d['strehl']:.3f}", fontsize=8)
    ax2.tick_params(labelsize=7)
    # 3) Vergleich
    ax3 = fig.add_subplot(gs[1, 1]); vgl = zelle_vergleich(D_mm, t_mm, mat, refokus, lam_nm)
    farben = ["#c0392b" if v["n"] == n else "#8aa5b8" for v in vgl]
    vals = [v["rms_nm"] for v in vgl]
    bars = ax3.bar([f"{v['n']}" for v in vgl], vals, color=farben)
    ax3.set_yscale("log"); ax3.set_ylabel("RMS Oberfläche [nm]", fontsize=8); ax3.set_xlabel("Auflagepunkte", fontsize=8)
    ax3.tick_params(labelsize=8); ax3.grid(alpha=0.3, which="both", axis="y")
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.1f}", ha="center", fontsize=7)
    ax3.set_title(f"Vergleich bei t = {t_mm:g} mm, {mat.split(' (')[0].split(' /')[0]}", fontsize=9)
    fig.suptitle("Spiegelzelle – Lage der Auflagepunkte (Eigengewicht, Zenit, dünne Platte)", fontsize=11)
    return d


def zelle_text(d, t_mm, mat):
    """Maße und Bewertung als Text (mm)."""
    n = d["n"]; m = d["masse"]; R = d["R"]
    z = [f"{n} Auflagepunkte · Ø {2*R:g} mm · Dicke {t_mm:g} mm · {mat}",
         f"Last je Punkt ≈ {d['last_pro_punkt_N']:.1f} N (≈ {d['last_pro_punkt_N']/9.81:.2f} kg); Wippen/Dreiecke sind im Drehpunkt bzw. Schwerpunkt zu lagern, dann tragen alle Punkte gleich.",
         ""]
    if n == 3:
        z += [f"Auflagekreis: r = {m['r']:.1f} mm ({d['par']['r']:.3f}·R), Punkte bei 90°, 210°, 330° (gleichseitiges Dreieck, Seite {m['abstand']:.1f} mm)."]
    elif n == 6:
        z += [f"Punktkreis: r = {m['r']:.1f} mm ({d['par']['r']:.3f}·R), regelmäßiges Sechseck.",
              f"3 Wippen tangential, Abstand Punkt–Punkt (Wippenlänge) {m['wippe']:.1f} mm, Drehpunkt auf r = {m['rp']:.1f} mm (je 120°)."]
    elif n == 9:
        z += [f"3 Dreiecke, Drehpunkt (Schwerpunkt) auf r = {m['rp']:.1f} mm bei 90°, 210°, 330°.",
              f"Umkreisradius {m['rho']:.1f} mm, Seitenlänge {m['seite']:.1f} mm; eine Ecke zeigt zur Spiegelmitte (innerer Punkt r = {m['r_innen']:.1f} mm), "
              f"die beiden äußeren liegen auf r = {m['r_aussen']:.1f} mm."]
    else:
        z += [f"3 Wippen, Drehpunkt auf r = {m['rp']:.1f} mm bei 90°, 210°, 330°, Wippenlänge (Dreieck–Dreieck) {m['wippe']:.1f} mm, tangential.",
              f"Je Wippenende ein Dreieck: Umkreisradius {m['rho']:.1f} mm, Seite {m['seite']:.1f} mm, jeweils eine Ecke zeigt zur Spiegelmitte; äußerster Punkt r = {m['r_aussen']:.1f} mm."]
    z += ["", f"Durchbiegung (Oberfläche): RMS {d['rms_nm']:.1f} nm → Wellenfront RMS {d['wf_rms_nm']:.1f} nm (λ/{d['wf_lambda']:.0f}), Strehl ≈ {d['strehl']:.3f}.",
          f"zum Vergleich: ohne Refokus-Abzug {d['rms_ohne_nm']:.1f} nm, mit Refokus-Abzug {d['rms_refokus_nm']:.1f} nm.",
          "Gilt für Zenit-Lage (Eigengewicht senkrecht), dünne Platte ohne Schubverformung, vollflächiger Spiegel; Seitenlagerung und Wärmespannungen sind nicht enthalten.",
          "", "Nr.   Radius [mm]   Winkel [°]"]
    for i, (r, w) in enumerate(zip(d["radius_mm"], d["winkel"])):
        z.append(f"{i+1:>2}   {r:>9.1f}   {w:>9.1f}")
    return "\n".join(z)
