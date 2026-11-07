"""Kern: Spiegelvergleich – absolutes Q_abs, Karten, Break-even, Planetenbild mit Augenfilter.
(v8.0.0: aus newton_spiegel_rechner.py v7.11.0 ausgegliedert, Code unverändert)"""
import numpy as np
from kern_bau import fangspiegel_d
from kern_optik import _L_eff, _QABS_N, _Wb_von_S, _strehl_exakt, _wellenfronten, csf_L, mtf_aus_wellenfront, mtf_aus_wellenfront_pupille, mtf_para_obstr, seeing_mtf


_QABS_MSR = {}
_QABS_MO = {}




def _qabs_mo(eps: float) -> np.ndarray:
    key = round(min(max(float(eps), 0.0), 0.95), 4)
    if key not in _QABS_MO:
        nu = np.linspace(0.0, 1.0, _QABS_N)
        _QABS_MO[key] = np.array([mtf_para_obstr(x, key) for x in nu])
    return _QABS_MO[key]


_QABS_MTF = {}
_QABS_RHO = np.linspace(0.0, 1.0, 400)
_QABS_NUG = np.linspace(0.0, 1.0, _QABS_N)


def _qabs_mtf(S: float, eps: float, rho_s: float = 1.0) -> np.ndarray:
    """Gesamt-MTF (Aberration + Ringblende) für Strehl S und Obstruktion eps,
    exakt über die 2D-Pupillen-Autokorrelation (v7.0.0-beta). Ersetzt das
    1D-Näherungsmodell mtf_sph_rel·mtf_para_obstr, das bei S<~0,9 die MTF
    überschätzt (Strehl-Kontrolle: FFT reproduziert S exakt).
    rho_s < 1: nur die zentrale Teilöffnung ρ ≤ rho_s wird genutzt (Augenpupille)."""
    key = (round(float(S), 4), round(min(max(float(eps), 0.0), 0.95), 4),
           round(min(max(float(rho_s), 0.02), 1.0), 3))
    if key not in _QABS_MTF:
        Wb = _Wb_von_S(min(max(key[0], 1e-6), 1.0))
        W = 4.0 * Wb * (_QABS_RHO ** 4 - _QABS_RHO ** 2)
        _QABS_MTF[key] = mtf_aus_wellenfront_pupille(_QABS_RHO, W, key[1], key[2])
    return _QABS_MTF[key]


def _qabs_basis(D_mm, d_fang_mm, V, lam_nm, seeing, mit_mo=True, L=None):
    """Strehl-unabhängiger Teil: (x_ang [cyc/″], Gewicht) für Spiegel-Ø D."""
    fc = D_mm / (lam_nm * 1e-6 * 206265.0)             # cyc/arcsec
    # Nur der vom Auge sichtbare Bereich (≤70 cyc/° am Auge) wird integriert: sonst wird bei
    # großem D / kleinem V das Gitter zu grob (CSF-Gewicht sitzt bei ν ≪ 1).
    nu_max = min(1.0, 70.0 * V / 3600.0 / fc)
    nu = np.linspace(0.0, nu_max, _QABS_N)
    x  = nu * fc
    w  = csf_L(x * 3600.0 / V, L)                      # v7.5.0: L=None -> photopisch (wie bisher)
    see = seeing_mtf(x * 206265.0, seeing, lam_nm)
    return x, (_qabs_mo(d_fang_mm / D_mm) if mit_mo else 1.0) * w * see, nu


def _rho_s(D_mm, V, d_pupil):
    """Genutzter Anteil des Pupillenradius: min(1, V·d_pupil/D); d_pupil<=0 → 1."""
    if not d_pupil or d_pupil <= 0:
        return 1.0
    return float(min(1.0, V * d_pupil / D_mm))


def _mtf_eff(D_mm, S, d_fang_mm, V, d_pupil=0.0, wf=None, lam_nm=550.0):
    """(MTF-Array, genutzter Durchmesser) bei Vergrößerung V und Augenpupille
    d_pupil.  wf: optionale Foucault-Wellenfront (dict rho, W, lam)."""
    rs = _rho_s(D_mm, V, d_pupil)
    eps = d_fang_mm / D_mm
    if wf is not None:
        W = np.asarray(wf["W"], float) * (wf.get("lam", lam_nm) / lam_nm)
        m = mtf_aus_wellenfront_pupille(wf["rho"], W, min(eps, 0.95), rs)
    else:
        m = _qabs_mtf(S, eps, rs)
    return m, D_mm * rs


def qabs_zaehler(D_mm, S, d_fang_mm, V, lam_nm=550.0, seeing=0.0,
                 mtf_nu=None, d_pupil=0.0, wf=None, L=None) -> float:
    """∫ MTF_ges(ν) · CSF(ν/V) dν  über die WINKELfrequenz (absolut).
    d_pupil: Augenpupille [mm] (0 = aus): bei Austrittspupille D/V > d_pupil
    nutzt das Auge nur die Teilöffnung V·d_pupil (blendet den Spiegel ab).
    wf: optionale Foucault-Wellenfront; mtf_nu: fertige Gesamt-MTF (nur ohne Pupille)."""
    _tr = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if mtf_nu is not None and wf is None and _rho_s(D_mm, V, d_pupil) >= 0.999:
        x, base, nu = _qabs_basis(D_mm, d_fang_mm, V, lam_nm, seeing, mit_mo=False, L=L)
        return float(_tr(np.interp(nu, _QABS_NUG, mtf_nu) * base, x))
    m, D_use = _mtf_eff(D_mm, S, d_fang_mm, V, d_pupil, wf, lam_nm)
    x, base, nu = _qabs_basis(D_use, d_fang_mm, V, lam_nm, seeing, mit_mo=False, L=L)
    return float(_tr(np.interp(nu, _QABS_NUG, m) * base, x))


def qabs(D_mm, S, d_fang_mm, V, ref_D=200.0, ref_S=1.0, ref_d=0.0,
         lam_nm=550.0, seeing=0.0, mtf_nu=None, d_pupil=0.0, wf=None, L=None) -> float:
    """Absolutes Q_abs: sichtbare Detailübertragung relativ zu einem festen
    Referenzspiegel (Standard: idealer 200-mm ohne Fangspiegel = 1; der
    Referenzspiegel wird von derselben Augenpupille abgeblendet)."""
    return (qabs_zaehler(D_mm, S, d_fang_mm, V, lam_nm, seeing, mtf_nu, d_pupil, wf, L) /
            max(qabs_zaehler(ref_D, ref_S, ref_d, V, lam_nm, seeing, None, d_pupil, None, L), 1e-12))


def _wf_kugel(Wb: float, lam_nm: float = 550.0) -> dict:
    """Wellenfront der Kugel am besten Fokus als Eingabe für qabs(wf=...)."""
    rho = np.linspace(0.0, 1.0, 600)
    return dict(rho=rho, W=4.0 * Wb * (rho ** 4 - rho ** 2), lam=lam_nm)


def qvis_pupille_kurve(D_mm: float, S, V_arr, d_pupil: float,
                       lam_nm: float = 550.0, wf=None, L=None) -> np.ndarray:
    """Q_vis(V) MIT Abblenden durch die Augenpupille (v7.2.0).

    Relativ zu einem idealen Spiegel gleicher Öffnung, der von derselben
    Pupille abgeblendet wird (Paraboloid = 1).  Ist die Austrittspupille D/V
    größer als d_pupil, nutzt das Auge nur die Mittelzone ρ_s = V·d_p/D; die
    Aberration wird nur über diese Teilöffnung gerechnet (mit Refokus).
    wf: Wellenfront der Kugel (dict) oder None -> S-Modell (Kugelwellenfront
    zu S, wie beim Schieber).  Ohne Seeing, ohne Fangspiegel.
    L: ROHE Objekt-Leuchtdichte; die Netzhautleuchtdichte L_eff(V) (Étendue) wird je Stützstelle gebildet (v7.6.0).
    Auswertung auf einem groben V-Gitter (verdichtet um V = D/d_p), dazwischen
    linear interpoliert."""
    V_arr = np.asarray(V_arr, dtype=float)
    if d_pupil <= 0:
        return np.full_like(V_arr, np.nan)
    v0, v1 = float(V_arr.min()), float(V_arr.max())
    gitter = list(np.linspace(v0, v1, 40))
    vs = D_mm / d_pupil
    gitter += [v for v in np.linspace(0.75 * vs, 1.45 * vs, 22) if v0 <= v <= v1]
    gitter = np.unique(np.round(gitter, 3))
    q = np.array([qabs(D_mm, S, 0.0, float(v), D_mm, 1.0, 0.0, lam_nm, 0.0,
                       d_pupil=d_pupil, wf=wf, L=_L_eff(L, D_mm, v, d_pupil)) for v in gitter])
    return np.interp(V_arr, gitter, q)


def qabs_karte(Ds, Ss, ratio_N, V, ref, lam_nm=550.0, seeing=0.0,
               buet=50.0, tubus=0.0, bild=10.0, d_pupil=0.0) -> np.ndarray:
    """Q_abs auf einem D×S-Gitter; Fangspiegel aus D und f=ratio_N·D."""
    _tr = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nenner = qabs_zaehler(ref[0], ref[1], ref[2], V, lam_nm, seeing, None, d_pupil)
    Z = np.zeros((len(Ss), len(Ds)))
    for j, D in enumerate(Ds):
        d = fangspiegel_d(D, ratio_N * D, buet, tubus, bild)
        rs = _rho_s(D, V, d_pupil)
        x, base, nu = _qabs_basis(D * rs, d, V, lam_nm, seeing, mit_mo=False)
        for i, S in enumerate(Ss):
            Z[i, j] = (_tr(np.interp(nu, _QABS_NUG, _qabs_mtf(S, d / D, rs)) * base, x)
                       / max(nenner, 1e-12))
    return Z


def qabs_breakeven_S(D2, d2, target, V, lam_nm=550.0, seeing=0.0,
                     ref=(200.0, 1.0, 0.0), d_pupil=0.0):
    """Kleinstes S des Spiegels (D2, Fangspiegel d2), das Q_abs >= target hält.
    Rückgabe: S, oder None (selbst S=1 reicht nicht), oder 0.3 (≤0,30)."""
    f = lambda S: qabs(D2, S, d2, V, ref[0], ref[1], ref[2], lam_nm, seeing,
                       d_pupil=d_pupil)
    if f(1.0) < target:
        return None
    if f(0.3) >= target:
        return 0.3
    lo, hi = 0.3, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def qabs_aequiv_D(q_ziel, V, lam_nm=550.0, seeing=0.0, ref=(200.0, 1.0, 0.0),
                  d_pupil=0.0):
    """Durchmesser [mm] eines IDEALEN Spiegels ohne Fangspiegel mit Q_abs = q_ziel
    (Bisektion).  None, wenn q_ziel im Bereich 20…2000 mm nicht erreicht wird."""
    f = lambda D: qabs(D, 1.0, 0.0, V, ref[0], ref[1], ref[2], lam_nm, seeing,
                       d_pupil=d_pupil)
    lo, hi = 20.0, 2000.0
    if q_ziel < f(lo) or q_ziel > f(hi):
        return None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if f(mid) < q_ziel:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def planet_objekt(n=384, px_arcsec=0.1, seed=7):
    """Synthetisches Planetenscheibchen (Ø 24″) mit Bändern, ovalem Fleck und
    feinen Details bis ≈0,1″ — nur zur Veranschaulichung (v7.0.0-beta)."""
    rng = np.random.default_rng(seed)
    y, x = (np.mgrid[0:n, 0:n] - n / 2.0) * px_arcsec
    r = np.hypot(x, y); R = 12.0
    disk = (r < R).astype(float)
    lat = y / R
    img = 0.55 + 0.18 * np.sin(lat * 9.0) + 0.10 * np.sin(lat * 23.0 + 1.0)
    img += 0.08 * np.sin(lat * 41.0 + 2.0)
    img += -0.30 * np.exp(-(((x - 4.0) / 2.6) ** 2 + ((y + 4.5) / 1.3) ** 2))   # roter Fleck
    for _ in range(140):                                                       # feine Details
        cx, cy = rng.uniform(-R, R, 2)
        s = rng.uniform(0.12, 0.28)
        img += rng.choice([-1, 1]) * rng.uniform(0.08, 0.2) * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    for k, sep in enumerate([0.25, 0.4, 0.6, 0.9]):                           # Doppelpunkte
        yy = -8.0 + 2.4 * k
        for sx in (-sep / 2, sep / 2):
            img -= 0.35 * np.exp(-((x + 4.0 - sx) ** 2 + (y - yy) ** 2) / (2 * 0.09 ** 2))
    img *= (1.0 - 0.35 * (r / R) ** 2)                                         # Randabdunklung
    img = np.clip(img, 0.0, 1.0) * disk
    # Strichgitter-Testtafel links neben der Scheibe (Perioden in Bogensekunden):
    # zeigt Auflösung und Mittelfrequenz-Kontrast, wo sich S bemerkbar macht
    for per, y0 in [(0.30, -9.0), (0.45, -4.5), (0.70, 0.0), (1.10, 4.5)]:
        m = (x > -19.0) & (x < -14.5) & (y > y0) & (y < y0 + 3.0)
        img[m] = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * (x[m] + 19.0) / per))
    return img


def augen_filter(f_cyc_arcsec, V: float):
    """Augen-Tiefpass bei Vergrößerung V (v7.0.0-beta): CSF (Mannos/Sakrison) normiert auf ihr
    Maximum, nur oberhalb des Maximums wirksam (darunter 1, damit Scheibe/Bänder
    nicht als Kantenbild erscheinen).  f im Himmel [cyc/″] → Auge [cyc/°] = f·3600/V."""
    fe = np.maximum(np.asarray(f_cyc_arcsec, float) * 3600.0 / max(V, 1e-9), 1e-6)
    g = np.linspace(0.2, 60.0, 600)
    cg = 2.6 * (0.0192 + 0.114 * g) * np.exp(-((0.114 * g) ** 1.1))
    fpk, cpk = float(g[int(np.argmax(cg))]), float(cg.max())
    c = 2.6 * (0.0192 + 0.114 * fe) * np.exp(-((0.114 * fe) ** 1.1))
    return np.where(fe > fpk, c / cpk, 1.0)


def bild_mit_mtf(obj, px_arcsec, D_mm, S, d_fang_mm, lam_nm=550.0, seeing=0.0,
                 mtf_nu=None, V_auge=None):
    """Objekt mit der Gesamt-MTF (Aberration × Ringblende × Seeing) falten."""
    n0 = obj.shape[0]
    n = 2 * n0                                            # Null-Auffüllung gegen Umlauf-Artefakte
    pad = np.zeros((n, n)); pad[n0 // 2:n0 // 2 + n0, n0 // 2:n0 // 2 + n0] = obj
    fx = np.fft.fftfreq(n, d=px_arcsec)                   # cyc/arcsec
    fr = np.hypot(*np.meshgrid(fx, fx))
    fc = D_mm / (lam_nm * 1e-6 * 206265.0)
    nu = np.linspace(0.0, 1.0, _QABS_N)
    m1 = mtf_nu if mtf_nu is not None else _qabs_mtf(S, d_fang_mm / D_mm)
    mtf = np.interp(fr / fc, nu, m1, right=0.0) * seeing_mtf(fr * 206265.0, seeing, lam_nm)
    if V_auge:                                   # Auge bei Vergrößerung V
        mtf = mtf * augen_filter(fr, V_auge)
    out = np.real(np.fft.ifft2(np.fft.fft2(pad) * mtf))
    return out[n0 // 2:n0 // 2 + n0, n0 // 2:n0 // 2 + n0]


def vergleich_achsen(fig, extra=False):
    """Raster: oben Q_abs(V) und Karte, Mitte drei Bildfelder, optional unten
    MTF- und Wellenfront-Vergleich (wenn eine Foucault-Wellenfront benutzt wird)."""
    if extra:
        gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 1.0, 1.1])
    else:
        gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
    ax1 = fig.add_subplot(gs[0, 0:3]); ax2 = fig.add_subplot(gs[0, 3:6])
    axb = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
           fig.add_subplot(gs[1, 4:6])]
    axw = ([fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6])]
           if extra else None)
    return ax1, ax2, axb, axw


def _wf_mtf(wf, D, d_fang, lam):
    """MTF-Gitter aus einer Foucault-Wellenfront (dict rho, W, lam) für den
    Spiegel (D, Fangspiegel d_fang); W [λ_Messung] → [λ_vis]."""
    if wf is None:
        return None
    W = np.asarray(wf["W"], float) * (wf.get("lam", lam) / lam)
    return mtf_aus_wellenfront(wf["rho"], W, min(d_fang / D, 0.95))


def vergleich_zeichnen(ax1, ax2, p: dict, axb=None, axw=None):
    """Zeichnet links Q_abs(V) für Spiegel 1/2, rechts die D×S-Karte, darunter
    Bildsimulation und (mit Foucault-Wellenfront) MTF/Wellenfront-Vergleich.
    p: V, seeing, refD, refS, N, D1, S1, D2, S2, lam, buet, tubus, bild,
       optional wf1/wf2 = dict(rho, W, lam) aus dem Foucault-Test.
    Rückgabe: Textzeilen (Ergebnis)."""
    lam, see, V = p["lam"], p["seeing"], p["V"]
    ref = (p["refD"], p["refS"], 0.0)
    dp = float(p.get("pupille", 0.0) or 0.0)
    kw = dict(lam_nm=lam, seeing=see, d_pupil=dp)
    geo = (p["buet"], p["tubus"], p["bild"])
    p = dict(p)
    for _k in ("1", "2"):                       # Option: Spiegel sphärisch (aus D, f/D)
        if p.get("sph" + _k):
            _Dm = p["D" + _k]
            _, _, _Wb, _, _ = _wellenfronten(_Dm, p["N"] * _Dm, lam)
            _r = np.linspace(0.0, 1.0, 600)
            p["wf" + _k] = dict(rho=_r, W=4.0 * _Wb * (_r ** 4 - _r ** 2), lam=lam, name="sphärische")
            p["S" + _k] = _strehl_exakt(_Wb)
    wf1, wf2 = p.get("wf1"), p.get("wf2")
    _nm = lambda w: (w or {}).get("name", "Foucault")
    d1 = fangspiegel_d(p["D1"], p["N"] * p["D1"], *geo)
    d2 = fangspiegel_d(p["D2"], p["N"] * p["D2"], *geo)
    m1 = _wf_mtf(wf1, p["D1"], d1, lam)
    m2 = _wf_mtf(wf2, p["D2"], d2, lam)
    Vs = np.linspace(30, 500, 24)
    for (D, S, d, m, col, nm, wf) in [
            (p["D1"], p["S1"], d1, m1, "#1565c0", "Spiegel 1", wf1),
            (p["D2"], p["S2"], d2, m2, "#d84315", "Spiegel 2", wf2)]:
        quelle = f"{_nm(wf)}-Wellenfront" if wf is not None else f"S={S:.2f}"
        ax1.plot(Vs, [qabs(D, S, d, v, *ref, wf=wf, **kw) for v in Vs], color=col,
                 lw=1.8, label=f"{nm}: {D:.0f} mm  {quelle}  (Fangsp. {d:.0f} mm)")
    ax1.axhline(1.0, color="#555", ls=":", lw=1.0,
                label=f"Referenz: idealer {p['refD']:.0f} mm (S={p['refS']:.2f})")
    ax1.axvline(V, color="#999", lw=0.8, ls=":")
    ax1.set_xlabel("Vergrößerung V [×]"); ax1.set_ylabel("Q_abs  (Referenz = 1)")
    ax1.set_title("Q_abs(V)" + (f"  —  Seeing {see:.1f}\"" if see > 0 else "  —  ohne Seeing"),
                  fontsize=10)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7.5, loc="best")
    lo, hi = ax1.get_ylim()
    if hi - lo < 0.3:                       # keine übertriebene Vergrößerung kleiner Unterschiede
        mid = 0.5 * (lo + hi); ax1.set_ylim(mid - 0.15, mid + 0.15)

    Ds = np.linspace(100, 500, 17); Ss = np.linspace(0.5, 1.0, 11)
    Z = qabs_karte(Ds, Ss, p["N"], V, ref, lam, see, *geo, d_pupil=dp)
    cf = ax2.contourf(Ds, Ss, Z, levels=14, cmap="viridis")
    ax2.figure.colorbar(cf, ax=ax2, label="Q_abs")
    q1 = qabs(p["D1"], p["S1"], d1, V, *ref, wf=wf1, **kw)
    q2 = qabs(p["D2"], p["S2"], d2, V, *ref, wf=wf2, **kw)
    try:
        cs = ax2.contour(Ds, Ss, Z, levels=[q1], colors="white", linewidths=2.4)
        ax2.clabel(cs, fmt=lambda v: f"= Spiegel 1 ({v:.2f})", fontsize=8)
    except Exception:
        pass
    ax2.plot([p["D1"]], [min(max(p["S1"], 0.5), 1.0)], "o", color="#1565c0",
             mec="white", ms=9, label="Spiegel 1")
    ax2.plot([p["D2"]], [min(max(p["S2"], 0.5), 1.0)], "s", color="#d84315",
             mec="white", ms=9, label="Spiegel 2")
    ax2.set_xlabel(f"Öffnung D [mm]  (f/{p['N']:g}, Fangspiegel automatisch; Karte = S-Modell)")
    ax2.set_ylabel("Strehl S")
    ax2.set_title(f"Q_abs-Karte bei {V:.0f}×  —  weiße Linie = gleich gut wie Spiegel 1",
                  fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")

    # äquivalente ideale Durchmesser
    e1 = qabs_aequiv_D(q1, V, lam, see, ref, dp)
    e2 = qabs_aequiv_D(q2, V, lam, see, ref, dp)
    _et = lambda e: "außerhalb 20–2000 mm" if e is None else f"{e:.0f} mm"

    # Bildvergleich
    if axb is not None:
        px = 0.1
        obj = planet_objekt(384, px)
        va = V if p.get("auge", True) else None
        mi1, Du1 = _mtf_eff(p["D1"], p["S1"], d1, V, dp, wf1, lam)
        mi2, Du2 = _mtf_eff(p["D2"], p["S2"], d2, V, dp, wf2, lam)
        b1 = bild_mit_mtf(obj, px, Du1, p["S1"], d1, lam, see, mtf_nu=mi1, V_auge=va)
        b2 = bild_mit_mtf(obj, px, Du2, p["S2"], d2, lam, see, mtf_nu=mi2, V_auge=va)
        vmax = float(obj.max())
        # Darstellung im AUGENmaßstab: der Planet wird mit V größer (feste 1 ′-Augenauflösung
        # ⇒ bei kleinem V ist er klein und wirkt scharf, aber ohne Feinstruktur).
        augenmass = bool(va) and p.get("augenmass", True)
        if augenmass:
            h = 19.2 * V / 60.0                      # halbe Feldbreite des Bildes am Auge [′]
            ext = [-h, h, -h, h]
            win = max(60.0, h)                       # Fenster ±60′ (wächst erst ab V>190)
        else:
            ext = [-19.2, 19.2, -19.2, 19.2]
            win = 19.2
        t1 = f"{_nm(wf1)}-Wellenfront" if wf1 is not None else f"S={p['S1']:.2f}"
        t2 = f"{_nm(wf2)}-Wellenfront" if wf2 is not None else f"S={p['S2']:.2f}"
        for a, im, tt in [(axb[0], b1, f"Spiegel 1  ({p['D1']:.0f} mm, {t1})"),
                          (axb[1], b2, f"Spiegel 2  ({p['D2']:.0f} mm, {t2})")]:
            a.set_facecolor("black")
            a.imshow(im, cmap="gray", vmin=0, vmax=vmax, extent=ext, interpolation="bicubic")
            a.set_xlim(-win, win); a.set_ylim(-win, win)
            a.set_title(tt, fontsize=9); a.set_xticks([]); a.set_yticks([])
        diff = b2 - b1
        lim = max(float(np.max(np.abs(diff))), 1e-3)
        axb[2].set_facecolor("#f7f7f7")
        axb[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext,
                      interpolation="bicubic")
        axb[2].set_xlim(-win, win); axb[2].set_ylim(-win, win)
        axb[2].set_title(f"Differenz Sp.2 − Sp.1  (rot = 2 heller, Skala ±{lim*100:.0f} % der Helligkeit)",
                         fontsize=9)
        axb[2].set_xticks([]); axb[2].set_yticks([])
        axb[0].set_xlabel("Planet Ø 24″ + Gitter 0,30/0,45/0,70/1,10″;  "
                          + ((f"Auge bei {V:.0f}×, Augenmaßstab (Fenster ±{win:.0f}′)" if augenmass
                              else f"Auge bei {V:.0f}×, Himmelsmaßstab") if va else "ohne Augen-Filter"),
                          fontsize=7.5)

    # Wellenfront-/MTF-Vergleich (nur mit Foucault-Daten)
    zusatz = []
    if axw is not None:
        nu = np.linspace(0.0, 1.0, _QABS_N)
        for (wf, D, S, d, m, nm, col) in [(wf1, p["D1"], p["S1"], d1, m1, "Spiegel 1", "#1565c0"),
                                          (wf2, p["D2"], p["S2"], d2, m2, "Spiegel 2", "#d84315")]:
            if wf is None:
                continue
            m_s = _qabs_mtf(S, d / D)
            m_id = _qabs_mtf(1.0, d / D)
            axw[0].plot(nu, m, color=col, lw=2.0, label=f"{nm}: aus {_nm(wf)}-Wellenfront")
            axw[0].plot(nu, m_s, color=col, lw=1.3, ls="--",
                        label=f"{nm}: S-Modell (S={S:.3f})")
            axw[0].plot(nu, m_id, color="#2e7d32", lw=1.0, ls=":",
                        label="ideal mit Fangspiegel" if nm == "Spiegel 1" else None)
            axw[1].plot(wf["rho"], np.asarray(wf["W"]) * (wf.get("lam", lam) / lam),
                        color=col, lw=1.8, label=f"{nm}: W(ρ) bester Fokus")
            qS = qabs(D, S, d, V, *ref, **kw)
            qW = qabs(D, S, d, V, *ref, wf=wf, **kw)
            zusatz.append(f"{nm}: Q_abs aus {_nm(wf)}-Wellenfront {qW:.3f}  vs. S-Modell (S={S:.3f}) {qS:.3f}  "
                          f"(Differenz {100*(qW/max(qS,1e-12)-1):+.1f} %)")
        axw[0].set_xlabel("normierte Ortsfrequenz ν/ν_c"); axw[0].set_ylabel("MTF")
        axw[0].set_title("MTF: echte Wellenfrontform vs. S-Modell (inkl. Fangspiegel)", fontsize=10)
        axw[0].grid(alpha=0.3); axw[0].legend(fontsize=7.5)
        axw[1].set_xlabel("Pupillenradius ρ"); axw[1].set_ylabel("W [λ]")
        axw[1].set_title("Wellenfront (Foucault-Test bzw. Kugel; kolbenfrei, bester Fokus)", fontsize=10)
        axw[1].grid(alpha=0.3); axw[1].legend(fontsize=7.5)

    be = qabs_breakeven_S(p["D2"], d2, q1, V, lam, see, ref, dp)
    if be is None:
        bt = "selbst ein perfekter Spiegel 2 erreicht Spiegel 1 nicht"
    elif be <= 0.3001:
        bt = "Spiegel 2 erreicht Spiegel 1 selbst bei S ≤ 0,30 (Modell dort unsicher)"
    else:
        bt = f"Spiegel 2 braucht mindestens S = {be:.2f}, um Spiegel 1 zu erreichen"
    pup = []
    if dp > 0:
        for nm, Dm in (("Spiegel 1", p["D1"]), ("Spiegel 2", p["D2"])):
            ep = Dm / V
            pup.append(f"Augenpupille {dp:.1f} mm bei {V:.0f}×: {nm} hat Austrittspupille {ep:.1f} mm → "
                       + (f"Auge blendet auf {V*dp:.0f} mm ab ({100*V*dp/Dm:.0f} % der Öffnung)"
                          if ep > dp else "volle Öffnung wird genutzt"))
    return (pup + [f"V={V:.0f}×   Seeing {'aus' if see <= 0 else f'{see:.1f}″'}   Referenz: idealer {p['refD']:.0f} mm = 1",
             f"Spiegel 1: {p['D1']:.0f} mm, S={p['S1']:.2f}{(' (' + _nm(wf1) + ')') if wf1 is not None else ''}, Fangspiegel {d1:.0f} mm  →  Q_abs = {q1:.3f}",
             f"Spiegel 2: {p['D2']:.0f} mm, S={p['S2']:.2f}{(' (' + _nm(wf2) + ')') if wf2 is not None else ''}, Fangspiegel {d2:.0f} mm  →  Q_abs = {q2:.3f}   "
             f"(Verhältnis 2/1 = {q2/max(q1,1e-12):.2f})",
             f"Entspricht einem idealen Spiegel (ohne Fangspiegel) von:  Spiegel 1 ≙ {_et(e1)},  Spiegel 2 ≙ {_et(e2)}",
             bt] + zusatz)
