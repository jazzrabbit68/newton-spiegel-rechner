"""
Newton-Teleskop: Kontrast- und Schärfeverlust sphärischer Hauptspiegel
=======================================================================
Streamlit-Version v6.1.0 (2026-06-21)
Entspricht newton_spiegel_rechner.py v6.1.0

Abhängigkeiten: pip install streamlit matplotlib numpy scipy
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
from functools import lru_cache

# kein scipy nötig – eigene goldene-Schnitt-Minimierung
_SCIPY_OK = True  # immer True

def _minimize_scalar_bounded(f, bounds, tol=1e-8, maxiter=200):
    """Goldene-Schnitt-Suche auf Intervall [a, b]."""
    a, b = bounds
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    for _ in range(maxiter):
        if abs(b - a) < tol:
            break
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    class _Res:
        x = (a + b) / 2
    return _Res()

VERSION = "6.1.0 (2026-06-21)"
EYE_RES = 60.0   # Augenauflösung [arcsec]
BG  = "#f5f5f5"
ACC = "#534AB7"
COR = "#D85A30"
GRN = "#0F6E56"

# ══════════════════════════════════════════════════════════════════════════════
# Kernrechnung (identisch mit newton_spiegel_rechner.py)
# ══════════════════════════════════════════════════════════════════════════════

def _wellenfronten(D_mm: float, f_mm: float, lam_nm: float):
    lam_mm = lam_nm * 1e-6
    Wp   = D_mm**4 / (1024.0 * f_mm**3 * lam_mm)
    Wb   = Wp / 4.0
    Wrms = Wb / (1.5 * math.sqrt(5))
    S    = math.exp(-(2 * math.pi * Wrms) ** 2)
    return lam_mm, Wp, Wb, Wrms, S


@st.cache_data(max_entries=2048)
def _strehl_exakt(Wb: float, n: int = 2000) -> float:
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    rho   = np.linspace(0.0, 1.0, n)
    W     = 4.0 * Wb * (rho**4 - rho**2)
    W_m   = float(_trapz(W * rho, rho) / 0.5)
    Wc    = W - W_m
    phase = 2.0 * math.pi * Wc
    A_re  = float(_trapz(np.cos(phase) * rho, rho)) / 0.5
    A_im  = float(_trapz(np.sin(phase) * rho, rho)) / 0.5
    return A_re**2 + A_im**2


def _Wb_von_S(S: float, n: int = 60) -> float:
    if S >= 0.9999:
        return 0.0
    lo, hi = 0.0, 0.44
    for _ in range(n):
        mid = (lo + hi) / 2.0
        if _strehl_exakt(mid) > S:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _kennzahlen_von_S(D_mm: float, S: float, Wb: float) -> dict:
    sqrtS       = math.sqrt(max(S, 1e-9))
    Deff_k      = D_mm * S**0.25
    Deff_s      = D_mm * sqrtS
    theta_ideal = 116.0 / D_mm
    theta_eff   = theta_ideal / sqrtS
    V_krit_vis  = D_mm * 0.7 * sqrtS
    Q02 = mtf_sph_rel(0.2, Wb)
    Q04 = mtf_sph_rel(0.4, Wb)
    Q06 = mtf_sph_rel(0.6, Wb)
    Qshape  = 0.4*Q02 + 0.4*Q04 + 0.2*Q06
    Qvis    = S * Qshape
    Deff_vis = D_mm * Qvis**0.25
    return dict(strehl=S, Wb=Wb,
                Deff_k=Deff_k,   loss_k=D_mm - Deff_k,
                Deff_s=Deff_s,   loss_s=D_mm - Deff_s,
                Deff_vis=Deff_vis, loss_vis=D_mm - Deff_vis,
                theta_ideal=theta_ideal, theta_eff=theta_eff,
                V_krit_vis=V_krit_vis,
                Q02=Q02, Q04=Q04, Q06=Q06, Qshape=Qshape, Qvis=Qvis)


def berechne(D_mm: float, f_mm: float, lam_nm: float) -> dict:
    lam_mm, Wp, Wb, Wrms, S_mar = _wellenfronten(D_mm, f_mm, lam_nm)
    S_exakt = _strehl_exakt(Wb)
    r = _kennzahlen_von_S(D_mm, S_exakt, Wb)
    r["strehl_marechal"] = S_mar
    d_blur_mm = D_mm**3 / (64.0 * f_mm**2)
    d_blur_as = d_blur_mm / f_mm * 206265.0
    theta_geo = math.sqrt(r["theta_ideal"]**2 + d_blur_as**2)
    r.update(
        N=f_mm/D_mm, D_inch=D_mm/25.4, lam_mm=lam_mm,
        Wp=Wp, Wrms=Wrms,
        r_airy_as=1.22 * lam_mm / D_mm * 206265.0,
        d_blur_mm=d_blur_mm, d_blur_as=d_blur_as,
        aufl_verlust_pct=(1.0 - r["theta_ideal"] / theta_geo) * 100.0,
    )
    return r


def berechne_vergr(D_mm, f_mm, lam_nm, V, eye_res_as=60.0):
    r = berechne(D_mm, f_mm, lam_nm)
    theta_auge = eye_res_as / V
    theta_para = max(r["theta_ideal"], theta_auge)
    theta_sph  = max(r["theta_eff"],   theta_auge)
    Deff_para  = min(D_mm, 116.0 / theta_para)
    Deff_sph   = min(D_mm, 116.0 / theta_sph)
    r.update(theta_auge=theta_auge, theta_para=theta_para, theta_sph=theta_sph,
             Deff_para=Deff_para, Deff_sph=Deff_sph, verlust=Deff_para - Deff_sph)
    return r


def v_kritisch(D_mm, f_mm, lam_nm, eye_res_as=60.0):
    r = berechne(D_mm, f_mm, lam_nm)
    return dict(Vk_sph=r["V_krit_vis"], Vk_para=D_mm*0.7,
                Vmax=D_mm/0.5, Vmin=D_mm/7.0)


def v_krit_blur_direkt(D_mm, f_mm, eye_res_as=60.0):
    d_blur_mm     = D_mm**3 / (64.0 * f_mm**2)
    alpha_blur_as = d_blur_mm / f_mm * 206265.0
    if alpha_blur_as <= 0:
        return 1e9
    return eye_res_as / alpha_blur_as


def v_kritisch_strehl(D_mm, strehl):
    return dict(Vk_sph=D_mm * 0.7 * math.sqrt(max(strehl, 1e-9)),
                Vk_para=D_mm * 0.7, Vmax=D_mm / 0.5, Vmin=D_mm / 7.0)


def v_krit_aus_qvis(D_mm, f_mm, lam_nm, rel_schwelle=0.90,
                    V_min=30.0, V_max=600.0, n_pts=60):
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu     = np.linspace(0.0, 1.0, n_pts)
    mp     = np.array([mtf_para(f)        for f in nu])
    msr    = np.array([mtf_sph_rel(f, Wb) for f in nu])
    def qvis_rel(V):
        w = np.array([csf(fi * 3600.0 / V) for fi in nu * fc])
        return float(_trapz(msr * mp * w, nu) / max(_trapz(mp * w, nu), 1e-12))
    q_low = qvis_rel(V_min)
    if q_low < rel_schwelle:
        return V_min
    q_high = qvis_rel(V_max)
    if q_high >= rel_schwelle:
        return V_max
    lo, hi = V_min, V_max
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if qvis_rel(mid) >= rel_schwelle:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def v_krit_aus_qvis_von_Wb(D_mm, Wb, lam_nm, rel_schwelle=0.90,
                            V_min=30.0, V_max=600.0, n_pts=60):
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu     = np.linspace(0.0, 1.0, n_pts)
    mp     = np.array([mtf_para(f)        for f in nu])
    msr    = np.array([mtf_sph_rel(f, Wb) for f in nu])
    def qvis_rel(V):
        w = np.array([csf(fi * 3600.0 / V) for fi in nu * fc])
        return float(_trapz(msr * mp * w, nu) / max(_trapz(mp * w, nu), 1e-12))
    q_low = qvis_rel(V_min)
    if q_low < rel_schwelle:
        return V_min
    q_high = qvis_rel(V_max)
    if q_high >= rel_schwelle:
        return V_max
    lo, hi = V_min, V_max
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if qvis_rel(mid) >= rel_schwelle:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def v_halb_exakt(D_mm, f_mm, lam_nm, V_min=10.0, V_max=800.0, n=50):
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    S_ex     = _strehl_exakt(Wb)
    schwelle = (1.0 + S_ex) / 2.0
    fc       = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz   = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu       = np.linspace(0.0, 1.0, 80)
    mp       = np.array([mtf_para(f)        for f in nu])
    msr      = np.array([mtf_sph_rel(f, Wb) for f in nu])
    def q(V):
        w = np.array([csf(fi * 3600.0 / V) for fi in nu * fc])
        return float(_trapz(msr * mp * w, nu) / max(_trapz(mp * w, nu), 1e-12))
    if q(V_min) <= schwelle: return V_min
    if q(V_max) >= schwelle: return V_max
    lo, hi = V_min, V_max
    for _ in range(n):
        mid = (lo + hi) / 2.0
        if q(mid) > schwelle: lo = mid
        else: hi = mid
    return (lo + hi) / 2.0


def mtf_para(f: float) -> float:
    """MTF einer kreisförmigen Apertur (Einheitskreis).
    OTF(f) = (2/π)·(acos(f) − f·√(1−f²)),  normiert auf OTF(0)=1.
    Die Formel ergibt bei f=0 bereits 1.0; kein weiterer Normierungsfaktor nötig.
    """
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    return (2.0 / math.pi) * (math.acos(f) - f * math.sqrt(max(0.0, 1.0 - f**2)))


def mtf_para_obstr(f: float, eps: float = 0.0) -> float:
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    def seg(r, fv):
        """Kreissegment-Integralterm; fv=0 ergibt Kreisfläche."""
        if fv <= 0:
            # f→0: ganzer Kreisanteil = π*r² (normiert auf Einheitskreis)
            return math.pi * r**2 if r <= 1.0 else math.pi
        hf = fv / 2.0
        if r <= hf: return 0.0
        x1 = max(-1.0, min(1.0, (r**2 + hf**2 - 1.0) / (2.0 * r * hf)))
        x2 = max(-1.0, min(1.0, (r**2 + 1.0 - hf**2) / (2.0 * r)))
        disc = max(0.0, (r + hf + 1.0) * (r + hf - 1.0) *
                        (1.0 - r + hf) * (1.0 + r - hf))
        return r**2 * math.acos(x1) + hf**2 * math.acos(x2) - 0.5 * math.sqrt(disc)
    def ring_area(r1, r2, fv):
        return seg(r2, fv) - seg(r1, fv) if r2 > r1 else 0.0
    A_full = ring_area(eps, 1.0, f)
    A_norm = ring_area(eps, 1.0, 0.0)
    if A_norm < 1e-10: return 0.0
    return A_full / A_norm


def mtf_sph_rel(f: float, Wb: float, n: int = 80, paraxial: bool = False) -> float:
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    lim = math.sqrt(max(0.0, 1.0 - (f/2)**2))
    re = im = norm = 0.0
    Wp = Wb * 8
    for i in range(n):
        x  = -lim + (2*i + 1) / (2*n) * 2*lim
        x1 = x - f/2
        x2 = x + f/2
        if x1**2 >= 1 or x2**2 >= 1:
            continue
        h  = min(math.sqrt(max(0.0, 1 - x1**2)),
                 math.sqrt(max(0.0, 1 - x2**2)))
        if paraxial:
            W1 = Wp * (x1**4)
            W2 = Wp * (x2**4)
        else:
            W1 = Wp * (x1**4 - x1**2)
            W2 = Wp * (x2**4 - x2**2)
        dW = 2 * math.pi * (W1 - W2)
        re   += math.cos(dW) * h
        im   += math.sin(dW) * h
        norm += h
    if norm < 1e-10:
        return 0.0
    return math.sqrt(re**2 + im**2) / norm


def csf(f_cpd: float) -> float:
    f = max(f_cpd, 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))


_L_REF = 100.0


def csf_lum(f_cpd: float, L: float) -> float:
    L = max(L, 1e-4)
    f_scale = (L / _L_REF) ** 0.1
    a_scale = (L / _L_REF) ** 0.5
    return csf(f_cpd * f_scale) * a_scale


def perceived_quality(D_mm, f_mm, lam_nm, V, n_pts=120):
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    S = _strehl_exakt(Wb)
    fc = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu = np.linspace(0.0, 1.0, n_pts)
    mp  = np.array([mtf_para(f)        for f in nu])
    msr = np.array([mtf_sph_rel(f, Wb) for f in nu])
    msa = msr * mp * S
    w   = np.array([csf(fi * 3600.0 / V) for fi in nu * fc])
    Q_raw = float(_trapz(msa * w, nu) / max(_trapz(mp * w, nu), 1e-12))
    V_krit = D_mm * 0.7 * math.sqrt(max(S, 1e-9))
    wv = 1.0 / (1.0 + (V_krit / V) ** 2)
    return float(Q_raw + (1.0 - Q_raw) * (1.0 - wv))


def perceived_quality_exakt_kurven(D_mm, f_mm, lam_nm, V_arr, n_pts=80):
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    lam_mm  = lam_nm * 1e-6
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    fc      = D_mm / (lam_mm * 206265)
    nu      = np.linspace(0.0, 1.0, n_pts)
    mp      = np.array([mtf_para(f)        for f in nu])
    msr     = np.array([mtf_sph_rel(f, Wb) for f in nu])
    V_arr   = np.asarray(V_arr)
    f_cpd   = np.maximum(nu[:, None] * fc * 3600.0 / V_arr[None, :], 1e-6)
    csf_mat = 2.6 * (0.0192 + 0.114 * f_cpd) * np.exp(-((0.114 * f_cpd) ** 1.1))
    num = _trapz(msr[:, None] * mp[:, None] * csf_mat, nu, axis=0)
    den = _trapz(mp[:, None] * csf_mat, nu, axis=0)
    return num / np.maximum(den, 1e-12)


def perceived_quality_exakt_kurven_von_Wb(D_mm, Wb, lam_nm, V_arr, n_pts=80):
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    fc      = D_mm / (lam_nm * 1e-6 * 206265)
    nu      = np.linspace(0.0, 1.0, n_pts)
    mp      = np.array([mtf_para(f)        for f in nu])
    msr     = np.array([mtf_sph_rel(f, Wb) for f in nu])
    V_arr   = np.asarray(V_arr)
    f_cpd   = np.maximum(nu[:, None] * fc * 3600.0 / V_arr[None, :], 1e-6)
    csf_mat = 2.6 * (0.0192 + 0.114 * f_cpd) * np.exp(-((0.114 * f_cpd) ** 1.1))
    num = _trapz(msr[:, None] * mp[:, None] * csf_mat, nu, axis=0)
    den = _trapz(mp[:, None] * csf_mat, nu, axis=0)
    return num / np.maximum(den, 1e-12)


def kurven_N(D_mm, lam_nm, N_min=3.0, N_max=15.0, schritte=150):
    ns = np.linspace(N_min, N_max, schritte)
    strehls, deff_ks, aufl_verluste, strehls_exakt = [], [], [], []
    for n in ns:
        r = berechne(D_mm, D_mm * n, lam_nm)
        strehls.append(r["strehl"])
        deff_ks.append(r["Deff_k"])
        aufl_verluste.append(r["aufl_verlust_pct"])
        strehls_exakt.append(_strehl_exakt(r["Wb"]))
    return ns, np.array(strehls), np.array(deff_ks), np.array(aufl_verluste), np.array(strehls_exakt)


def d_fang_min(D_mm, f_mm):
    return (D_mm**2 + 100.0 * D_mm) / (2.0 * f_mm) + 10.0


def Q_vis_blende_batch(D_mm, f_mm, lam_nm, V_arr, L_arr, D_ref=200.0, n=60):
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    d_fang  = d_fang_min(D_ref, f_mm)
    eps     = min(d_fang / D_mm, 0.99)
    fc_orig = D_ref / (lam_nm * 1e-6 * 206265)
    fc_bl   = D_mm  / (lam_nm * 1e-6 * 206265)
    f_ratio = fc_bl / fc_orig
    nu      = np.linspace(0.0, 1.0, n)
    nu_bl   = nu / max(f_ratio, 1e-6)
    mp  = np.array([mtf_para_obstr(min(x, 1.0), eps) for x in nu_bl])
    ms  = np.array([mtf_sph_rel(min(x, 1.0), Wb)     for x in nu_bl])
    mask = nu > f_ratio; mp[mask] = 0.0; ms[mask] = 0.0
    ms_mp = ms * mp
    V_arr = np.asarray(V_arr); L_arr = np.asarray(L_arr)
    L_ok  = (L_arr * (D_mm / D_ref) ** 2)[None, None, :]
    nu_g  = nu[:, None, None]
    V_g   = V_arr[None, :, None]
    f_cpd = nu_g * fc_orig * 3600.0 / V_g
    L_s   = np.maximum(L_ok, 1e-4)
    f_sc  = (L_s / 100.0) ** 0.1
    a_sc  = (L_s / 100.0) ** 0.5
    f_eff = np.maximum(f_cpd * f_sc, 1e-6)
    csf_m = 2.6 * (0.0192 + 0.114 * f_eff) * np.exp(-((0.114 * f_eff) ** 1.1)) * a_sc
    return _trapz(ms_mp[:, None, None] * csf_m, nu, axis=0)


def beurteilung(strehl):
    if strehl >= 0.95:
        return "✓ Sehr gut — nahezu gleichwertig mit Parabolspiegel (Strehl ≥ 0.95)", "#2e7d32"
    elif strehl >= 0.80:
        return "⚠  Noch beugungsbegrenzt (Rayleigh), spürbarer Kontrastverlust bei Planeten  (0.80 ≤ Strehl < 0.95)", "#e65100"
    else:
        return "✗  Nicht beugungsbegrenzt — erheblicher Kontrast- und Schärfeverlust, Parabolspiegel dringend empfohlen (Strehl < 0.80)", "#c62828"


# Foucault-Funktionen
def foucault_zonen(D_mm, n=4):
    R = D_mm / 2.0
    return R * np.sqrt((2 * np.arange(1, n + 1) - 1) / (2 * n))


def foucault_zonengrenzen(D_mm, n=4):
    R = D_mm / 2.0
    return R * np.sqrt(np.arange(1, n + 1) / n)


def foucault_wellenfront(r_zonen, df_zonen, f_mm, lam_nm):
    lam_mm      = lam_nm * 1e-6
    dl_para     = r_zonen**2 / (4.0 * f_mm)
    dl_para_rel = dl_para - dl_para[0]
    delta_l     = dl_para_rel - df_zonen
    return delta_l * r_zonen**2 / (8.0 * f_mm**2 * lam_mm)


def foucault_kennzahlen(W_zonen, r_zonen, D_mm, lam_nm=None):
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    R    = D_mm / 2.0
    rho_z = r_zonen / R
    rho_fit = np.concatenate([[0.0], rho_z])
    W_fit   = np.concatenate([[0.0], W_zonen])
    A = np.column_stack([rho_fit**4, rho_fit**2])
    coeffs, _, _, _ = np.linalg.lstsq(A, W_fit, rcond=None)
    a4, a2 = coeffs
    rho_f = np.linspace(0.0, 1.0, 2000)
    W_f   = a4 * rho_f**4 + a2 * rho_f**2

    def wrms_sq(d):
        Wt = W_f + d * rho_f**2
        Wm = _trapz(Wt * rho_f, rho_f) / 0.5
        Wc = Wt - Wm
        return _trapz(Wc**2 * rho_f, rho_f) / 0.5

    res    = _minimize_scalar_bounded(wrms_sq, bounds=(-200, 200))
    W_best = W_f + res.x * rho_f**2
    Wm     = _trapz(W_best * rho_f, rho_f) / 0.5
    W_c    = W_best - Wm
    W_rms  = math.sqrt(_trapz(W_c**2 * rho_f, rho_f) / 0.5)
    W_ptv  = float(W_c.max() - W_c.min())
    phase  = 2.0 * math.pi * W_c
    A_re   = _trapz(np.cos(phase) * rho_f, rho_f) / 0.5
    A_im   = _trapz(np.sin(phase) * rho_f, rho_f) / 0.5
    S_exakt = A_re**2 + A_im**2
    Wb_fit  = abs(a4) / 4.0
    S_mar   = math.exp(-(2 * math.pi * W_rms)**2)
    Deff_k  = D_mm * S_exakt**0.25

    Glas_f = Glas_zonen = None
    Glas_ptv_nm = W_min = None
    Glas_volumen_mm3 = None
    if lam_nm is not None:
        W_min       = float(W_c.min())
        Glas_f      = (W_c - W_min) * lam_nm / 2.0
        W_c_zonen_f = a4 * rho_z**4 + (a2 + res.x) * rho_z**2 - Wm
        Glas_zonen  = (W_c_zonen_f - W_min) * lam_nm / 2.0
        Glas_ptv_nm = float(Glas_f.max())
        Glas_mm     = Glas_f * 1e-6
        integrand   = Glas_mm * rho_f
        integral    = _trapz(integrand, rho_f)
        Glas_volumen_mm3 = 2 * math.pi * (R ** 2) * integral

    return dict(
        W_zonen=W_zonen, W_c_zonen=W_zonen - float(np.mean(W_zonen)),
        W_rms=W_rms, W_ptv=W_ptv,
        Wb_fit=Wb_fit, a4=a4, a2=a2,
        S_mar=S_mar, S_exakt=S_exakt, Deff_k=Deff_k,
        Glas_f=Glas_f, Glas_zonen=Glas_zonen,
        Glas_ptv_nm=Glas_ptv_nm, W_min=W_min, lam_nm=lam_nm,
        Glas_volumen_mm3=Glas_volumen_mm3,
        rho_f=rho_f, W_best=W_best, W_c=W_c)


# ══════════════════════════════════════════════════════════════════════════════
# Plot-Funktionen (matplotlib → Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

def _ax_fmt(ax):
    ax.grid(True, color="#ddd", lw=0.5)
    ax.set_facecolor(BG)


def plot_strehl(D, f, lam, S_slide=None):
    ns, sts, _, _, sts_exakt = kurven_N(D, lam)
    N_akt    = f / D
    r_akt    = berechne(D, f, lam)
    S_akt    = r_akt["strehl"]
    S_akt_mar = r_akt["strehl_marechal"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    ax.fill_between(ns, S_akt, 1.0, color=ACC, alpha=0.10, label="Bereich Paraboloid→Sphäre")
    ax.axhline(1.00, color=GRN,       lw=1.5, ls="-",  label="Paraboloid (S=1.0)")
    ax.plot(ns, sts,       color=ACC,       lw=2.0, ls="-",  label="Strehl Maréchal (Näherung)")
    ax.plot(ns, sts_exakt, color="#534AB7", lw=1.8, ls="--", label="Strehl exakt (Pupillenintegral)")
    ax.axhline(0.80, color="#BA7517", lw=1.2, ls="--", label="Rayleigh S=0.80")
    ax.axhline(0.95, color="#888",    lw=1.0, ls=":",  label="S=0.95")
    ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
    ax.scatter([N_akt], [S_akt_mar], color=ACC,       s=60, zorder=5)
    ax.scatter([N_akt], [S_akt],    color="#534AB7", s=40, zorder=5, marker="D")
    ax.annotate(f"f/{N_akt:.1f}  S={S_akt_mar:.3f} (Maréchal) / S={S_akt:.3f} (exakt)",
                xy=(N_akt, S_akt_mar), xytext=(8, 10),
                textcoords="offset points", fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="-", color="#aaa"))
    if S_slide is not None and S_slide > S_akt_mar + 0.005:
        ax.axhline(S_slide, color=COR, lw=1.5, ls="-.", label=f"Verbessert S={S_slide:.3f}")
        ax.annotate(f"S={S_slide:.3f}", xy=(14.5, S_slide), fontsize=9, color=COR, va="bottom")
    ax.set_xlim(3, 15); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=10)
    ax.set_ylabel("Strehl-Quotient", fontsize=10)
    ax.set_title(f"Strehl vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=9, loc="lower right")
    _ax_fmt(ax); fig.tight_layout()
    return fig


def plot_oeffnung(D, f, lam, S_slide=None):
    ns, _, deff_ks, aufl_verluste, _ = kurven_N(D, lam)
    N_akt   = f / D
    r_akt   = berechne(D, f, lam)
    Dk_akt  = r_akt["Deff_k"]
    avp_akt = r_akt["aufl_verlust_pct"]
    Dk_slide = D * S_slide**0.25 if S_slide is not None else None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    ax.fill_between(ns, deff_ks, D, color=ACC, alpha=0.10, label="Kontrast-Verlust gg. Paraboloid")
    ax.axhline(D, color="#555", lw=1.5, ls="-", label=f"Paraboloid = {D:.0f}mm")
    ax.plot(ns, deff_ks, color=ACC, lw=2, label="Eff. Öffnung (Kontrast)")
    ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
    ax.scatter([N_akt], [Dk_akt], color=ACC, s=60, zorder=5)
    ax.annotate(f"D_eff={Dk_akt:.0f}mm",
                xy=(N_akt, Dk_akt), xytext=(N_akt+0.3, Dk_akt-15), fontsize=8, color=ACC)
    if Dk_slide is not None and Dk_slide > Dk_akt + 1:
        ax.scatter([N_akt], [Dk_slide], color=GRN, s=120, zorder=6, marker="*",
                   label=f"Schieber D_eff={Dk_slide:.0f}mm")
    ax.set_xlim(3, 15)
    ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=10)
    ax.set_ylabel("Effektive Öffnung Kontrast [mm]", fontsize=10, color=ACC)
    ax.tick_params(axis="y", labelcolor=ACC)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.fill_between(ns, aufl_verluste, 0, color=COR, alpha=0.08)
    ax2.plot(ns, aufl_verluste, color=COR, lw=2, label="Aufl.verlust gg. Paraboloid [%]")
    ax2.scatter([N_akt], [avp_akt], color=COR, s=60, zorder=5)
    ax2.annotate(f"{avp_akt:.0f}%",
                 xy=(N_akt, avp_akt), xytext=(N_akt+0.3, avp_akt+2), fontsize=8, color=COR)
    ax2.axhline(50, color=COR, lw=0.8, ls=":", alpha=0.5)
    ax2.set_ylabel("Auflösungsverlust [%]", fontsize=10, color=COR)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor=COR)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="center right")
    ax.set_title(f"Eff. Öffnung & Aufl.verlust vs. f/D  (D={D:.0f}mm)", fontsize=10)
    _ax_fmt(ax); fig.tight_layout()
    return fig


def plot_deff_D(D_akt, N_akt, S_slide=None):
    D_arr    = np.linspace(50, 400, 300)
    N_list   = [5, 6, 7, 8, 10]
    colors_N = {5:"#c62828", 6:"#D85A30", 7:"#c8a020", 8:"#2e7d32", 10:"#1565c0"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    for N in N_list:
        loss_pct, loss_qvis = [], []
        for D in D_arr:
            r = berechne(D, D * N, 550.0)
            loss_pct.append((1 - r["strehl"]**0.25) * 100)
            loss_qvis.append((1 - r["Qvis"]**0.25) * 100)
        col = colors_N[N]
        ax.plot(D_arr, loss_pct,  color=col, lw=2,   label=f"f/{N}")
        ax.plot(D_arr, loss_qvis, color=col, lw=1.2, ls="--", alpha=0.6)
        ax.text(D_arr[-1] + 3, loss_pct[-1], f"f/{N}",
                color=col, fontsize=9, va="center", fontweight="bold")

    r_akt    = berechne(D_akt, D_akt * N_akt, 550.0)
    S_akt    = r_akt["strehl"]
    loss_akt = (1 - S_akt**0.25) * 100
    ax.scatter([D_akt], [loss_akt], color=ACC, s=80, zorder=6)
    ax.annotate(f"D={D_akt:.0f}mm f/{N_akt:.1f}  -{loss_akt:.0f}% Öffnung",
                xy=(D_akt, loss_akt), xytext=(D_akt + 15, loss_akt + 5),
                fontsize=9, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))
    if S_slide is not None and S_slide > S_akt + 0.01:
        loss_slide = (1 - S_slide**0.25) * 100
        ax.scatter([D_akt], [loss_slide], color=GRN, s=80, zorder=7, marker="*")
        ax.annotate(f"Verbessert S={S_slide:.2f}  -{loss_slide:.0f}%",
                    xy=(D_akt, loss_slide), xytext=(D_akt + 15, loss_slide - 6),
                    fontsize=9, color=GRN, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=GRN, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRN, alpha=0.85))
        ax.annotate("", xy=(D_akt, loss_slide), xytext=(D_akt, loss_akt),
                    arrowprops=dict(arrowstyle="->", color=GRN, lw=1.5))

    ax.axhline(20, color="#BA7517", lw=1.5, ls="--", label="20% Verlust (spürbar)")
    ax.axhline(50, color="#c62828", lw=1.0, ls=":",  label="50% Verlust (erheblich)")
    ax.text(52, 21, "20% Verlust", fontsize=8, color="#BA7517", fontweight="bold")
    ax.text(52, 51, "50% Verlust", fontsize=8, color="#c62828", fontweight="bold")
    rayleigh_loss = (1 - 0.8**0.25)*100
    ax.axhline(rayleigh_loss, color="#888", lw=1.0, ls=":")
    ax.text(52, rayleigh_loss + 1, "Rayleigh S=0.80", fontsize=8, color="#888")
    ax.set_xlim(50, 420); ax.set_ylim(0, min(100, max(90, loss_akt + 15)))
    ax.set_xlabel("Öffnung D [mm]", fontsize=10)
    ax.set_ylabel("Öffnungsverlust [%]", fontsize=10)
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0],[0], color="#555", lw=2,            label="D × S^0.25  (Kontrast)"),
        Line2D([0],[0], color="#555", lw=1.2, ls="--", label="D × Q_vis^0.25  (visuell)"),
    ]
    ax.legend(handles=proxy, fontsize=9, loc="upper left")
    ax.set_title("Öffnungsverlust vs. Öffnung", fontsize=9)
    _ax_fmt(ax); fig.tight_layout()
    return fig


def plot_beugung(D_akt, f_akt):
    lam_mm = 550e-6
    f_arr  = np.linspace(200, 2000, 500)

    def D_grenz(S):
        Wb = _Wb_von_S(S)
        return (Wb * 4.0 * 1024 * lam_mm) ** 0.25 * f_arr ** 0.75

    D_95 = D_grenz(0.95)
    D_80 = D_grenz(0.80)
    D_50 = D_grenz(0.50)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.fill_between(f_arr, 0,    D_95, color="#2e7d32", alpha=0.10)
    ax.fill_between(f_arr, D_95, D_80, color="#f9a825", alpha=0.12)
    ax.fill_between(f_arr, D_80, 300,  color="#c62828", alpha=0.07)
    ax.text(300,  40,  "S ≥ 0.95  (sehr gut)",              color="#2e7d32", fontsize=8)
    ax.text(300, 155,  "0.80 ≤ S < 0.95  (gut)",            color="#c8860a", fontsize=8)
    ax.text(300, 255,  "S < 0.80  (nicht beugungsbegrenzt)", color="#c62828", fontsize=8)
    ax.plot(f_arr, D_95, color="#1565c0", lw=2,   label="S=0.95")
    ax.plot(f_arr, D_80, color="#2e7d32", lw=2.5, label="S=0.80  (Rayleigh)")
    ax.plot(f_arr, D_50, color="#BA7517", lw=1.5, ls="--", label="S=0.50")
    for N in [5, 6, 7, 8, 10]:
        D_fN = np.minimum(f_arr / N, 300)
        ax.plot(f_arr, D_fN, color="#ccc", lw=0.8, ls="--")
        y_label = 1900.0 / N
        if y_label < 285:
            ax.text(1920, y_label, f"f/{N}", color="#999", fontsize=8, va="center")
    ax.scatter([f_akt], [D_akt], color=ACC, s=100, zorder=7)
    x_off = -200 if f_akt > 1500 else 80
    ax.annotate(f"D={D_akt:.0f}mm  f/{f_akt/D_akt:.1f}",
                xy=(f_akt, D_akt), xytext=(f_akt + x_off, D_akt + 25),
                fontsize=9, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))
    ax.set_xlim(200, 2000); ax.set_ylim(0, 300)
    ax.set_xlabel("Brennweite f [mm]", fontsize=10)
    ax.set_ylabel("Öffnung D [mm]", fontsize=10)
    ax.set_title("Beugungsgrenze sphärischer Spiegel  (λ=550nm)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    _ax_fmt(ax); fig.tight_layout()
    return fig


def plot_mtf(D, f, lam, S_slide=None, paraxial=False, objekte="planeten"):
    lam_mm   = lam * 1e-6
    _, _, Wb, _, _ = _wellenfronten(D, f, lam)
    strehl   = _strehl_exakt(Wb)
    fc       = D / (lam_mm * 206265)
    fc_fokal = 1.0 / (lam_mm * (f / D))
    N_KURVE  = 300
    Wb_k     = Wb

    nu_k2  = np.linspace(0, 1, N_KURVE)
    mp_k2  = np.array([mtf_para(nu)                              for nu in nu_k2])
    msr_k2 = np.array([mtf_sph_rel(nu, Wb_k, paraxial=paraxial) for nu in nu_k2])
    ms_k2  = mp_k2 * msr_k2

    fokus_label = "Paraxialer Fokus  (ρ⁴)" if paraxial else "Best Focus  (ρ⁴−ρ²)"

    all_details = [
        ("Jupiter Gürtel",    5.0,      "#4a90d9", "planeten"),
        ("Jupiter Festons",   1.5,      "#4a90d9", "planeten"),
        ("Mars Pol",          1.0,      "#d94a4a", "planeten"),
        ("Saturn Cassini",    0.5,      "#c8a020", "planeten"),
        ("Trapezium AB (~9\")",9.0,     "#8e44ad", "deepsky"),
        ("M42 Filamente",     3.0,      "#8e44ad", "deepsky"),
        ("Trapezium CD (~2\")",2.0,     "#8e44ad", "deepsky"),
        ("M13 Halo-Sterne",   2.5,      "#c0392b", "deepsky"),
        ("M13 Außensterne",   1.5,      "#c0392b", "deepsky"),
        ("M13 Kern",          0.7,      "#c0392b", "deepsky"),
        ("Dawes Grenze",      116.0/D,  "#888",    "beide"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    ax.plot(nu_k2, mp_k2,  color=GRN, lw=2.2, ls="-",  label="Paraboloid  (Beugungsgrenze)")
    ax.plot(nu_k2, ms_k2,  color=COR, lw=2.2, ls="-",  label=f"Sphäre absolut  (S={strehl:.3f})")
    ax.plot(nu_k2, msr_k2, color=ACC, lw=1.5, ls="--", label="Sphäre relativ")

    if S_slide is not None and S_slide > strehl + 0.01:
        Wb_s    = _Wb_von_S(S_slide)
        msr_s_k = np.array([mtf_sph_rel(nu, Wb_s, paraxial=paraxial) for nu in nu_k2])
        ax.plot(nu_k2, mp_k2 * msr_s_k, color="#1a7abf", lw=1.8, ls="-.",
                label=f"Schieber absolut  (S={S_slide:.2f})")

    for label, d_as, col, kat in all_details:
        if objekte != "alle" and kat not in ("beide", objekte):
            continue
        nu_det = (1.0 / (2.0 * d_as)) / fc
        if nu_det >= 1.0:
            continue
        ax.axvline(nu_det, color=col, lw=0.8, ls=":", alpha=0.7)
        mp_v = float(np.interp(nu_det, nu_k2, mp_k2))
        ax.text(nu_det + 0.005, mp_v - 0.07, label, fontsize=7, color=col, rotation=90, va="top")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    ax2x = ax.twiny()
    ax2x.set_facecolor("none")
    # Obere Achse in Lp/mm: nu * fc_fokal
    ax2x.set_xlim(0, fc_fokal)
    ax2x.set_xlabel("Räumliche Frequenz [Lp/mm] (fokal)", fontsize=9)
    ax2x.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Normierte Ortsfrequenz ν  (0=DC, 1=Beugungsgrenze)", fontsize=9)
    ax.set_ylabel("Kontrastübertragung (MTF)", fontsize=10)
    ax.set_title(
        f"MTF [{fokus_label}] — D={D:.0f}mm  f/{f/D:.1f}  Strehl={strehl:.3f}  "
        f"fc={fc_fokal:.1f} Lp/mm  λ={lam:.0f}nm", fontsize=9)
    fokus_color = "#2e7d32" if paraxial else ACC
    ax.text(0.98, 0.97,
            fokus_label + ("\nSacek §4.7" if paraxial else "\ntelescope-optics.net"),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=fokus_color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=fokus_color, alpha=0.85))
    ax.legend(fontsize=9, loc="lower left")
    _ax_fmt(ax); fig.tight_layout()
    return fig


def plot_wahrnehmung(D, f, lam, S_slide=None, S_real=None):
    _, _, Wb, _, S_mar = _wellenfronten(D, f, lam)
    S_exakt    = _strehl_exakt(Wb)
    V_krit     = D * 0.7 * math.sqrt(max(S_exakt, 1e-9))
    V_max_plot = 400.0
    V_arr      = np.linspace(30, V_max_plot, 150)

    Qe_sph  = perceived_quality_exakt_kurven(D, f, lam, V_arr)
    f_para  = D * 50.0
    Qe_para = perceived_quality_exakt_kurven(D, f_para, lam, V_arr)

    def Q_naeh(V_a, S):
        Vk = D * 0.7 * math.sqrt(max(S, 1e-9))
        w  = 1.0 / (1.0 + (Vk / np.maximum(V_a, 1e-9))**2)
        return S + (1 - S) * (1 - w)

    Qn_sph  = Q_naeh(V_arr, S_exakt)
    Qn_para = Q_naeh(V_arr, 1.0)

    has_slide = (S_slide is not None and S_real is not None
                 and S_slide > S_mar + 0.01)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.fill_between(V_arr, Qe_sph, Qn_sph, color="#534AB7", alpha=0.12,
                    label="Abweichung exakt ↔ Näherung")
    ax.fill_between(V_arr, Qe_sph, Qe_para, color=COR, alpha=0.10,
                    label="Wahrnehmungsverlust")
    ax.plot(V_arr, Qe_para, color=GRN, lw=2.0, ls="-",  label="Paraboloid  exakt")
    ax.plot(V_arr, Qn_para, color=GRN, lw=1.0, ls="--", alpha=0.4)
    ax.plot(V_arr, Qe_sph,  color=COR, lw=2.2, ls="-",
            label=f"Sphäre f/{f/D:.1f}  S={S_exakt:.3f}  exakt")
    ax.plot(V_arr, Qn_sph,  color=COR, lw=1.5, ls="--", alpha=0.75,
            label=f"Sphäre  Näherung Q̃  (S={S_exakt:.3f})")

    if has_slide:
        Wb_s     = _Wb_von_S(S_slide)
        S_ex_s   = _strehl_exakt(Wb_s)
        Qe_slide = perceived_quality_exakt_kurven_von_Wb(D, Wb_s, lam, V_arr)
        Qn_slide = Q_naeh(V_arr, S_ex_s)
        ax.plot(V_arr, Qe_slide, color=ACC, lw=2.0, ls="-",
                label=f"Schieber exakt  S={S_slide:.3f}")
        ax.plot(V_arr, Qn_slide, color=ACC, lw=1.2, ls="--", alpha=0.7)

    ax.axhline(S_exakt, color=COR, lw=1.0, ls=":", alpha=0.75)
    ax.text(V_max_plot * 0.98, S_exakt + 0.012, f"Asymptote Q̃→S={S_exakt:.3f}",
            fontsize=8, color=COR, ha="right", alpha=0.85)

    for V_m, col, ls, lbl in [
            (D / 7.0, "#888", ":", f"V_min={D/7:.0f}×"),
            (V_krit,  ACC,    "--", f"V_krit={V_krit:.0f}×")]:
        if 30 <= V_m <= V_max_plot:
            ax.axvline(V_m, color=col, lw=1.0, ls=ls, alpha=0.65)
            ax.text(V_m + 2, 0.03, lbl, fontsize=7, color=col, rotation=90, va="bottom")

    for q_thresh, col, lbl in [(0.9, GRN, "Q=0.90"), (0.7, "#BA7517", "Q=0.70")]:
        ax.axhline(q_thresh, color=col, lw=1.0, ls=":", alpha=0.7)
        ax.text(32, q_thresh + 0.008, lbl, fontsize=8, color=col)

    max_abw = float(np.max(np.abs(Qe_sph - Qn_sph)))
    ax.set_xlim(30, V_max_plot); ax.set_ylim(0, 1.12)
    ax.set_xlabel("Vergrößerung V  [×]", fontsize=10)
    ax.set_ylabel("Visueller Qualitätsindex  Q_vis", fontsize=10)
    ax.set_title(
        f"Visuelle Wahrnehmung  —  D={D:.0f}mm  f/{f/D:.1f}"
        f"  |  S_exakt={S_exakt:.3f}  max.Abw.={max_abw:.3f}", fontsize=9)
    ax.legend(fontsize=8, loc="lower right",
              title="— exakt (MTF×CSF)   - - Näherung Q̃(V)", title_fontsize=7)
    _ax_fmt(ax); fig.tight_layout()
    return fig


def plot_blende(D, f, lam, D_blend):
    V_arr  = np.linspace(30, 400, 150)
    n_pts  = 150
    V_max_plot = 400.0

    def S_von_D(Db):
        _, _, Wb_, _, _ = _wellenfronten(Db, f, lam)
        return _strehl_exakt(Wb_)

    S_orig  = S_von_D(D)
    S_blend = S_von_D(D_blend)
    d_fang  = d_fang_min(D, f)
    eps_orig = d_fang / D

    def Q_naeh_arr(V_a, S, Deff):
        Vk = Deff * 0.7 * math.sqrt(max(S, 1e-9))
        w  = 1.0 / (1.0 + (Vk / np.maximum(V_a, 1e-9))**2)
        return S + (1.0 - S) * (1.0 - w)

    Q_orig  = Q_naeh_arr(V_arr, S_orig,  D)
    Q_bl    = Q_naeh_arr(V_arr, S_blend, D_blend)

    V_min_orig   = D / 7.0
    V_max_AP     = D / 0.5
    V_krit_orig  = D * 0.7 * math.sqrt(max(S_orig, 1e-9))
    Vk_orig      = V_krit_orig
    V_max_orig   = min(V_max_AP, 2 * V_krit_orig)
    V_min_blend  = D_blend / 7.0
    V_max_AP_bl  = D_blend / 0.5
    V_max_blend  = min(V_max_AP_bl, 2 * D_blend * 0.7 * math.sqrt(max(S_blend, 1e-9)))

    # Optimale Blende je Vergrößerung
    D_arr_opt = np.linspace(D * 0.4, D, 40)
    Q_opt_line = np.zeros(len(V_arr))
    D_opt_arr  = np.zeros(len(V_arr))
    for j, V in enumerate(V_arr):
        q_best = -1.0; d_best = D
        for Db in D_arr_opt:
            s_ = S_von_D(Db)
            q_ = (s_ + (1-s_) * (1 - 1/(1+(Db*0.7*math.sqrt(max(s_,1e-9))/max(V,1))**2)))
            if q_ > q_best:
                q_best = q_; d_best = Db
        Q_opt_line[j] = q_best; D_opt_arr[j] = d_best

    mask_sinn  = (V_arr >= V_min_orig) & (V_arr <= V_max_orig)
    D_opt_vk   = float(D_opt_arr[np.argmin(np.abs(V_arr - Vk_orig))]) if len(V_arr) else D
    Q_opt_vk   = float(Q_opt_line[np.argmin(np.abs(V_arr - Vk_orig))]) if len(V_arr) else 0
    D_opt_mean = float(D_opt_arr[mask_sinn].mean()) if mask_sinn.any() else D_opt_vk

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    fig.patch.set_facecolor(BG)

    # Oberes Panel
    ax.plot(V_arr, Q_orig, color=ACC, lw=2.2, ls="-",
            label=f"Original D={D:.0f}mm  ε={eps_orig:.3f}  S={S_orig:.3f}")
    if D_blend < D * 0.98:
        ax.plot(V_arr, Q_bl, color=COR, lw=2.0, ls="-.",
                label=f"Abgeblendet D={D_blend:.0f}mm  ε={d_fang/D_blend:.3f}  S={S_blend:.3f}")
    ax.plot(V_arr, Q_opt_line, color=GRN, lw=1.8, ls="-.",
            label=f"Optimale Blende (max Q)  ⌀≈{D_opt_mean:.0f}mm")

    for q_thresh, col, lbl in [(0.9, GRN, "Q=0.90"), (0.7, "#BA7517", "Q=0.70")]:
        ax.axhline(q_thresh, color=col, lw=0.8, ls=":", alpha=0.6)
        ax.text(32, q_thresh + 0.008, lbl, fontsize=7, color=col, alpha=0.8)
    if 30 <= Vk_orig <= V_max_plot:
        ax.axvline(Vk_orig, color=COR, lw=1.0, ls="--", alpha=0.5)
        ax.text(Vk_orig + 2, 0.03, f"V½={Vk_orig:.0f}×",
                fontsize=7, color=COR, rotation=90, va="bottom", alpha=0.7)
    ax.set_xlim(30, V_max_plot)
    ax.set_ylim(0, min(1.15, max(float(Q_orig.max()), float(Q_bl.max()) if D_blend < D*0.98 else 0) * 1.15))
    ax.set_ylabel("Q_vis normiert auf Paraboloid  (MTF×CSF)", fontsize=9)
    rayleigh_ok = S_orig >= 0.80
    titel_farbe = "#2e7d32" if rayleigh_ok else "#c62828"
    rayleigh_text = ("✓ Rayleigh S ≥ 0.80 — Abblenden nicht sinnvoll"
                     if rayleigh_ok else
                     f"✗ Rayleigh nicht erfüllt (S={S_orig:.3f}) — Abblenden kann helfen")
    ax.text(0.02, 0.97, rayleigh_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color=titel_farbe,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=titel_farbe, alpha=0.9))
    ax.set_title(
        f"Blendeneffekt  —  D={D:.0f}mm  f/{f/D:.1f}  S={S_orig:.3f}"
        f"  |  Blende={D_blend:.0f}mm ({D_blend/D*100:.0f}%)"
        f"  |  d_fang={d_fang:.0f}mm  ε={eps_orig:.3f}→{d_fang/D_blend:.3f}", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    _ax_fmt(ax)

    # Unteres Panel: Gain-Faktor
    objekte_lum = [
        ("Jupiter",           6000, "#e65100"),
        ("Mars/Saturn",        500, "#c8860a"),
        ("Schwacher Planet",    50, "#1565c0"),
        ("Heller Nebel",         5, "#6a1b9a"),
        ("Schwaches Deep-Sky",   0.5, "#1b5e20"),
    ]
    L_arr_lum = [L for _, L, _ in objekte_lum]
    Q_bl_mat  = Q_vis_blende_batch(D_blend, f, lam, V_arr, L_arr_lum, D_ref=D)
    Q_ori_mat = Q_vis_blende_batch(D,       f, lam, V_arr, L_arr_lum, D_ref=D)
    gain_mat  = Q_bl_mat / np.maximum(Q_ori_mat, 1e-12)

    for li, (name, L_obj, col) in enumerate(objekte_lum):
        gain = gain_mat[:, li]
        ls = "-" if L_obj >= 500 else ("--" if L_obj >= 5 else ":")
        ax2.plot(V_arr, gain, color=col, lw=1.8, ls=ls, label=f"{name}  (L≈{L_obj:.0f} cd/m²)")

    ax2.axhline(1.0, color="#888", lw=1.2)
    ax2.text(V_max_plot * 0.98, 1.02, "Abblenden lohnt sich ↑", fontsize=8, color="#2e7d32", ha="right")
    ax2.text(V_max_plot * 0.98, 0.96, "Abblenden schadet ↓",    fontsize=8, color="#c62828", ha="right")
    y_max = min(float(gain_mat.max()) * 1.1, 4.0)
    y_min = max(float(gain_mat.min()) * 0.95, 0.3)
    ax2.set_ylim(y_min, y_max)
    ax2.fill_between(V_arr, 1.0, y_max, color="#2e7d32", alpha=0.04)
    ax2.fill_between(V_arr, y_min, 1.0, color="#c62828", alpha=0.06)
    if 30 <= Vk_orig <= V_max_plot:
        ax2.axvline(Vk_orig, color=COR, lw=1.0, ls="--", alpha=0.5)
    ax2.set_xlim(30, V_max_plot)
    ax2.set_xlabel("Vergrößerung V  [×]", fontsize=10)
    ax2.set_ylabel("Gain-Faktor  Q_blend / Q_orig", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    _ax_fmt(ax2)

    fig.tight_layout()
    return fig


def plot_foucault(D, f, lam, r_zon, df_arr, kz):
    R     = D / 2.0
    rho_z = r_zon / R
    _COR = "#E65100"; _ACC = "#1565c0"; _GRN = "#2e7d32"

    fig, (ax_wf, ax_info) = plt.subplots(1, 2, figsize=(10, 4.5),
                                          gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(BG)
    ax_wf.set_facecolor(BG); ax_info.set_facecolor(BG)

    rho_fine = kz["rho_f"]
    W_fine   = kz["W_c"]
    ax_wf.plot(rho_fine, W_fine, color=_ACC, lw=2.0,
               label=f"Wellenfront (Polynom-Fit, a4={kz['a4']:.3f})")

    W_z    = kz["W_zonen"]
    Glas_z = kz.get("Glas_zonen")
    W_min  = kz.get("W_min")
    if Glas_z is not None and W_min is not None:
        y_punkte = Glas_z * 2.0 / lam + W_min
        ax_wf.scatter(rho_z, y_punkte, color=_COR, s=50, zorder=5,
                      label=f"Messzonen (n={len(r_zon)})")
        for i, (rho, yp) in enumerate(zip(rho_z, y_punkte)):
            wc  = W_z[i] - float(np.mean(W_z))
            ax_wf.annotate(f"Z{i+1}: {Glas_z[i]:.0f} nm\n({wc:+.3f}λ)",
                           xy=(rho, yp), xytext=(0, 10),
                           textcoords="offset points", fontsize=7, ha="center", color=_COR)
    else:
        Wm_disc  = float(np.mean(W_z))
        W_c_disc = W_z - Wm_disc
        ax_wf.scatter(rho_z, W_c_disc, color=_COR, s=50, zorder=5)

    ax_wf.axhline(0, color="#999", lw=0.8)
    ax_wf.axhline(+0.125, color=_ACC, lw=0.7, ls=":", alpha=0.7, label="±λ/8 (Rayleigh)")
    ax_wf.axhline(-0.125, color=_ACC, lw=0.7, ls=":", alpha=0.7)
    ax_wf.set_xlim(-0.02, 1.05)
    ax_wf.set_xlabel("Normierter Zonenradius ρ = r/R", fontsize=8)
    ax_wf.set_ylabel("Wellenfrontfehler W(ρ)  [Einheiten von λ]", fontsize=8)
    ax_wf.set_title("Gemessene Wellenfront (bester Fokus, pistonbereinigt)", fontsize=8, fontweight="bold")
    ax_wf.legend(fontsize=7, loc="lower center")
    ax_wf.grid(True, alpha=0.25)

    if W_min is not None:
        zu_glas = lambda w: (np.asarray(w) - W_min) * lam / 2.0
        zu_lam  = lambda g: np.asarray(g) * 2.0 / lam + W_min
        ax_glas = ax_wf.secondary_yaxis("right", functions=(zu_glas, zu_lam))
        ax_glas.set_ylabel("Materialabtrag [nm Glas]  (0 = am wenigsten bearbeiteter Punkt)",
                           fontsize=8, color="#6a1b9a")
        ax_glas.tick_params(colors="#6a1b9a", labelsize=7)

    # Kennzahlentafel
    ax_info.axis("off")
    rayleigh_ok = kz["S_exakt"] >= 0.80
    s_col = _GRN if kz["S_exakt"] >= 0.95 else "#e65100" if rayleigh_ok else "#c62828"
    zeilen = [
        ("Öffnung / f-Zahl",   f"D = {D:.0f} mm  f/{f/D:.1f}"),
        ("Wellenlänge",         f"λ = {lam:.0f} nm"),
        ("", ""),
        ("W_rms",               f"{kz['W_rms']:.4f} λ"),
        ("W_PtV",               f"{kz['W_ptv']:.4f} λ"),
        ("Wb (aus Fit)",        f"{kz['Wb_fit']:.4f} λ"),
        ("", ""),
        ("Abtrag RMS (Glas)",   f"{kz['W_rms']*lam/2.0:.1f} nm"
                                if kz.get("Glas_ptv_nm") is not None else "–"),
        ("Abtrag PtV (Glas)",   f"{kz['Glas_ptv_nm']:.1f} nm"
                                if kz.get("Glas_ptv_nm") is not None else "–"),
        ("Abtrag V (Glas)",     f"{kz['Glas_volumen_mm3']:.1f} mm³"
                                if kz.get("Glas_volumen_mm3") is not None else "–"),
        ("", ""),
        ("Strehl  Maréchal",    f"{kz['S_mar']:.4f}"),
        ("Strehl  exakt",       f"{kz['S_exakt']:.4f}"),
        ("D_eff  (Kontrast)",   f"{kz['Deff_k']:.1f} mm"),
        ("", ""),
        ("Beurteilung", "✓ Rayleigh erfüllt (S ≥ 0.80)" if rayleigh_ok
                        else "✗ Rayleigh nicht erfüllt"),
    ]
    y = 0.97
    for lbl, val in zeilen:
        if not lbl and not val:
            y -= 0.04; continue
        is_strehl  = lbl.startswith("Strehl")
        is_verdict = lbl == "Beurteilung"
        ax_info.text(0.03, y, lbl + ":", transform=ax_info.transAxes,
                     fontsize=8, color="#555", va="top")
        ax_info.text(0.52, y, val, transform=ax_info.transAxes,
                     fontsize=9,
                     fontweight="bold" if (is_strehl or is_verdict) else "normal",
                     color=s_col if (is_strehl or is_verdict) else "#222",
                     va="top",
                     fontfamily="monospace" if not is_verdict else "sans-serif")
        y -= 0.075
    ax_info.set_title("Auswertung", fontsize=8, fontweight="bold")
    fig.suptitle(
        f"Foucault-Auswertung  —  D={D:.0f}mm  f/{f/D:.1f}  "
        f"S={kz['S_exakt']:.4f}  W_rms={kz['W_rms']:.4f}λ",
        fontsize=8, fontweight="bold")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit-App
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title=f"Newton-Spiegel  v{VERSION}",
        page_icon="🔭",
        layout="wide",
    )

    st.title(f"🔭 Newton-Spiegel: Kontrast- und Schärfeverlust  v{VERSION}")
    st.caption("Sphärischer Hauptspiegel — Wellenfrontfehler, Strehl, MTF, CSF-Wahrnehmung, Foucault")

    # ── Sidebar: Teleskop-Parameter ──────────────────────────────────────────
    st.sidebar.header("Teleskop-Parameter")

    D   = st.sidebar.slider("Öffnung D [mm]",       min_value=50,   max_value=600, value=200, step=5)
    f   = st.sidebar.slider("Brennweite f [mm]",    min_value=200,  max_value=6000, value=1000, step=50)
    lam = st.sidebar.slider("Wellenlänge λ [nm]",   min_value=400,  max_value=700, value=550, step=10)

    N = f / D
    st.sidebar.markdown(f"**f/D = {N:.2f}**")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Strehl-Schieber (Verbesserung)")
    S_pct = st.sidebar.slider(
        "Relative Verbesserung zu Paraboloid [%]", 0, 100, 0,
        help="0% = reale Sphäre, 100% = Paraboloid. Simuliert Figurierungsfortschritt.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Abblendung")
    D_blend = st.sidebar.slider("Blende D [mm]", min_value=50, max_value=D, value=D, step=5)

    # ── Rechnung ─────────────────────────────────────────────────────────────
    r_real     = berechne(D, f, lam)
    S_real     = r_real["strehl"]
    S_mar      = r_real["strehl_marechal"]
    S_slide    = S_real + (S_pct / 100.0) * (1.0 - S_real)
    S_slide    = min(max(S_slide, S_real), 1.0)
    Deff_k_sl  = D * S_slide**0.25
    Vk         = v_kritisch(D, f, lam)
    Vk_slide   = v_kritisch_strehl(D, S_slide)
    rv         = berechne_vergr(D, f, lam, 150.0, EYE_RES)
    Vk_naeh    = Vk["Vk_sph"]
    Vk_naeh_s  = Vk_slide["Vk_sph"]
    Vk_blur    = v_krit_blur_direkt(D, f, EYE_RES)
    Vk_exakt   = v_krit_aus_qvis(D, f, lam, rel_schwelle=0.90)
    Wb_s       = _Wb_von_S(S_slide)
    Vk_exakt_s = v_krit_aus_qvis_von_Wb(D, Wb_s, lam, rel_schwelle=0.90)
    Vk_halb    = v_halb_exakt(D, f, lam)
    Qp_80      = perceived_quality(D, f, lam, 30.0)
    Qp_200     = perceived_quality(D, f, lam, 80.0)
    Qp_350     = perceived_quality(D, f, lam, 160.0)
    Qp_V       = perceived_quality(D, f, lam, 150.0)

    verdict_text, verdict_color = beurteilung(S_real)

    # ── Kennzahlen-Übersicht (kompakt) ──────────────────────────────────────
    def _qc(q):
        return "#2e7d32" if q >= 0.90 else "#e65100" if q >= 0.70 else "#c62828"

    def _vk_ex_text(v):
        if v <= 30:  return "< 30x"
        if v >= 600: return ">= 600x"
        return f"{v:.0f}x"

    delta_s    = S_real - S_mar
    delta_halb = Vk_halb - Vk_naeh
    avp        = r_real["aufl_verlust_pct"]
    alpha_blur = D**3 / (64 * f**2) / f * 206265

    s_col = "#2e7d32" if S_real >= 0.95 else "#e65100" if S_real >= 0.80 else "#c62828"

    st.markdown(
        f"<div style='background:{verdict_color};color:white;padding:5px 10px;"
        f"border-radius:5px;font-size:0.82em;font-weight:bold;margin-bottom:5px'>"
        f"{verdict_text}</div>",
        unsafe_allow_html=True)

    def _row(lbl, val, color="#222"):
        return (f"<tr>"
                f"<td style='color:#666;font-size:0.76em;padding:1px 10px 1px 2px;white-space:nowrap'>{lbl}</td>"
                f"<td style='font-weight:bold;font-size:0.76em;color:{color};font-family:monospace;white-space:nowrap'>{val}</td>"
                f"</tr>")

    def _hdr(title):
        return (f"<tr><td colspan='2' style='font-weight:bold;font-size:0.78em;"
                f"padding:5px 0 1px;color:#333;border-bottom:1px solid #ddd'>{title}</td></tr>")

    rows  = _hdr("Kontrast / Wellenfront")
    rows += _row("f/D",                  f"{N:.2f}")
    rows += _row("W_PtV paraxial",       f"{r_real['Wp']:.4f} &#955;")
    rows += _row("W_PtV best focus",     f"{r_real['Wb']:.4f} &#955;")
    rows += _row("W_RMS",                f"{r_real['Wrms']:.5f} &#955;")
    rows += _row("Strehl Marechal",      f"{S_mar:.4f} &#8594; {S_slide:.4f}")
    rows += _row("Strehl exakt",
                 f"{S_real:.4f}  (&#916; Mar. {delta_s:+.4f})", s_col)
    rows += _row("D_eff Kontrast",       f"{r_real['Deff_k']:.1f} &#8594; {Deff_k_sl:.1f} mm")
    rows += _hdr("Vergr&#246;&#223;erung")
    rows += _row("V&#189;  D&#183;0.7&#183;&#8730;S",
                 f"{Vk_naeh:.0f}x &#8594; {Vk_naeh_s:.0f}x  | exakt: {Vk_halb:.0f}x  (&#916; {delta_halb:+.0f}x)")
    rows += _row("V_krit Blur (geom.)",
                 f"{Vk_blur:.0f}x  &#945;_blur={alpha_blur:.1f}\"" if Vk_blur > 1 else "< 1x - Blur dominiert immer")
    rows += _row("V_krit MTF&#215;CSF >10%",
                 f"{_vk_ex_text(Vk_exakt)} &#8594; {_vk_ex_text(Vk_exakt_s)}")
    rows += _row("Aufl.verlust",         f"{avp:.1f}%  (Blur {r_real['d_blur_as']:.1f}\")")
    rows += _row("Verlust bei 150x",     f"{rv['verlust']:.1f} mm")
    rows += _row("V_krit / Vmax",
                 f"{Vk['Vk_sph']:.0f}x &#8594; {Vk_slide['Vk_sph']:.0f}x  | Para: {Vk['Vk_para']:.0f}x  | Vmax: {Vk['Vmax']:.0f}x",
                 "#534AB7")
    rows += _hdr("Q_perc (CSF-gewichtet)")
    for lbl_q, q in [("Q bei  30x", Qp_80), ("Q bei  80x", Qp_200),
                     ("Q bei 160x", Qp_350), ("Q bei 150x", Qp_V)]:
        rows += _row(lbl_q, f"{q:.3f}", _qc(q))

    st.markdown(
        f"<table style='border-collapse:collapse;width:100%;margin-bottom:4px'>{rows}</table>",
        unsafe_allow_html=True)

    st.markdown("---")

    # ── Diagramm-Tabs ────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Strehl vs. f/D",
        "Eff. Öffnung vs. f/D",
        "D_eff vs. Öffnung",
        "Beugungsgrenze",
        "MTF",
        "Wahrnehmung",
        "Blende",
        "Foucault",
    ])

    with tabs[0]:
        st.pyplot(plot_strehl(D, f, lam, S_slide if S_pct > 0 else None))

    with tabs[1]:
        st.pyplot(plot_oeffnung(D, f, lam, S_slide if S_pct > 0 else None))

    with tabs[2]:
        st.pyplot(plot_deff_D(D, N, S_slide if S_pct > 0 else None))

    with tabs[3]:
        st.pyplot(plot_beugung(D, f))

    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            fokus_mode = st.radio("Fokusebene:",
                                  ["Best Focus  (ρ⁴−ρ²)", "Paraxialer Fokus  (ρ⁴)"],
                                  horizontal=True)
        with c2:
            objekte_mode = st.radio("Objekte:", ["Planeten", "Deep-Sky (M42)", "Alle"],
                                    horizontal=True)
        paraxial  = "Paraxial" in fokus_mode
        objekte_k = {"Planeten": "planeten", "Deep-Sky (M42)": "deepsky", "Alle": "alle"}[objekte_mode]
        st.pyplot(plot_mtf(D, f, lam, S_slide if S_pct > 0 else None,
                           paraxial=paraxial, objekte=objekte_k))

    with tabs[5]:
        st.pyplot(plot_wahrnehmung(D, f, lam,
                                   S_slide if S_pct > 0 else None,
                                   S_real))

    with tabs[6]:
        st.pyplot(plot_blende(D, f, lam, D_blend))

    with tabs[7]:
        # Foucault-Tab
        st.subheader("Foucault-Auswertung")
        lam_fc = 550.0   # fix
        fc1, fc2 = st.columns([1, 3])
        with fc1:
            n_zonen = st.radio("Zonen:", [4, 6, 8], horizontal=True, index=0)
            st.caption(f"λ = {lam_fc:.0f} nm (fix)")

        R       = D / 2.0
        r_zon   = foucault_zonen(D, n_zonen)
        r_grenz = foucault_zonengrenzen(D, n_zonen)
        Rc      = 2.0 * f
        df_para_abs = r_zon**2 / (2.0 * Rc)
        df_para     = df_para_abs - df_para_abs[0]

        # Info-Zeile
        info_parts = [f"Z{i+1}: r={rv_:.1f}mm  ΔF_soll={dp:.3f}mm"
                      for i, (rv_, dp) in enumerate(zip(r_zon, df_para))]
        st.caption("Gleichflächige Zonen | Parabolspiegel-Sollwerte rel. Zone 1 "
                   "(+ = Messerschneide weiter weg, FigureXP-Konvention):  " +
                   "  |  ".join(info_parts))
        st.caption("ΔF=0 → Kugel  |  ΔF=ΔF_soll → Parabel  |  Zone 1 = Referenz, fest 0")

        # Eingabetabelle
        st.markdown("**ΔF-Messwerte [mm]  (relativ zu Zone 1):**")

        # Session-State für Messwerte
        key_prefix = f"fc_{D}_{f}_{n_zonen}"
        df_vals = []
        cols_hdr = st.columns([1, 2, 2, 2, 2, 2])
        cols_hdr[0].markdown("**Zone**")
        cols_hdr[1].markdown("**Radius [mm]**")
        cols_hdr[2].markdown("**Grenze innen–außen [mm]**")
        cols_hdr[3].markdown("**ΔF_soll [mm]**")
        cols_hdr[4].markdown("**ΔF gemessen [mm]**")
        cols_hdr[5].markdown("**Abw. [mm]**")

        for i in range(n_zonen):
            r_innen  = r_grenz[i-1] if i > 0 else 0.0
            r_aussen = r_grenz[i]
            dp       = df_para[i]
            sk = f"{key_prefix}_z{i}"
            default_val = 0.0 if i == 0 else round(float(dp), 3)
            if sk not in st.session_state:
                st.session_state[sk] = default_val

            cols = st.columns([1, 2, 2, 2, 2, 2])
            cols[0].markdown(f"**Z{i+1}**")
            cols[1].write(f"{r_zon[i]:.2f}")
            cols[2].write(f"{r_innen:.2f} – {r_aussen:.2f}")
            cols[3].write(f"{dp:.3f}")
            if i == 0:
                cols[4].write("0.000  *(Referenz)*")
                df_vals.append(0.0)
            else:
                val = cols[4].number_input(
                    f"Z{i+1}",
                    value=st.session_state[sk],
                    min_value=-50.0, max_value=50.0, step=0.001,
                    format="%.3f",
                    label_visibility="collapsed",
                    key=sk)
                df_vals.append(val)
                abw = val - dp
                col_txt = "🟢" if abs(abw) < 0.01 else "🟡" if abs(abw) < 0.1 else "🔴"
                cols[5].write(f"{col_txt} {abw:+.3f}")

        # Buttons: Parabel / Kugel
        # WICHTIG: Buttons müssen VOR den number_input-Widgets gerendert werden,
        # damit session_state-Schreibvorgänge nicht mit laufenden Widget-Keys kollidieren.
        bk1, bk2 = st.columns(2)
        if bk1.button("↺  Alle auf Parabel-Sollwerte"):
            for i in range(1, n_zonen):
                k = f"{key_prefix}_z{i}"
                v = round(float(df_para[i]), 3)
                # Widget-Key löschen, dann neu setzen – vermeidet Streamlit-Konflikt
                if k in st.session_state:
                    del st.session_state[k]
                st.session_state[k] = v
            st.rerun()
        if bk2.button("✕  Alle auf Null (Kugel)"):
            for i in range(1, n_zonen):
                k = f"{key_prefix}_z{i}"
                if k in st.session_state:
                    del st.session_state[k]
                st.session_state[k] = 0.0
            st.rerun()

        # Berechnen
        df_arr = np.array(df_vals)
        try:
            W_z = foucault_wellenfront(r_zon, df_arr, f, lam_fc)
            kz  = foucault_kennzahlen(W_z, r_zon, D, lam_fc)

            rayleigh_ok = kz["S_exakt"] >= 0.80
            s_color = "green" if kz["S_exakt"] >= 0.95 else "orange" if rayleigh_ok else "red"
            status_icon = "✓" if rayleigh_ok else "✗"
            st.markdown(
                f"<div style='background-color:{'#e8f5e9' if rayleigh_ok else '#ffebee'};"
                f"padding:8px;border-radius:6px;margin:8px 0'>"
                f"<b style='color:{s_color}'>{status_icon} "
                f"S_exakt = {kz['S_exakt']:.4f} &nbsp;|&nbsp; "
                f"W_rms = {kz['W_rms']:.4f}λ &nbsp;|&nbsp; "
                f"W_PtV = {kz['W_ptv']:.4f}λ &nbsp;|&nbsp; "
                f"Abtrag PtV = {kz['Glas_ptv_nm']:.1f} nm &nbsp;|&nbsp; "
                f"Abtrag V = {kz['Glas_volumen_mm3']:.1f} mm³ &nbsp;|&nbsp; "
                f"D_eff = {kz['Deff_k']:.1f} mm &nbsp;|&nbsp; "
                f"{'✓ Rayleigh OK' if rayleigh_ok else '✗ Rayleigh nicht erfüllt'}"
                f"</b></div>",
                unsafe_allow_html=True)

            st.pyplot(plot_foucault(D, f, lam_fc, r_zon, df_arr, kz))

        except ImportError as e:
            st.error(f"scipy fehlt: {e}")
        except Exception as e:
            st.error(f"Foucault-Berechnungsfehler: {e}")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Newton-Spiegel v{VERSION} · "
        "Formeln: Sacek (telescope-optics.net) · Suiter (Star Testing) · Barten 1999 (CSF) · "
        "Strehl: exaktes Pupillenintegral"
    )


if __name__ == "__main__" or True:
    main()
