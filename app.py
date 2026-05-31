"""
Newton-Teleskop Rechner - Streamlit-Version
============================================
Alle Kernfunktionen unverändert aus newton_spiegel_rechner.py übernommen.
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from functools import lru_cache
import streamlit as st

# ── Farben ────────────────────────────────────────────────────────────────────
ACC = "#534AB7"
COR = "#D85A30"
GRN = "#0F6E56"

VERSION = "4.3.0 (2026-05-30)"

# ═════════════════════════════════════════════════════════════════════════════
# KERNFUNKTIONEN (identisch mit newton_spiegel_rechner.py)
# ═════════════════════════════════════════════════════════════════════════════

def _wellenfronten(D_mm, f_mm, lam_nm):
    """Gemeinsame Basis: Wp, Wb, Wrms, Strehl — einmal berechnet, überall genutzt."""
    lam_mm = lam_nm * 1e-6
    Wp   = D_mm**4 / (1024.0 * f_mm**3 * lam_mm)
    Wb   = Wp / 4.0
    Wrms = Wb / (1.5 * math.sqrt(5))
    S    = math.exp(-(2 * math.pi * Wrms) ** 2)
    return lam_mm, Wp, Wb, Wrms, S

def _Wb_von_S(S):
    """Wb direkt aus Strehl — Umkehrung der Maréchal-Näherung."""
    Wrms = math.sqrt(-math.log(max(S, 1e-9))) / (2.0 * math.pi)
    return Wrms * 1.5 * math.sqrt(5)

def _kennzahlen_von_S(D_mm, S, Wb):
    """Effektive Öffnungen und Vergrößerungsgrenzen aus S und Wb."""
    sqrtS       = math.sqrt(max(S, 1e-9))
    Deff_k      = D_mm * S**0.25
    Deff_s      = D_mm * sqrtS              # = 116 / theta_eff
    theta_ideal = 116.0 / D_mm
    theta_eff   = theta_ideal / sqrtS
    V_krit_vis  = D_mm * 0.7 * sqrtS
    Q02 = mtf_sph_rel(0.2, Wb); Q04 = mtf_sph_rel(0.4, Wb); Q06 = mtf_sph_rel(0.6, Wb)
    Qshape   = 0.4*Q02 + 0.4*Q04 + 0.2*Q06
    Qvis     = S * Qshape
    Deff_vis = D_mm * Qvis**0.25
    return dict(strehl=S, Wb=Wb,
                Deff_k=Deff_k,   loss_k=D_mm - Deff_k,
                Deff_s=Deff_s,   loss_s=D_mm - Deff_s,
                Deff_vis=Deff_vis, loss_vis=D_mm - Deff_vis,
                theta_ideal=theta_ideal, theta_eff=theta_eff,
                V_krit_vis=V_krit_vis,
                Q02=Q02, Q04=Q04, Q06=Q06, Qshape=Qshape, Qvis=Qvis)

def berechne(D_mm, f_mm, lam_nm):
    lam_mm, Wp, Wb, Wrms, S = _wellenfronten(D_mm, f_mm, lam_nm)
    r = _kennzahlen_von_S(D_mm, S, Wb)
    # Geometrischer Blur (nur Fotografie, Wb bereits bekannt)
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

def berechne_von_S(D_mm, S, lam_nm=550.0):
    """Alle Kennzahlen direkt aus S — kein Umweg über strehl_to_f."""
    return _kennzahlen_von_S(D_mm, S, _Wb_von_S(S))

def berechne_vergr(D_mm, f_mm, lam_nm, V, eye_res_as=16.0):
    r = berechne(D_mm, f_mm, lam_nm)
    theta_auge = eye_res_as / V
    theta_para = max(r["theta_ideal"], theta_auge)
    theta_sph  = max(r["theta_eff"],   theta_auge)
    Deff_para  = min(D_mm, 116.0 / theta_para)
    Deff_sph   = min(D_mm, 116.0 / theta_sph)
    r.update(theta_auge=theta_auge, theta_para=theta_para, theta_sph=theta_sph,
             Deff_para=Deff_para, Deff_sph=Deff_sph, verlust=Deff_para - Deff_sph)
    return r

def v_kritisch(D_mm, f_mm, lam_nm, eye_res_as=16.0):
    r = berechne(D_mm, f_mm, lam_nm)
    return dict(Vk_sph=r["V_krit_vis"], Vk_para=D_mm*0.7,
                Vmax=D_mm/0.5, Vmin=D_mm/7.0)

def kurven_N(D_mm, lam_nm, N_min=3.0, N_max=15.0, schritte=400):
    ns = np.linspace(N_min, N_max, schritte)
    strehls, deff_ks, aufl_verluste = [], [], []
    for n in ns:
        r = berechne(D_mm, D_mm * n, lam_nm)
        strehls.append(r["strehl"]); deff_ks.append(r["Deff_k"])
        aufl_verluste.append(r["aufl_verlust_pct"])
    return ns, np.array(strehls), np.array(deff_ks), np.array(aufl_verluste)

def strehl_to_f(S, D_mm, lam_nm=550.0):
    """Äquivalente Brennweite zu einem Strehl-Wert (nur für Info-Anzeige)."""
    if S >= 0.9999: return D_mm * 50.0
    Wb  = _Wb_von_S(S)
    Wp  = Wb * 4.0
    f3  = D_mm**4 / (1024.0 * Wp * lam_nm * 1e-6)
    return D_mm * (f3 ** (1.0/3.0)) / D_mm

def mtf_para(f):
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    return (2/math.pi) * (math.acos(f) - f * math.sqrt(1 - f**2))

@lru_cache(maxsize=2048)
def mtf_sph_rel(f, Wb, n=80):
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    lim = math.sqrt(max(0.0, 1.0 - (f/2)**2))
    re = im = norm = 0.0
    for i in range(n):
        x  = -lim + (2*i + 1) / (2*n) * 2*lim
        x1 = x - f/2; x2 = x + f/2
        if x1**2 >= 1 or x2**2 >= 1: continue
        h  = min(math.sqrt(max(0.0, 1 - x1**2)), math.sqrt(max(0.0, 1 - x2**2)))
        dW = 2*math.pi * (Wb*(x2**4 - x2**2) - Wb*(x1**4 - x1**2))
        re += math.cos(dW) * h; im += math.sin(dW) * h; norm += h
    if norm < 1e-10: return 0.0
    return math.sqrt(re**2 + im**2) / norm

def _mtf_arrays(Wb, S, fc, n_pts):
    """MTF-Arrays aus Wb und S — gemeinsame Basis für mtf_kurven und mtf_kurven_von_S."""
    freqs_norm = np.linspace(0, 1, n_pts)
    freqs_as   = freqs_norm * fc
    mp  = np.array([mtf_para(f)        for f in freqs_norm])
    msr = np.array([mtf_sph_rel(f, Wb) for f in freqs_norm])
    msa = msr * mp * S
    return freqs_as, mp, msa, msr

def mtf_kurven(D_mm, f_mm, lam_nm=550.0, n_pts=200):
    _, _, Wb, _, S = _wellenfronten(D_mm, f_mm, lam_nm)
    fc = D_mm / (lam_nm * 1e-6 * 206265)
    freqs_as, mp, msa, msr = _mtf_arrays(Wb, S, fc, n_pts)
    return freqs_as, fc, mp, msa, msr, S

def mtf_kurven_von_S(D_mm, S, lam_nm=550.0, n_pts=200):
    """Wie mtf_kurven(), aber S direkt einsetzen."""
    Wb = _Wb_von_S(S)
    fc = D_mm / (lam_nm * 1e-6 * 206265)
    freqs_as, mp, msa, msr = _mtf_arrays(Wb, S, fc, n_pts)
    return freqs_as, fc, mp, msa, msr, S

def csf(f_cpd):
    f = max(f_cpd, 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))

def mtf_eye(f_cpd):
    return float(np.exp(-(f_cpd / 30.0) ** 1.3))

def _perceived_quality_kern(freqs_as, fc_scope, mp_arr, msa_arr, S, V, D_mm, use_csf=True):
    """Gemeinsamer Q_perc-Kern — aus msa/mp-Arrays direkt, kein Doppelrechnen."""
    nu        = freqs_as / fc_scope
    f_cpd_arr = freqs_as * 3600.0 / V
    weight    = np.array([csf(fi) if use_csf else mtf_eye(fi) for fi in f_cpd_arr])
    _trapz    = getattr(np, "trapezoid", getattr(np, "trapz", None))
    Q_raw  = float(_trapz(msa_arr * weight, nu) / max(_trapz(mp_arr * weight, nu), 1e-12))
    V_krit = D_mm * 0.7 * math.sqrt(max(S, 1e-9))
    w      = 1.0 / (1.0 + (V_krit / V) ** 2)
    return float(Q_raw + (1.0 - Q_raw) * (1.0 - w))

def perceived_quality(D_mm, f_mm, lam_nm, V, n_pts=28, use_csf=True):
    freqs_as, fc, mp, msa, _, S = mtf_kurven(D_mm, f_mm, lam_nm, n_pts)
    return _perceived_quality_kern(freqs_as, fc, mp, msa, S, V, D_mm, use_csf)

def perceived_quality_von_S(D_mm, S, lam_nm, V, n_pts=28, use_csf=True):
    freqs_as, fc, mp, msa, _, S = mtf_kurven_von_S(D_mm, S, lam_nm, n_pts)
    return _perceived_quality_kern(freqs_as, fc, mp, msa, S, V, D_mm, use_csf)

def perceived_quality_kurven(D_mm, f_mm, lam_nm, V_arr, use_csf=True):
    return np.array([perceived_quality(D_mm, f_mm, lam_nm, V, use_csf=use_csf) for V in V_arr])

def beurteilung(strehl):
    if strehl >= 0.95:
        return "✓ Sehr gut — nahezu gleichwertig mit Parabolspiegel (Strehl ≥ 0.95)", "#2e7d32"
    elif strehl >= 0.80:
        return ("⚠  Noch beugungsbegrenzt (Maréchal), spürbarer Kontrastverlust bei Planeten  (0.80 ≤ Strehl < 0.95)"), "#e65100"
    else:
        return ("✗  Nicht beugungsbegrenzt — erheblicher Kontrast- und Schärfeverlust, Parabolspiegel dringend empfohlen (Strehl < 0.80)"), "#c62828"

def ax_fmt(ax):
    ax.grid(True, color="#e5e5e5", lw=0.3)
    ax.tick_params(axis="both", labelsize=5, length=2, width=0.5)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMM-FUNKTIONEN (matplotlib, ohne Canvas)
# ═════════════════════════════════════════════════════════════════════════════

def fig_strehl(D, f, lam, S_slide):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    ns, sts, _, _ = kurven_N(D, lam)
    r = berechne(D, f, lam)
    N_akt = r["N"]; S_akt = r["strehl"]
    ax.fill_between(ns, S_akt, 1.0, color=ACC, alpha=0.10)
    ax.axhline(1.00, color=GRN,      lw=0.9, ls="-",  label="Paraboloid (S=1.0)")
    ax.plot(ns, sts, color=ACC, lw=2, label="Strehl (sphärisch)")
    ax.axhline(0.986, color="#1565c0", lw=0.7, ls=":",  label="Rayleigh-Grenze  S≈0.986  (Wp=λ/4)")
    ax.axhline(0.80,  color="#BA7517", lw=0.7, ls="--", label="Maréchal  S=0.80  (beugungsbegrenzt)")
    ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
    ax.scatter([N_akt], [S_akt], color=ACC, s=16, zorder=5)
    ax.annotate(f"f/{N_akt:.1f}  S={S_akt:.3f}", xy=(N_akt, S_akt),
                xytext=(8, 10), textcoords="offset points", fontsize=5, color="#333",
                arrowprops=dict(arrowstyle="-", color="#aaa"))
    if S_slide > S_akt + 0.01:
        ax.axhline(S_slide, color=COR, lw=0.9, ls="-.", label=f"Schieber S={S_slide:.3f}")
    ax.set_xlim(3, 15); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=5)
    ax.set_ylabel("Strehl-Quotient", fontsize=5)
    ax.set_title(f"Strehl vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=5, loc="lower right"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_oeffnung(D, f, lam, S_slide):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    ns, _, deff_ks, aufl_verluste = kurven_N(D, lam)
    r = berechne(D, f, lam)
    N_akt = r["N"]; Dk = r["Deff_k"]; Aufl = r["aufl_verlust_pct"]

    # Linke Achse: Deff_k in mm (blau/ACC)
    ax.fill_between(ns, deff_ks, D, color=ACC, alpha=0.10, label="Kontrast-Verlust (Fläche)")
    ax.axhline(D, color="#555", lw=0.9, ls="-", label=f"Paraboloid = {D:.0f} mm")
    ax.plot(ns, deff_ks, color=ACC, lw=2, label=f"D\u2091\u2091\u2096 Kontrast = D·S\u2070'\u00b2\u2075 [mm]")
    ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
    ax.scatter([N_akt], [Dk], color=ACC, s=18, zorder=5,
               label=f"Messpunkt: {Dk:.0f} mm  (f/{N_akt:.1f})")
    if S_slide > r["strehl"] + 0.01:
        r_sl = berechne_von_S(D, S_slide, lam)
        ax.scatter([N_akt], [r_sl["Deff_k"]], color=ACC, s=28, marker="*", zorder=6,
                   label=f"Schieber Kontrast: {r_sl['Deff_k']:.0f} mm")
    ax.set_xlim(3, 15)
    ax.set_ylim(0, D * 1.08)
    ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=5)
    ax.set_ylabel("Effektive Öffnung D\u2091\u2091\u2096 [mm]", fontsize=5, color=ACC)
    ax.tick_params(axis="y", labelcolor=ACC)
    ax.spines["left"].set_color(ACC)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # Rechte Achse: geometrischer Auflösungsverlust in % (orange/COR)
    ax2 = ax.twinx(); ax2.set_facecolor("none")
    ax2.plot(ns, aufl_verluste, color=COR, lw=1.5, ls="--",
             label="Geom. Aufl.verlust [%]\n(Fotografie, nicht visuell)")
    ax2.scatter([N_akt], [Aufl], color=COR, s=18, zorder=5,
                label=f"Messpunkt: {Aufl:.1f}%")
    if S_slide > r["strehl"] + 0.01:
        # Auflösungsverlust ist geometrisch (f-abhängig) → bleibt beim aktuellen Spiegel,
        # da der Schieber nur S verändert, nicht die Brennweite
        pass
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Geom. Auflösungsverlust [%]", fontsize=5, color=COR)
    ax2.tick_params(axis="y", labelcolor=COR)
    ax2.spines["right"].set_color(COR)

    ax.set_title(f"Eff. Öffnung & Auflösungsverlust vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=5)
    # Legenden kombinieren
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=4.5, loc="lower left")
    ax_fmt(ax); ax_fmt(ax2)
    fig.tight_layout(); return fig

def fig_deff_D(D_akt, N_akt, S_slide):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    D_arr    = np.linspace(50, 400, 300)
    colors_N = {5:"#c62828", 6:"#D85A30", 7:"#c8a020", 8:"#2e7d32", 10:"#1565c0"}
    for N in [5, 6, 7, 8, 10]:
        loss_pct, loss_qvis = [], []
        for D in D_arr:
            r = berechne(D, D * N, 550.0)
            loss_pct.append((1 - r["strehl"]**0.25) * 100)
            loss_qvis.append((1 - r["Qvis"]**0.25) * 100)
        col = colors_N[N]
        ax.plot(D_arr, loss_pct,  color=col, lw=2,   label=f"f/{N}")
        ax.plot(D_arr, loss_qvis, color=col, lw=1.2, ls="--", alpha=0.6)
        ax.text(D_arr[-1]+3, loss_pct[-1], f"f/{N}", color=col, fontsize=5, va="center", fontweight="bold")
    r_akt  = berechne(D_akt, D_akt * N_akt, 550.0)
    S_a    = r_akt["strehl"]
    loss_a = (1 - S_a**0.25) * 100
    ax.scatter([D_akt], [loss_a], color=ACC, s=20, zorder=6)
    ax.annotate(f"D={D_akt:.0f}mm f/{N_akt:.1f}  -{loss_a:.0f}%",
                xy=(D_akt, loss_a), xytext=(D_akt+15, loss_a+5), fontsize=5,
                color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))
    if S_slide > S_a + 0.01:
        loss_sl = (1-S_slide**0.25)*100
        ax.scatter([D_akt], [loss_sl], color=GRN, s=20, marker="*", zorder=7)
        ax.annotate(f"Schieber S={S_slide:.2f}  -{loss_sl:.0f}%",
                    xy=(D_akt, loss_sl), xytext=(D_akt+15, loss_sl-6), fontsize=5,
                    color=GRN, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=GRN, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRN, alpha=0.85))
        ax.annotate("", xy=(D_akt, loss_sl), xytext=(D_akt, loss_a),
                    arrowprops=dict(arrowstyle="->", color=GRN, lw=0.9))
    ax.axhline(20, color="#BA7517", lw=0.9, ls="--", label="20% Verlust")
    ax.axhline(50, color="#c62828", lw=1.0, ls=":",  label="50% Verlust")
    ax.set_xlim(50, 420); ax.set_ylim(0, min(100, max(90, loss_a+15)))
    ax.set_xlabel("Öffnung D [mm]", fontsize=5)
    ax.set_ylabel("Öffnungsverlust [%]", fontsize=5)
    ax.set_title("Öffnungsverlust vs. Öffnung", fontsize=5)
    ax.legend(fontsize=5, loc="upper left"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_beugung(D_akt, f_akt):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    lam_mm = 550e-6
    f_arr  = np.linspace(200, 2000, 500)
    def D_grenz(S):
        Wb = _Wb_von_S(S)
        return (Wb * 4.0 * 1024 * lam_mm) ** 0.25 * f_arr ** 0.75
    D_95 = D_grenz(0.95); D_80 = D_grenz(0.80); D_50 = D_grenz(0.50)
    ax.fill_between(f_arr, 0,    D_95, color="#2e7d32", alpha=0.10)
    ax.fill_between(f_arr, D_95, D_80, color="#f9a825", alpha=0.12)
    ax.fill_between(f_arr, D_80, 300,  color="#c62828", alpha=0.07)
    ax.text(300, 40,  "S >= 0.95  (sehr gut)",            color="#2e7d32", fontsize=5)
    ax.text(300, 155, "0.80 <= S < 0.95  (gut)",          color="#c8860a", fontsize=5)
    ax.text(300, 255, "S < 0.80  (nicht beugungsbegrenzt)", color="#c62828", fontsize=5)
    ax.plot(f_arr, D_95, color="#1565c0", lw=2,   label="S=0.95")
    ax.plot(f_arr, D_80, color="#2e7d32", lw=0.9, label="S=0.80 (Rayleigh)")
    ax.plot(f_arr, D_50, color="#BA7517", lw=0.9, ls="--", label="S=0.50")
    for N in [5, 6, 7, 8, 10]:
        D_fN = np.minimum(f_arr / N, 300)
        ax.plot(f_arr, D_fN, color="#ccc", lw=0.8, ls="--")
        y_label = 1900.0 / N
        if y_label < 285:
            ax.text(1920, y_label, f"f/{N}", color="#999", fontsize=5, va="center")
    ax.scatter([f_akt], [D_akt], color=ACC, s=25, zorder=7)
    x_off = -200 if f_akt > 1500 else 80
    ax.annotate(f"D={D_akt:.0f}mm  f/{f_akt/D_akt:.1f}",
                xy=(f_akt, D_akt), xytext=(f_akt+x_off, D_akt+25),
                fontsize=5, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))
    ax.set_xlim(200, 2000); ax.set_ylim(0, 300)
    ax.set_xlabel("Brennweite f [mm]", fontsize=5)
    ax.set_ylabel("Öffnung D [mm]", fontsize=5)
    ax.set_title("Beugungsgrenze sphärischer Spiegel  (λ=550nm)", fontsize=5)
    ax.legend(fontsize=5, loc="upper left"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_mtf(D, f, lam, S_slide, V, modus="gesamt"):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    N_PTS = 200
    freqs_as, fc, mp_arr, msa_arr, msr_arr, strehl = mtf_kurven(D, f, lam, n_pts=N_PTS)
    freqs_norm = freqs_as / fc
    nu_axis    = freqs_norm

    planet_details = [
        ("Jupiter\nGürtel", 5.0,  "#4a90d9"),
        ("Jupiter\nFestons",1.5,  "#4a90d9"),
        ("Mars\nPol",       1.0,  "#d94a4a"),
        ("Saturn\nCassini", 0.5,  "#c8a020"),
        (f"Dawes\n{116/D:.2f}\"", 116.0/D, "#888"),
    ]

    # ── Klassischer Kurven-Modus ───────────────────────────────────────────────
    if modus == "klassisch":
        N_KURVE  = 300
        nu_k     = np.linspace(0, 1, N_KURVE)
        lam_mm   = lam * 1e-6
        Wp_k     = D**4 / (1024.0 * f**3 * lam_mm)
        Wb_k     = Wp_k / 4.0
        fc_fokal = 1.0 / (lam_mm * (f / D))
        mp_k  = np.array([mtf_para(nu)          for nu in nu_k])
        msr_k = np.array([mtf_sph_rel(nu, Wb_k) for nu in nu_k])
        ax.plot(nu_k, mp_k,     color=GRN, lw=1.5, ls="-",
                label="Paraboloid  (Beugungsgrenze)")
        ax.plot(nu_k, mp_k*msr_k, color=COR, lw=1.5, ls="-",
                label=f"Sphäre absolut  (S={strehl:.3f})")
        ax.plot(nu_k, msr_k,    color=ACC, lw=1.2, ls="--",
                label="Sphäre relativ  (norm. gg. Paraboloid)")
        if S_slide > strehl + 0.01:
            _, _, _, _, msr_s_k, _ = mtf_kurven_von_S(D, S_slide, lam, n_pts=N_KURVE)
            ax.plot(nu_k, mp_k*msr_s_k, color="#1a7abf", lw=1.2, ls="-.",
                    label=f"Schieber absolut  (S={S_slide:.2f})")
        for label, d_as, col in planet_details:
            nu_det = (1.0/(2.0*d_as)) / fc
            if nu_det >= 1.0: continue
            ax.axvline(nu_det, color=col, lw=0.7, ls=":", alpha=0.7)
            ax.text(nu_det+0.005, float(np.interp(nu_det, nu_k, mp_k))-0.07,
                    label.replace("\n"," "), fontsize=4, color=col, rotation=90, va="top")
        ax2x = ax.twiny(); ax2x.set_facecolor("none")
        ax2x.set_xlim(0, fc_fokal)
        ax2x.set_xlabel("Räumliche Frequenz [Lp/mm] (fokal)", fontsize=4)
        ax2x.tick_params(axis="x", labelsize=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Normierte Frequenz ν", fontsize=5)
        ax.set_ylabel("Kontrastübertragung (MTF)", fontsize=5)
        ax.set_title(f"MTF-Kurven — D={D:.0f}mm f/{f/D:.1f}  Strehl={strehl:.3f}  "
                     f"fc={fc_fokal:.1f}Lp/mm", fontsize=5)
        ax.text(0.98, 0.97, "Relativ = wie telescope-optics.net",
                transform=ax.transAxes, ha="right", va="top", fontsize=4,
                color=ACC, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ACC, alpha=0.8))
        ax.legend(fontsize=4, loc="lower left"); ax_fmt(ax)
        fig.tight_layout(); return fig

    # ── Balken-Modi: absolut / gesamt / relativ ────────────────────────────────
    labels, mp_vals = [], []
    ms_vals_mtf, ms_vals_ges = [], []
    verlust_mtf, verlust_ges = [], []
    bar_colors, detail_as   = [], []
    for label, d_as, col in planet_details:
        fn = 1.0 / (2.0 * d_as * fc)
        if fn >= 1.0: continue
        mp_v   = float(np.interp(fn, nu_axis, mp_arr))
        msr_v  = float(np.interp(fn, nu_axis, msr_arr))
        ms_mtf = mp_v * msr_v
        ms_ges = mp_v * msr_v * strehl
        vl_mtf = (mp_v - ms_mtf) / mp_v * 100.0 if mp_v > 1e-9 else 0.0
        vl_ges = (mp_v - ms_ges) / mp_v * 100.0 if mp_v > 1e-9 else 0.0
        labels.append(label); mp_vals.append(mp_v)
        ms_vals_mtf.append(ms_mtf); ms_vals_ges.append(ms_ges)
        verlust_mtf.append(vl_mtf); verlust_ges.append(vl_ges)
        bar_colors.append(col); detail_as.append(d_as)

    ms_vals     = ms_vals_ges if modus == "gesamt" else ms_vals_mtf
    verlust_vals = verlust_ges if modus == "gesamt" else verlust_mtf

    ms_s = None
    if S_slide > strehl + 0.01:
        _, fc_s, mp_s_arr, _, msr_s_arr, _ = mtf_kurven_von_S(D, S_slide, lam, n_pts=N_PTS)
        nu_s = np.linspace(0, 1, len(msr_s_arr))
        ms_s = []
        for d_as in detail_as:
            fn  = 1.0 / (2.0 * d_as * fc)
            mp_v  = float(np.interp(fn, nu_axis, mp_arr))
            msr_v = float(np.interp(fn, nu_s, msr_s_arr))
            ms_s.append(mp_v * msr_v * S_slide if modus == "gesamt" else mp_v * msr_v)

    n = len(labels); x = np.arange(n)

    if modus in ("absolut", "gesamt"):
        w = 0.28 if ms_s else 0.35
        ax.bar(x-w/2, mp_vals, w, label="Paraboloid", color="#888", alpha=0.75)
        ax.bar(x+w/2, ms_vals, w, label=f"Sphäre (S={strehl:.3f})", color=COR, alpha=0.85)
        if ms_s:
            ax.bar(x+w/2, ms_s, w, label=f"Schieber (S={S_slide:.2f})", color=GRN, alpha=0.75)
        for i, (mp_v, ms_v) in enumerate(zip(mp_vals, ms_vals)):
            ax.text(i-w/2, mp_v+0.02, f"{mp_v:.2f}", ha="center", fontsize=5, color="#444")
            ax.text(i+w/2, ms_v+0.02, f"{ms_v:.2f}", ha="center", fontsize=5,
                    color=COR, fontweight="bold")
        ax.axhline(0.2, color="#BA7517", lw=0.7, ls="-", zorder=5)
        ax.text(0.01, 0.215, "20%-Schwelle", transform=ax.get_yaxis_transform(),
                fontsize=5, fontweight="bold", color="#BA7517")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Kontrastübertragung", fontsize=5)
        if modus == "absolut":
            ax.set_title(f"MTF absolut (Standard, ohne Strehl) — "
                         f"D={D:.0f}mm f/{f/D:.1f}  Strehl={strehl:.3f}", fontsize=5)
            ax.text(0.02, 0.97, f"Strehl={strehl:.3f} nicht eingerechnet\n"
                    "Vergleichbar mit Zemax / telescope-optics.net",
                    transform=ax.transAxes, ha="left", va="top", fontsize=4,
                    color=ACC, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ACC, alpha=0.8))
        else:
            ax.set_title(f"MTF × Strehl (Beobachtungsqualität) — "
                         f"D={D:.0f}mm f/{f/D:.1f}  Strehl={strehl:.3f}", fontsize=5)
            ax.text(0.02, 0.97, f"Strehl={strehl:.3f} eingerechnet\n= effektiver Kontrast am Okular",
                    transform=ax.transAxes, ha="left", va="top", fontsize=4,
                    color=COR, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COR, alpha=0.8))

    else:  # relativ
        w = 0.4 if not ms_s else 0.3
        ax.bar(x, verlust_vals, width=w, color=bar_colors, alpha=0.85,
               label=f"Sphäre (S={strehl:.3f})")
        if ms_s:
            verlust_s = [(mp_v-ms_v)/max(mp_v,1e-9)*100
                         for mp_v, ms_v in zip(mp_vals, ms_s)]
            ax.bar(x+w, verlust_s, width=w, color=GRN, alpha=0.75,
                   label=f"Schieber (S={S_slide:.2f})")
            for i, v in enumerate(verlust_s):
                ax.text(i+w, v+0.3, f"{v:.1f}%", ha="center", fontsize=5,
                        color=GRN, fontweight="bold")
        for i, v in enumerate(verlust_vals):
            ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=5,
                    fontweight="bold", color=bar_colors[i])
        ymax = max(verlust_vals)*1.35+3 if verlust_vals else 30
        ax.axhline(20, color="#BA7517", lw=0.7, ls="-", zorder=5)
        ax.text(0.01, 21.5, "20%-Schwelle", transform=ax.get_yaxis_transform(),
                fontsize=5, fontweight="bold", color="#BA7517")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Kontrastverlust gg. Paraboloid [%]", fontsize=5)
        ax.set_title(f"Kontrastverlust (MTF-Form, ohne Strehl) — "
                     f"D={D:.0f}mm f/{f/D:.1f}  Strehl={strehl:.3f}", fontsize=5)

    # CSF-Overlay (nur Balkenmodi)
    if detail_as:
        csf_vals = [csf(3600.0/(2.0*d*V)) for d in detail_as]
        ax2 = ax.twinx(); ax2.set_facecolor("none")
        csf_col = "#1a7abf"
        ax2.plot(x, csf_vals, color=csf_col, lw=1.3, ls="-.", marker="o", ms=3, zorder=8,
                 label=f"CSF  ({V:.0f}×)")
        for i, cv in enumerate(csf_vals):
            ax2.text(i, cv+0.03, f"{cv:.2f}", ha="center", fontsize=5,
                     color=csf_col, fontweight="bold")
        ax2.set_ylim(0, 1.35)
        ax2.set_ylabel("Augenempfindlichkeit CSF", fontsize=5, color=csf_col)
        ax2.tick_params(axis="y", labelcolor=csf_col)
        ax2.spines["right"].set_color(csf_col)
        ax2.legend(fontsize=5, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5, linespacing=1.3)
    ax.legend(fontsize=5, loc="upper right"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_wahrnehmung(D, f, lam, V_slider, S_slide, S_real):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    V_arr = np.linspace(30, 250, 100)
    Qp_sph  = perceived_quality_kurven(D, f, lam, V_arr)
    Qp_para = perceived_quality_kurven(D, D*50, lam, V_arr)
    has_slide = S_slide > S_real + 0.01
    if has_slide:
        Qp_slide = np.array([perceived_quality_von_S(D, S_slide, lam, V) for V in V_arr])
        ax.fill_between(V_arr, Qp_sph,   Qp_para, color=COR, alpha=0.10)
        ax.fill_between(V_arr, Qp_slide, Qp_para, color=GRN, alpha=0.10)
    else:
        ax.fill_between(V_arr, Qp_sph, Qp_para, color=COR, alpha=0.13, label="Wahrnehmungsverlust")
    ax.plot(V_arr, Qp_para, color=GRN, lw=0.7, ls="--", label="Paraboloid (Referenz)")
    ax.plot(V_arr, Qp_sph,  color=COR, lw=0.7, ls=":",  label=f"Sphäre f/{f/D:.1f}  S={S_real:.3f}")
    if has_slide:
        ax.plot(V_arr, Qp_slide, color=ACC, lw=0.9, label=f"Schieber  S={S_slide:.3f}")
    q_curve = Qp_slide if has_slide else Qp_sph
    for V_mark, col, ls in [(30, "#c8a020", ":"), (80, "#888", "-."), (160, "#c62828", ":")]:
        idx = int(round((V_mark-30)/(250-30)*(len(V_arr)-1)))
        q_m = float(q_curve[idx])
        ax.axvline(V_mark, color=col, lw=0.7, ls=ls, alpha=0.6)
        ax.scatter([V_mark], [q_m], color=col, s=14, zorder=6)
        ax.annotate(f"{V_mark}×\n{q_m:.2f}", xy=(V_mark, q_m),
                    xytext=(V_mark+7, q_m-0.05), fontsize=5, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.80))
    q_v = perceived_quality_von_S(D, S_slide, lam, V_slider) if has_slide else perceived_quality(D, f, lam, V_slider)
    ax.axvline(V_slider, color=ACC, lw=1.0, ls="-", alpha=0.9)
    ax.scatter([V_slider], [q_v], color=ACC, s=22, zorder=7)
    x_off = -65 if V_slider > 200 else 12
    ax.annotate(f"V={V_slider:.0f}×\nQ={q_v:.3f}", xy=(V_slider, q_v),
                xytext=(V_slider+x_off, q_v+0.04), fontsize=5, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.88))
    for q_thresh, col, label in [(0.9, GRN, "Q=0.90 (sehr gut)"), (0.7, "#BA7517", "Q=0.70 (spürbar)")]:
        ax.axhline(q_thresh, color=col, lw=1.0, ls=":", alpha=0.7)
        ax.text(32, q_thresh+0.008, label, fontsize=5, color=col, alpha=0.85)
    ax.set_xlim(30, 250); ax.set_ylim(0, 1.12)
    ax.set_xlabel("Vergrößerung V  [×]", fontsize=5)
    ax.set_ylabel("Wahrgenommener Qualitätsindex Q_perc", fontsize=5)
    ax.set_title(f"Wahrgenommener Kontrast (CSF-gewichtet)  -  D={D:.0f}mm  f/{f/D:.1f}", fontsize=5)
    ax.legend(fontsize=5, loc="lower right"); ax_fmt(ax)
    fig.tight_layout(); return fig


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Newton-Spiegel Rechner", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem}</style>",
            unsafe_allow_html=True)
st.title("Newton-Spiegel Rechner")
st.caption(f"v{VERSION}  ·  Kontrast- und Schärfeverlust sphärischer Hauptspiegel")

with st.expander("ℹ️ Über dieses Programm"):
    st.caption(f"Version {VERSION}")
    import re as _re, os as _os
    _wp_candidates = ["whitepaper.html", "/app/whitepaper.html"]
    _wp_path = next((p for p in _wp_candidates if _os.path.exists(p)), None)
    if _wp_path:
        _wp_html = open(_wp_path, encoding="utf-8").read()
        _wp_html = _re.sub(r'@page\s*\{[^}]*\}', '', _wp_html)
        _body = _re.search(r'<body[^>]*>(.*?)</body>', _wp_html, _re.DOTALL)
        _style = _re.search(r'<style>(.*?)</style>', _wp_html, _re.DOTALL)
        _css = _style.group(1) if _style else ""
        st.html(f"<style>{_css}</style>{_body.group(1)}" if _body else _wp_html)
    else:
        st.warning("whitepaper.html nicht gefunden — bitte Datei im Programmverzeichnis ablegen.")

with st.expander("📐 Technische Dokumentation"):
    doc_html = open("/app/documentation_v43.html").read() if __import__('os').path.exists("/app/documentation_v43.html") else open("documentation_v43.html").read()
    # Entferne @page CSS (nicht relevant im Browser) und body-Margins
    import re
    doc_html = re.sub(r'@page\s*\{[^}]*\}', '', doc_html)
    doc_html = re.sub(r'margin:\s*-20mm[^;]+;', 'margin: 0;', doc_html)
    # Nur den Body-Inhalt extrahieren
    body_match = re.search(r'<body[^>]*>(.*?)</body>', doc_html, re.DOTALL)
    if body_match:
        body_content = body_match.group(1)
        style_match  = re.search(r'<style>(.*?)</style>', doc_html, re.DOTALL)
        style        = style_match.group(1) if style_match else ""
        st.html(f"<style>{style}</style>{body_content}")
    else:
        st.html(doc_html)

# ── Seitenleiste: Eingaben ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Teleskop-Parameter")
    D   = st.slider("Öffnung D [mm]",       50,  500, 200, step=10)
    f   = st.slider("Brennweite f [mm]",    200, 4000, 1000, step=50)
    lam = 550.0  # fest

    st.divider()
    st.subheader("Spiegel-Qualität")
    S_pct = st.slider("Strehl-Schieber [%]", 0, 95, 0, step=5,
                      help="0 % = theoretische Sphäre aus D/f  |  95 % = fast Paraboloid")

    st.divider()
    st.subheader("Beobachtung")
    V_slider = st.slider("Vergrößerung [×]", 30, 250, 150, step=10)
    eye_res  = st.slider("Augen-Auflösung [\"]", 20, 180, 60, step=5)

# ── Berechnungen ──────────────────────────────────────────────────────────────
r      = berechne(D, f, lam)
S_real = r["strehl"]
S_slide = S_real + (S_pct / 100.0) * (1.0 - S_real)
S_slide = min(S_slide, 0.9999)

rv  = berechne_vergr(D, f, lam, V_slider, eye_res)
Vk  = v_kritisch(D, f, lam, eye_res)

Qp_30  = perceived_quality(D, f, lam, 30.0)
Qp_80  = perceived_quality(D, f, lam, 80.0)
Qp_160 = perceived_quality(D, f, lam, 160.0)
Qp_V   = perceived_quality(D, f, lam, V_slider)

verdict_text, verdict_color = beurteilung(S_real)

# ── Ergebnisse kompakt ───────────────────────────────────────────────────────
border = verdict_color  # already a hex color string
st.markdown(
    f"<div style='background:#f0f0f0;padding:7px 14px;border-radius:6px;"
    f"border-left:4px solid {border};font-size:0.85em;margin-bottom:8px'>"
    f"{verdict_text}</div>", unsafe_allow_html=True)

def row(label, val, sub=""):
    sub_html = f"<span style='color:#888;font-size:0.8em'> {sub}</span>" if sub else ""
    return f"<td style='padding:2px 14px 2px 0;white-space:nowrap'><b>{label}</b></td><td style='padding:2px 20px 2px 0;white-space:nowrap'>{val}{sub_html}</td>"

st.markdown(f"""
<table style='font-size:0.88em;border-collapse:collapse;width:100%'>
<tr>
  {row("f/D", f"{r['N']:.1f}")}
  {row("Strehl", f"{S_real:.4f}")}
  {row("D_eff Kontrast", f"{r['Deff_k']:.1f} mm", f"−{r['loss_k']:.1f} mm")}
  {row("D_eff Schärfe", f"{r['Deff_s']:.1f} mm", f"−{r['loss_s']:.1f} mm")}
  {row("V_krit", f"{Vk['Vk_sph']:.0f}×")}
</tr>
<tr>
  {row("Q₀.₂", f"{r['Q02']:.4f}")}
  {row("Q₀.₄", f"{r['Q04']:.4f}")}
  {row("Q₀.₆", f"{r['Q06']:.4f}")}
  {row("Q_vis", f"{r['Qvis']:.4f}")}
  {row(f"D_eff Sphäre ({V_slider}×)", f"{rv['Deff_sph']:.1f} mm", f"−{rv['verlust']:.1f} mm")}
</tr>
<tr>
  {row("Q_perc 30×",  f"{Qp_30:.3f}")}
  {row("Q_perc 80×",  f"{Qp_80:.3f}")}
  {row("Q_perc 160×", f"{Qp_160:.3f}")}
  {row(f"Q_perc {V_slider}×", f"{Qp_V:.3f}")}
  <td></td><td></td>
</tr>
</table>
""", unsafe_allow_html=True)

st.divider()

# ── Diagramme ─────────────────────────────────────────────────────────────────
st.subheader("Diagramme")
diagramm = st.selectbox("Diagramm wählen", [
    "Wahrnehmung",
    "MTF",
    "Strehl vs. f/D",
    "Eff. Öffnung vs. f/D",
    "D_eff vs. Öffnung",
    "Beugungsgrenze",
])

if diagramm == "Strehl vs. f/D":
    fig = fig_strehl(D, f, lam, S_slide)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

elif diagramm == "Eff. Öffnung vs. f/D":
    fig = fig_oeffnung(D, f, lam, S_slide)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

elif diagramm == "D_eff vs. Öffnung":
    fig = fig_deff_D(D, r["N"], S_slide)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

elif diagramm == "Beugungsgrenze":
    fig = fig_beugung(D, f)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

elif diagramm == "MTF":
    mtf_modus = st.radio(
        "Darstellung",
        ["MTF × Strehl  (Beobachtungsqualität)",
         "MTF absolut  (Standard, ohne Strehl)",
         "Kontrastverlust %",
         "Klassisch  (Kurven, Vergleich mit Optikrechnern)"],
        horizontal=True, index=0, label_visibility="collapsed")
    modus_key = {"MTF × Strehl  (Beobachtungsqualität)":        "gesamt",
                 "MTF absolut  (Standard, ohne Strehl)":        "absolut",
                 "Kontrastverlust %":                           "relativ",
                 "Klassisch  (Kurven, Vergleich mit Optikrechnern)": "klassisch"}[mtf_modus]
    fig = fig_mtf(D, f, lam, S_slide, V_slider, modus=modus_key)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

elif diagramm == "Wahrnehmung":
    fig = fig_wahrnehmung(D, f, lam, V_slider, S_slide, S_real)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.caption("Formeln: Maréchal-Näherung * MTF via Pupillen-Autokorrelation * CSF nach van Meeteren * "
           "Quellen: telescope-optics.net * gordtulloch.com")
