"""
Newton-Teleskop Rechner — Streamlit-Version
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

# ═════════════════════════════════════════════════════════════════════════════
# KERNFUNKTIONEN (identisch mit newton_spiegel_rechner.py)
# ═════════════════════════════════════════════════════════════════════════════

def berechne(D_mm, f_mm, lam_nm):
    N      = f_mm / D_mm
    D_inch = D_mm / 25.4
    lam_mm = lam_nm * 1e-6
    Wp     = D_mm**4 / (1024.0 * f_mm**3 * lam_mm)
    Wb     = Wp / 4.0
    Wrms   = Wb / (1.5 * math.sqrt(5))
    S      = math.exp(-(2 * math.pi * Wrms) ** 2)
    Deff_k = D_mm * S**0.25
    r_airy_as   = 1.22 * lam_mm / D_mm * 206265.0
    blur_airy   = Wb / 2.44
    blur_as     = blur_airy * 2.0 * r_airy_as
    theta_ideal = 116.0 / D_mm
    theta_eff   = math.sqrt(theta_ideal**2 + blur_as**2)
    Deff_s      = 116.0 / theta_eff
    Q02 = mtf_sph_rel(0.2, Wb)
    Q04 = mtf_sph_rel(0.4, Wb)
    Q06 = mtf_sph_rel(0.6, Wb)
    Qshape = 0.4*Q02 + 0.4*Q04 + 0.2*Q06
    Qvis = S * Qshape
    Deff_vis = D_mm * Qvis**0.25
    return dict(
        N=N, D_inch=D_inch, Wp=Wp, Wb=Wb, Wrms=Wrms,
        strehl=S, Deff_k=Deff_k, loss_k=D_mm - Deff_k,
        r_airy_as=r_airy_as, blur_airy=blur_airy, blur_as=blur_as,
        theta_ideal=theta_ideal, theta_eff=theta_eff,
        Deff_s=Deff_s, loss_s=D_mm - Deff_s,
        Q02=Q02, Q04=Q04, Q06=Q06, Qshape=Qshape,
        Qvis=Qvis, Deff_vis=Deff_vis, loss_vis=D_mm - Deff_vis,
    )

def berechne_vergr(D_mm, f_mm, lam_nm, V, eye_res_as=60.0):
    r  = berechne(D_mm, f_mm, lam_nm)
    theta_auge = eye_res_as / V
    theta_para = max(r["theta_ideal"], theta_auge)
    theta_sph  = max(r["theta_eff"],   theta_auge)
    Deff_para = min(D_mm, 116.0 / theta_para)
    Deff_sph  = min(D_mm, 116.0 / theta_sph)
    return dict(theta_auge=theta_auge,
                theta_para=theta_para, theta_sph=theta_sph,
                Deff_para=Deff_para, Deff_sph=Deff_sph,
                verlust=Deff_para - Deff_sph, **r)

def v_kritisch(D_mm, f_mm, lam_nm, eye_res_as=60.0):
    r = berechne(D_mm, f_mm, lam_nm)
    return dict(
        Vk_sph  = eye_res_as / r["theta_eff"],
        Vk_para = eye_res_as / r["theta_ideal"],
        Vmax    = D_mm / 0.5,
        Vmin    = D_mm / 7.0,
    )

def kurven_N(D_mm, lam_nm, N_min=3.0, N_max=15.0, schritte=400):
    ns = np.linspace(N_min, N_max, schritte)
    strehls, deff_ks, deff_ss = [], [], []
    for n in ns:
        r = berechne(D_mm, D_mm * n, lam_nm)
        strehls.append(r["strehl"]); deff_ks.append(r["Deff_k"]); deff_ss.append(r["Deff_s"])
    return ns, np.array(strehls), np.array(deff_ks), np.array(deff_ss)

def strehl_to_f(S, D_mm, lam_nm=550.0):
    lam_mm = lam_nm * 1e-6
    if S >= 0.9999:
        return D_mm * 50.0
    Wrms = math.sqrt(-math.log(max(S, 1e-6))) / (2.0 * math.pi)
    Wb   = Wrms * 1.5 * math.sqrt(5)
    Wp   = Wb * 4.0
    f3   = D_mm**4 / (1024.0 * Wp * lam_mm)
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

def mtf_kurven(D_mm, f_mm, lam_nm=550.0, n_pts=60):
    Wp     = D_mm**4 / (1024.0 * f_mm**3 * (lam_nm*1e-6))
    Wb     = Wp / 4.0
    Wrms   = Wb / (1.5 * math.sqrt(5))
    strehl = math.exp(-(2 * math.pi * Wrms)**2)
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    freqs_norm = np.linspace(0, 1, n_pts)
    freqs_as   = freqs_norm * fc
    mp  = np.array([mtf_para(f)        for f in freqs_norm])
    msr = np.array([mtf_sph_rel(f, Wb) for f in freqs_norm])
    msa = msr * mp * strehl
    return freqs_as, fc, mp, msa, msr, strehl

def csf(f_cpd):
    f = max(f_cpd, 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))

def mtf_eye(f_cpd):
    return float(np.exp(-(f_cpd / 30.0) ** 1.3))

def perceived_quality(D_mm, f_mm, lam_nm, V, n_pts=120, use_csf=True):
    freqs_as, fc_scope, mp_arr, msa_arr, msr_arr, strehl = mtf_kurven(D_mm, f_mm, lam_nm, n_pts)
    nu        = freqs_as / fc_scope
    f_cpd_arr = freqs_as * 3600.0 / V
    weight    = np.array([csf(fi) if use_csf else mtf_eye(fi) for fi in f_cpd_arr])
    msa_scope = msr_arr * mp_arr * strehl
    _trapz    = getattr(np, "trapezoid", getattr(np, "trapz", None))
    num   = _trapz(msa_scope * weight, nu)
    denom = _trapz(mp_arr    * weight, nu)
    Q_raw = float(num / denom) if denom > 1e-12 else 0.0
    lam_mm    = lam_nm * 1e-6
    Wp        = D_mm**4 / (1024.0 * f_mm**3 * lam_mm)
    Wb        = Wp / 4.0
    r_airy    = 1.22 * lam_mm / D_mm * 206265.0
    blur_as   = (Wb / 2.44) * 2.0 * r_airy
    theta_eff = math.sqrt((116.0/D_mm)**2 + blur_as**2)
    V_krit    = 60.0 / theta_eff
    w = 1.0 / (1.0 + (V_krit / V) ** 2)
    return float(Q_raw + (1.0 - Q_raw) * (1.0 - w))

def perceived_quality_kurven(D_mm, f_mm, lam_nm, V_arr, use_csf=True):
    return np.array([perceived_quality(D_mm, f_mm, lam_nm, V, use_csf=use_csf) for V in V_arr])

def beurteilung(strehl):
    if strehl >= 0.95:
        return "✅ Sehr gut — nahezu gleichwertig mit Parabolspiegel (Strehl ≥ 0.95)", "green"
    elif strehl >= 0.80:
        return "⚠️ Noch beugungsbegrenzt (Rayleigh), spürbarer Kontrastverlust (0.80 ≤ Strehl < 0.95)", "orange"
    else:
        return "❌ Nicht beugungsbegrenzt — erheblicher Kontrast- und Schärfeverlust (Strehl < 0.80)", "red"

def ax_fmt(ax):
    ax.grid(True, color="#ddd", lw=0.5)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMM-FUNKTIONEN (matplotlib, ohne Canvas)
# ═════════════════════════════════════════════════════════════════════════════

def fig_strehl(D, f, lam, S_slide):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    ns, sts, _, _ = kurven_N(D, lam)
    r = berechne(D, f, lam)
    N_akt = r["N"]; S_akt = r["strehl"]
    ax.fill_between(ns, S_akt, 1.0, color=ACC, alpha=0.10)
    ax.axhline(1.00, color=GRN,      lw=1.5, ls="-",  label="Paraboloid (S=1.0)")
    ax.plot(ns, sts, color=ACC, lw=2, label="Strehl (sphärisch)")
    ax.axhline(0.80, color="#BA7517", lw=1.2, ls="--", label="Rayleigh S=0.80")
    ax.axhline(0.95, color="#888",    lw=1.0, ls=":",  label="S=0.95")
    ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
    ax.scatter([N_akt], [S_akt], color=ACC, s=60, zorder=5)
    ax.annotate(f"f/{N_akt:.1f}  S={S_akt:.3f}", xy=(N_akt, S_akt),
                xytext=(8, 10), textcoords="offset points", fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="-", color="#aaa"))
    if S_slide > S_akt + 0.01:
        ax.axhline(S_slide, color=COR, lw=1.5, ls="-.", label=f"Schieber S={S_slide:.3f}")
    ax.set_xlim(3, 15); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=10)
    ax.set_ylabel("Strehl-Quotient", fontsize=10)
    ax.set_title(f"Strehl vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=9, loc="lower right"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_oeffnung(D, f, lam, S_slide):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    ns, _, deff_ks, deff_ss = kurven_N(D, lam)
    r = berechne(D, f, lam)
    N_akt = r["N"]; Dk = r["Deff_k"]; Ds = r["Deff_s"]
    ax.fill_between(ns, deff_ks, D, color=ACC, alpha=0.10, label="Kontrast-Verlust")
    ax.fill_between(ns, deff_ss, D, color=COR, alpha=0.10, label="Schärfe-Verlust")
    ax.axhline(D, color="#555", lw=1.5, ls="-", label=f"Paraboloid = {D:.0f}mm")
    ax.plot(ns, deff_ks, color=ACC, lw=2, label="Eff. Öffnung (Kontrast)")
    ax.plot(ns, deff_ss, color=COR, lw=2, label="Eff. Öffnung (Schärfe)")
    ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
    ax.scatter([N_akt], [Dk], color=ACC, s=60, zorder=5)
    ax.scatter([N_akt], [Ds], color=COR, s=60, zorder=5)
    if S_slide > r["strehl"] + 0.01:
        f_sl = strehl_to_f(S_slide, D, lam)
        r_sl = berechne(D, f_sl, lam)
        ax.scatter([N_akt], [r_sl["Deff_k"]], color=ACC, s=120, marker="*", zorder=6,
                   label=f"Schieber Kontrast {r_sl['Deff_k']:.0f}mm")
        ax.scatter([N_akt], [r_sl["Deff_s"]], color=COR, s=120, marker="*", zorder=6,
                   label=f"Schieber Schärfe {r_sl['Deff_s']:.0f}mm")
    ax.set_xlim(3, 15)
    ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=10)
    ax.set_ylabel("Effektive Öffnung [mm]", fontsize=10)
    ax.set_title(f"Eff. Öffnung vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=9, loc="lower right"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_deff_D(D_akt, N_akt, S_slide):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    D_arr = np.linspace(50, 400, 300)
    colors_N = {5:"#c62828", 6:"#D85A30", 7:"#c8a020", 8:"#2e7d32", 10:"#1565c0"}
    for N in [5, 6, 7, 8, 10]:
        loss_pct = []
        for D in D_arr:
            Wp = D**4 / (1024.0*(D*N)**3*550e-6); Wb=Wp/4
            S = math.exp(-(2*math.pi*Wb/(1.5*math.sqrt(5)*4))**2)
            Wrms = Wb/(1.5*math.sqrt(5)); S = math.exp(-(2*math.pi*Wrms)**2)
            loss_pct.append((1-S**0.25)*100)
        col = colors_N[N]
        ax.plot(D_arr, loss_pct, color=col, lw=2, label=f"f/{N}")
        ax.text(D_arr[-1]+3, loss_pct[-1], f"f/{N}", color=col, fontsize=9, va="center", fontweight="bold")
    f_akt = D_akt*N_akt
    Wp_a = D_akt**4/(1024.0*f_akt**3*550e-6); Wb_a=Wp_a/4
    Wrms_a = Wb_a/(1.5*math.sqrt(5)); S_a = math.exp(-(2*math.pi*Wrms_a)**2)
    loss_a = (1-S_a**0.25)*100
    ax.scatter([D_akt], [loss_a], color=ACC, s=80, zorder=6)
    ax.annotate(f"D={D_akt:.0f}mm f/{N_akt:.1f}  -{loss_a:.0f}%",
                xy=(D_akt, loss_a), xytext=(D_akt+15, loss_a+5), fontsize=9,
                color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))
    if S_slide > S_a + 0.01:
        loss_sl = (1-S_slide**0.25)*100
        ax.scatter([D_akt], [loss_sl], color=GRN, s=80, marker="*", zorder=7)
        ax.annotate(f"Schieber S={S_slide:.2f}  -{loss_sl:.0f}%",
                    xy=(D_akt, loss_sl), xytext=(D_akt+15, loss_sl-6), fontsize=9,
                    color=GRN, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=GRN, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRN, alpha=0.85))
        ax.annotate("", xy=(D_akt, loss_sl), xytext=(D_akt, loss_a),
                    arrowprops=dict(arrowstyle="->", color=GRN, lw=1.5))
    ax.axhline(20, color="#BA7517", lw=1.5, ls="--", label="20% Verlust")
    ax.axhline(50, color="#c62828", lw=1.0, ls=":",  label="50% Verlust")
    ax.set_xlim(50, 420); ax.set_ylim(0, min(100, max(90, loss_a+15)))
    ax.set_xlabel("Öffnung D [mm]", fontsize=10)
    ax.set_ylabel("Öffnungsverlust [%]", fontsize=10)
    ax.set_title("Öffnungsverlust vs. Öffnung", fontsize=10)
    ax.legend(fontsize=9, loc="upper left"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_beugung(D_akt, f_akt):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    lam_mm = 550e-6
    f_arr  = np.linspace(200, 2000, 500)
    def D_grenz(S):
        Wrms = math.sqrt(-math.log(S)) / (2*math.pi)
        Wp   = Wrms * 1.5 * math.sqrt(5) * 4
        return (Wp * 1024 * lam_mm) ** 0.25 * f_arr ** 0.75
    D_95 = D_grenz(0.95); D_80 = D_grenz(0.80); D_50 = D_grenz(0.50)
    ax.fill_between(f_arr, 0,    D_95, color="#2e7d32", alpha=0.10)
    ax.fill_between(f_arr, D_95, D_80, color="#f9a825", alpha=0.12)
    ax.fill_between(f_arr, D_80, 300,  color="#c62828", alpha=0.07)
    ax.text(300, 40,  "S ≥ 0.95  (sehr gut)",            color="#2e7d32", fontsize=8)
    ax.text(300, 155, "0.80 ≤ S < 0.95  (gut)",          color="#c8860a", fontsize=8)
    ax.text(300, 255, "S < 0.80  (nicht beugungsbegrenzt)", color="#c62828", fontsize=8)
    ax.plot(f_arr, D_95, color="#1565c0", lw=2,   label="S=0.95")
    ax.plot(f_arr, D_80, color="#2e7d32", lw=2.5, label="S=0.80 (Rayleigh)")
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
                xy=(f_akt, D_akt), xytext=(f_akt+x_off, D_akt+25),
                fontsize=9, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))
    ax.set_xlim(200, 2000); ax.set_ylim(0, 300)
    ax.set_xlabel("Brennweite f [mm]", fontsize=10)
    ax.set_ylabel("Öffnung D [mm]", fontsize=10)
    ax.set_title("Beugungsgrenze sphärischer Spiegel  (λ=550nm)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_mtf(D, f, lam, S_slide, V):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    freqs_as, fc, mp_arr, msa, msr_arr, strehl = mtf_kurven(D, f, lam)
    planet_details = [
        ("Jupiter\nGürtel", 5.0,  "#4a90d9"),
        ("Jupiter\nFestons",1.5,  "#4a90d9"),
        ("Mars\nPol",       1.0,  "#d94a4a"),
        ("Saturn\nCassini", 0.5,  "#c8a020"),
        (f"Dawes\n{116/D:.2f}\"", 116.0/D, "#888"),
    ]
    def mtf_at(fn, arr):
        idx = min(int(round(fn*(len(arr)-1))), len(arr)-1)
        return arr[idx]
    labels, mp_vals, ms_vals, bar_colors, detail_as = [], [], [], [], []
    for label, d_as, col in planet_details:
        fn = 1.0 / (2.0 * d_as * fc)
        if fn >= 1.0: continue
        mp_v  = mtf_para(fn)
        msr_v = mtf_at(fn, msr_arr)
        strehl_eff = 1.0 - (1.0 - strehl) * fn
        ms_v  = msr_v * mp_v * strehl_eff
        labels.append(label); mp_vals.append(mp_v); ms_vals.append(ms_v)
        bar_colors.append(col); detail_as.append(d_as)
    ms_s = None
    if S_slide > strehl + 0.01:
        f_sl = strehl_to_f(S_slide, D, lam)
        _, _, _, _, msr_s, _ = mtf_kurven(D, f_sl, lam)
        ms_s = []
        for d_as in detail_as:
            fn = 1.0/(2.0*d_as*fc)
            msr_v = mtf_at(fn, msr_s)
            strehl_eff = 1.0 - (1.0 - S_slide) * fn
            ms_s.append(msr_v * mtf_para(fn) * strehl_eff)
    n = len(labels); x = np.arange(n)
    w = 0.28 if ms_s else 0.35
    ax.bar(x - w/2, mp_vals, w, label="Paraboloid", color="#888", alpha=0.75)
    ax.bar(x + w/2, ms_vals, w, label=f"Sphäre (S={strehl:.3f})", color=COR, alpha=0.85)
    if ms_s:
        ax.bar(x + w/2, ms_s, w, label=f"Schieber (S={S_slide:.2f})", color=GRN, alpha=0.75)
    for i, (mp_v, ms_v) in enumerate(zip(mp_vals, ms_vals)):
        ax.text(i-w/2, mp_v+0.02, f"{mp_v:.2f}", ha="center", fontsize=8, color="#444")
        ax.text(i+w/2, ms_v+0.02, f"{ms_v:.2f}", ha="center", fontsize=8, color=COR, fontweight="bold")
    ax.axhline(0.2, color="#BA7517", lw=2.0, ls="-", zorder=5)
    ax.text(0.01, 0.215, "20%-Schwelle", transform=ax.get_yaxis_transform(),
            fontsize=9, fontweight="bold", color="#BA7517")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Kontrastübertragung (MTF)", fontsize=10)
    ax.set_title(f"MTF je Detail — D={D:.0f}mm f/{f/D:.1f}  Strehl={strehl:.3f}", fontsize=10)
    # CSF-Overlay
    if detail_as:
        csf_vals = [csf(3600.0/(2.0*d_as*V)) for d_as in detail_as]
        ax2 = ax.twinx(); ax2.set_facecolor("none")
        csf_col = "#1a7abf"
        ax2.plot(x, csf_vals, color=csf_col, lw=2.2, ls="-.", marker="o", ms=6, zorder=8,
                 label=f"CSF  ({V:.0f}×)")
        for i, cv in enumerate(csf_vals):
            ax2.text(i, cv+0.03, f"{cv:.2f}", ha="center", fontsize=8, color=csf_col, fontweight="bold")
        ax2.set_ylim(0, 1.35)
        ax2.set_ylabel("Augenempfindlichkeit CSF", fontsize=9, color=csf_col)
        ax2.tick_params(axis="y", labelcolor=csf_col)
        ax2.spines["right"].set_color(csf_col)
        ax2.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, linespacing=1.3)
    ax.legend(fontsize=9, loc="upper right"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_wahrnehmung(D, f, lam, V_slider, S_slide, S_real):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    V_arr = np.linspace(30, 250, 100)
    Qp_sph  = perceived_quality_kurven(D, f, lam, V_arr)
    Qp_para = perceived_quality_kurven(D, D*50, lam, V_arr)
    has_slide = S_slide > S_real + 0.01
    if has_slide:
        f_sl = strehl_to_f(S_slide, D, lam)
        Qp_slide = perceived_quality_kurven(D, f_sl, lam, V_arr)
        ax.fill_between(V_arr, Qp_sph,   Qp_para, color=COR, alpha=0.10)
        ax.fill_between(V_arr, Qp_slide, Qp_para, color=GRN, alpha=0.10)
    else:
        ax.fill_between(V_arr, Qp_sph, Qp_para, color=COR, alpha=0.13, label="Wahrnehmungsverlust")
    ax.plot(V_arr, Qp_para, color=GRN, lw=2.0, ls="--", label="Paraboloid (Referenz)")
    ax.plot(V_arr, Qp_sph,  color=COR, lw=2.0, ls=":",  label=f"Sphäre f/{f/D:.1f}  S={S_real:.3f}")
    if has_slide:
        ax.plot(V_arr, Qp_slide, color=ACC, lw=2.5, label=f"Schieber  S={S_slide:.3f}")
    q_curve = Qp_slide if has_slide else Qp_sph
    for V_mark, col, ls in [(30, "#c8a020", ":"), (80, "#888", "-."), (160, "#c62828", ":")]:
        idx = int(round((V_mark-30)/(250-30)*(len(V_arr)-1)))
        q_m = float(q_curve[idx])
        ax.axvline(V_mark, color=col, lw=1.2, ls=ls, alpha=0.6)
        ax.scatter([V_mark], [q_m], color=col, s=50, zorder=6)
        ax.annotate(f"{V_mark}×\n{q_m:.2f}", xy=(V_mark, q_m),
                    xytext=(V_mark+7, q_m-0.05), fontsize=8, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.80))
    q_v = perceived_quality(D, f_sl if has_slide else f, lam, V_slider)
    ax.axvline(V_slider, color=ACC, lw=1.8, ls="-", alpha=0.9)
    ax.scatter([V_slider], [q_v], color=ACC, s=90, zorder=7)
    x_off = -65 if V_slider > 200 else 12
    ax.annotate(f"V={V_slider:.0f}×\nQ={q_v:.3f}", xy=(V_slider, q_v),
                xytext=(V_slider+x_off, q_v+0.04), fontsize=9, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.88))
    for q_thresh, col, label in [(0.9, GRN, "Q=0.90 (sehr gut)"), (0.7, "#BA7517", "Q=0.70 (spürbar)")]:
        ax.axhline(q_thresh, color=col, lw=1.0, ls=":", alpha=0.7)
        ax.text(32, q_thresh+0.008, label, fontsize=8, color=col, alpha=0.85)
    ax.set_xlim(30, 250); ax.set_ylim(0, 1.12)
    ax.set_xlabel("Vergrößerung V  [×]", fontsize=10)
    ax.set_ylabel("Wahrgenommener Qualitätsindex Q_perc", fontsize=10)
    ax.set_title(f"Wahrgenommener Kontrast (CSF-gewichtet)  —  D={D:.0f}mm  f/{f/D:.1f}", fontsize=10)
    ax.legend(fontsize=9, loc="lower right"); ax_fmt(ax)
    fig.tight_layout(); return fig


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Newton-Spiegel Rechner", layout="wide", page_icon="🔭")
st.title("🔭 Newton-Spiegel Rechner")
st.caption("Kontrast- und Schärfeverlust sphärischer Hauptspiegel")

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

# ── Ergebnisse oben ──────────────────────────────────────────────────────────
st.markdown(f"<div style='background:#f0f0f0;padding:10px 16px;border-radius:8px;"
            f"border-left:5px solid {'#2e7d32' if verdict_color=='green' else '#e65100' if verdict_color=='orange' else '#c62828'};'>"
            f"{verdict_text}</div>", unsafe_allow_html=True)
st.write("")

# Karten Zeile 1: Grunddaten
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("f/D", f"{r['N']:.1f}")
c2.metric("Strehl", f"{S_real:.4f}")
c3.metric("D_eff Kontrast", f"{r['Deff_k']:.1f} mm", delta=f"-{r['loss_k']:.1f} mm")
c4.metric("D_eff Schärfe", f"{r['Deff_s']:.1f} mm", delta=f"-{r['loss_s']:.1f} mm")
c5.metric("V_krit", f"{Vk['Vk_sph']:.0f}×")

# Karten Zeile 2: Q-Indizes
c1, c2, c3, c4 = st.columns(4)
c1.metric("Q₀.₂", f"{r['Q02']:.4f}")
c2.metric("Q₀.₄", f"{r['Q04']:.4f}")
c3.metric("Q₀.₆", f"{r['Q06']:.4f}")
c4.metric("Q_vis", f"{r['Qvis']:.4f}")

# Karten Zeile 3: Schärfe bei gewählter Vergrößerung
st.subheader(f"Schärfe bei {V_slider}×")
c1, c2 = st.columns(2)
c1.metric("D_eff Sphäre", f"{rv['Deff_sph']:.1f} mm")
c2.metric("Verlust", f"{rv['verlust']:.1f} mm",
          delta=f"-{rv['verlust']:.1f} mm" if rv['verlust'] > 0 else None,
          delta_color="inverse")

# Karten Zeile 4: Wahrnehmung
def qcolor(q): return "normal" if q >= 0.9 else "off"
st.subheader("Wahrgenommener Kontrast Q_perc (CSF-gewichtet)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Q_perc bei 30×",   f"{Qp_30:.3f}")
c2.metric("Q_perc bei 80×",   f"{Qp_80:.3f}")
c3.metric("Q_perc bei 160×",  f"{Qp_160:.3f}")
c4.metric(f"Q_perc bei {V_slider}×", f"{Qp_V:.3f}")

st.divider()

# ── Diagramme ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Strehl vs. f/D",
    "Eff. Öffnung vs. f/D",
    "D_eff vs. Öffnung",
    "Beugungsgrenze",
    "MTF",
    "Wahrnehmung",
])

with tab1:
    st.pyplot(fig_strehl(D, f, lam, S_slide), use_container_width=True)

with tab2:
    st.pyplot(fig_oeffnung(D, f, lam, S_slide), use_container_width=True)

with tab3:
    st.pyplot(fig_deff_D(D, r["N"], S_slide), use_container_width=True)

with tab4:
    st.pyplot(fig_beugung(D, f), use_container_width=True)

with tab5:
    st.pyplot(fig_mtf(D, f, lam, S_slide, V_slider), use_container_width=True)

with tab6:
    st.pyplot(fig_wahrnehmung(D, f, lam, V_slider, S_slide, S_real), use_container_width=True)

st.divider()
st.caption("Formeln: Maréchal-Näherung · MTF via Pupillen-Autokorrelation · CSF nach van Meeteren · "
           "Quellen: telescope-optics.net · gordtulloch.com")
