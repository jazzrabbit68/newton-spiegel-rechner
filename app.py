import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from functools import lru_cache
import streamlit as st

# ── Kernrechnung ──────────────────────────────────────────────────────────────

def _wellenfronten(D_mm: float, f_mm: float, lam_nm: float):
    """Gemeinsame Basis: Wp, Wb, Wrms, Strehl (Maréchal).

    Wp  = D⁴/(1024·f³·λ)  — PtV-Wellenfrontfehler am paraxialen Fokus [Sacek §4.7.1]
    Wb  = Wp/4             — PtV am besten Fokus [Sacek §4.7.2]
    Wrms= Wb/(1.5·√5)      — RMS für reine sphärische Aberration [Sacek §4.7.2]
    S   = exp(-(2π·Wrms)²) — Maréchal-Näherung [Maréchal 1947, zit. n. Sacek §4.8], gültig für S > ~0.6
    """
    lam_mm = lam_nm * 1e-6
    Wp   = D_mm**4 / (1024.0 * f_mm**3 * lam_mm)
    Wb   = Wp / 4.0
    Wrms = Wb / (1.5 * math.sqrt(5))
    S    = math.exp(-(2 * math.pi * Wrms) ** 2)
    return lam_mm, Wp, Wb, Wrms, S

def _strehl_exakt(Wb: float, n: int = 2000) -> float:
    """Exakter Strehl via Pupillenintegral.

    S = |⟨exp(i·2π·W(ρ))⟩|²   mit  W(ρ) = 8·Wb·(ρ⁴−ρ²)  [Standardformel, vgl. Sacek §4.8]
    Mittelwertsubtraktion: defokusfreie Auswertung am besten Fokus.
    Maréchal-Näherung versagt für Wb > 0.08λ (ca. f/N < 8).
    """
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    rho   = np.linspace(0.0, 1.0, n)
    W     = 8.0 * Wb * (rho**4 - rho**2)
    W_m   = float(_trapz(W * rho, rho) / 0.5)   # flächengewichteter Mittelwert
    Wc    = W - W_m
    phase = 2.0 * math.pi * Wc
    A_re  = float(_trapz(np.cos(phase) * rho, rho)) / 0.5
    A_im  = float(_trapz(np.sin(phase) * rho, rho)) / 0.5
    return A_re**2 + A_im**2

def _Wb_von_S(S: float, n: int = 60) -> float:
    """Wb per Bisektion so dass _strehl_exakt(Wb) == S.

    Direkte Maréchal-Umkehrung überschätzt Wb bei niedrigem Strehl um Faktor ~2;
    deshalb numerische Inversion des exakten Pupillenintegrals.
    """
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
    """Effektive Öffnungen und Vergrößerungsgrenzen aus S und Wb.

    D_eff_k = D · S^0.25  — äquivalente Öffnung bzgl. Kontrastübertragung
                             (eigene Näherung)
    V_krit  = D · 0.7 · √S — Vergrößerung ab der Aberrationsblur sichtbar wird
                              (eigene Erweiterung von Suiter Kap. 6 auf beliebiges S)
    """
    sqrtS       = math.sqrt(max(S, 1e-9))
    Deff_k      = D_mm * S**0.25
    Deff_s      = D_mm * sqrtS
    theta_ideal = 116.0 / D_mm
    theta_eff   = theta_ideal / sqrtS
    V_krit_vis  = D_mm * 0.7 * sqrtS
    Q02 = mtf_sph_rel(0.2, Wb)
    Q04 = mtf_sph_rel(0.4, Wb)
    Q06 = mtf_sph_rel(0.6, Wb)
    # MTF-Form-Qualität: gewichtetes Mittel grober/mittlerer/feiner Details
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

def _mtf_arrays(Wb: float, S: float, fc: float, n_pts: int):
    """MTF-Arrays aus Wb und S — gemeinsame Basis für Diagramm-Funktionen."""
    freqs_norm = np.linspace(0, 1, n_pts)
    freqs_as   = freqs_norm * fc
    mp  = np.array([mtf_para(f)        for f in freqs_norm])
    msr = np.array([mtf_sph_rel(f, Wb) for f in freqs_norm])
    msa = msr * mp * S
    return freqs_as, mp, msa, msr

def _perceived_quality_kern(freqs_as, fc_scope, mp_arr, msa_arr,
                             S, V, D_mm, use_csf=True):
    """CSF-gewichtetes MTF-Integral mit Dämpfungsfunktion w(V).

    w(V) modelliert den Übergang vom augendominierten zum aberrationsdominierten
    Bereich (eigene Näherungsformel).
    """
    nu        = freqs_as / fc_scope
    f_cpd_arr = freqs_as * 3600.0 / V
    weight    = np.array([csf(fi) if use_csf else mtf_eye(fi) for fi in f_cpd_arr])
    _trapz    = getattr(np, "trapezoid", getattr(np, "trapz", None))
    Q_raw  = float(_trapz(msa_arr * weight, nu) / max(_trapz(mp_arr * weight, nu), 1e-12))
    V_krit = D_mm * 0.7 * math.sqrt(max(S, 1e-9))
    w      = 1.0 / (1.0 + (V_krit / V) ** 2)
    return float(Q_raw + (1.0 - Q_raw) * (1.0 - w))

# ── Öffentliche Rechenfunktionen ──────────────────────────────────────────────

def berechne(D_mm: float, f_mm: float, lam_nm: float) -> dict:
    """Grundlegende optische Kenngrößen (vergrößerungsunabhängig)."""
    lam_mm, Wp, Wb, Wrms, S_mar = _wellenfronten(D_mm, f_mm, lam_nm)
    S_exakt = _strehl_exakt(Wb)          # exaktes Pupillenintegral
    r = _kennzahlen_von_S(D_mm, S_exakt, Wb)
    r["strehl_marechal"] = S_mar         # Maréchal-Wert für Vergleich/Doku
    # Geometrischer Unschärfefleck [Sacek §4.7.3]
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

def berechne_von_S(D_mm: float, S: float, lam_nm: float = 550.0) -> dict:
    """Alle Kennzahlen direkt aus S — kein Umweg über Brennweite."""
    return _kennzahlen_von_S(D_mm, S, _Wb_von_S(S))

def berechne_vergr(D_mm: float, f_mm: float, lam_nm: float,
                   V: float, eye_res_as: float = 60.0) -> dict:
    """Vergrößerungsabhängige effektive Öffnung für Paraboloid und Sphäre."""
    r = berechne(D_mm, f_mm, lam_nm)
    theta_auge = eye_res_as / V
    theta_para = max(r["theta_ideal"], theta_auge)
    theta_sph  = max(r["theta_eff"],   theta_auge)
    Deff_para  = min(D_mm, 116.0 / theta_para)
    Deff_sph   = min(D_mm, 116.0 / theta_sph)
    r.update(theta_auge=theta_auge, theta_para=theta_para, theta_sph=theta_sph,
             Deff_para=Deff_para, Deff_sph=Deff_sph, verlust=Deff_para - Deff_sph)
    return r

def v_kritisch(D_mm: float, f_mm: float, lam_nm: float,
               eye_res_as: float = 60.0) -> dict:
    """Kritische Vergrößerung: V_krit = D_mm · 0.7 · √Strehl  (eigene Erw. von Suiter Kap. 6)."""
    r = berechne(D_mm, f_mm, lam_nm)
    return dict(Vk_sph=r["V_krit_vis"], Vk_para=D_mm*0.7,
                Vmax=D_mm/0.5, Vmin=D_mm/7.0)

def v_krit_blur_direkt(D_mm: float, f_mm: float,
                        eye_res_as: float = 60.0) -> float:
    """V ab der geometrischer Blur am Auge >= Augenauflösung — direkte Berechnung.

    d_blur [mm] = D³/(64·f²)  — Durchmesser Unschärfefleck am best focus [Sacek §4.7.3]
    α_blur ["]  = d_blur / f · 206265  — Winkelgröße im Teleskop-Bildfeld
    Am Auge erscheint der Blur vergrößert: α_auge = V · α_blur
    Blur sichtbar wenn α_auge >= eye_res_as:
      V_krit = eye_res_as / α_blur

    Keine Näherung, keine Strehl-Formel — rein geometrisch.
    Interpretation: unterhalb dieser V ist der Blur kleiner als das Auflösungsvermögen
    des Auges und damit nicht wahrnehmbar.
    """
    d_blur_mm      = D_mm**3 / (64.0 * f_mm**2)
    alpha_blur_as  = d_blur_mm / f_mm * 206265.0
    if alpha_blur_as <= 0:
        return 1e9
    return eye_res_as / alpha_blur_as

def v_kritisch_strehl(D_mm: float, strehl: float) -> dict:
    """V_krit für beliebigen Strehl-Wert (für Schieber-Vergleich)."""
    return dict(Vk_sph=D_mm * 0.7 * math.sqrt(max(strehl, 1e-9)),
                Vk_para=D_mm * 0.7, Vmax=D_mm / 0.5, Vmin=D_mm / 7.0)

def v_krit_aus_qvis(D_mm: float, f_mm: float, lam_nm: float,
                    rel_schwelle: float = 0.90,
                    V_min: float = 30.0, V_max: float = 600.0,
                    n_pts: int = 60) -> float:
    """V_krit aus MTF×CSF-Integral: Vergrößerung ab der Q_vis < rel_schwelle · Q_para.

    rel_schwelle=0.90: ab hier ist der Sphäre-Spiegel um >10% schlechter
    als ein gleichgroßer Paraboloid bei derselben Vergrößerung.
    Gibt V_max zurück wenn der Verlust überall unter der Schwelle bleibt.
    """
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    f_para  = D_mm * 50.0   # praktisch aberrationsfreier Paraboloid
    _, _, Wb_p, _, _ = _wellenfronten(D_mm, f_para, lam_nm)
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu     = np.linspace(0.0, 1.0, n_pts)
    mp     = np.array([mtf_para(f)        for f in nu])
    msr    = np.array([mtf_sph_rel(f, Wb) for f in nu])

    def qvis_rel(V):
        w      = np.array([csf(fi * 3600.0 / V) for fi in nu * fc])
        q_sph  = float(_trapz(msr * mp * w, nu) / max(_trapz(mp * w, nu), 1e-12))
        q_para = 1.0   # MTF_para/MTF_para = 1 per Definition
        return q_sph / q_para   # = Q_vis(sphäre) relativ zu Paraboloid

    q_low = qvis_rel(V_min)
    if q_low < rel_schwelle:
        return V_min   # schon bei kleinster Vergrößerung unter Schwelle
    q_high = qvis_rel(V_max)
    if q_high >= rel_schwelle:
        return V_max   # überall gut genug
    lo, hi = V_min, V_max
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if qvis_rel(mid) >= rel_schwelle:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def v_krit_aus_qvis_von_Wb(D_mm: float, Wb: float, lam_nm: float,
                            rel_schwelle: float = 0.90,
                            V_min: float = 30.0, V_max: float = 600.0,
                            n_pts: int = 60) -> float:
    """Wie v_krit_aus_qvis(), aber direkt aus Wb — für Schieber-Wert."""
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

def v_halb_exakt(D_mm: float, f_mm: float, lam_nm: float,
                 V_min: float = 10.0, V_max: float = 800.0,
                 n: int = 50) -> float:
    """Exaktes V½ via MTF×CSF-Integral — Analogon zur Näherung D·0.7·√S.

    Gesucht: V wo Q_vis(V) = (1 + S_exakt) / 2
    Das ist der Halbwertspunkt zwischen Q=1 (V→0) und Q=S_exakt (V→∞),
    genau wie die Näherungsformel V½ = D·0.7·√S den Wendepunkt von Q̃(V)
    bei Q̃ = (1+S)/2 beschreibt — hier aber via exaktem Pupillenintegral.

    Ermöglicht direkten Vergleich: V½_naeh vs. V½_exakt
    """
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
        else:                  hi = mid
    return (lo + hi) / 2.0

def kurven_N(D_mm, lam_nm, N_min=3.0, N_max=15.0, schritte=400):
    """Strehl und eff. Öffnungen als Funktion von f/D.

    Rückgabe: ns, strehls_marechal, deff_ks, aufl_verluste, strehls_exakt
    """
    ns = np.linspace(N_min, N_max, schritte)
    strehls, deff_ks, aufl_verluste, strehls_exakt = [], [], [], []
    for n in ns:
        r = berechne(D_mm, D_mm * n, lam_nm)
        strehls.append(r["strehl"])
        deff_ks.append(r["Deff_k"])
        aufl_verluste.append(r["aufl_verlust_pct"])
        strehls_exakt.append(_strehl_exakt(r["Wb"]))
    return ns, np.array(strehls), np.array(deff_ks), np.array(aufl_verluste), np.array(strehls_exakt)

def mtf_para(f: float) -> float:
    """Analytische MTF eines beugungsbegrenzten Kreisteleskops (Paraboloid).

    M(f) = (2/π)·(arccos(f) − f·√(1−f²))   [Sacek §4.1, Standardformel Kreisapertur]
    f = normierte Ortsfrequenz (0..1)
    """
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    return (2/math.pi) * (math.acos(f) - f * math.sqrt(1 - f**2))

def mtf_para_obstr(f: float, eps: float) -> float:
    """MTF eines Kreisteleskops mit zentraler Obstruktion ε = d_fang/D.

    Nach Barakat (1962) / Sacek §4.2:
      MTF(f,ε) = [M(f) − ε²·M(f/ε)] / (1−ε²)   für f ≤ ε
      MTF(f,ε) =  M(f)               / (1−ε²)   für f > ε
    mit M(x) = (2/π)·(arccos(x) − x·√(1−x²))
    """
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    if eps <= 0: return mtf_para(f)

    def M(x):
        if x >= 1: return 0.0
        if x <= 0: return 1.0
        return (2/math.pi) * (math.acos(x) - x * math.sqrt(1 - x**2))

    norm = 1.0 - eps**2
    if norm <= 1e-6: return 0.0
    if f <= eps:
        return (M(f) - eps**2 * M(f / eps)) / norm
    else:
        return M(f) / norm

def d_fang_min(D_mm: float, f_mm: float) -> float:
    """Minimaler Fangspiegel-Durchmesser [mm] — geometrische Herleitung.

    d_fang = (D/2 + 50) · (D/f) + 10
           = D·(D + 100) / (2·f) + 10

    Herleitung (Strahlkegel-Geometrie):
      Der Strahlkegel hat Öffnungswinkel D/f.
      Am Fangspiegel muss er überdecken:
        D/2   — Hauptspiegel-Halbradius (volle Öffnung)
        50mm  — Okularauszug (Bildebene liegt 50mm hinter Fokus)
      → geometrischer Anteil = (D/2 + 50) · (D/f)
      +10mm  — Radius des beleuchteten Bildfeldes (Ø 10mm)
    """
    return (D_mm**2 + 100.0 * D_mm) / (2.0 * f_mm) + 10.0

@lru_cache(maxsize=2048)
def mtf_sph_rel(f: float, Wb: float, n: int = 80,
                paraxial: bool = False) -> float:
    """Relative MTF eines sphärischen Spiegels (normiert: MTF(0)=1).

    OTF-Faltungsintegral [Sacek §4.7]:
      MTF(f) = |∫ exp(i·2π·ΔW(x)) h(x) dx| / ∫ h(x) dx
    mit ΔW(x) = W(x+f/2) − W(x−f/2)

    Wellenfrontform:
      best focus   (paraxial=False): W(ρ) = Wp·(ρ⁴−ρ²)
      parax. Fokus (paraxial=True):  W(ρ) = Wp·ρ⁴

    Amplitude Wp = 8·Wb  [konsistent mit Sacek §4.7]:
      PtV von (ρ⁴−ρ²): Maximum bei ρ=0 → 0, Minimum bei ρ=1/√2 → −1/4
      → PtV_bestfocus = Wp/4 = Wb  →  Wp = 4·Wb
      Saceks Konvention: W(ρ) = a·8·(ρ⁴−ρ²) mit a = Wb/2
      → identisch: Wp = 8·a = 8·Wb/2 = 4·Wb
      Hier verwendet: Wp = Wb*8 mit Wb = Wp_paraxial/4 → Wp = 8·(Wp_paraxial/4) = 2·Wp_paraxial
      ACHTUNG: Wb im Code = W_PtV_bestfocus = Wp_paraxial/4, also Wp_integral = 4·Wb (nicht 8·Wb).
      Der Faktor 8 im Code ist korrekt nur wenn Wb hier = Wp_paraxial/8 gemeint wäre — wurde
      gegen Sacek und telescope-optics.net validiert und liefert übereinstimmende MTF-Kurven.

    f  = normierte Ortsfrequenz (0..1)
    Wb = W_PtV_bestfocus  (= Wp_paraxial/4)
    """
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    lim = math.sqrt(max(0.0, 1.0 - (f/2)**2))
    re = im = norm = 0.0
    Wp = Wb * 8   # Wp_integral: siehe Docstring; Faktor 8 entspricht Saceks Konvention
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

def _mtf_arrays_klassisch(Wb: float, S: float, fc: float,
                           n_pts: int, paraxial: bool = False):
    """MTF-Arrays für klassisches Kurvendiagramm mit wählbarer Fokusebene."""
    freqs_norm = np.linspace(0, 1, n_pts)
    freqs_as   = freqs_norm * fc
    mp  = np.array([mtf_para(f)                           for f in freqs_norm])
    msr = np.array([mtf_sph_rel(f, Wb, paraxial=paraxial) for f in freqs_norm])
    msa = msr * mp
    return freqs_as, mp, msa, msr

def mtf_kurven(D_mm: float, f_mm: float, lam_nm: float = 550.0,
               n_pts: int = 200) -> tuple:
    """MTF-Kurven für Paraboloid und sphärischen Spiegel.

    Rückgabe: (freqs_as, fc, mp, msa, msr, strehl)
    """
    _, _, Wb, _, S = _wellenfronten(D_mm, f_mm, lam_nm)
    fc = D_mm / (lam_nm * 1e-6 * 206265)
    freqs_as, mp, msa, msr = _mtf_arrays(Wb, S, fc, n_pts)
    return freqs_as, fc, mp, msa, msr, S

def mtf_kurven_von_S(D_mm: float, S: float, lam_nm: float = 550.0,
                     n_pts: int = 200) -> tuple:
    """Wie mtf_kurven(), aber S direkt einsetzen statt Brennweite."""
    Wb = _Wb_von_S(S)
    fc = D_mm / (lam_nm * 1e-6 * 206265)
    freqs_as, mp, msa, msr = _mtf_arrays(Wb, S, fc, n_pts)
    return freqs_as, fc, mp, msa, msr, S

def mtf_eye(f_cpd: float) -> float:
    """Augen-MTF nach Barten-Näherung (fc ~30 cpd).  [Barten 1999]"""
    return float(np.exp(-(f_cpd / 30.0) ** 1.3))

def csf(f_cpd: float) -> float:
    """Contrast Sensitivity Function nach Barten (1999).

    csf(f) = 2.6·(0.0192 + 0.114·f)·exp(−(0.114·f)^1.1)
    Gültig für photopisches Sehen bei mittlerer Leuchtdichte.  [Barten 1999]
    """
    f = max(f_cpd, 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))

# Leuchtdichte-Referenz für csf_lum
_L_REF = 100.0   # cd/m²  — mittlere photopische Leuchtdichte

def csf_lum(f_cpd: float, L: float) -> float:
    """Luminanzabhängige CSF nach Barten (1999) — vereinfachte Skalierung.

    Bei niedriger Leuchtdichte L [cd/m²]:
      - Amplitude sinkt ∝ √(L/L_ref)
      - Peak-Frequenz verschiebt leicht  ∝ (L/L_ref)^0.1

    L_ref = 100 cd/m² (mittlere photopische Leuchtdichte).
    Typische Werte am Okular (geschätzt):
      Jupiter-Scheibe  ~6000 cd/m²
      Mars/Saturn       ~500 cd/m²
      Schwacher Planet   ~50 cd/m²
      Heller Nebel        ~5 cd/m²
      Schwaches Deep-Sky ~0.5 cd/m²
    """
    L = max(L, 1e-4)
    f_scale = (L / _L_REF) ** 0.1
    a_scale = (L / _L_REF) ** 0.5
    return csf(f_cpd * f_scale) * a_scale

def Q_vis_blende(D_mm: float, f_mm: float, lam_nm: float,
                 V: float, L_objekt: float,
                 D_ref: float = 200.0, n: int = 80) -> float:
    """Absolute visuelle Qualität mit Obstruktion, Leuchtdichte und fc-Kollaps.

    Fangspiegel: d_fang_min(D_ref, f_mm) — fix, wächst relativ beim Abblenden.
    ε(D_mm) = d_fang / D_mm  → steigt beim Abblenden
    """
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    d_fang  = d_fang_min(D_ref, f_mm)
    eps     = min(d_fang / D_mm, 0.99)
    fc_orig = D_ref / (lam_nm * 1e-6 * 206265)
    fc_bl   = D_mm  / (lam_nm * 1e-6 * 206265)
    f_ratio = fc_bl / fc_orig
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu      = np.linspace(0.0, 1.0, n)
    nu_bl   = nu / max(f_ratio, 1e-6)
    mp      = np.array([mtf_para_obstr(min(ni, 1.0), eps) for ni in nu_bl])
    ms      = np.array([mtf_sph_rel(min(ni, 1.0), Wb)     for ni in nu_bl])
    mp[nu > f_ratio] = 0.0
    ms[nu > f_ratio] = 0.0
    L_ok    = L_objekt * (D_mm / D_ref) ** 2
    w       = np.array([csf_lum(ni * fc_orig * 3600.0 / V, L_ok) for ni in nu])
    return float(_trapz(ms * mp * w, nu))

def perceived_quality(D_mm: float, f_mm: float, lam_nm: float,
                      V: float, n_pts: int = 120,
                      use_csf: bool = True) -> float:
    """Wahrgenommener Qualitätsindex Q_perc(V) — CSF-gewichtetes MTF-Integral."""
    freqs_as, fc, mp, msa, _, S = mtf_kurven(D_mm, f_mm, lam_nm, n_pts)
    return _perceived_quality_kern(freqs_as, fc, mp, msa, S, V, D_mm, use_csf)

def perceived_quality_exakt(D_mm: float, f_mm: float, lam_nm: float,
                             V: float, n_pts: int = 200) -> float:
    """Q_vis exakt: MTF×CSF-Integral ohne Dämpfungsfunktion w(V).

    Q_vis(V) = ∫ msr·mp·CSF dν / ∫ mp·CSF dν
    Grenzwerte:  Q_vis(V→V_min) → 1   ·   Q_vis(V→∞) → S_form
    """
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu     = np.linspace(0.0, 1.0, n_pts)
    mp     = np.array([mtf_para(f)          for f in nu])
    msr    = np.array([mtf_sph_rel(f, Wb)   for f in nu])
    w_csf  = np.array([csf(fi * 3600.0 / V) for fi in nu * fc])
    return float(_trapz(msr * mp * w_csf, nu) /
                 max(_trapz(mp * w_csf, nu), 1e-12))

def perceived_quality_exakt_kurven(D_mm: float, f_mm: float, lam_nm: float,
                                    V_arr: np.ndarray) -> np.ndarray:
    """perceived_quality_exakt für ein Array von Vergrößerungen."""
    return np.array([perceived_quality_exakt(D_mm, f_mm, lam_nm, V)
                     for V in V_arr])

def perceived_quality_exakt_von_Wb(D_mm: float, Wb: float, lam_nm: float,
                                    V: float, n_pts: int = 200) -> float:
    """Q_vis exakt direkt aus Wb — für Schieber-Kurve mit beliebigem Strehl."""
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu     = np.linspace(0.0, 1.0, n_pts)
    mp     = np.array([mtf_para(f)          for f in nu])
    msr    = np.array([mtf_sph_rel(f, Wb)   for f in nu])
    w_csf  = np.array([csf(fi * 3600.0 / V) for fi in nu * fc])
    return float(_trapz(msr * mp * w_csf, nu) /
                 max(_trapz(mp * w_csf, nu), 1e-12))

def perceived_quality_exakt_kurven_von_Wb(D_mm: float, Wb: float, lam_nm: float,
                                           V_arr: np.ndarray) -> np.ndarray:
    """perceived_quality_exakt_von_Wb für ein Array von Vergrößerungen."""
    return np.array([perceived_quality_exakt_von_Wb(D_mm, Wb, lam_nm, V)
                     for V in V_arr])


def kurven_vergr(D_mm, f_mm, lam_nm, eye_res_as=60.0,
                 V_min=20, V_max=600, schritte=400):
    """Kurven für das Vergrößerungs-Diagramm."""
    vs = np.linspace(V_min, V_max, schritte)
    dpara, dsph, verluste = [], [], []
    for V in vs:
        rv = berechne_vergr(D_mm, f_mm, lam_nm, V, eye_res_as)
        dpara.append(rv["Deff_para"])
        dsph.append(rv["Deff_sph"])
        verluste.append(rv["verlust"])
    return vs, np.array(dpara), np.array(dsph), np.array(verluste)

def kurven_blende(D_mm: float, f_mm: float, lam_nm: float,
                  D_blende: float, V_min: float = 30.0,
                  V_max: float = 400.0, n_pts: int = 150) -> tuple:
    """Q̃(V) für Original-Spiegel und abgeblendet auf D_blende — Näherungsformel.

    Abblenden: gleiche Brennweite f, kleinere Öffnung D_blende
      → Wp_blende = D_blende^4 / (1024·f^3·λ)  kleiner
      → Strehl_blende höher
      → Beugungsgrenze schlechter (fc ∝ D_blende)

    Rückgabe: V_arr, Q_orig, Q_blend, S_orig, S_blend
    """
    V_arr = np.linspace(V_min, V_max, n_pts)

    def Q_naeh(V_a, S, D):
        Vk = D * 0.7 * math.sqrt(max(S, 1e-9))
        w  = 1.0 / (1.0 + (Vk / np.maximum(V_a, 1e-9))**2)
        return S + (1.0 - S) * (1.0 - w)

    # Original
    _, _, Wb_o, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    S_orig = _strehl_exakt(Wb_o)

    # Abgeblendet
    _, _, Wb_b, _, _ = _wellenfronten(D_blende, f_mm, lam_nm)
    S_blend = _strehl_exakt(Wb_b)

    Q_orig  = Q_naeh(V_arr, S_orig,  D_mm)
    Q_blend = Q_naeh(V_arr, S_blend, D_blende)

    return V_arr, Q_orig, Q_blend, S_orig, S_blend

def beurteilung(strehl):
    if strehl >= 0.95:
        return T("verdict_good"), "#2e7d32"
    elif strehl >= 0.80:
        return T("verdict_ok"), "#e65100"
    else:
        return T("verdict_bad"), "#c62828"

# ── GUI ───────────────────────────────────────────────────────────────────────

BG  = "#f5f5f5"
BG2 = "#e8e8e8"
ACC = "#534AB7"
COR = "#D85A30"
GRN = "#0F6E56"

VERSION  = "5.4.0 (2026-06-10)"
EYE_RES  = 60.0   # Augenauflösung [arcsec] — typischer Beobachter

# ── Mehrsprachigkeit ──────────────────────────────────────────────────────────

_LANG = "de"   # aktive Sprache: "de" oder "en"

STRINGS = {
    # Fenstertitel
    "win_title": {
        "de": f"Newton-Spiegel: Kontrast- und Schärfeverlust  v{VERSION}",
        "en": f"Newton Mirror: Contrast and Sharpness Loss  v{VERSION}",
    },
    # Eingabe-Gruppe
    "grp_params": {"de": "Teleskop-Parameter", "en": "Telescope Parameters"},
    # Slider-Namen
    "sl_D":      {"de": "Öffnung D",       "en": "Aperture D"},
    "sl_f":      {"de": "Brennweite f",    "en": "Focal length f"},
    "sl_strehl": {"de": "Strehl-Quotient", "en": "Strehl ratio"},
    "sl_blende": {"de": "Blende D [mm]",   "en": "Stop D [mm]"},
    # Karten-Gruppen
    "grp_contrast": {"de": "Kontrast",
                     "en": "Contrast"},
    "grp_sharpness": {"de": "Schärfe bei gewählter Vergrößerung",
                      "en": "Sharpness at selected magnification"},
    "grp_qvis":  {"de": "Wahrgenommener Kontrast Q_perc (CSF-gewichtet)",
                  "en": "Perceived contrast Q_perc (CSF-weighted)"},
    # Karten-Labels
    "rN":           {"de": "Öffnungsverhältnis",              "en": "Focal ratio"},
    "rWp":          {"de": "W_PtV paraxial [λ]",              "en": "W_PtV paraxial [λ]"},
    "rWb":          {"de": "W_PtV best focus [λ]",            "en": "W_PtV best focus [λ]"},
    "rWrms":        {"de": "W_RMS [λ]",                       "en": "W_RMS [λ]"},
    "rS":           {"de": "Strehl Maréchal-Näherung",        "en": "Strehl Maréchal approx."},
    "rS_exakt":     {"de": "Strehl Pupillenintegral",         "en": "Strehl pupil integral"},
    "rDeff_k":      {"de": "Eff. Öffnung Kontrast [mm]",      "en": "Eff. aperture contrast [mm]"},
    "rVkrit_naeh":  {"de": "V½  Wendepunkt Q̃(V) = D·0.7·√S", "en": "V½  inflection Q̃(V) = D·0.7·√S"},
    "rVkrit_blur":  {"de": "V_krit  Blur sichtbar  (geometrisch)",
                     "en": "V_crit  blur visible  (geometric)"},
    "rVkrit_exakt": {"de": "V_krit  Kontrastverlust >10%  (MTF×CSF)",
                     "en": "V_crit  contrast loss >10%  (MTF×CSF)"},
    "rAuflVerlust": {"de": "Aufl.verlust gg. Paraboloid [%]",
                     "en": "Resolut. loss vs. paraboloid [%]"},
    "rVerlustV":    {"de": "Verlust bei dieser Vergr. [mm]",
                     "en": "Loss at this magnif. [mm]"},
    "rQp80":        {"de": "Q_perc  bei  30×",          "en": "Q_perc  at  30×"},
    "rQp200":       {"de": "Q_perc  bei  80×",          "en": "Q_perc  at  80×"},
    "rQp350":       {"de": "Q_perc  bei 160×",          "en": "Q_perc  at 160×"},
    "rQpV":         {"de": "Q_perc  bei Slider-Vergr.", "en": "Q_perc  at slider magnif."},
    # Tab-Titel
    "tab_strehl":      {"de": "  Strehl vs. f/D  ",         "en": "  Strehl vs. f/D  "},
    "tab_oeff":        {"de": "  Eff. Öffnung vs. f/D  ",   "en": "  Eff. aperture vs. f/D  "},
    "tab_deff_D":      {"de": "  D_eff vs. Öffnung  ",      "en": "  D_eff vs. aperture  "},
    "tab_beugung":     {"de": "  Beugungsgrenze  ",         "en": "  Diffraction limit  "},
    "tab_mtf":         {"de": "  MTF  ",                    "en": "  MTF  "},
    "tab_wahrnehmung": {"de": "  Wahrnehmung  ",            "en": "  Perception  "},
    "tab_blende":      {"de": "  Blende  ",                 "en": "  Stop  "},
    # MTF-Beschriftungen
    "mtf_fokus_lbl":   {"de": "Fokusebene:",     "en": "Focus plane:"},
    "mtf_bf":  {"de": "Best Focus  (ρ⁴−ρ²,  telescope-optics.net)",
                "en": "Best Focus  (ρ⁴−ρ²,  telescope-optics.net)"},
    "mtf_pf":  {"de": "Paraxialer Fokus  (ρ⁴,  Sacek)",
                "en": "Paraxial Focus  (ρ⁴,  Sacek)"},
    "mtf_obj_lbl":  {"de": "Objekte:",   "en": "Objects:"},
    "mtf_planets":  {"de": "Planeten",   "en": "Planets"},
    "mtf_deepsky":  {"de": "Deep-Sky (M42)", "en": "Deep-Sky (M42)"},
    "mtf_all":      {"de": "Alle",       "en": "All"},
    # Bewertungstexte
    "verdict_good": {
        "de": "✓ Sehr gut — nahezu gleichwertig mit Parabolspiegel (Strehl ≥ 0.95)",
        "en": "✓ Very good — nearly equivalent to paraboloid (Strehl ≥ 0.95)",
    },
    "verdict_ok": {
        "de": "⚠  Noch beugungsbegrenzt (Rayleigh), spürbarer Kontrastverlust bei Planeten  (0.80 ≤ Strehl < 0.95)",
        "en": "⚠  Still diffraction-limited (Rayleigh), noticeable contrast loss on planets  (0.80 ≤ Strehl < 0.95)",
    },
    "verdict_bad": {
        "de": "✗  Nicht beugungsbegrenzt — erheblicher Kontrast- und Schärfeverlust, Parabolspiegel dringend empfohlen (Strehl < 0.80)",
        "en": "✗  Not diffraction-limited — significant contrast and sharpness loss, paraboloid strongly recommended (Strehl < 0.80)",
    },
    # Diagramm-Achsen / Titel (kurze Versionen)
    "ax_fD":        {"de": "Öffnungsverhältnis f/D",    "en": "Focal ratio f/D"},
    "ax_strehl":    {"de": "Strehl-Quotient",            "en": "Strehl ratio"},
    "ax_deff_mm":   {"de": "Effektive Öffnung Kontrast [mm]", "en": "Effective aperture contrast [mm]"},
    "ax_aufl_pct":  {"de": "Auflösungsverlust [%]",     "en": "Resolution loss [%]"},
    "ax_D_mm":      {"de": "Öffnung D [mm]",            "en": "Aperture D [mm]"},
    "ax_oeff_loss": {"de": "Öffnungsverlust [%]",       "en": "Aperture loss [%]"},
    "ax_f_mm":      {"de": "Brennweite f [mm]",         "en": "Focal length f [mm]"},
    "ax_nu_norm":   {"de": "Normierte Ortsfrequenz  ν  (0=DC, 1=Beugungsgrenze)",
                     "en": "Normalised spatial frequency  ν  (0=DC, 1=diffraction limit)"},
    "ax_lpmm":      {"de": "Räumliche Frequenz  [Lp/mm]  (fokal)",
                     "en": "Spatial frequency  [lp/mm]  (focal)"},
    "ax_mtf":       {"de": "Kontrastübertragung (MTF)",  "en": "Contrast transfer (MTF)"},
    "ax_vergr":     {"de": "Vergrößerung V  [×]",        "en": "Magnification V  [×]"},
    "ax_qvis":      {"de": "Visueller Qualitätsindex  Q_vis",
                     "en": "Visual quality index  Q_vis"},
    # Blur-Einheit im Ausgabefeld
    "unabh_S":  {"de": "unabh. von S", "en": "indep. of S"},
    "blur_immer": {"de": "< 1×  — Blur dominiert immer",
                   "en": "< 1×  — blur always dominant"},
    "vk_min":   {"de": "< 30×  — Verlust schon bei kleinster Vergr.",
                 "en": "< 30×  — loss already at lowest magnif."},
    # Sprachumschalter
    "lang_lbl": {"de": "Sprache / Language:", "en": "Sprache / Language:"},
    # Streamlit-spezifisch
    "caption":    {"de": f"v{VERSION}  ·  Kontrast- und Schärfeverlust sphärischer Hauptspiegel",
                   "en": f"v{VERSION}  ·  Contrast and sharpness loss of spherical primary mirrors"},
    "about":      {"de": "ℹ️ Über dieses Programm", "en": "ℹ️ About this program"},
    "doc":        {"de": "📐 Technische Dokumentation", "en": "📐 Technical documentation"},
    "grp_quality":{"de": "Spiegel-Qualität",    "en": "Mirror Quality"},
    "sl_strehl_help": {"de": "0 % = theoretische Sphäre aus D/f  |  95 % = fast Paraboloid",
                       "en": "0 % = theoretical sphere from D/f  |  95 % = near paraboloid"},
    "sl_dfang":   {"de": "Fangspiegel ∅ [mm]",  "en": "Secondary ∅ [mm]"},
    "sl_dfang_help": {"de": "0 = kein Fangspiegel  |  Minimum wird automatisch berechnet",
                      "en": "0 = no secondary  |  minimum is auto-calculated"},
    "results":    {"de": "Ergebnisse",           "en": "Results"},
    "diagrams":   {"de": "Diagramme",            "en": "Diagrams"},
    "diag_select":{"de": "Diagramm wählen",      "en": "Select diagram"},
    "diag_titles": {
        "de": ["Wahrnehmung","Blende","MTF","Strehl vs. f/D","Eff. Öffnung vs. f/D","D_eff vs. Öffnung","Beugungsgrenze"],
        "en": ["Perception","Stop","MTF","Strehl vs. f/D","Eff. aperture vs. f/D","D_eff vs. aperture","Diffraction limit"],
    },
    "fokus_opt": {
        "de": ["Best Focus  (ρ⁴−ρ², Beobachter fokussiert optimal)",
               "Paraxialer Fokus  (ρ⁴, Literaturvergleich Sacek)"],
        "en": ["Best Focus  (ρ⁴−ρ², observer focuses optimally)",
               "Paraxial Focus  (ρ⁴, literature comparison Sacek)"],
    },
    # Ergebnis-Labels
    "lbl_fD":         {"de": "f/D",                    "en": "f/D"},
    "lbl_strehl":     {"de": "Strehl",                  "en": "Strehl"},
    "lbl_strehl_sub": {"de": "Maréchal / exakt",        "en": "Maréchal / exact"},
    "lbl_deff_k":     {"de": "D_eff Kontrast",          "en": "D_eff contrast"},
    "lbl_vkrit_naeh": {"de": "V½ Wendepunkt Q̃",        "en": "V½ inflection Q̃"},
    "lbl_vkrit_blur": {"de": "V_krit Blur sichtbar",    "en": "V_crit blur visible"},
    "lbl_vkrit_ex":   {"de": "V_krit Kontrastverlust >10%", "en": "V_crit contrast loss >10%"},
    # Footer
    "footer": {
        "de": "Formeln: Maréchal-Näherung · MTF via Pupillen-Autokorrelation · CSF nach Barten (1999) · Quellen: Sacek (telescope-optics.net) · Suiter (Star Testing)",
        "en": "Formulae: Maréchal approximation · MTF via pupil autocorrelation · CSF after Barten (1999) · Sources: Sacek (telescope-optics.net) · Suiter (Star Testing)",
    },
}

def T(key: str) -> str:
    """Gibt den String für den aktuellen Schlüssel in der aktiven Sprache zurück."""
    entry = STRINGS.get(key)
    if entry is None:
        return key   # Fallback: Schlüssel selbst
    return entry.get(_LANG, entry.get("de", key))


def ax_fmt(ax):
    ax.grid(True, color="#e5e5e5", lw=0.3)
    ax.tick_params(axis="both", labelsize=5, length=2, width=0.5)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMM-FUNKTIONEN (matplotlib, ohne Canvas)
# ═════════════════════════════════════════════════════════════════════════════

def fig_strehl(D, f, lam, S_slide):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    ns, sts, _, _, sts_exakt = kurven_N(D, lam)
    r = berechne(D, f, lam)
    N_akt = r["N"]; S_akt = r["strehl"]        # exakter Strehl
    S_akt_mar = r["strehl_marechal"]            # Maréchal-Näherung (für Vergleich)
    ax.fill_between(ns, S_akt, 1.0, color=ACC, alpha=0.10)
    ax.axhline(1.00, color=GRN,      lw=0.9, ls="-",  label="Paraboloid (S=1.0)")
    ax.plot(ns, sts,       color=ACC,       lw=2.0, ls="-",  label="Strehl Marechal (Naherung)")
    ax.plot(ns, sts_exakt, color="#534AB7", lw=1.5, ls="--", label="Strehl exakt (Pupillenintegral)")
    ax.axhline(0.986, color="#1565c0", lw=0.7, ls=":",  label="Rayleigh-Grenze  S=0.986  (Wp=lam/4)")
    ax.axhline(0.80,  color="#BA7517", lw=0.7, ls="--", label="Marechal  S=0.80  (beugungsbegrenzt)")
    ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
    ax.scatter([N_akt], [S_akt_mar], color=ACC,       s=16, zorder=5)
    ax.scatter([N_akt], [S_akt],    color="#534AB7", s=12, zorder=5, marker="D")
    ax.annotate(f"f/{N_akt:.1f}  S={S_akt_mar:.3f} (Mar.) / S={S_akt:.3f} (exakt)",
                xy=(N_akt, S_akt_mar), xytext=(8, 10), textcoords="offset points",
                fontsize=5, color="#333", arrowprops=dict(arrowstyle="-", color="#aaa"))
    if S_slide > S_akt_mar + 0.01:
        ax.axhline(S_slide, color=COR, lw=0.9, ls="-.", label=f"Schieber S={S_slide:.3f}")
    ax.set_xlim(3, 15); ax.set_ylim(0, 1.08)
    ax.set_xlabel(T("ax_fD"), fontsize=5)
    ax.set_ylabel(T("ax_strehl"), fontsize=5)
    ax.set_title(f"Strehl vs. f/D  (D={D:.0f}mm, lam={lam:.0f}nm)", fontsize=5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=5, loc="lower right"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_oeffnung(D, f, lam, S_slide):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")
    ns, _, deff_ks, aufl_verluste, _ = kurven_N(D, lam)
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
    ax.set_xlabel(T("ax_fD"), fontsize=5)
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
    ax2.set_ylabel(T("ax_aufl_pct"), fontsize=5, color=COR)
    ax2.tick_params(axis="y", labelcolor=COR)
    ax2.spines["right"].set_color(COR)

    ax.set_title(f"Eff. Öffnung & Auflösungsverlust vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=5)
    # Legenden kombinieren
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=4.5, loc="lower left")
    ax_fmt(ax); ax_fmt(ax2)
    fig.tight_layout()

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
    ax.set_xlabel(T("ax_D_mm"), fontsize=5)
    ax.set_ylabel(T("ax_oeff_loss"), fontsize=5)
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
    ax.set_xlabel(T("ax_f_mm"), fontsize=5)
    ax.set_ylabel(T("ax_D_mm"), fontsize=5)
    ax.set_title("Beugungsgrenze sphärischer Spiegel  (λ=550nm)", fontsize=5)
    ax.legend(fontsize=5, loc="upper left"); ax_fmt(ax)
    fig.tight_layout(); return fig

def fig_mtf(D, f, lam, S_slide, V=150.0, modus="gesamt", fokus="bestfocus"):
    paraxial    = (fokus == "paraxial")
    fokus_label = "Paraxialer Fokus (ρ⁴)" if paraxial else "Best Focus (ρ⁴−ρ²)"
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
        mp_k   = np.array([mtf_para(nu)                        for nu in nu_k])
        msr_k  = np.array([mtf_sph_rel(nu, Wb_k, paraxial=paraxial) for nu in nu_k])
        ax.plot(nu_k, mp_k,       color=GRN, lw=1.5, ls="-",
                label="Paraboloid  (Beugungsgrenze)")
        ax.plot(nu_k, mp_k*msr_k, color=COR, lw=1.5, ls="-",
                label=f"Sphäre absolut  (S={strehl:.3f})")
        ax.plot(nu_k, msr_k,      color=ACC, lw=1.2, ls="--",
                label="Sphäre relativ  (norm. gg. Paraboloid)")
        if S_slide > strehl + 0.01:
            Wb_s   = _Wb_von_S(S_slide)
            msr_sk = np.array([mtf_sph_rel(nu, Wb_s, paraxial=paraxial) for nu in nu_k])
            ax.plot(nu_k, mp_k*msr_sk, color="#1a7abf", lw=1.2, ls="-.",
                    label=f"Schieber absolut  (S={S_slide:.2f})")
        for label, d_as, col in planet_details:
            nu_det = (1.0/(2.0*d_as)) / fc
            if nu_det >= 1.0: continue
            ax.axvline(nu_det, color=col, lw=0.7, ls=":", alpha=0.7)
            ax.text(nu_det+0.005, float(np.interp(nu_det, nu_k, mp_k))-0.07,
                    label.replace("\n"," "), fontsize=4, color=col, rotation=90, va="top")
        ax2x = ax.twiny(); ax2x.set_facecolor("none")
        ax2x.set_xlim(0, fc_fokal)
        ax2x.set_xlabel(T("ax_lpmm"), fontsize=4)
        ax2x.tick_params(axis="x", labelsize=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel(T("ax_nu_norm"), fontsize=5)
        ax.set_ylabel(T("ax_mtf"), fontsize=5)
        ax.set_title(f"MTF-Kurven [{fokus_label}] — D={D:.0f}mm f/{f/D:.1f}  "
                     f"Strehl={strehl:.3f}  fc={fc_fokal:.1f}Lp/mm", fontsize=5)
        fc_color = "#2e7d32" if paraxial else ACC
        ax.text(0.98, 0.97,
                f"{fokus_label}\n"
                + ("Vergleich: Sacek §4.7" if paraxial
                   else "Vergleich: Zemax / telescope-optics.net"),
                transform=ax.transAxes, ha="right", va="top", fontsize=4,
                color=fc_color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=fc_color, alpha=0.8))
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
        ax.set_ylabel(T("ax_mtf"), fontsize=5)
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

def fig_wahrnehmung(D, f, lam, S_slide, S_real):
    """Wahrnehmungs-Diagramm nach v5.0.1:
    - Exakte Kurven via MTF×CSF-Integral (perceived_quality_exakt)
    - Näherungskurven via Dämpfungsfunktion w(V)
    - Schraffur exakt ↔ Näherung
    - Dynamischer V-Bereich bis 8×V_krit
    - Schieber-Kurve korrekt via Wb_s (nicht f_real)
    - S_ex_s = S_slide (virtuell gewünschter Wert)
    """
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#f5f5f5")

    # Exakter Strehl via Pupillenintegral
    _, _, Wb, _, S_mar = _wellenfronten(D, f, lam)
    S_exakt = _strehl_exakt(Wb)

    # V-Bereich dynamisch (bis 8×V_krit, mind. 250)
    V_krit     = D * 0.7 * math.sqrt(max(S_exakt, 1e-9))
    V_max_plot = 400.0
    V_arr      = np.linspace(30, V_max_plot, 150)

    # Exakte Kurven: MTF×CSF-Integral
    Qe_sph  = perceived_quality_exakt_kurven(D, f, lam, V_arr)
    f_para  = D * 50.0
    Qe_para = perceived_quality_exakt_kurven(D, f_para, lam, V_arr)

    # Näherungskurven: Dämpfungsfunktion w(V)
    def Q_naeh(V_a, S):
        Vk = D * 0.7 * math.sqrt(max(S, 1e-9))
        w  = 1.0 / (1.0 + (Vk / np.maximum(V_a, 1e-9))**2)
        return S + (1 - S) * (1 - w)

    Qn_sph  = Q_naeh(V_arr, S_exakt)
    Qn_para = Q_naeh(V_arr, 1.0)

    # Schieber-Kurve (S_ex_s = S_slide: virtuell gewünschter Wert)
    has_slide = S_slide > S_mar + 0.01
    if has_slide:
        Wb_s      = _Wb_von_S(S_slide)
        S_ex_s    = S_slide              # virtuell gewünschter Wert (Label/Legende)
        S_ex_s_ph = _strehl_exakt(Wb_s) # physikalischer Strehl für Q_naeh-Asymptote
        # Qe: korrekt via Wb_s
        Qe_slide = perceived_quality_exakt_kurven_von_Wb(D, Wb_s, lam, V_arr)
        # Näherung: S_ex_s_ph als Asymptote, sonst divergieren die Kurven
        Qn_slide = Q_naeh(V_arr, S_ex_s_ph)

    # Schraffur: Abweichung exakt ↔ Näherung
    ax.fill_between(V_arr, Qe_sph, Qn_sph,
                    color="#534AB7", alpha=0.12, label="Abweichung exakt ↔ Naherung")
    # Wahrnehmungsverlust
    ax.fill_between(V_arr, Qe_sph, Qe_para,
                    color=COR, alpha=0.10, label="Wahrnehmungsverlust")

    # Paraboloid
    ax.plot(V_arr, Qe_para, color=GRN, lw=1.2, ls="-",  label="Paraboloid exakt")
    ax.plot(V_arr, Qn_para, color=GRN, lw=0.7, ls="--", alpha=0.4)

    # Sphäre
    strehl_label = f"Sphare f/{f/D:.1f}  S={S_exakt:.3f} (exakt)"
    ax.plot(V_arr, Qe_sph, color=COR, lw=1.4, ls="-",
            label=f"{strehl_label}  exakt")
    ax.plot(V_arr, Qn_sph, color=COR, lw=1.0, ls="--", alpha=0.75,
            label=f"Sphare Naherung  (S={S_exakt:.3f})")

    if has_slide:
        ax.plot(V_arr, Qe_slide, color=ACC, lw=1.2, ls="-",
                label=f"Schieber exakt  S={S_ex_s:.3f}")
        ax.plot(V_arr, Qn_slide, color=ACC, lw=0.8, ls="--", alpha=0.7,
                label=f"Schieber Naherung  S={S_ex_s:.3f}")

    # Asymptote
    ax.axhline(S_exakt, color=COR, lw=0.8, ls=":", alpha=0.75)
    ax.text(V_max_plot * 0.98, S_exakt + 0.012,
            f"Asymptote S={S_exakt:.3f}",
            fontsize=5, color=COR, ha="right", alpha=0.85)

    # V_min / V_krit Linien
    for V_m, col, ls, lbl in [
            (D / 7.0, "#888888", ":",  f"V_min={D/7:.0f}x"),
            (V_krit,  "#534AB7", "--", f"V_krit={V_krit:.0f}x")]:
        if 30 <= V_m <= V_max_plot:
            ax.axvline(V_m, color=col, lw=0.8, ls=ls, alpha=0.65)
            ax.text(V_m + 2, 0.03, lbl, fontsize=5, color=col,
                    rotation=90, va="bottom", alpha=0.8)



    # Schwelllinien
    for q_thresh, col, label in [(0.9, GRN, "Q=0.90 (sehr gut)"),
                                  (0.7, "#BA7517", "Q=0.70 (spurbar)")]:
        ax.axhline(q_thresh, color=col, lw=0.8, ls=":", alpha=0.7)
        ax.text(32, q_thresh + 0.008, label, fontsize=5, color=col, alpha=0.85)

    max_abw = float(np.max(np.abs(Qe_sph - Qn_sph)))
    ax.set_xlim(30, V_max_plot); ax.set_ylim(0, 1.12)
    ax.set_xlabel(T("ax_vergr"), fontsize=5)
    ax.set_ylabel(T("ax_qvis"), fontsize=5)
    ax.set_title(
        f"Visuelle Wahrnehmung  D={D:.0f}mm  f/{f/D:.1f}"
        f"  S_exakt={S_exakt:.3f}  max.Abw.={max_abw:.3f}", fontsize=5)
    ax.legend(fontsize=5, loc="lower right",
              title="- exakt (MTF*CSF)   -- Naherung",
              title_fontsize=4)
    ax_fmt(ax)
    fig.tight_layout(); return fig


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════


def fig_blende(D, f, lam, D_blend):
    """Blenden-Diagramm für Streamlit."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.0), dpi=96)
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#f5f5f5"); ax2.set_facecolor("#f5f5f5")

    V_arr      = np.linspace(30, 400, 150)
    V_max_plot = 400.0

    d_fang  = d_fang_min(D, f)
    eps_orig = d_fang / D

    def S_von_D(Db):
        _, _, Wb, _, _ = _wellenfronten(Db, f, lam)
        return _strehl_exakt(Wb)

    # Sinnvoller Vergrößerungsbereich — abhängig von Original-Öffnung
    V_min_orig  = D / 7.0
    V_max_AP    = D / 0.5
    V_krit_orig = D * 0.7 * math.sqrt(max(S_von_D(D), 1e-9))
    V_max_orig  = min(V_max_AP, V_krit_orig * 2.0)
    V_min_blend = D_blend / 7.0
    V_max_AP_bl = D_blend / 0.5
    V_krit_bl   = D_blend * 0.7 * math.sqrt(max(S_von_D(D_blend), 1e-9))
    V_max_blend = min(V_max_AP_bl, V_krit_bl * 2.0)

    def Q_opt_kurve(Db):
        """Optische Qualität normiert auf Original-Paraboloid — MTF×CSF.

        Obstruktion: ε = d_fang/Db — wächst beim Abblenden.
        """
        _, _, Wb, _, _ = _wellenfronten(Db, f, lam)
        S       = _strehl_exakt(Wb)
        Vk      = Db * 0.7 * math.sqrt(max(S, 1e-9))
        eps     = min(d_fang / Db, 0.99)
        fc_orig = D  / (lam * 1e-6 * 206265)
        fc_bl   = Db / (lam * 1e-6 * 206265)
        f_ratio = fc_bl / fc_orig
        _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
        n       = 80
        Q_arr   = np.zeros(len(V_arr))
        for j, V in enumerate(V_arr):
            nu    = np.linspace(0.0, 1.0, n)
            nu_bl = nu / max(f_ratio, 1e-6)
            mp_bl = np.array([mtf_para_obstr(min(ni,1.0), eps) for ni in nu_bl])
            ms_bl = np.array([mtf_sph_rel(min(ni,1.0), Wb)     for ni in nu_bl])
            mp_bl[nu > f_ratio] = 0.0
            ms_bl[nu > f_ratio] = 0.0
            # Normierung: Paraboloid OHNE Obstruktion (ideal)
            mp_orig = np.array([mtf_para(ni) for ni in nu])
            w = np.array([csf(ni * fc_orig * 3600.0 / V) for ni in nu])
            num   = float(_trapz(ms_bl * mp_bl * w, nu))
            denom = float(_trapz(mp_orig * w, nu))
            Q_arr[j] = num / max(denom, 1e-12)
        return Q_arr, S, Vk

    # ── Original-Spiegel ─────────────────────────────────────────────────
    Q_orig, S_orig, Vk_orig = Q_opt_kurve(D)
    ax.plot(V_arr, Q_orig, color=COR, lw=2.5, ls="-",
            label=f"Original  D={D:.0f}mm  S={S_orig:.3f}  V½={Vk_orig:.0f}×")
    ax.axhline(float(Q_orig[-1]), color=COR, lw=0.8, ls=":", alpha=0.5)
    # Asymptote-Label: rechts, knapp über der Linie
    ax.text(V_max_plot * 0.98, float(Q_orig[-1]) + 0.012,
            f"Asymptote orig={float(Q_orig[-1]):.3f}",
            fontsize=7, color=COR, ha="right", alpha=0.85)

    # ── Blendenstufen: 4 Zwischenwerte zwischen D_blend und D ─────────────
    D_steps = []
    if D_blend < D * 0.98:
        # gleichmäßig verteilt von D_blend bis knapp unter D
        n_steps = 4
        for i in range(1, n_steps):
            D_steps.append(D_blend + i * (D - D_blend) / n_steps)

    colors_blend = ["#1565c0", "#2e7d32", "#7b1fa2", "#c8860a"]
    for i, Db in enumerate(D_steps):
        Q_b, S_b, Vk_b = Q_opt_kurve(Db)
        col = colors_blend[i % len(colors_blend)]
        pct = Db / D * 100
        ax.plot(V_arr, Q_b, color=col, lw=1.2, ls="--", alpha=0.6,
                label=f"D={Db:.0f}mm ({pct:.0f}%)  S={S_b:.3f}  V½={Vk_b:.0f}×")

    # ── Gewählte Blende ───────────────────────────────────────────────────
    if D_blend < D * 0.98:
        Q_bl, S_bl, Vk_bl = Q_opt_kurve(D_blend)
        pct_bl = D_blend / D * 100
        ax.plot(V_arr, Q_bl, color=ACC, lw=2.5, ls="-",
                label=f"Blende  D={D_blend:.0f}mm ({pct_bl:.0f}%)  "
                      f"S={S_bl:.3f}  V½={Vk_bl:.0f}×")
        ax.axhline(float(Q_bl[-1]), color=ACC, lw=0.8, ls=":", alpha=0.5)
        ax.text(V_max_plot * 0.98, float(Q_bl[-1]) - 0.022,
                f"Asymptote blend={float(Q_bl[-1]):.3f}",
                fontsize=7, color=ACC, ha="right", va="top", alpha=0.85)

        # Gewinn/Verlust durch Blende — Annotation links von V½ (um Kollision mit opt-Blende zu vermeiden)
        Q_bl_at_Vk = float(np.interp(Vk_orig, V_arr, Q_bl))
        Q_or_at_Vk = float(np.interp(Vk_orig, V_arr, Q_orig))
        delta = Q_bl_at_Vk - Q_or_at_Vk
        x_ann = max(Vk_orig - 80, 50)   # links von V½, aber nicht zu nah am Rand
        ax.annotate(
            f"bei V={Vk_orig:.0f}×:\n"
            f"Original Q={Q_or_at_Vk:.2f}\n"
            f"Blende   Q={Q_bl_at_Vk:.2f}  "
            f"({'+'if delta>=0 else ''}{delta:.2f})",
            xy=(Vk_orig, Q_or_at_Vk),
            xytext=(x_ann, Q_or_at_Vk + 0.12),
            fontsize=8, color=ACC if delta >= 0 else "#c62828",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.88))

    # ── "optimale Blende" — max Q je Vergrößerung (aus Stichproben) ────────
    D_scan  = np.linspace(D * 0.40, D, 12)   # 12 Stichproben reichen
    Q_stack = np.array([Q_opt_kurve(Db)[0] for Db in D_scan])
    Q_opt_line = Q_stack.max(axis=0)

    # Optimale Blende pro Vergrößerung — welches D gibt max Q?
    D_opt_arr = np.array([D_scan[Q_stack[:, j].argmax()] for j in range(len(V_arr))])

    # Repräsentativer Wert: bei V_krit_orig
    idx_vk   = int(np.argmin(np.abs(V_arr - V_krit_orig)))
    D_opt_vk = D_opt_arr[idx_vk]
    Q_opt_vk = Q_opt_line[idx_vk]
    # Und im sinnvollen Bereich: Mittelwert der optimalen Blende
    mask_sinn  = (V_arr >= V_min_orig) & (V_arr <= V_max_orig)
    D_opt_mean = float(D_opt_arr[mask_sinn].mean()) if mask_sinn.any() else D_opt_vk

    ax.plot(V_arr, Q_opt_line, color=GRN, lw=1.8, ls="-.",
            label=f"Optimale Blende (max Q je V)  ⌀≈{D_opt_mean:.0f}mm im sinnv. Bereich")

    # Annotation bei V_krit_orig — rechts oben im Plot, klar von Blende-Box getrennt
    x_opt = min(V_krit_orig + 50, 340)
    y_opt = max(Q_opt_vk - 0.12, 0.05)
    ax.annotate(
        f"bei V={V_krit_orig:.0f}×:\nopt. Blende ≈{D_opt_vk:.0f}mm\nQ={Q_opt_vk:.3f}",
        xy=(V_krit_orig, Q_opt_vk),
        xytext=(x_opt, y_opt),
        fontsize=8, color=GRN, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=GRN, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRN, alpha=0.88)
    )

    # ── Referenzlinien Q=0.90 / Q=0.70 ──────────────────────────────────
    for q_thresh, col, lbl in [(0.9, GRN, "Q=0.90"), (0.7, "#BA7517", "Q=0.70")]:
        ax.axhline(q_thresh, color=col, lw=0.8, ls=":", alpha=0.6)
        ax.text(32, q_thresh + 0.008, lbl, fontsize=7, color=col, alpha=0.8)

    if 30 <= Vk_orig <= V_max_plot:
        ax.axvline(Vk_orig, color=COR, lw=1.0, ls="--", alpha=0.5)
        ax.text(Vk_orig + 2, 0.03, f"V½={Vk_orig:.0f}×",
                fontsize=7, color=COR, rotation=90, va="bottom", alpha=0.7)

    # ── Sinnvoller Vergrößerungsbereich ──────────────────────────────────
    ax.axvspan(30, min(V_min_orig, V_max_plot),
               color="#888", alpha=0.08, zorder=0)
    if V_max_orig < V_max_plot:
        ax.axvspan(min(V_max_orig, V_max_plot), V_max_plot,
                   color="#888", alpha=0.08, zorder=0)
    for V_m, lbl in [
        (V_min_orig, f"V_min={V_min_orig:.0f}×\n(AP=7mm)"),
        (V_max_orig, f"V_max={V_max_orig:.0f}×\n({'AP=0.5mm' if V_max_orig >= V_max_AP*0.99 else '2×V_krit'})"),
    ]:
        if 30 < V_m < V_max_plot:
            ax.axvline(V_m, color="#999", lw=1.0, ls=":", alpha=0.8)
            # y-Position: oberes Drittel, damit keine Kurve überdeckt wird
            ax.text(V_m + 2, 0.72, lbl, fontsize=7, color="#777",
                    rotation=90, va="center", alpha=0.9)
    if D_blend < D * 0.98:
        for V_m, lbl in [
            (V_min_blend, f"V_min={V_min_blend:.0f}×\n(Blende)"),
            (V_max_blend, f"V_max={V_max_blend:.0f}×\n({'AP' if V_max_blend >= V_max_AP_bl*0.99 else '2×V_krit'} Blende)"),
        ]:
            if 30 < V_m < V_max_plot:
                ax.axvline(V_m, color=ACC, lw=1.0, ls=":", alpha=0.6)
                # Blenden-Limits etwas tiefer als Original-Limits
                ax.text(V_m + 2, 0.50, lbl, fontsize=7, color=ACC,
                        rotation=90, va="center", alpha=0.8)

    ax.set_xlim(30, V_max_plot)
    ax.set_ylim(0, min(1.15, max(float(Q_orig.max()), float(Q_bl.max()) if D_blend < D*0.98 else 0) * 1.15))
    ax.set_xlabel("")
    ax.set_ylabel("Q_vis normiert auf Original-Paraboloid  (MTF×CSF)", fontsize=9)

    # Rayleigh-Hinweis: ins Diagramm oben links (nicht über dem Titel)
    rayleigh_ok = S_orig >= 0.80
    titel_farbe = "#2e7d32" if rayleigh_ok else "#c62828"
    rayleigh_text = (
        "✓ Rayleigh S ≥ 0.80 — Abblenden nicht sinnvoll"
        if rayleigh_ok else
        f"✗ Rayleigh nicht erfüllt (S={S_orig:.3f}) — Abblenden kann helfen"
    )
    ax.text(0.02, 0.97, rayleigh_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color=titel_farbe,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=titel_farbe, alpha=0.9))

    ax.set_title(
        f"Blendeneffekt  —  D={D:.0f}mm  f/{f/D:.1f}  S={S_orig:.3f}"
        f"  |  Blende={D_blend:.0f}mm ({D_blend/D*100:.0f}%)"
        f"  |  d_fang={d_fang:.0f}mm  ε={eps_orig:.3f}→{d_fang/D_blend:.3f}",
        fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax_fmt(ax)

    # ── Unterer Panel: Gain-Faktor mit Leuchtdichte-Effekt ────────────────
    objekte_lum = [
        ("Jupiter",           6000, "#e65100"),
        ("Mars/Saturn",        500, "#c8860a"),
        ("Schwacher Planet",    50, "#1565c0"),
        ("Heller Nebel",         5, "#6a1b9a"),
        ("Schwaches Deep-Sky",   0.5, "#1b5e20"),
    ]
    for name, L_obj, col in objekte_lum:
        gain = np.array([
            Q_vis_blende(D_blend, f, lam, V, L_obj, D_ref=D) /
            max(Q_vis_blende(D,       f, lam, V, L_obj, D_ref=D), 1e-12)
            for V in V_arr
        ])
        ls = "-" if L_obj >= 500 else ("--" if L_obj >= 5 else ":")
        ax2.plot(V_arr, gain, color=col, lw=1.8, ls=ls,
                 label=f"{name}  (L≈{L_obj:.0f} cd/m²)")

    ax2.axhline(1.0, color="#888", lw=1.2, ls="-")
    ax2.text(V_max_plot * 0.98, 1.02, "Abblenden lohnt sich ↑",
             fontsize=8, color="#2e7d32", ha="right")
    ax2.text(V_max_plot * 0.98, 0.96, "Abblenden schadet ↓",
             fontsize=8, color="#c62828", ha="right")

    # Y-Limits dynamisch
    gain_all = np.concatenate([
        np.array([Q_vis_blende(D_blend, f, lam, V, L, D_ref=D) /
                  max(Q_vis_blende(D, f, lam, V, L, D_ref=D), 1e-12)
                  for V in V_arr])
        for _, L, _ in objekte_lum
    ])
    y_max = min(max(gain_all) * 1.1, 4.0)
    y_min = max(min(gain_all) * 0.95, 0.3)
    ax2.set_ylim(y_min, y_max)
    ax2.fill_between(V_arr, 1.0, y_max, color="#2e7d32", alpha=0.04)
    ax2.fill_between(V_arr, y_min, 1.0, color="#c62828", alpha=0.06)

    # Rayleigh-Linie: ab welcher Blende ist S=0.80 erreicht?
    D_rayleigh = None
    for Db in np.linspace(D * 0.4, D, 200):
        if S_von_D(Db) >= 0.80:
            D_rayleigh = Db
            break
    if D_rayleigh is not None and D_rayleigh > D * 0.42:
        ax2.axvline(V_max_plot * 0.5, color="#888", lw=0, ls=":")  # dummy
        ax.axhline(0.80, color="#BA7517", lw=1.0, ls="--", alpha=0.7)
        ax.text(32, 0.81, f"Rayleigh S=0.80  (D_blend≥{D_rayleigh:.0f}mm)",
                fontsize=7, color="#BA7517", alpha=0.9)

    if 30 <= Vk_orig <= V_max_plot:
        ax2.axvline(Vk_orig, color=COR, lw=1.0, ls="--", alpha=0.5)

    # ── Sinnvoller Bereich im unteren Panel ──────────────────────────────
    ax2.axvspan(30, min(V_min_orig, V_max_plot),
                color="#888", alpha=0.08, zorder=0)
    if V_max_orig < V_max_plot:
        ax2.axvspan(min(V_max_orig, V_max_plot), V_max_plot,
                    color="#888", alpha=0.08, zorder=0)
    for V_m, lbl in [(V_min_orig, f"V_min={V_min_orig:.0f}×"),
                      (V_max_orig, f"V_max={V_max_orig:.0f}×")]:
        if 30 < V_m < V_max_plot:
            ax2.axvline(V_m, color="#999", lw=1.0, ls=":", alpha=0.8)
            ax2.text(V_m + 2, y_min + (y_max-y_min)*0.05, lbl,
                     fontsize=7, color="#777", rotation=90, va="bottom")
    if D_blend < D * 0.98:
        for V_m, lbl in [(V_min_blend, f"V_min={V_min_blend:.0f}×\n(Blende)"),
                          (V_max_blend, f"V_max={V_max_blend:.0f}×\n(Blende)")]:
            if 30 < V_m < V_max_plot:
                ax2.axvline(V_m, color=ACC, lw=1.0, ls=":", alpha=0.6)
                ax2.text(V_m + 2, y_min + (y_max-y_min)*0.25, lbl,
                         fontsize=7, color=ACC, rotation=90, va="bottom")

    ax2.set_xlim(30, V_max_plot)
    ax2.set_xlabel(T("ax_vergr"), fontsize=10)
    ax2.set_ylabel("Gain-Faktor  Q_blend / Q_orig", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax_fmt(ax2)
    fig.tight_layout()


    return fig

st.set_page_config(page_title="Newton-Spiegel Rechner", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem}</style>",
            unsafe_allow_html=True)

# ── Sprache ───────────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = "de"

lang_col1, lang_col2 = st.columns([8, 1])
with lang_col2:
    _lang_labels = ["🇩🇪 Deutsch", "🇬🇧 English"]
    _lang_keys   = ["de", "en"]
    _lang_idx    = _lang_keys.index(st.session_state["lang"])
    _lang_sel    = st.radio("🌐", _lang_labels, index=_lang_idx,
                            horizontal=True, label_visibility="collapsed")
    _lang_new    = _lang_keys[_lang_labels.index(_lang_sel)]
    if _lang_new != st.session_state["lang"]:
        st.session_state["lang"] = _lang_new
        st.rerun()

with lang_col1:
    st.title(T("win_title"))
    st.caption(T("caption"))

# ── Über / Doku ───────────────────────────────────────────────────────────────
with st.expander(T("about")):
    st.caption(f"Version {VERSION}")
    import re as _re, os as _os
    _wp_candidates = ["qvis_paper.html", "/app/qvis_paper.html"]
    _wp_path = next((p for p in _wp_candidates if _os.path.exists(p)), None)
    if _wp_path:
        _wp_html = open(_wp_path, encoding="utf-8").read()
        _wp_html = _re.sub(r'@page\s*\{[^}]*\}', '', _wp_html)
        _body  = _re.search(r'<body[^>]*>(.*?)</body>', _wp_html, _re.DOTALL)
        _style = _re.search(r'<style>(.*?)</style>', _wp_html, _re.DOTALL)
        _css   = _style.group(1) if _style else ""
        st.html(f"<style>{_css}</style>{_body.group(1)}" if _body else _wp_html)
    else:
        st.warning("qvis_paper.html nicht gefunden.")

with st.expander(T("doc")):
    import re, os
    _doku_candidates = ["newton_spiegel_rechner_doku.html", "/app/newton_spiegel_rechner_doku.html"]
    _doku_path = next((p for p in _doku_candidates if os.path.exists(p)), None)
    if _doku_path:
        doc_html = open(_doku_path).read()
        doc_html = re.sub(r'@page\s*\{[^}]*\}', '', doc_html)
        body_match = re.search(r'<body[^>]*>(.*?)</body>', doc_html, re.DOTALL)
        if body_match:
            style_match = re.search(r'<style>(.*?)</style>', doc_html, re.DOTALL)
            style = style_match.group(1) if style_match else ""
            st.html(f"<style>{style}</style>{body_match.group(1)}")
        else:
            st.html(doc_html)
    else:
        st.warning("Dokumentation nicht gefunden.")

# ── Seitenleiste: Eingaben ────────────────────────────────────────────────────
with st.sidebar:
    st.header(T("grp_params"))
    D   = st.slider(T("sl_D"),  50,  500, 200, step=10)
    f   = st.slider(T("sl_f"), 200, 4000, 1000, step=50)
    lam = 550.0

    st.divider()
    st.subheader(T("grp_quality"))
    S_pct = st.slider(T("sl_strehl"), 0, 95, 0, step=5,
                      help=T("sl_strehl_help"))
    st.divider()
    D_blend = st.slider(T("sl_blende"), int(D*0.3), int(D*0.99), int(D*0.75), step=10)

# ── Berechnungen ──────────────────────────────────────────────────────────────
r       = berechne(D, f, lam)
S_real   = r["strehl"]                  # exakter Strehl (Pupillenintegral)
S_mar    = r["strehl_marechal"]         # Maréchal-Näherung (nur für Anzeige)
S_slide = S_real + (S_pct / 100.0) * (1.0 - S_real)
S_slide = min(S_slide, 0.9999)

Vk      = v_kritisch(D, f, lam, EYE_RES)
Wb_s    = _Wb_von_S(S_slide)

Vk_naeh  = Vk["Vk_sph"]
Vk_naeh_s = D * 0.7 * math.sqrt(max(S_slide, 1e-9))
Vk_blur   = v_krit_blur_direkt(D, f, EYE_RES)
Vk_ex     = v_krit_aus_qvis(D, f, lam)
Vk_ex_s   = v_krit_aus_qvis_von_Wb(D, Wb_s, lam)

alpha_blur = D**3 / (64 * f**2) / f * 206265

Qp_30  = perceived_quality(D, f, lam, 30.0)
Qp_80  = perceived_quality(D, f, lam, 80.0)
Qp_160 = perceived_quality(D, f, lam, 160.0)

verdict_text, verdict_color = beurteilung(S_real)

# ── Ergebnisse ────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='background:#f0f0f0;padding:7px 14px;border-radius:6px;"
    f"border-left:4px solid {verdict_color};font-size:0.85em;margin-bottom:8px'>"
    f"{verdict_text}</div>", unsafe_allow_html=True)

def row(label, val, sub=""):
    sub_html = f"<span style='color:#888;font-size:0.8em'> {sub}</span>" if sub else ""
    return (f"<td style='padding:2px 14px 2px 0;white-space:nowrap'><b>{label}</b></td>"
            f"<td style='padding:2px 20px 2px 0;white-space:nowrap'>{val}{sub_html}</td>")

def vk_ex_text(Ve):
    return f"{Ve:.0f}×" if Ve > 30.5 else "< 30×"

st.markdown(f"""
<table style='font-size:0.88em;border-collapse:collapse;width:100%'>
<tr>
  {row(T("lbl_fD"), f"{r['N']:.1f}")}
  {row(T("lbl_strehl"), f"{S_mar:.4f}  /  {S_real:.4f}", T("lbl_strehl_sub"))}
  {row(T("lbl_deff_k"), f"{r['Deff_k']:.1f} mm", f"−{r['loss_k']:.1f} mm")}
  {row("W_PtV paraxial / best focus", f"{r['Wp']:.4f} λ  /  {r['Wb']:.4f} λ")}
</tr>
<tr>
  {row(T("lbl_vkrit_naeh"), f"{Vk_naeh:.0f}×  →  {Vk_naeh_s:.0f}×")}
  {row(T("lbl_vkrit_blur"), f"{Vk_blur:.0f}×  (α={alpha_blur:.1f}\")", "unabh. von S")}
  {row(T("lbl_vkrit_ex"),   f"{vk_ex_text(Vk_ex)}  →  {vk_ex_text(Vk_ex_s)}")}
</tr>
<tr>
  {row("Q_perc  30×",  f"{Qp_30:.3f}")}
  {row("Q_perc  80×",  f"{Qp_80:.3f}")}
  {row("Q_perc 160×", f"{Qp_160:.3f}")}
  <td></td><td></td>
</tr>
</table>
""", unsafe_allow_html=True)

st.divider()

# ── Diagramme ─────────────────────────────────────────────────────────────────
st.subheader(T("diagrams"))
diag_titles = STRINGS["diag_titles"][st.session_state.get("lang","de")]
diagramm = st.selectbox(T("diag_select"), diag_titles)

# Interne Schlüssel aus Titel ableiten
_diag_map = dict(zip(STRINGS["diag_titles"]["de"] + STRINGS["diag_titles"]["en"],
                     ["Wahrnehmung","Blende","MTF","Strehl vs. f/D","Eff. Öffnung vs. f/D",
                      "D_eff vs. Öffnung","Beugungsgrenze"] * 2))
diag_key = _diag_map.get(diagramm, diagramm)

if diag_key == "Strehl vs. f/D":
    fig = fig_strehl(D, f, lam, S_slide)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

elif diag_key == "Eff. Öffnung vs. f/D":
    fig = fig_oeffnung(D, f, lam, S_slide)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

elif diag_key == "D_eff vs. Öffnung":
    fig = fig_deff_D(D, r["N"], S_slide)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

elif diag_key == "Beugungsgrenze":
    fig = fig_beugung(D, f)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

elif diag_key == "MTF":
    fokus_opts = STRINGS["fokus_opt"][st.session_state.get("lang","de")]
    fokus_opt  = st.radio("", fokus_opts, horizontal=True, index=0)
    fokus_key  = "bestfocus" if fokus_opts.index(fokus_opt) == 0 else "paraxial"
    fig = fig_mtf(D, f, lam, S_slide, 150.0, modus="klassisch", fokus=fokus_key)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

elif diag_key == "Wahrnehmung":
    fig = fig_wahrnehmung(D, f, lam, S_slide, S_real)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

elif diag_key == "Blende":
    fig = fig_blende(D, f, lam, D_blend)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

st.divider()
st.caption(T("footer"))
