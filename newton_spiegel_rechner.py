"""
Newton-Teleskop: Kontrast- und Schärfeverlust sphärischer Hauptspiegel V5.1
============================================================================
Berücksichtigt:
  - Wellenfrontfehler und Strehl-Quotient (Kontrast)
  - Geometrischen Unschärfefleck am besten Fokus (Schärfe)
  - Vergrößerungsabhängigen Schärfeverlust durch das Auge

Formeln:
  Kontrast (Strehl):
    W_PtV (paraxial)  = D⁴ / (1024·f³·λ)         [Sacek §4.7.1]
    W_PtV (best focus)= W_PtV_paraxial / 4         [Sacek §4.7.2]
    W_RMS             = W_PtV_bestfocus / (1.5·√5) [Sacek §4.7.2]
    Strehl (Näherung) = exp(-(2π·W_RMS)²)          [Maréchal 1947, zit. n. Sacek §4.8]
    Strehl (exakt)    = |⟨exp(i·2π·W(ρ))⟩|²        [Standardformel, vgl. Sacek §4.8]
    D_eff (Kontrast)  = D · S^0.25                 [eigene Näherung]

  Schärfe:
    r_Airy [arcsec]   = 1.22·λ/D·206265
    d_blur [mm]       = D³ / (64·f²)               [Sacek §4.7.3]
    Dawes [arcsec]    = 116 / D[mm]

  Kritische Vergrößerung:
    V_krit            = D[mm] · 0.7 · √Strehl      [eigene Erw. von Suiter Kap. 6]

  MTF:
    Paraboloid:  analytisch (Kreisapertur)          [Sacek §4.1]
    Sphäre rel.: OTF-Faltungsintegral, numerisch    [Sacek §4.7]
      Wellenfrontform W(ρ) = Wp·(ρ⁴−ρ²)  [best focus]
                   oder Wp·ρ⁴             [paraxialer Fokus]
      Amplitude: Wp = 8·Wb
        (PtV von ρ⁴−ρ² liegt bei ρ=1/√2: Minimum = −1/4
         → Wb = Wp/4  →  Wp = 4·Wb = 8·Wb_bestfocus; konsistent mit Sacek)

  Wahrnehmung:
    CSF nach Barten (1999):
      csf(f) = 2.6·(0.0192 + 0.114·f)·exp(−(0.114·f)^1.1)   [Barten 1999]
    Q_vis = ∫(MTF_sph · MTF_para · CSF) dν / ∫(MTF_para · CSF) dν
    Q̃(V) = S + (1−S)·(1−w(V))   [eigene Näherungsformel]
      mit w(V) = 1 / (1 + (V_krit/V)²)

Quellen:
  [Sacek]    V. Sacek, "Notes on Amateur Telescope Optics",
             https://www.telescope-optics.net  (laufend aktualisiert)
  [Suiter]   H.R. Suiter, "Star Testing Astronomical Telescopes",
             2nd ed., Willmann-Bell (2009)
  [Barten]   P.G.J. Barten, "Contrast Sensitivity of the Human Eye and Its Effects
             on Image Quality", SPIE Press (1999), ISBN 978-0-8194-3296-4

Abhängigkeiten: pip install matplotlib numpy
"""

import math
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import numpy as np
from functools import lru_cache


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


@lru_cache(maxsize=1024)
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
    "tab_blende":      {"de": "  Blende  ",                 "en": "  Stop-down  "},
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
}


def T(key: str) -> str:
    """Gibt den String für den aktuellen Schlüssel in der aktiven Sprache zurück."""
    entry = STRINGS.get(key)
    if entry is None:
        return key   # Fallback: Schlüssel selbst
    return entry.get(_LANG, entry.get("de", key))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(T("win_title"))
        self.resizable(True, True)
        self.configure(bg=BG)
        self._build_ui()
        self._aktualisieren()

    def _sprache_wechseln(self, lang):
        global _LANG
        _LANG = lang
        self.title(T("win_title"))
        self._texte_aktualisieren()
        self._aktualisieren()

    def _texte_aktualisieren(self):
        """Alle statischen Label-Texte nach Sprachwechsel neu setzen."""
        # Slider-Namen
        for key, (lbl, unit) in self._slider_labels.items():
            pass   # Slider-Label-Texte werden über _slider_label_widgets gesetzt
        for key, widget in self._slider_name_widgets.items():
            widget.config(text=T(key))
        # Karten-Überschriften
        for key, widget in self._grp_widgets.items():
            widget.config(text=T(key))
        # Karten-Labels
        for key, widget in self._karten_name_widgets.items():
            widget.config(text=T(key))
        # Tab-Titel
        for i, key in enumerate(self._tab_keys):
            self._nb.tab(i, text=T(key))
        # MTF-Beschriftungen
        for key, widget in self._mtf_label_widgets.items():
            widget.config(text=T(key))

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=3)

        left  = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Sprachumschalter ─────────────────────────────────────────────────
        lang_frame = tk.Frame(left, bg=BG)
        lang_frame.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 0))
        tk.Label(lang_frame, text=T("lang_lbl"), bg=BG,
                 font=("Helvetica", 9), fg="#888").pack(side="left", padx=(0, 6))
        self._var_lang = tk.StringVar(value=_LANG)
        for code, label in [("de", "Deutsch"), ("en", "English")]:
            tk.Radiobutton(lang_frame, text=label, variable=self._var_lang,
                           value=code, bg=BG, fg="#555",
                           activebackground=BG, selectcolor=BG2,
                           font=("Helvetica", 9),
                           command=lambda c=code: self._sprache_wechseln(c)
                           ).pack(side="left", padx=4)

        # ── Eingaben ─────────────────────────────────────────────────────────
        grp_lbl = tk.Label(left, text=T("grp_params"),
                           font=("Helvetica", 11, "bold"), bg=BG)
        grp_lbl.grid(row=1, column=0, columnspan=3, sticky="w", **pad)
        self._grp_widgets = {"grp_params": grp_lbl}
        self._slider_name_widgets = {}

        self.var_D      = tk.DoubleVar(value=200)
        self.var_f      = tk.DoubleVar(value=1000)
        self.var_strehl = tk.DoubleVar(value=0)
        self.var_blende = tk.DoubleVar(value=150)
        self.var_lam    = tk.DoubleVar(value=550)

        slider_defs = [
            ("sl_D",      self.var_D,      50,  500, 10, "mm"),
            ("sl_f",      self.var_f,      200, 4000, 50, "mm"),
            ("sl_strehl", self.var_strehl,   0,  100,  5, ""),
            ("sl_blende", self.var_blende,  30,  490, 10, "mm"),
        ]
        self._slider_labels = {}
        for i, (key, var, mn, mx, res, unit) in enumerate(slider_defs, 2):
            lbl_name = tk.Label(left, text=T(key), bg=BG, width=16, anchor="w")
            lbl_name.grid(row=i, column=0, sticky="w", **pad)
            self._slider_name_widgets[key] = lbl_name
            tk.Scale(left, from_=mn, to=mx, resolution=res,
                     orient="horizontal", variable=var, length=205,
                     command=self._slider_changed,
                     bg=BG, highlightthickness=0,
                     troughcolor="#ddd", activebackground=ACC
                     ).grid(row=i, column=1, **pad)
            lbl_val = tk.Label(left, text="", bg=BG, width=9, anchor="w")
            lbl_val.grid(row=i, column=2, sticky="w", **pad)
            self._slider_labels[key] = (lbl_val, unit)

        row = len(slider_defs) + 2

        # ── Ergebniskarten ────────────────────────────────────────────────────
        self._res = {}
        self._karten_name_widgets = {}

        karten_gruppen = [
            ("grp_contrast", [
                ("rN",           "rN"),
                ("rWp",          "rWp"),
                ("rWb",          "rWb"),
                ("rWrms",        "rWrms"),
                ("rS",           "rS"),
                ("rS_exakt",     "rS_exakt"),
                ("rDeff_k",      "rDeff_k"),
                ("rVkrit_naeh",  "rVkrit_naeh"),
                ("rVkrit_blur",  "rVkrit_blur"),
                ("rVkrit_exakt", "rVkrit_exakt"),
            ]),
            ("grp_sharpness", [
                ("rAuflVerlust", "rAuflVerlust"),
                ("rVerlustV",    "rVerlustV"),
            ]),
            ("grp_qvis", [
                ("rQp80",  "rQp80"),
                ("rQp200", "rQp200"),
                ("rQp350", "rQp350"),
                ("rQpV",   "rQpV"),
            ]),
        ]

        for grp_key, karten in karten_gruppen:
            grp_w = tk.Label(left, text=T(grp_key),
                             font=("Helvetica", 11, "bold"), bg=BG)
            grp_w.grid(row=row, column=0, columnspan=3, sticky="w", **pad)
            self._grp_widgets[grp_key] = grp_w
            row += 1
            row = self._karten_grid(left, karten, row, self._res,
                                    self._karten_name_widgets, pad)

        self.lbl_verdict = tk.Label(left, text="", bg=BG,
                                    font=("Helvetica", 10),
                                    wraplength=400, justify="left")
        self.lbl_verdict.grid(row=row, column=0, columnspan=3,
                              sticky="w", padx=10, pady=(8, 2))
        row += 1
        self.lbl_vkrit = tk.Label(left, text="", bg=BG,
                                   font=("Helvetica", 11, "bold"),
                                   wraplength=400, justify="left")
        self.lbl_vkrit.grid(row=row, column=0, columnspan=3,
                             sticky="w", padx=10, pady=(0, 8))

        # ── Diagramm-Tabs ─────────────────────────────────────────────────────
        self._nb = ttk.Notebook(right)
        self._nb.pack(fill="both", expand=True)
        self._tab_keys = ["tab_strehl","tab_oeff","tab_deff_D",
                          "tab_beugung","tab_mtf","tab_wahrnehmung","tab_blende"]
        self._mtf_label_widgets = {}

        tab_attrs = [
            ("_tab_strehl",       "tab_strehl"),
            ("_tab_oeff",         "tab_oeff"),
            ("_tab_deff_D",       "tab_deff_D"),
            ("_tab_beugung",      "tab_beugung"),
            ("_tab_mtf",          "tab_mtf"),
            ("_tab_wahrnehmung",  "tab_wahrnehmung"),
            ("_tab_blende",       "tab_blende"),
        ]
        self._tab_keys = ["tab_strehl","tab_oeff","tab_deff_D",
                          "tab_beugung","tab_mtf","tab_wahrnehmung","tab_blende"]
        for attr, tab_key in tab_attrs:
            frm = tk.Frame(self._nb, bg=BG)
            self._nb.add(frm, text=T(tab_key))
            if attr == "_tab_mtf":
                btn_frame = tk.Frame(frm, bg=BG)
                btn_frame.pack(side="top", fill="x", padx=8, pady=(6, 0))
                self._mtf_modus = tk.StringVar(value="klassisch")
                row1b = tk.Frame(btn_frame, bg=BG)
                row1b.pack(side="top", anchor="w", pady=(2, 0))
                lbl_fok = tk.Label(row1b, text=T("mtf_fokus_lbl"), bg=BG,
                                   fg="#888", font=("Helvetica", 9))
                lbl_fok.pack(side="left", padx=(0, 6))
                self._mtf_label_widgets["mtf_fokus_lbl"] = lbl_fok
                self._mtf_fokus = tk.StringVar(value="bestfocus")
                for val, key in [("bestfocus", "mtf_bf"), ("paraxial", "mtf_pf")]:
                    rb = tk.Radiobutton(row1b, text=T(key), variable=self._mtf_fokus,
                                        value=val, bg=BG, fg="#666",
                                        activebackground=BG, selectcolor=BG2,
                                        font=("Helvetica", 9),
                                        command=self._aktualisieren)
                    rb.pack(side="left", padx=8)
                    self._mtf_label_widgets[key] = rb
                row2 = tk.Frame(btn_frame, bg=BG)
                row2.pack(side="top", anchor="w", pady=(3, 0))
                lbl_obj = tk.Label(row2, text=T("mtf_obj_lbl"), bg=BG, fg=ACC,
                                   font=("Helvetica", 10, "bold"))
                lbl_obj.pack(side="left", padx=(0, 6))
                self._mtf_label_widgets["mtf_obj_lbl"] = lbl_obj
                self._mtf_objekte = tk.StringVar(value="planeten")
                for val, key in [("planeten", "mtf_planets"),
                                  ("deepsky",  "mtf_deepsky"),
                                  ("alle",     "mtf_all")]:
                    rb2 = tk.Radiobutton(row2, text=T(key), variable=self._mtf_objekte,
                                         value=val, bg=BG, fg="#8e44ad",
                                         activebackground=BG, selectcolor=BG2,
                                         font=("Helvetica", 10),
                                         command=self._aktualisieren)
                    rb2.pack(side="left", padx=8)
                    self._mtf_label_widgets[key] = rb2
                fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))
            elif attr == "_tab_wahrnehmung":
                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))
            elif attr == "_tab_blende":
                fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.0), dpi=96)
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG); ax2.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, ax2, canvas))
            else:
                fs = (7.5, 4.8) if attr == "_tab_beugung" else (5.6, 4.2)
                fig, ax = plt.subplots(figsize=fs, dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

    def _karten_grid(self, parent, karten, start_row, store, name_widgets, pad):
        is_odd_last = len(karten) % 2 == 1
        for j, (str_key, attr) in enumerate(karten):
            r    = start_row + j // 2
            last = is_odd_last and j == len(karten) - 1
            c    = 0 if last else (j % 2) * 2
            cs   = 4 if last else 2
            frm  = tk.Frame(parent, bg=BG2, relief="flat", bd=0, padx=8, pady=5)
            frm.grid(row=r, column=c, columnspan=cs, sticky="ew", padx=6, pady=2)
            lbl_name = tk.Label(frm, text=T(str_key), bg=BG2,
                                font=("Helvetica", 9), fg="#555")
            lbl_name.pack(anchor="w")
            name_widgets[str_key] = lbl_name
            lbl_val = tk.Label(frm, text="–", bg=BG2, font=("Helvetica", 13, "bold"))
            lbl_val.pack(anchor="w")
            store[attr] = lbl_val
        return start_row + math.ceil(len(karten) / 2)

    # ── Aktualisierung ────────────────────────────────────────────────────────

    def _slider_changed(self, _=None):
        """Entprellt Slider-Events."""
        if hasattr(self, "_update_job") and self._update_job:
            self.after_cancel(self._update_job)
        self._update_job = self.after(150, self._aktualisieren)

    def _aktualisieren(self):
        D   = self.var_D.get()
        f   = self.var_f.get()
        lam = self.var_lam.get()
        D_bl = min(self.var_blende.get(), D - 10)  # Blende nie größer als Spiegel

        r_real = berechne(D, f, lam)
        S_real = r_real["strehl"]

        S_pct   = self.var_strehl.get()
        S_slide = S_real + (S_pct / 100.0) * (1.0 - S_real)
        S_slide = min(max(S_slide, S_real), 1.0)

        Deff_k_slide = D * S_slide**0.25
        Vk_slide     = v_kritisch_strehl(D, S_slide)

        for key, (lbl, unit) in self._slider_labels.items():
            if key == "sl_D":
                lbl.config(text=f"{D:.0f} mm")
            elif key == "sl_f":
                lbl.config(text=f"{f:.0f} mm  (f/{f/D:.1f})")
            elif key == "sl_strehl":
                lbl.config(text=f"{S_slide:.3f}  "
                                 f"({'Paraboloid' if S_slide >= 0.9999 else 'Sphäre'})")
            elif key == "sl_blende":
                lbl.config(text=f"{D_bl:.0f} mm  (f/{f/D_bl:.1f})")

        r  = r_real
        rv = berechne_vergr(D, f, lam, 150.0, EYE_RES)
        Vk = v_kritisch(D, f, lam, EYE_RES)

        rv2 = self._res
        rv2["rN"].config(    text=f"f/{r['N']:.2f}")
        rv2["rWp"].config(   text=f"{r['Wp']:.4f} λ")
        rv2["rWb"].config(   text=f"{r['Wb']:.4f} λ")
        rv2["rWrms"].config( text=f"{r['Wrms']:.5f} λ")
        rv2["rS"].config(    text=f"{S_real:.4f}  →  {S_slide:.4f}")

        S_mar = r["strehl_marechal"]
        _col_ex = "#2e7d32" if S_real >= 0.95 else "#e65100" if S_real >= 0.80 else "#c62828"
        rv2["rS_exakt"].config(text=f"{S_real:.4f}  (Δ vs. Mar. {S_real - S_mar:+.4f})", fg=_col_ex)
        rv2["rDeff_k"].config( text=f"{r['Deff_k']:.1f} → {Deff_k_slide:.1f} mm")

        Vk_naeh  = Vk["Vk_sph"]
        Vk_blur  = v_krit_blur_direkt(D, f, EYE_RES)
        Vk_exakt = v_krit_aus_qvis(D, f, lam, rel_schwelle=0.90)
        Vk_halb  = v_halb_exakt(D, f, lam)

        # Schieber-Werte
        Vk_naeh_s  = Vk_slide["Vk_sph"]
        Wb_s       = _Wb_von_S(S_slide)
        Vk_exakt_s = v_krit_aus_qvis_von_Wb(D, Wb_s, lam, rel_schwelle=0.90)
        # Blur-direkt: rein geometrisch, unabhängig vom Strehl → kein Schieber-Wert

        delta_halb = Vk_halb - Vk_naeh
        delta_ex   = Vk_exakt - Vk_naeh
        delta_ex_s = Vk_exakt_s - Vk_naeh_s
        col_naeh   = "#534AB7"
        col_blur   = "#2e7d32" if Vk_blur > 50 else "#e65100" if Vk_blur > 20 else "#c62828"
        col_ex     = "#2e7d32" if abs(delta_ex) < 30 else "#e65100"

        rv2["rVkrit_naeh"].config(
            text=f"{Vk_naeh:.0f}×  →  {Vk_naeh_s:.0f}×"
                 f"  |  exakt: {Vk_halb:.0f}×  (Δ {delta_halb:+.0f}×)",
            fg=col_naeh)
        rv2["rVkrit_blur"].config(
            text=(f"{Vk_blur:.0f}×  (α_blur={D**3/(64*f**2)/f*206265:.1f}\"  — unabh. von S)"
                  if Vk_blur > 1 else "< 1×  — Blur dominiert immer"),
            fg=col_blur)

        def _vk_ex_text(Ve):
            return f"{Ve:.0f}×" if Ve > 30.5 else "< 30×"

        rv2["rVkrit_exakt"].config(
            text=f"{_vk_ex_text(Vk_exakt)}  →  {_vk_ex_text(Vk_exakt_s)}",
            fg=col_ex)

        avp = r_real["aufl_verlust_pct"]
        rv2["rAuflVerlust"].config(
            text=f"{avp:.1f}%  (Blur {r_real['d_blur_as']:.1f}\")",
            fg="#c62828" if avp > 80 else "#e65100" if avp > 40 else "#2e7d32")
        rv2["rVerlustV"].config(text=f"{rv['verlust']:.1f} mm",
                                fg="#c62828" if rv["verlust"] > 5 else "#333")

        text, farbe = beurteilung(S_real)
        self.lbl_verdict.config(text=text, fg=farbe)
        self.lbl_vkrit.config(
            text=f"⚡ Real: V_krit={Vk['Vk_sph']:.0f}×  →  "
                 f"Verbessert (S={S_slide:.2f}): V_krit={Vk_slide['Vk_sph']:.0f}×  |  "
                 f"Paraboloid: V_krit={Vk['Vk_para']:.0f}×  |  V_max={Vk['Vmax']:.0f}×",
            fg="#534AB7")

        Qp_80  = perceived_quality(D, f, lam, 30.0)
        Qp_200 = perceived_quality(D, f, lam, 80.0)
        Qp_350 = perceived_quality(D, f, lam, 160.0)
        Qp_V   = perceived_quality(D, f, lam, 150.0)

        def _qcolor(q):
            return "#2e7d32" if q >= 0.90 else "#e65100" if q >= 0.70 else "#c62828"

        rv2["rQp80"].config( text=f"{Qp_80:.3f}",  fg=_qcolor(Qp_80))
        rv2["rQp200"].config(text=f"{Qp_200:.3f}", fg=_qcolor(Qp_200))
        rv2["rQp350"].config(text=f"{Qp_350:.3f}", fg=_qcolor(Qp_350))
        rv2["rQpV"].config(  text=f"{Qp_V:.3f}  (bei 150×)",
                             fg=_qcolor(Qp_V))

        self._diagramm_strehl(D, f, lam, r["N"], S_real, S_slide)
        self._diagramm_oeffnung(D, f, lam, r["N"], r["Deff_k"],
                                S_sph=S_real, Dk_slide=Deff_k_slide)
        self._diagramm_deff_D(D, r["N"], S_slide)
        self._diagramm_beugung(D, f)
        self._diagramm_mtf(D, f, lam, S_slide)
        self._diagramm_wahrnehmung(D, f, lam, S_slide=S_slide, S_real=S_real)
        self._diagramm_blende(D, f, lam, D_bl)

    # ── Diagramme ─────────────────────────────────────────────────────────────

    def _ax_fmt(self, ax):
        ax.grid(True, color="#ddd", lw=0.5)

    def _diagramm_strehl(self, D, f, lam, N_akt, S_akt, S_sph=None):
        fig, ax, canvas = self._tab_strehl
        ax.cla()
        ns, sts, _, _, sts_exakt = kurven_N(D, lam)
        _, _, Wb_akt, _, S_akt_mar = _wellenfronten(D, f, lam)  # S_akt_mar = Maréchal
        ax.fill_between(ns, S_sph if S_sph else S_akt, 1.0,
                        color=ACC, alpha=0.10, label="Bereich Paraboloid→Sphäre")
        ax.axhline(1.00, color=GRN,       lw=1.5, ls="-",  label="Paraboloid (S=1.0)")
        ax.plot(ns, sts,       color=ACC,       lw=2.0, ls="-",  label="Strehl Maréchal (Näherung)")
        ax.plot(ns, sts_exakt, color="#534AB7", lw=1.8, ls="--", label="Strehl exakt (Pupillenintegral)")
        ax.axhline(0.80, color="#BA7517", lw=1.2, ls="--", label="Rayleigh S=0.80")
        ax.axhline(0.95, color="#888",    lw=1.0, ls=":",  label="S=0.95")
        ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
        ax.scatter([N_akt], [S_akt_mar], color=ACC,       s=60, zorder=5)
        ax.scatter([N_akt], [S_akt],    color="#534AB7", s=40, zorder=5, marker="D")
        ax.annotate(f"f/{N_akt:.1f}  S={S_akt_mar:.3f} (Maréchal) / "
                    f"S={S_akt:.3f} (exakt)",
                    xy=(N_akt, S_akt_mar), xytext=(8, 10),
                    textcoords="offset points", fontsize=9, color="#333",
                    arrowprops=dict(arrowstyle="-", color="#aaa"))
        if S_sph and S_sph > S_akt_mar:
            ax.axhline(S_sph, color=COR, lw=1.5, ls="-.",
                       label=f"Verbessert S={S_sph:.3f}")
            ax.annotate(f"S={S_sph:.3f}", xy=(14.5, S_sph),
                        fontsize=9, color=COR, va="bottom")
        ax.set_xlim(3, 15); ax.set_ylim(0, 1.08)
        ax.set_xlabel(T("ax_fD"), fontsize=10)
        ax.set_ylabel(T("ax_strehl"), fontsize=10)
        ax.set_title(f"Strehl vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=10)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=9, loc="lower right")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_oeffnung(self, D, f, lam, N_akt, Dk_akt,
                            Ds_akt=None, S_sph=None, Dk_slide=None, Ds_slide=None):
        fig, ax, canvas = self._tab_oeff
        ax.cla()
        for ax2_old in fig.get_axes()[1:]:
            fig.delaxes(ax2_old)

        ns, _, deff_ks, aufl_verluste, _ = kurven_N(D, lam)
        r_akt   = berechne(D, f, lam)
        avp_akt = r_akt["aufl_verlust_pct"]

        ax.fill_between(ns, deff_ks, D, color=ACC, alpha=0.10,
                        label="Kontrast-Verlust gg. Paraboloid")
        ax.axhline(D, color="#555", lw=1.5, ls="-", label=f"Paraboloid = {D:.0f}mm")
        ax.plot(ns, deff_ks, color=ACC, lw=2, label="Eff. Öffnung (Kontrast)")
        ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
        ax.scatter([N_akt], [Dk_akt], color=ACC, s=60, zorder=5)
        ax.annotate(f"D_eff={Dk_akt:.0f}mm",
                    xy=(N_akt, Dk_akt), xytext=(N_akt+0.3, Dk_akt-15),
                    fontsize=8, color=ACC)
        if Dk_slide is not None and Dk_slide > Dk_akt + 1:
            ax.scatter([N_akt], [Dk_slide], color=GRN, s=120, zorder=6,
                       marker="*", label=f"Schieber D_eff={Dk_slide:.0f}mm")
        ax.set_xlim(3, 15)
        ax.set_xlabel(T("ax_fD"), fontsize=10)
        ax.set_ylabel(T("ax_deff_mm"), fontsize=10, color=ACC)
        ax.tick_params(axis="y", labelcolor=ACC)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        ax2.fill_between(ns, aufl_verluste, 0, color=COR, alpha=0.08)
        ax2.plot(ns, aufl_verluste, color=COR, lw=2,
                 label="Aufl.verlust gg. Paraboloid [%]")
        ax2.scatter([N_akt], [avp_akt], color=COR, s=60, zorder=5)
        ax2.annotate(f"{avp_akt:.0f}%",
                     xy=(N_akt, avp_akt), xytext=(N_akt+0.3, avp_akt+2),
                     fontsize=8, color=COR, fontweight="bold")
        ax2.axhline(50, color=COR, lw=0.8, ls=":", alpha=0.5)
        ax2.set_ylabel(T("ax_aufl_pct"), fontsize=10, color=COR)
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis="y", labelcolor=COR)
        ax2.spines["right"].set_color(COR)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="center right")
        ax.set_title(f"Eff. Öffnung & Aufl.verlust vs. f/D  (D={D:.0f}mm)", fontsize=10)
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_deff_D(self, D_akt, N_akt, S_slide=None):
        fig, ax, canvas = self._tab_deff_D
        ax.cla()

        D_arr    = np.linspace(50, 400, 300)
        N_list   = [5, 6, 7, 8, 10]
        colors_N = {5:"#c62828", 6:"#D85A30", 7:"#c8a020", 8:"#2e7d32", 10:"#1565c0"}

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
        ax.axhline((1 - 0.8**0.25)*100, color="#888", lw=1.0, ls=":",
                   label=f"Rayleigh S=0.80 ({(1-0.8**0.25)*100:.0f}% Verlust)")
        ax.text(52, (1-0.8**0.25)*100 + 1, "Rayleigh S=0.80", fontsize=8, color="#888")

        ax.set_xlim(50, 420)
        ax.set_ylim(0, min(100, max(90, loss_akt + 15)))
        ax.set_xlabel(T("ax_D_mm"), fontsize=10)
        ax.set_ylabel(T("ax_oeff_loss"), fontsize=10)
        from matplotlib.lines import Line2D
        proxy = [
            Line2D([0],[0], color="#555", lw=2,            label="D × S^0.25  (Kontrast)"),
            Line2D([0],[0], color="#555", lw=1.2, ls="--", label="D × Q_vis^0.25  (visuell)"),
        ]
        ax.legend(handles=proxy, fontsize=9, loc="upper left")
        ax.set_title("Öffnungsverlust vs. Öffnung — "
                     "durchgezogen: D×S^0.25  |  gestrichelt: D×Q_vis^0.25", fontsize=9)
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_beugung(self, D_akt, f_akt):
        fig, ax, canvas = self._tab_beugung
        ax.cla()

        lam_mm = 550e-6
        f_arr  = np.linspace(200, 2000, 500)

        def D_grenz(S):
            Wb = _Wb_von_S(S)
            return (Wb * 4.0 * 1024 * lam_mm) ** 0.25 * f_arr ** 0.75

        D_95 = D_grenz(0.95)
        D_80 = D_grenz(0.80)
        D_50 = D_grenz(0.50)

        ax.fill_between(f_arr, 0,    D_95, color="#2e7d32", alpha=0.10)
        ax.fill_between(f_arr, D_95, D_80, color="#f9a825", alpha=0.12)
        ax.fill_between(f_arr, D_80, 300,  color="#c62828", alpha=0.07)

        ax.text(300,  40,  "S ≥ 0.95  (sehr gut)",              color="#2e7d32", fontsize=8, alpha=0.9)
        ax.text(300, 155,  "0.80 ≤ S < 0.95  (gut)",            color="#c8860a", fontsize=8, alpha=0.9)
        ax.text(300, 255,  "S < 0.80  (nicht beugungsbegrenzt)", color="#c62828", fontsize=8, alpha=0.9)

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
        ax.set_xlabel(T("ax_f_mm"), fontsize=10)
        ax.set_ylabel(T("ax_D_mm"), fontsize=10)
        ax.set_title("Beugungsgrenze sphärischer Spiegel  (λ=550nm)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_mtf(self, D, f, lam, S_slide=None, V=150.0):
        fig, ax, canvas = self._tab_mtf
        ax.cla()
        for ax2_old in fig.get_axes()[1:]:
            fig.delaxes(ax2_old)

        freqs_as, fc, mp_arr, msa, msr_arr, strehl = mtf_kurven(D, f, lam)

        objekte = self._mtf_objekte.get()
        all_details = [
            ("Jupiter\nGürtel",      5.0,      "#4a90d9", "planeten"),
            ("Jupiter\nFestons",     1.5,      "#4a90d9", "planeten"),
            ("Mars\nPol",            1.0,      "#d94a4a", "planeten"),
            ("Saturn\nCassini",      0.5,      "#c8a020", "planeten"),
            ("Trapezium\nAB (~9\")", 9.0,      "#8e44ad", "deepsky"),
            ("M42\nFilamente",       3.0,      "#8e44ad", "deepsky"),
            ("Trapezium\nCD (~2\")", 2.0,      "#8e44ad", "deepsky"),
            ("M13\nHalo-Sterne",     2.5,      "#c0392b", "deepsky"),
            ("M13\nAußensterne",     1.5,      "#c0392b", "deepsky"),
            ("M13\nKern",            0.7,      "#c0392b", "deepsky"),
            ("Dawes\nGrenze",        116.0/D,  "#888",    "beide"),
        ]

        paraxial    = getattr(self, '_mtf_fokus',
                              tk.StringVar(value="bestfocus")).get() == "paraxial"
        fokus_label = "Paraxialer Fokus  (ρ⁴)" if paraxial else "Best Focus  (ρ⁴−ρ²)"
        N_KURVE     = 300
        lam_mm      = lam * 1e-6
        Wb_k        = D**4 / (1024.0 * f**3 * lam_mm) / 4.0
        fc_fokal    = 1.0 / (lam_mm * (f / D))

        nu_k2  = np.linspace(0, 1, N_KURVE)
        mp_k2  = np.array([mtf_para(nu)                              for nu in nu_k2])
        msr_k2 = np.array([mtf_sph_rel(nu, Wb_k, paraxial=paraxial) for nu in nu_k2])
        ms_k2  = mp_k2 * msr_k2

        ax.plot(nu_k2, mp_k2,  color=GRN, lw=2.2, ls="-",
                label="Paraboloid  (Beugungsgrenze)")
        ax.plot(nu_k2, ms_k2,  color=COR, lw=2.2, ls="-",
                label=f"Sphäre absolut  (S={strehl:.3f})")
        ax.plot(nu_k2, msr_k2, color=ACC, lw=1.5, ls="--",
                label="Sphäre relativ  (normiert gg. Paraboloid)")

        if S_slide is not None and S_slide > strehl + 0.01:
            Wb_s    = _Wb_von_S(S_slide)
            msr_s_k = np.array([mtf_sph_rel(nu, Wb_s, paraxial=paraxial)
                                 for nu in nu_k2])
            ax.plot(nu_k2, mp_k2 * msr_s_k, color="#1a7abf", lw=1.8, ls="-.",
                    label=f"Schieber absolut  (S={S_slide:.2f})")

        for label, d_as, col, *_ in all_details:
            nu_det = (1.0 / (2.0 * d_as)) / fc
            if nu_det >= 1.0:
                continue
            ax.axvline(nu_det, color=col, lw=0.8, ls=":", alpha=0.7)
            mp_v = float(np.interp(nu_det, nu_k2, mp_k2))
            ax.text(nu_det + 0.005, mp_v - 0.07,
                    label.replace("\n", " "), fontsize=7, color=col, rotation=90, va="top")

        ax2x = ax.twiny()
        ax2x.set_facecolor("none")
        ax2x.set_xlim(0, fc_fokal)
        ax2x.set_xlabel(T("ax_lpmm"), fontsize=9)
        ax2x.tick_params(axis="x", labelsize=8)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel(T("ax_nu_norm"), fontsize=9)
        ax.set_ylabel(T("ax_mtf"), fontsize=10)
        ax.set_title(
            f"MTF-Kurven [{fokus_label}] — D={D:.0f}mm  f/{f/D:.1f}  "
            f"Strehl={strehl:.3f}  fc={fc_fokal:.1f} Lp/mm  λ={lam:.0f}nm",
            fontsize=9)
        fokus_color = "#2e7d32" if paraxial else ACC
        ax.text(0.98, 0.97,
                f"{fokus_label}\n"
                + ("Vergleichbar mit Sacek §4.7" if paraxial
                   else "Vergleichbar mit Zemax / telescope-optics.net"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=fokus_color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=fokus_color, alpha=0.85))
        ax.legend(fontsize=9, loc="lower left")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_wahrnehmung(self, D, f, lam,
                               S_slide=None, S_real=None):
        """Visueller Qualitätsindex Q_vis vs. Vergrößerung.

        Zwei Kurven pro Spiegel:
          · Q_vis exakt  = MTF×CSF-Integral ohne w(V)         (durchgezogen)
          · Q̃ Näherung  = S + (1−S)·(1−w(V))  [eigene Formel] (gestrichelt)
        """
        fig, ax, canvas = self._tab_wahrnehmung
        ax.cla()

        _, _, Wb, _, S_mar = _wellenfronten(D, f, lam)
        S_exakt = _strehl_exakt(Wb)

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
        if has_slide:
            Wb_s      = _Wb_von_S(S_slide)
            S_ex_s    = S_slide
            S_ex_s_ph = _strehl_exakt(Wb_s)
            Qe_slide  = perceived_quality_exakt_kurven_von_Wb(D, Wb_s, lam, V_arr)
            Qn_slide  = Q_naeh(V_arr, S_ex_s_ph)

        strehl_label = f"Sphäre f/{f/D:.1f}  S={S_exakt:.3f} (exakt)"

        ax.fill_between(V_arr, Qe_sph, Qn_sph,
                        color="#534AB7", alpha=0.12, label="Abweichung exakt ↔ Näherung")
        ax.fill_between(V_arr, Qe_sph, Qe_para,
                        color=COR, alpha=0.10, label="Wahrnehmungsverlust")
        ax.plot(V_arr, Qe_para, color=GRN, lw=2.0, ls="-",  label="Paraboloid  exakt")
        ax.plot(V_arr, Qn_para, color=GRN, lw=1.0, ls="--", alpha=0.4)
        ax.plot(V_arr, Qe_sph,  color=COR, lw=2.2, ls="-",
                label=f"{strehl_label}  exakt")
        ax.plot(V_arr, Qn_sph,  color=COR, lw=1.5, ls="--", alpha=0.75,
                label=f"Sphäre  Näherung Q̃  (S={S_exakt:.3f})")

        if has_slide:
            ax.plot(V_arr, Qe_slide, color=ACC, lw=2.0, ls="-",
                    label=f"Schieber exakt  S={S_ex_s:.3f}")
            ax.plot(V_arr, Qn_slide, color=ACC, lw=1.2, ls="--", alpha=0.7,
                    label=f"Schieber Näherung  S={S_ex_s:.3f}")

        ax.axhline(S_exakt, color=COR, lw=1.0, ls=":", alpha=0.75)
        ax.text(V_max_plot * 0.98, S_exakt + 0.012,
                f"Asymptote Q̃→S={S_exakt:.3f}",
                fontsize=8, color=COR, ha="right", alpha=0.85)

        for V_m, col, ls, lbl in [
                (D / 7.0, "#888888", ":",  f"V_min={D/7:.0f}×"),
                (V_krit,  "#534AB7", "--", f"V_krit={V_krit:.0f}×")]:
            if 30 <= V_m <= V_max_plot:
                ax.axvline(V_m, color=col, lw=1.0, ls=ls, alpha=0.65)
                ax.text(V_m + 2, 0.03, lbl,
                        fontsize=7, color=col, rotation=90, va="bottom", alpha=0.8)

        for q_thresh, col, lbl in [(0.9, GRN, "Q=0.90 (sehr gut)"),
                                    (0.7, "#BA7517", "Q=0.70 (spürbar)")]:
            ax.axhline(q_thresh, color=col, lw=1.0, ls=":", alpha=0.7)
            ax.text(32, q_thresh + 0.008, lbl, fontsize=8, color=col, alpha=0.85)

        max_abw = float(np.max(np.abs(Qe_sph - Qn_sph)))
        ax.set_xlim(30, V_max_plot); ax.set_ylim(0, 1.12)
        ax.set_xlabel(T("ax_vergr"), fontsize=10)
        ax.set_ylabel(T("ax_qvis"), fontsize=10)
        ax.set_title(
            f"Visuelle Wahrnehmung  —  D={D:.0f}mm  f/{f/D:.1f}"
            f"  |  S_exakt={S_exakt:.3f}  max.Abw.={max_abw:.3f}",
            fontsize=9)
        ax.legend(fontsize=8, loc="lower right",
                  title="— exakt (MTF×CSF)   - - Näherung Q̃(V)",
                  title_fontsize=7)
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()


    def _diagramm_blende(self, D, f, lam, D_blend):
        """Blenden-Diagramm: Q̃(V) für Original und Blendenstufen + Leuchtdichte-Effekt."""
        fig, ax, ax2, canvas = self._tab_blende
        ax.cla(); ax2.cla()

        V_arr      = np.linspace(30, 400, 80)   # 80 Punkte reichen für Plot (war 150)
        V_max_plot = 400.0

        d_fang   = d_fang_min(D, f)
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

        _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
        _N_MTF  = 40                                    # MTF-Stützpunkte (war 80)
        _nu     = np.linspace(0.0, 1.0, _N_MTF)        # einmal berechnen, nicht per V

        # MTF-Arrays für Original-Paraboloid (hängen nicht von V ab) — einmal berechnen
        _mp_orig = np.array([mtf_para(ni) for ni in _nu])
        _fc_orig = D / (lam * 1e-6 * 206265)

        _q_cache: dict = {}   # lokaler Cache: Db → (Q_arr, S, Vk)

        def Q_opt_kurve(Db):
            """Optische Qualität normiert auf Original-Paraboloid — MTF×CSF.

            Obstruktion: ε = d_fang/Db — wächst beim Abblenden.
            Ergebnis wird gecacht: gleiche Blende wird nicht zweimal gerechnet.
            """
            key = round(Db, 2)
            if key in _q_cache:
                return _q_cache[key]

            _, _, Wb, _, _ = _wellenfronten(Db, f, lam)
            S       = _strehl_exakt(Wb)          # gecacht durch @lru_cache
            Vk      = Db * 0.7 * math.sqrt(max(S, 1e-9))
            eps     = min(d_fang / Db, 0.99)
            fc_bl   = Db / (lam * 1e-6 * 206265)
            f_ratio = fc_bl / _fc_orig

            # Blenden-MTF-Arrays einmal für alle V berechnen
            nu_bl = _nu / max(f_ratio, 1e-6)
            mp_bl = np.array([mtf_para_obstr(min(ni, 1.0), eps) for ni in nu_bl])
            ms_bl = np.array([mtf_sph_rel(min(ni, 1.0), Wb)     for ni in nu_bl])
            cutoff_mask = _nu > f_ratio
            mp_bl[cutoff_mask] = 0.0
            ms_bl[cutoff_mask] = 0.0
            mtf_bl = ms_bl * mp_bl   # kombinierte Blenden-MTF (V-unabhängig)

            Q_arr = np.zeros(len(V_arr))
            for j, V in enumerate(V_arr):
                w     = np.array([csf(ni * _fc_orig * 3600.0 / V) for ni in _nu])
                num   = float(_trapz(mtf_bl  * w, _nu))
                denom = float(_trapz(_mp_orig * w, _nu))
                Q_arr[j] = num / max(denom, 1e-12)

            result = (Q_arr, S, Vk)
            _q_cache[key] = result
            return result

        # ── Original-Spiegel ─────────────────────────────────────────────────
        Q_orig, S_orig, Vk_orig = Q_opt_kurve(D)
        ax.plot(V_arr, Q_orig, color=COR, lw=2.5, ls="-",
                label=f"Original  D={D:.0f}mm  S={S_orig:.3f}  V½={Vk_orig:.0f}×")
        ax.axhline(float(Q_orig[-1]), color=COR, lw=0.8, ls=":", alpha=0.5)
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
            # Asymptote blend: unterhalb der Linie, damit kein Overlap mit orig
            ax.text(V_max_plot * 0.98, float(Q_bl[-1]) - 0.022,
                    f"Asymptote blend={float(Q_bl[-1]):.3f}",
                    fontsize=7, color=ACC, ha="right", va="top", alpha=0.85)

            # Gewinn/Verlust durch Blende — links von V½ (weg von opt-Blende-Annotation)
            Q_bl_at_Vk = float(np.interp(Vk_orig, V_arr, Q_bl))
            Q_or_at_Vk = float(np.interp(Vk_orig, V_arr, Q_orig))
            delta = Q_bl_at_Vk - Q_or_at_Vk
            x_ann = max(Vk_orig - 80, 50)
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
        D_scan  = np.linspace(D * 0.40, D, 8)    # 8 Stichproben reichen (war 12)
        Q_stack = np.array([Q_opt_kurve(Db)[0] for Db in D_scan])
        Q_opt_line = Q_stack.max(axis=0)

        # Optimale Blende pro Vergrößerung — welches D gibt max Q?
        D_opt_arr = np.array([D_scan[Q_stack[:, j].argmax()] for j in range(len(V_arr))])

        # Repräsentativer Wert: bei V_krit_orig
        idx_vk  = int(np.argmin(np.abs(V_arr - V_krit_orig)))
        D_opt_vk = D_opt_arr[idx_vk]
        Q_opt_vk = Q_opt_line[idx_vk]
        # Und im sinnvollen Bereich: Mittelwert der optimalen Blende
        mask_sinn = (V_arr >= V_min_orig) & (V_arr <= V_max_orig)
        D_opt_mean = float(D_opt_arr[mask_sinn].mean()) if mask_sinn.any() else D_opt_vk

        ax.plot(V_arr, Q_opt_line, color=GRN, lw=1.8, ls="-.",
                label=f"Optimale Blende (max Q je V)  ⌀≈{D_opt_mean:.0f}mm im sinnv. Bereich")

        # Annotation opt. Blende: rechts und unterhalb, weg von Blende-Box
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
        if V_min_orig > 32:
            ax.text((30 + min(V_min_orig, V_max_plot)) / 2, 0.96,
                    "nicht sinnvoll\n(zu niedrig)",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color="#999", style="italic", zorder=1)
        if V_max_orig < V_max_plot:
            ax.axvspan(min(V_max_orig, V_max_plot), V_max_plot,
                       color="#888", alpha=0.08, zorder=0)
            ax.text((V_max_orig + V_max_plot) / 2, 0.96,
                    "nicht sinnvoll\n(zu hoch)",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color="#999", style="italic", zorder=1)
        for V_m, lbl in [
            (V_min_orig, f"V_min={V_min_orig:.0f}×\n(AP=7mm)"),
            (V_max_orig, f"V_max={V_max_orig:.0f}×\n({'AP=0.5mm' if V_max_orig >= V_max_AP*0.99 else '2×V_krit'})"),
        ]:
            if 30 < V_m < V_max_plot:
                ax.axvline(V_m, color="#999", lw=1.0, ls=":", alpha=0.8)
                ax.text(V_m + 2, 0.72, lbl, fontsize=7, color="#777",
                        rotation=90, va="center", alpha=0.9)
        if D_blend < D * 0.98:
            for V_m, lbl in [
                (V_min_blend, f"V_min={V_min_blend:.0f}×\n(Blende)"),
                (V_max_blend, f"V_max={V_max_blend:.0f}×\n({'AP' if V_max_blend >= V_max_AP_bl*0.99 else '2×V_krit'} Blende)"),
            ]:
                if 30 < V_m < V_max_plot:
                    ax.axvline(V_m, color=ACC, lw=1.0, ls=":", alpha=0.6)
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
        self._ax_fmt(ax)

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
        for Db in np.linspace(D * 0.4, D, 40):   # 40 reichen; _strehl_exakt gecacht
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
        if V_min_orig > 32:
            ax2.text((30 + min(V_min_orig, V_max_plot)) / 2, 0.96,
                     "nicht sinnvoll\n(zu niedrig)",
                     transform=ax2.get_xaxis_transform(), ha="center", va="top",
                     fontsize=7, color="#999", style="italic", zorder=1)
        if V_max_orig < V_max_plot:
            ax2.axvspan(min(V_max_orig, V_max_plot), V_max_plot,
                        color="#888", alpha=0.08, zorder=0)
            ax2.text((V_max_orig + V_max_plot) / 2, 0.96,
                     "nicht sinnvoll\n(zu hoch)",
                     transform=ax2.get_xaxis_transform(), ha="center", va="top",
                     fontsize=7, color="#999", style="italic", zorder=1)
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
        self._ax_fmt(ax2)

        fig.tight_layout(); canvas.draw_idle()


if __name__ == "__main__":
    app = App()
    app.mainloop()
