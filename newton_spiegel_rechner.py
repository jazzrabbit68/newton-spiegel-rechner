"""
Newton-Teleskop: Kontrast- und Schärfeverlust sphärischer Hauptspiegel
=======================================================================
Berücksichtigt:
  - Wellenfrontfehler und Strehl-Quotient (Kontrast)
  - Geometrischen Unschärfefleck am besten Fokus (Schärfe)
  - Vergrößerungsabhängigen Schärfeverlust durch das Auge

Formeln:
  Kontrast (Strehl):
    W_PtV (paraxial)  = 22 * D[inch] / N^3  [Einheiten λ=550nm]
    W_PtV (best focus)= W_PtV_paraxial / 4
    W_RMS             = W_PtV_bestfocus / (1.5 * sqrt(5))
    Strehl            = exp(-(2π * W_RMS)^2)   [Maréchal-Näherung]
    D_eff_kontrast    = D * sqrt(Strehl)

  Schärfe (statisch, vergrößerungsunabhängig):
    r_Airy [arcsec]   = 1.22 * λ / D * 206265
    blur [arcsec]     = (W_PtV_bestfocus / 2.44) * 2 * r_Airy
    Dawes [arcsec]    = 116 / D[mm]
    θ_eff             = sqrt(Dawes² + blur²)
    D_eff_schaerfe    = 116 / θ_eff

  Schärfe (vergrößerungsabhängig):
    Auflösung des Auges: ~60 arcsec (1 Bogenminute, Kontrastschwelle für Planetendetails)
    Im Objektraum zurückgerechnet: θ_auge = eye_res / V
    Parabolspiegel-Limit: θ_para = max(Dawes, θ_auge)
    Sphäre-Limit:         θ_sph  = max(sqrt(Dawes²+blur²), θ_auge)
    D_eff_para(V) = 116 / θ_para
    D_eff_sph(V)  = 116 / θ_sph
    Verlust(V)    = D_eff_para(V) - D_eff_sph(V)

    → Unter der kritischen Vergrößerung V_krit dominiert das Auge:
      kein Unterschied zwischen Paraboloid und Sphäre sichtbar.
    → Über V_krit ist der Aberrationsblur für das Auge erkennbar.

Quellen: telescope-optics.net, gordtulloch.com

Abhängigkeiten: pip install matplotlib
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

def berechne(D_mm: float, f_mm: float, lam_nm: float) -> dict:
    """Grundlegende optische Kenngrößen (vergrößerungsunabhängig)."""
    N      = f_mm / D_mm
    D_inch = D_mm / 25.4
    lam_mm = lam_nm * 1e-6

    # Wellenfrontfehler (klassische Formel, ZEMAX-verifiziert)
    Wp     = D_mm**4 / (1024.0 * f_mm**3 * lam_mm)     # W_PtV paraxial [λ]
    Wb     = Wp / 4.0                                    # W_PtV best focus [λ]
    Wrms   = Wb / (1.5 * math.sqrt(5))                  # W_RMS best focus [λ]
    S      = math.exp(-(2 * math.pi * Wrms) ** 2)       # Strehl (Maréchal)
    Deff_k = D_mm * S**0.25                              # D_eff = D * S^(1/4)

    # Schärfe: visuell relevante Größen
    r_airy_as   = 1.22 * lam_mm / D_mm * 206265.0
    theta_ideal = 116.0 / D_mm                         # Dawes-Grenze ["]
    # theta_eff für visuelle Wahrnehmung: Strehl-basiert
    # Begründung: Beobachter sieht PSF, nicht geometrischen Lichtkegel
    # theta_eff = Dawes / sqrt(S)  (aus Deff = D*sqrt(S) -> theta = 116/Deff)
    theta_eff   = theta_ideal / math.sqrt(max(S, 1e-6))
    # Geometrischer Blur (für Information, nicht für V_krit)
    d_blur_mm   = D_mm**3 / (64.0 * f_mm**2)
    d_blur_as   = d_blur_mm / f_mm * 206265.0
    # Auflösungsverlust aus geometrischem Blur (gibt an wie stark der Strahl aufgeweitet wird)
    theta_geo   = math.sqrt(theta_ideal**2 + d_blur_as**2)
    aufl_verlust_pct = (1.0 - theta_ideal / theta_geo) * 100.0
    # Kritische Vergrößerung: V_krit = D_mm * 0.7 * sqrt(S)
    # (Strehl-gewichtete nutzbare Max-Vergrößerung, empirisch nach Suiter)
    V_krit_vis  = D_mm * 0.7 * math.sqrt(max(S, 1e-6))

    # ── Visueller Qualitätsindex Q_vis ─────────────────────────────
    #
    # mtf_sph_rel(f, Wb) liefert den RELATIVEN Formverlust der MTF:
    #
    #   Q(f) = MTF_sphäre(f) / MTF_paraboloid(f)
    #
    # also nur die Veränderung der Kurvenform.
    #
    # Der absolute Kontrastverlust durch sphärische Aberration wird
    # zusätzlich durch den Strehl-Quotienten beschrieben.
    #
    # Deshalb:
    #   Qshape = gewichteter MTF-Formfaktor
    #   Qvis   = Strehl × Qshape
    #
    # Die Gewichtung ist heuristisch und orientiert sich grob an
    # visuellen Planetendetails:
    #
    #   f=0.2 → grobe Details
    #   f=0.4 → mittlere Details
    #   f=0.6 → feine Details
    #
    # Effektive Öffnung:
    #
    #   D_eff ~ D × Q^(1/4)
    #
    # analog zur bereits verwendeten Strehl-Skalierung:
    #
    #   D_eff = D × Strehl^(1/4)
    #
    # Dadurch werden Formverlust UND Energieverlust berücksichtigt.
    #
    Q02 = mtf_sph_rel(0.2, Wb)
    Q04 = mtf_sph_rel(0.4, Wb)
    Q06 = mtf_sph_rel(0.6, Wb)

    # rein relativer Formverlust der MTF
    Qshape = 0.4*Q02 + 0.4*Q04 + 0.2*Q06

    # absoluter visueller Qualitätsindex
    Qvis = S * Qshape

    # äquivalente effektive Öffnung
    Deff_vis = D_mm * Qvis**0.25

    return dict(
        N=N, D_inch=D_inch,
        Wp=Wp, Wb=Wb, Wrms=Wrms,
        strehl=S, Deff_k=Deff_k, loss_k=D_mm - Deff_k,
        r_airy_as=r_airy_as, d_blur_mm=d_blur_mm, d_blur_as=d_blur_as,
        theta_ideal=theta_ideal, theta_eff=theta_eff,
        aufl_verlust_pct=aufl_verlust_pct, V_krit_vis=V_krit_vis,
        Q02=Q02, Q04=Q04, Q06=Q06, Qshape=Qshape,
        Qvis=Qvis, Deff_vis=Deff_vis,
        loss_vis=D_mm - Deff_vis,
    )


def berechne_vergr(D_mm: float, f_mm: float, lam_nm: float,
                   V: float, eye_res_as: float = 60.0) -> dict:
    """
    Vergrößerungsabhängige effektive Öffnung für Paraboloid und Sphäre.

    eye_res_as : Kontrastschwelle des Auges für Planetendetails in Bogensekunden (Standard 60″ = 1′)
    V          : Vergrößerung
    """
    r  = berechne(D_mm, f_mm, lam_nm)
    theta_auge = eye_res_as / V           # Auge-Limit im Objektraum

    theta_para = max(r["theta_ideal"], theta_auge)
    theta_sph  = max(r["theta_eff"],   theta_auge)

    Deff_para = min(D_mm, 116.0 / theta_para)
    Deff_sph  = min(D_mm, 116.0 / theta_sph)
    verlust   = Deff_para - Deff_sph

    return dict(theta_auge=theta_auge,
                theta_para=theta_para, theta_sph=theta_sph,
                Deff_para=Deff_para,   Deff_sph=Deff_sph,
                verlust=verlust, **r)


def v_kritisch(D_mm: float, f_mm: float, lam_nm: float,
               eye_res_as: float = 60.0) -> dict:
    """
    Kritische Vergrößerung: V_krit = D_mm * 0.7 * sqrt(Strehl)
    (Strehl-gewichtete nutzbare Max-Vergrößerung nach Suiter)
    """
    r = berechne(D_mm, f_mm, lam_nm)
    return dict(
        Vk_sph  = r["V_krit_vis"],
        Vk_para = D_mm * 0.7,
        Vmax    = D_mm / 0.5,
        Vmin    = D_mm / 7.0,
    )



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


def kurven_N(D_mm, lam_nm, N_min=3.0, N_max=15.0, schritte=400):
    """Strehl und eff. Öffnungen als Funktion von N."""
    ns = np.linspace(N_min, N_max, schritte)
    strehls, deff_ks, aufl_verluste = [], [], []
    for n in ns:
        r = berechne(D_mm, D_mm * n, lam_nm)
        strehls.append(r["strehl"])
        deff_ks.append(r["Deff_k"])
        aufl_verluste.append(r["aufl_verlust_pct"])
    return ns, np.array(strehls), np.array(deff_ks), np.array(aufl_verluste)


def strehl_to_f(S: float, D_mm: float, lam_nm: float = 550.0) -> float:
    """Brennweite die einem bestimmten Strehl-Wert entspricht (sphärischer Spiegel).
    Inversion von: Wp = D^4/(1024*f^3*lam), Wb=Wp/4, Wrms=Wb/(1.5*sqrt(5))
    Strehl ≈ exp(-(2π·Wrms)²)  →  Wrms = sqrt(-ln(S)) / (2π)
    """
    lam_mm = lam_nm * 1e-6
    if S >= 0.9999:
        return D_mm * 50.0
    Wrms   = math.sqrt(-math.log(max(S, 1e-6))) / (2.0 * math.pi)
    Wb     = Wrms * 1.5 * math.sqrt(5)
    Wp     = Wb * 4.0
    f3     = D_mm**4 / (1024.0 * Wp * lam_mm)
    N      = (f3 ** (1.0/3.0)) / D_mm
    return D_mm * N


def v_kritisch_strehl(D_mm: float, strehl: float, theta_eff: float = None,
                      eye_res_as: float = 60.0) -> dict:
    """V_krit für beliebigen Strehl-Wert (für Schieber-Vergleich)."""
    return dict(
        Vk_sph  = D_mm * 0.7 * math.sqrt(max(strehl, 1e-6)),
        Vk_para = D_mm * 0.7,
        Vmax    = D_mm / 0.5,
        Vmin    = D_mm / 7.0,
    )


def mtf_para(f: float) -> float:
    """Analytische MTF eines beugungsbegrenzten Kreisteleskops (Paraboloid)."""
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    return (2/math.pi) * (math.acos(f) - f * math.sqrt(1 - f**2))

@lru_cache(maxsize=2048)
def mtf_sph_rel(f: float, Wb: float, n: int = 80) -> float:
    """
    Relative MTF eines sphärischen Spiegels (normiert: MTF(0)=1).
    Wb = W_PtV_bestfocus = W_PtV_paraxial / 4
    f  = normierte Ortsfrequenz (0..1, fc=Grenzfrequenz)
    Berechnung via 1D-Autokorrelationsintegral der Pupillenfunktion.
    """
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    lim = math.sqrt(max(0.0, 1.0 - (f/2)**2))
    re = im = norm = 0.0
    for i in range(n):
        x  = -lim + (2*i + 1) / (2*n) * 2*lim
        x1 = x - f/2
        x2 = x + f/2
        if x1**2 >= 1 or x2**2 >= 1:
            continue
        h  = min(math.sqrt(max(0.0, 1 - x1**2)),
                 math.sqrt(max(0.0, 1 - x2**2)))
        dW = 2*math.pi * (Wb*(x2**4 - x2**2) - Wb*(x1**4 - x1**2))
        re   += math.cos(dW) * h
        im   += math.sin(dW) * h
        norm += h
    if norm < 1e-10:
        return 0.0
    return math.sqrt(re**2 + im**2) / norm


def mtf_kurven(D_mm: float, f_mm: float, lam_nm: float = 550.0,
               n_pts: int = 60) -> tuple:
    """
    Berechnet MTF-Kurven für Paraboloid und sphärischen Spiegel.
    Rückgabe: (freqs, mtf_para_abs, mtf_sph_abs, mtf_sph_rel_arr)
      - mtf_para_abs: normiert auf 1 bei f=0
      - mtf_sph_abs:  Strehl × MTF_sph_relativ  (absolute Übertragung)
      - mtf_sph_rel_arr: nur der Formfaktor (normiert auf 1 bei f=0)
    Ortsfrequenz-Achse in Linien/arcsec:
      fc = D/(lam*206265) [L/arcsec] — Grenzfrequenz
    """
    N      = f_mm / D_mm
    D_inch = D_mm / 25.4
    Wp     = D_mm**4 / (1024.0 * f_mm**3 * (lam_nm*1e-6))     # W_PtV paraxial [λ]
    Wb     = Wp / 4.0                                          # W_PtV best focus [λ]
    Wrms   = Wb / (1.5 * math.sqrt(5))                        # W_RMS best focus [λ]
    strehl = math.exp(-(2 * math.pi * Wrms)**2)

    # Grenzfrequenz in Linien/arcsec
    fc = D_mm / (lam_nm * 1e-6 * 206265)

    freqs_norm = np.linspace(0, 1, n_pts)
    freqs_as   = freqs_norm * fc          # in L/arcsec

    mp  = np.array([mtf_para(f)           for f in freqs_norm])
    msr = np.array([mtf_sph_rel(f, Wb)    for f in freqs_norm])
    msa = msr * mp * strehl               # absolut: Strehl × rel × MTF_para

    return freqs_as, fc, mp, msa, msr, strehl


def mtf_eye(f_cpd: float) -> float:
    """
    Augen-MTF nach Barten-Näherung.
    f_cpd : Ortsfrequenz in cycles/degree (cpd)
    fc    : Cut-off ~30 cpd (Tages-Sehen)
    """
    fc = 30.0
    return float(np.exp(-(f_cpd / fc) ** 1.3))


def csf(f_cpd: float) -> float:
    """
    Contrast Sensitivity Function (CSF) nach van Meeteren / Watson-Ahumada.
    Modelliert die frequenzabhängige Empfindlichkeit des Auges.
    f_cpd : Ortsfrequenz in cycles/degree
    """
    f = max(f_cpd, 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))


def perceived_quality(D_mm: float, f_mm: float, lam_nm: float,
                      V: float, n_pts: int = 120,
                      use_csf: bool = True) -> float:
    """
    Wahrgenommener Qualitätsindex Q_perc(V).

    Modell:
      Q_raw  = CSF-gewichtetes MTF-Integral (Sphäre / Paraboloid)
               → misst den optischen Qualitätsverlust unabhängig von V

      Q_eff  = Q_raw + (1 - Q_raw) * (1 - w(V))

    Übergangsgewicht w(V):
      w = 1 / (1 + (V_krit / V)²)
      → w≈0 bei V << V_krit  (Auge limitiert → Aberration unsichtbar → Q_eff→1)
      → w≈1 bei V >> V_krit  (Teleskop limitiert → volle Aberration → Q_eff→Q_raw)

    V_krit = eye_res (60″) / theta_eff_sphäre
    """
    freqs_as, fc_scope, mp_arr, msa_arr, msr_arr, strehl = mtf_kurven(
        D_mm, f_mm, lam_nm, n_pts)

    nu = freqs_as / fc_scope

    # CSF-Gewichtung im Bildraum: f_cpd = f_obj[L/as] * 3600 / V
    f_cpd_arr = freqs_as * 3600.0 / V
    if use_csf:
        weight = np.array([csf(fi) for fi in f_cpd_arr])
    else:
        weight = np.array([mtf_eye(fi) for fi in f_cpd_arr])

    msa_scope = msr_arr * mp_arr * strehl

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    num   = _trapz(msa_scope * weight, nu)
    denom = _trapz(mp_arr    * weight, nu)
    Q_raw = float(num / denom) if denom > 1e-12 else 0.0

    # Kritische Vergrößerung: Strehl-basiert (konsistent mit v_kritisch)
    V_krit = D_mm * 0.7 * math.sqrt(max(strehl, 1e-6))

    # Übergangsgewicht
    w = 1.0 / (1.0 + (V_krit / V) ** 2)

    return float(Q_raw + (1.0 - Q_raw) * (1.0 - w))


def perceived_quality_kurven(D_mm: float, f_mm: float, lam_nm: float,
                              V_arr: np.ndarray,
                              use_csf: bool = True) -> np.ndarray:
    """Berechnet perceived_quality für ein Array von Vergrößerungen."""
    return np.array([perceived_quality(D_mm, f_mm, lam_nm, V, use_csf=use_csf)
                     for V in V_arr])


def beurteilung(strehl):
    if strehl >= 0.95:
        return "✓ Sehr gut — nahezu gleichwertig mit Parabolspiegel (Strehl ≥ 0.95)", "#2e7d32"
    elif strehl >= 0.80:
        return ("⚠  Noch beugungsbegrenzt (Rayleigh), spürbarer Kontrast-"
                "verlust bei Planeten  (0.80 ≤ Strehl < 0.95)"), "#e65100"
    else:
        return ("✗  Nicht beugungsbegrenzt — erheblicher Kontrast- und"
                " Schärfeverlust, Parabolspiegel dringend empfohlen"
                " (Strehl < 0.80)"), "#c62828"


# ── GUI ───────────────────────────────────────────────────────────────────────

BG  = "#f5f5f5"
BG2 = "#e8e8e8"
ACC = "#534AB7"
COR = "#D85A30"
GRN = "#0F6E56"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Newton-Spiegel: Kontrast- und Schärfeverlust")
        self.resizable(True, True)
        self.configure(bg=BG)
        self._build_ui()
        self._aktualisieren()

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

        # ── Eingaben ─────────────────────────────────────────────────────────
        tk.Label(left, text="Teleskop-Parameter",
                 font=("Helvetica", 11, "bold"), bg=BG
                 ).grid(row=0, column=0, columnspan=3, sticky="w", **pad)

        self.var_D      = tk.DoubleVar(value=200)
        self.var_f      = tk.DoubleVar(value=1000)
        self.var_strehl = tk.DoubleVar(value=0)    # 0=sphärisch, 100=Paraboloid
        self.var_eye    = tk.DoubleVar(value=60)
        self.var_lam    = tk.DoubleVar(value=550)

        self.var_vergr  = tk.DoubleVar(value=150)

        slider_defs = [
            ("Öffnung D",       self.var_D,      50,  500, 10,  "mm"),
            ("Brennweite f",    self.var_f,      200, 4000, 50, "mm"),
            ("Strehl-Quotient", self.var_strehl,   0,  95,   5,  ""),
            ("Auge-Auflösung",  self.var_eye,     20,  180,  5,  '"'),
            ("Vergrößerung",    self.var_vergr,   30,  250, 10,  "×"),
        ]
        self._slider_labels = {}
        for i, (name, var, mn, mx, res, unit) in enumerate(slider_defs, 1):
            tk.Label(left, text=name, bg=BG, width=16, anchor="w"
                     ).grid(row=i, column=0, sticky="w", **pad)
            tk.Scale(left, from_=mn, to=mx, resolution=res,
                     orient="horizontal", variable=var, length=205,
                     #command=lambda _: self._aktualisieren(),
                     command=self._slider_changed,
                     bg=BG, highlightthickness=0,
                     troughcolor="#ddd", activebackground=ACC
                     ).grid(row=i, column=1, **pad)
            lbl = tk.Label(left, text="", bg=BG, width=9, anchor="w")
            lbl.grid(row=i, column=2, sticky="w", **pad)
            self._slider_labels[name] = (lbl, unit)

        row = len(slider_defs) + 1

        # ── Ergebniskarten ────────────────────────────────────────────────────
  #      tk.Label(left, text="Rechenschritte",
  #               font=("Helvetica", 11, "bold"), bg=BG
  #               ).grid(row=row, column=0, columnspan=3, sticky="w", **pad)
  #      row += 1
  #      self.txt_steps = tk.Text(left, height=11, width=56,
  #                               font=("Courier", 9), bg=BG2,
  #                               relief="flat", state="disabled", wrap="none")
  #      self.txt_steps.grid(row=row, column=0, columnspan=3, **pad)
  #      row += 1

        # ── Ergebniskarten ────────────────────────────────────────────────────
        self._res = {}

        for titel, karten in [
            ("Kontrast", [
                ("Öffnungsverhältnis",         "rN"),
                ("W_PtV paraxial [λ]",         "rWp"),
                ("W_PtV best focus [λ]",       "rWb"),
                ("W_RMS [λ]",                  "rWrms"),
                ("Strehl best focus (visuell)", "rS"),
                ("Eff. Öffnung Kontrast [mm]", "rDeff_k"),
                ("V_krit Sphäre (Blur sichtb.)","rVkrit2"),
            ]),

            ("Visueller Qualitätsindex Q_vis (MTF + Strehl)", [
                ("Q(0.2) grobe Details",        "rQ02"),
                ("Q(0.4) mittlere Details",     "rQ04"),
                ("Q(0.6) feine Details",        "rQ06"),
                ("Q_shape (nur MTF-Form)",      "rQshape"),
                ("Q_vis (gewichtet 0.4/0.4/0.2)","rQvis"),
                ("D_eff visuell [mm]",          "rDeff_vis"),
            ]),
            ("Schärfe bei gewählter Vergrößerung", [
                ("Aufl.verlust gg. Paraboloid [%]",    "rAuflVerlust"),
                ("Verlust bei dieser Vergr. [mm]",    "rVerlustV"),
            ]),
            ("Wahrgenommener Kontrast Q_perc (CSF-gewichtet)", [
                ("Q_perc  bei  30×",                  "rQp80"),
                ("Q_perc  bei  80×",                  "rQp200"),
                ("Q_perc  bei 160×",                  "rQp350"),
                ("Q_perc  bei Slider-Vergr.",          "rQpV"),
            ]),
        ]:
            tk.Label(left, text=titel,
                     font=("Helvetica", 11, "bold"), bg=BG
                     ).grid(row=row, column=0, columnspan=3,
                            sticky="w", **pad)
            row += 1
            row = self._karten_grid(left, karten, row, self._res, pad)

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
        row += 1

        # ── Diagramm-Tabs ─────────────────────────────────────────────────────
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        tabs = [
            ("_tab_strehl",       "  Strehl vs. f/D  "),
            ("_tab_oeff",         "  Eff. Öffnung vs. f/D  "),
            ("_tab_deff_D",       "  D_eff vs. Öffnung  "),
            ("_tab_beugung",      "  Beugungsgrenze  "),
            ("_tab_mtf",          "  MTF  "),
            ("_tab_wahrnehmung",  "  Wahrnehmung  "),
        ]
        for attr, title in tabs:
            frm = tk.Frame(nb, bg=BG)
            nb.add(frm, text=title)
            if attr == "_tab_mtf":
                # Umschalter: absolut / relativ
                btn_frame = tk.Frame(frm, bg=BG)
                btn_frame.pack(side="top", fill="x", padx=8, pady=(6,0))
                # Zeile 1: Darstellungsmodus
                row1 = tk.Frame(btn_frame, bg=BG)
                row1.pack(side="top", anchor="w")
                tk.Label(row1, text="Darstellung:", bg=BG, fg=ACC,
                         font=("Helvetica", 10, "bold")).pack(side="left", padx=(0,6))
                self._mtf_modus = tk.StringVar(value="absolut")
                for val, text in [("absolut", "Absolut (Para vs. Sphäre)"),
                                   ("relativ", "Relativ (Kontrastverlust %)")]:
                    tk.Radiobutton(row1, text=text, variable=self._mtf_modus,
                                   value=val, bg=BG, fg=ACC,
                                   activebackground=BG, selectcolor=BG2,
                                   font=("Helvetica", 10),
                                   command=self._aktualisieren
                                   ).pack(side="left", padx=8)
                # Zeile 2: Objektkategorie
                row2 = tk.Frame(btn_frame, bg=BG)
                row2.pack(side="top", anchor="w", pady=(3,0))
                tk.Label(row2, text="Objekte:", bg=BG, fg=ACC,
                         font=("Helvetica", 10, "bold")).pack(side="left", padx=(0,6))
                self._mtf_objekte = tk.StringVar(value="planeten")
                for val, text in [("planeten", "Planeten"),
                                   ("deepsky",  "Deep-Sky (M42)"),
                                   ("alle",     "Alle")]:
                    tk.Radiobutton(row2, text=text, variable=self._mtf_objekte,
                                   value=val, bg=BG, fg="#8e44ad",
                                   activebackground=BG, selectcolor=BG2,
                                   font=("Helvetica", 10),
                                   command=self._aktualisieren
                                   ).pack(side="left", padx=8)
                fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=96)
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))
            elif attr == "_tab_wahrnehmung":
                fs = (8.0, 4.8)
                fig, ax = plt.subplots(figsize=fs, dpi=96)
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))
            else:
                fs = (7.5, 4.8) if attr == "_tab_beugung" else (5.6, 4.2)
                fig, ax = plt.subplots(figsize=fs, dpi=96)
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

    def _karten_grid(self, parent, karten, start_row, store, pad):
        is_odd_last = len(karten) % 2 == 1
        for j, (name, attr) in enumerate(karten):
            r    = start_row + j // 2
            last = is_odd_last and j == len(karten) - 1
            c    = 0 if last else (j % 2) * 2
            cs   = 4 if last else 2
            frm  = tk.Frame(parent, bg=BG2, relief="flat",
                            bd=0, padx=8, pady=5)
            frm.grid(row=r, column=c, columnspan=cs,
                     sticky="ew", padx=6, pady=2)
            tk.Label(frm, text=name, bg=BG2,
                     font=("Helvetica", 9), fg="#555").pack(anchor="w")
            lbl = tk.Label(frm, text="–", bg=BG2,
                           font=("Helvetica", 13, "bold"))
            lbl.pack(anchor="w")
            store[attr] = lbl
        return start_row + math.ceil(len(karten) / 2)

    # ── Aktualisierung ────────────────────────────────────────────────────────

    def _slider_changed(self, _=None):
        """Entprellt Slider-Events."""
        if hasattr(self, "_update_job") and self._update_job:
            self.after_cancel(self._update_job)
        self._update_job = self.after(150, self._aktualisieren)

    def _aktualisieren(self):
        D      = self.var_D.get()
        f      = self.var_f.get()          # FEST — ändert sich nicht
        eye    = self.var_eye.get()
        lam    = self.var_lam.get()        # intern 550 nm
        V_slider = self.var_vergr.get()    # Vergrößerung für Wahrnehmungs-Tab
        V      = V_slider                  # für Schärfe-Karten

        # Realer Spiegel (aus D und f)
        r_real = berechne(D, f, lam)
        S_real = r_real["strehl"]

        # Strehl-Schieber: 0 = sphärischer Wert, 100 = Paraboloid (S=1)
        S_pct   = self.var_strehl.get()
        S_slide = S_real + (S_pct / 100.0) * (1.0 - S_real)
        S_slide = min(max(S_slide, S_real), 1.0)

        # Hypothetischer verbesserter Spiegel: f_slide aus S_slide
        f_slide = strehl_to_f(S_slide, D, lam) if S_slide > S_real + 0.01 else f
        r_slide = berechne(D, f_slide, lam)
        Deff_k_slide = D * S_slide**0.25
        Vk_slide = v_kritisch_strehl(D, S_slide)

        # Schieber-Beschriftungen
        for name, (lbl, unit) in self._slider_labels.items():
            if name == "Öffnung D":
                lbl.config(text=f"{D:.0f} mm")
            elif name == "Brennweite f":
                lbl.config(text=f"{f:.0f} mm  (f/{f/D:.1f})")
            elif name == "Strehl-Quotient":
                lbl.config(text=f"{S_slide:.3f}  "
                                f"({'Paraboloid' if S_slide >= 0.9999 else 'Sphäre'})")
            elif name == "Auge-Auflösung":
                lbl.config(text=f'{eye:.0f}"')
            elif name == "Vergrößerung":
                lbl.config(text=f"{V_slider:.0f}×")

        r  = r_real
        rv = berechne_vergr(D, f, lam, V, eye)
        Vk = v_kritisch(D, f, lam, eye)

        # Rechenschritte
   #     lines = [
   #         f"Realer Spiegel: D={D:.0f}mm  f={f:.0f}mm  f/{r['N']:.2f}",
   #         f"── Kontrast ──────────────────────────────────────────",
   #         f"W_PtV(parax.)  = {r['Wp']:.4f} λ",
   #         f"W_PtV(best f.) = {r['Wb']:.4f} λ",
   #         f"W_RMS          = {r['Wrms']:.5f} λ",
   #         f"Strehl (real)  = {S_real:.4f}",
   #         f"D_eff(Kont.)   = {r['Deff_k']:.1f}mm  (-{r['loss_k']:.1f}mm)",
   #         f"── Schärfe ───────────────────────────────────────────",
   #         f"Blur(best f.)  = {r['blur_as']:.4f}arcsec  ({r['blur_airy']:.4f} Airy)",
   #         f"θ_eff          = {r['theta_eff']:.4f}arcsec",
   #         f"D_eff(Schärfe) = {r['Deff_s']:.1f}mm  (-{r['loss_s']:.1f}mm)",
   #         f"── Verbesserter Spiegel (Strehl={S_slide:.3f}) ────────────",
   #         f"Blur (skaliert)= {blur_slide:.4f} arcsec  (x{blur_scale:.3f})",
   #         f"D_eff(Kont.)   = {Deff_k_slide:.1f}mm",
   #         f"D_eff(Schärfe) = {Deff_s_slide:.1f}mm",
   #         f"V_krit Sphäre  = {Vk_slide['Vk_sph']:.0f}x  "
   #         f"(real: {Vk['Vk_sph']:.0f}x  Para: {Vk['Vk_para']:.0f}x)",
   #     ]
   #     self.txt_steps.config(state="normal")
   #     self.txt_steps.delete("1.0", "end")
   #     self.txt_steps.insert("end", "\n".join(lines))
   #     self.txt_steps.config(state="disabled")

        rv2 = self._res
        rv2["rN"].config(     text=f"f/{r['N']:.2f}")
        rv2["rWp"].config(    text=f"{r['Wp']:.4f} λ")
        rv2["rWb"].config(    text=f"{r['Wb']:.4f} λ")
        rv2["rWrms"].config(  text=f"{r['Wrms']:.5f} λ")
        rv2["rS"].config(     text=f"{S_real:.4f}  →  {S_slide:.4f}")
        rv2["rDeff_k"].config(text=f"{r['Deff_k']:.1f} → {Deff_k_slide:.1f} mm")
        rv2["rQ02"].config(    text=f"{r['Q02']:.4f}")
        rv2["rQ04"].config(    text=f"{r['Q04']:.4f}")
        rv2["rQ06"].config(    text=f"{r['Q06']:.4f}")
        rv2["rQshape"].config(text=f"{r['Qshape']:.4f}")
        rv2["rQvis"].config(   text=f"{r['Qvis']:.4f}",
                               fg="#2e7d32" if r["Qvis"]>=0.95 else
                               "#e65100" if r["Qvis"]>=0.80 else "#c62828")
        rv2["rDeff_vis"].config(text=f"{r['Deff_vis']:.1f} mm")
        rv2["rVkrit2"].config(text=f"{Vk['Vk_sph']:.0f}× → {Vk_slide['Vk_sph']:.0f}×",
                              fg="#534AB7")
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
        # ── Wahrgenommener Qualitätsindex (CSF-gewichtet) ────────────────────
        Qp_80  = perceived_quality(D, f, lam, 30.0)
        Qp_200 = perceived_quality(D, f, lam, 80.0)
        Qp_350 = perceived_quality(D, f, lam, 160.0)
        Qp_V   = perceived_quality(D, f, lam, V_slider)

        def _qcolor(q):
            return "#2e7d32" if q >= 0.90 else "#e65100" if q >= 0.70 else "#c62828"

        rv2["rQp80"].config( text=f"{Qp_80:.3f}",  fg=_qcolor(Qp_80))
        rv2["rQp200"].config(text=f"{Qp_200:.3f}", fg=_qcolor(Qp_200))
        rv2["rQp350"].config(text=f"{Qp_350:.3f}", fg=_qcolor(Qp_350))
        rv2["rQpV"].config(  text=f"{Qp_V:.3f}  (bei {V_slider:.0f}×)",
                            fg=_qcolor(Qp_V))

        self._diagramm_strehl(D, f, lam, r["N"], S_real, S_slide)
        self._diagramm_oeffnung(D, f, lam, r["N"], r["Deff_k"],
                                S_sph=S_real, Dk_slide=Deff_k_slide)
        self._diagramm_deff_D(D, r["N"], S_slide)
        self._diagramm_beugung(D, f)
        self._diagramm_mtf(D, f, lam, S_slide, V=V_slider)
        self._diagramm_wahrnehmung(D, f, lam, V_slider, S_slide=S_slide, S_real=S_real)


    # ── Diagramme ─────────────────────────────────────────────────────────────

    def _ax_fmt(self, ax):
        ax.grid(True, color="#ddd", lw=0.5)

    def _diagramm_strehl(self, D, f, lam, N_akt, S_akt, S_sph=None):
        fig, ax, canvas = self._tab_strehl
        ax.cla()
        ns, sts, _, _ = kurven_N(D, lam)
        # Schraffur: Bereich zwischen Paraboloid (S=1) und gewähltem Strehl
        ax.fill_between(ns, S_sph if S_sph else S_akt, 1.0,
                        color=ACC, alpha=0.10, label="Bereich Paraboloid→Sphäre")
        ax.axhline(1.00, color=GRN,      lw=1.5, ls="-",  label="Paraboloid (S=1.0)")
        ax.plot(ns, sts, color=ACC, lw=2, label="Strehl (sphärisch)")
        ax.axhline(0.80, color="#BA7517", lw=1.2, ls="--", label="Rayleigh S=0.80")
        ax.axhline(0.95, color="#888",    lw=1.0, ls=":",  label="S=0.95")
        ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
        ax.scatter([N_akt], [S_akt], color=ACC, s=60, zorder=5)
        ax.annotate(f"f/{N_akt:.1f}  S={S_akt:.3f}",
                    xy=(N_akt, S_akt), xytext=(8, 10),
                    textcoords="offset points", fontsize=9, color="#333",
                    arrowprops=dict(arrowstyle="-", color="#aaa"))
        if S_sph and S_sph > S_akt:
            ax.axhline(S_sph, color=COR, lw=1.5, ls="-.",
                       label=f"Verbessert S={S_sph:.3f}")
            ax.annotate(f"S={S_sph:.3f}", xy=(14.5, S_sph),
                        fontsize=9, color=COR, va="bottom")
        ax.set_xlim(3, 15); ax.set_ylim(0, 1.08)
        ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=10)
        ax.set_ylabel("Strehl-Quotient", fontsize=10)
        ax.set_title(f"Strehl vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=10)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=9, loc="lower right"); self._ax_fmt(ax)
       # fig.tight_layout(); canvas.draw()
        fig.tight_layout(); canvas.draw_idle()

    def _diagramm_oeffnung(self, D, f, lam, N_akt, Dk_akt, Ds_akt=None, S_sph=None, Dk_slide=None, Ds_slide=None):
        fig, ax, canvas = self._tab_oeff
        ax.cla()
        for ax2_old in fig.get_axes()[1:]:
            fig.delaxes(ax2_old)

        ns, _, deff_ks, aufl_verluste = kurven_N(D, lam)
        r_akt = berechne(D, f, lam)
        avp_akt = r_akt["aufl_verlust_pct"]

        # Linke Achse: Deff_k
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
        ax.set_xlabel("Öffnungsverhältnis f/D", fontsize=10)
        ax.set_ylabel("Effektive Öffnung Kontrast [mm]", fontsize=10, color=ACC)
        ax.tick_params(axis="y", labelcolor=ACC)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        # Rechte Achse: Auflösungsverlust %
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
        ax2.set_ylabel("Auflösungsverlust [%]", fontsize=10, color=COR)
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis="y", labelcolor=COR)
        ax2.spines["right"].set_color(COR)

        # Gemeinsame Legende
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="center right")

        ax.set_title(f"Eff. Öffnung & Aufl.verlust vs. f/D  (D={D:.0f}mm)",
                     fontsize=10)
        self._ax_fmt(ax)
        fig.tight_layout()
        canvas.draw_idle()

    def _diagramm_deff_D(self, D_akt, N_akt, S_slide=None):
        fig, ax, canvas = self._tab_deff_D
        ax.cla()

        lam_mm = 550e-6
        D_arr  = np.linspace(50, 400, 300)
        N_list = [5, 6, 7, 8, 10]
        colors_N = {5:"#c62828", 6:"#D85A30", 7:"#c8a020", 8:"#2e7d32", 10:"#1565c0"}

        for N in N_list:
            loss_pct = []
            for D in D_arr:
                f   = D * N
                Wp  = D**4 / (1024.0 * f**3 * 550e-6)
                Wb  = Wp / 4
                Wrms = Wb / (1.5 * math.sqrt(5))
                S   = math.exp(-(2 * math.pi * Wrms)**2)
                loss_pct.append((1 - S**0.25) * 100)
            col = colors_N[N]
            ax.plot(D_arr, loss_pct, color=col, lw=2,
                    label=f"f/{N}")
            # Label am Ende der Kurve
            ax.text(D_arr[-1] + 3, loss_pct[-1], f"f/{N}",
                    color=col, fontsize=9, va="center", fontweight="bold")
        # Q_vis Kurven (gestrichelt, als Vergleich)
        for N in N_list:
            loss_qvis = []
            for D in D_arr:
                f   = D * N
                Wp  = D**4 / (1024.0 * f**3 * 550e-6)
                Wb  = Wp / 4
                Wrms = Wb / (1.5 * math.sqrt(5))
                S   = math.exp(-(2 * math.pi * Wrms)**2)
                Q02 = mtf_sph_rel(0.2, Wb)
                Q04 = mtf_sph_rel(0.4, Wb)
                Q06 = mtf_sph_rel(0.6, Wb)
                Qv  = 0.1*Q02 + 0.1*Q04 + 0.8*Q06
                Qvis_abs = S * Qv
                loss_qvis.append((1 - Qvis_abs**0.25) * 100)
            col = colors_N[N]
            ax.plot(D_arr, loss_qvis, color=col, lw=1.2, ls="--", alpha=0.6)


        # Aktueller Spiegel markieren
        f_akt    = D_akt * N_akt
        Wp_akt   = D_akt**4 / (1024.0 * f_akt**3 * 550e-6)
        Wb_akt   = Wp_akt / 4
        Wrms_akt = Wb_akt / (1.5 * math.sqrt(5))
        S_akt    = math.exp(-(2 * math.pi * Wrms_akt)**2)
        loss_akt = (1 - S_akt**0.25) * 100
        ax.scatter([D_akt], [loss_akt], color=ACC, s=80, zorder=6)
        ax.annotate(f"D={D_akt:.0f}mm f/{N_akt:.1f}  -{loss_akt:.0f}% Öffnung",
                    xy=(D_akt, loss_akt),
                    xytext=(D_akt + 15, loss_akt + 5),
                    fontsize=9, color=ACC, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=ACC, alpha=0.85))

        # Verbesserter Spiegel (Strehl-Schieber)
        if S_slide is not None and S_slide > S_akt + 0.01:
            loss_slide = (1 - S_slide**0.25) * 100
            ax.scatter([D_akt], [loss_slide], color=GRN, s=80, zorder=7,
                       marker="*")
            ax.annotate(f"Verbessert S={S_slide:.2f}  -{loss_slide:.0f}%",
                        xy=(D_akt, loss_slide),
                        xytext=(D_akt + 15, loss_slide - 6),
                        fontsize=9, color=GRN, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=GRN, lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=GRN, alpha=0.85))
            # Verbindungslinie zwischen real und verbessert
            ax.annotate("", xy=(D_akt, loss_slide), xytext=(D_akt, loss_akt),
                        arrowprops=dict(arrowstyle="->", color=GRN, lw=1.5))

        # Referenzlinien
        ax.axhline(20, color="#BA7517", lw=1.5, ls="--",
                   label="20% Verlust (spürbar)")
        ax.axhline(50, color="#c62828", lw=1.0, ls=":",
                   label="50% Verlust (erheblich)")
        ax.text(52, 21, "20% Verlust", fontsize=8,
                color="#BA7517", fontweight="bold")
        ax.text(52, 51, "50% Verlust", fontsize=8,
                color="#c62828", fontweight="bold")

        # Rayleigh-Grenze (S=0.8 -> sqrt-Verlust = 1-sqrt(0.8) = 10.6%)
        ax.axhline((1 - 0.8**0.25)*100, color="#888", lw=1.0, ls=":",
                   label=f"Rayleigh S=0.80 ({(1-0.8**0.25)*100:.0f}% Verlust)")
        ax.text(52, (1-0.8**0.25)*100 + 1, "Rayleigh S=0.80",
                fontsize=8, color="#888")

        ax.set_xlim(50, 420)
        ax.set_ylim(0, min(100, max(90, loss_akt + 15)))
        ax.set_xlabel("Öffnung D [mm]", fontsize=10)
        ax.set_ylabel("Öffnungsverlust [%]", fontsize=10)
        # Legende-Proxy für die zwei Kurventypen
        from matplotlib.lines import Line2D
        proxy = [
            Line2D([0],[0], color="#555", lw=2,      label="D × S^0.25  (Kontrast)"),
            Line2D([0],[0], color="#555", lw=1.2, ls="--", label="D × Q_vis  (visuell, MTF-gewichtet)"),
        ]
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title("Öffnungsverlust vs. Öffnung — "
            "durchgezogen: D×Strehl^0.25  |  "
            "gestrichelt: D×Q_vis^0.25",
            fontsize=9
        )
        self._ax_fmt(ax)
        fig.tight_layout()
        #canvas.draw()
        canvas.draw_idle()


    def _diagramm_beugung(self, D_akt, f_akt):
        fig, ax, canvas = self._tab_beugung
        ax.cla()

        lam_mm = 550e-6
        f_arr  = np.linspace(200, 2000, 500)

        # Grenzlinien berechnen
        def D_grenz(S):
            Wrms = math.sqrt(-math.log(S)) / (2*math.pi)
            Wp   = Wrms * 1.5 * math.sqrt(5) * 4
            # Wp = D^4/(1024*f^3*lam) => D = (Wp*1024*lam)^0.25 * f^0.75
            return (Wp * 1024 * lam_mm) ** 0.25 * f_arr ** 0.75
        D_95 = D_grenz(0.95)
        D_80 = D_grenz(0.80)
        D_50 = D_grenz(0.50)

        # Flächen
        ax.fill_between(f_arr, 0,    D_95, color="#2e7d32", alpha=0.10)
        ax.fill_between(f_arr, D_95, D_80, color="#f9a825", alpha=0.12)
        ax.fill_between(f_arr, D_80, 300,  color="#c62828", alpha=0.07)

        # Zonentext innerhalb des Plots
        ax.text(300,  40,  "S >= 0.95  (sehr gut)",    color="#2e7d32", fontsize=8, alpha=0.9)
        ax.text(300, 155,  "0.80 <= S < 0.95  (gut)",  color="#c8860a", fontsize=8, alpha=0.9)
        ax.text(300, 255,  "S < 0.80  (nicht beugungsbegrenzt)", color="#c62828", fontsize=8, alpha=0.9)

        # Grenzlinien — Label als Legende, nicht außerhalb
        ax.plot(f_arr, D_95, color="#1565c0", lw=2,   label="S=0.95")
        ax.plot(f_arr, D_80, color="#2e7d32", lw=2.5, label="S=0.80  (Rayleigh)")
        ax.plot(f_arr, D_50, color="#BA7517", lw=1.5, ls="--", label="S=0.50")

        # f/D-Hilfslinien mit Label innerhalb
        for N in [5, 6, 7, 8, 10]:
            D_fN = np.minimum(f_arr / N, 300)
            ax.plot(f_arr, D_fN, color="#ccc", lw=0.8, ls="--")
            # Label bei f=1900 wenn D<290
            y_label = 1900.0 / N
            if y_label < 285:
                ax.text(1920, y_label, f"f/{N}",
                        color="#999", fontsize=8, va="center")

        # Aktueller Spiegel
        ax.scatter([f_akt], [D_akt], color=ACC, s=100, zorder=7)
        # Annotation: nach links wenn f > 1500, sonst nach rechts
        x_off = -200 if f_akt > 1500 else 80
        ax.annotate(f"D={D_akt:.0f}mm  f/{f_akt/D_akt:.1f}",
                    xy=(f_akt, D_akt),
                    xytext=(f_akt + x_off, D_akt + 25),
                    fontsize=9, color=ACC, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=ACC, alpha=0.85))

        ax.set_xlim(200, 2000)
        ax.set_ylim(0, 300)
        ax.set_xlabel("Brennweite f [mm]", fontsize=10)
        ax.set_ylabel("Öffnung D [mm]", fontsize=10)
        ax.set_title("Beugungsgrenze sphärischer Spiegel  (λ=550nm)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        self._ax_fmt(ax)
        fig.tight_layout()
        #canvas.draw()
        canvas.draw_idle()


    def _diagramm_mtf(self, D, f, lam, S_slide=None, V=150.0):
        fig, ax, canvas = self._tab_mtf
        ax.cla()

        # Zweite Y-Achse aufräumen falls vorhanden
        for ax2_old in fig.get_axes()[1:]:
            fig.delaxes(ax2_old)

        modus = self._mtf_modus.get()  # "absolut" oder "relativ"

        freqs_as, fc, mp_arr, msa, msr_arr, strehl = mtf_kurven(D, f, lam)

        objekte = self._mtf_objekte.get()
        all_details = [
            # Planeten
            ("Jupiter\nGürtel",       5.0,  "#4a90d9", "planeten"),
            ("Jupiter\nFestons",      1.5,  "#4a90d9", "planeten"),
            ("Mars\nPol",             1.0,  "#d94a4a", "planeten"),
            ("Saturn\nCassini",       0.5,  "#c8a020", "planeten"),
            # Deep-Sky M42
            ("Trapezium\nAB (~9\")",  9.0,  "#8e44ad", "deepsky"),
            ("M42\nFilamente",        3.0,  "#8e44ad", "deepsky"),
            ("Trapezium\nCD (~2\")",  2.0,  "#8e44ad", "deepsky"),
            # Deep-Sky M13
            ("M13\nHalo-Sterne",      2.5,  "#c0392b", "deepsky"),
            ("M13\nAußensterne",      1.5,  "#c0392b", "deepsky"),
            ("M13\nKern",             0.7,  "#c0392b", "deepsky"),
            # Immer
            ("Dawes\nGrenze",         116.0/D, "#888", "beide"),
        ]
        planet_details = [(l,d,c) for l,d,c,k in all_details
                          if k == objekte or k == "beide"
                          or objekte == "alle"]

        def mtf_at(fn, msr_curve):
            idx = min(int(round(fn * (len(msr_curve)-1))), len(msr_curve)-1)
            return msr_curve[idx]

        labels, mp_vals, ms_vals, verlust_vals, bar_colors = [], [], [], [], []
        detail_sizes_as = []
        for label, d_as, col in planet_details:
            fn = 1.0 / (2.0 * d_as * fc)
            if fn >= 1.0: continue
            mp_v  = mtf_para(fn)
            msr_v = mtf_at(fn, msr_arr)
            strehl_eff = 1.0 - (1.0 - strehl) * fn   # fn = nu_v/nu_c
            ms_v  = msr_v * mp_v * strehl_eff
            labels.append(label)
            mp_vals.append(mp_v)
            ms_vals.append(ms_v)
            verlust_vals.append((1 - msr_v) * 100)
            bar_colors.append(col)
            detail_sizes_as.append(d_as)

        n = len(labels)
        x = np.arange(n)

        # Verbesserten Spiegel berechnen
        ms_s = None
        verlust_s = None
        if S_slide is not None and S_slide > strehl + 0.01:
            f_slide = strehl_to_f(S_slide, D, lam)
            _, _, _, _, msr_s_arr, _ = mtf_kurven(D, f_slide, lam)
            ms_s = [mtf_at(1/(2*d*fc), msr_s_arr) * mtf_para(1/(2*d*fc))
                    for _,d,_ in planet_details if 1/(2*d*fc) < 1]
            verlust_s = [(1 - mtf_at(1/(2*d*fc), msr_s_arr)) * 100
                         for _,d,_ in planet_details if 1/(2*d*fc) < 1]

        if modus == "absolut":
            w = 0.28 if ms_s else 0.35
            ax.bar(x - w/2, mp_vals, w, label="Paraboloid", color="#888", alpha=0.75)
            ax.bar(x + w/2, ms_vals, w, label=f"Sphäre (S={strehl:.3f})",
                   color=COR, alpha=0.85)
            if ms_s:
                ax.bar(x + w/2, ms_s, w, label=f"Verbessert (S={S_slide:.2f})",
                       color=GRN, alpha=0.75)
            for i, (mp_v, ms_v) in enumerate(zip(mp_vals, ms_vals)):
                ax.text(i-w/2, mp_v+0.02, f"{mp_v:.2f}", ha="center", va="bottom",
                        fontsize=8, color="#444")
                ax.text(i+w/2, ms_v+0.02, f"{ms_v:.2f}", ha="center", va="bottom",
                        fontsize=8, color=COR, fontweight="bold")
            ax.axhline(0.2, color="#BA7517", lw=2.0, ls="-", zorder=5)
            ax.text(0.01, 0.215, "20%-Schwelle", transform=ax.get_yaxis_transform(),
                    fontsize=9, fontweight="bold", color="#BA7517")
            ax.set_ylim(0, 1.18)
            ax.set_ylabel("Kontrastübertragung (MTF)", fontsize=10)
            ax.set_title(f"Absolute MTF je Detail — D={D:.0f}mm f/{f/D:.1f}  Strehl={strehl:.3f}",
                         fontsize=10)
            ax.text(0.02, 0.97, f"Strehl = {strehl:.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9, color=COR,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COR, alpha=0.8))

        else:  # relativ
            w = 0.4 if not ms_s else 0.3
            bars = ax.bar(x, verlust_vals, width=w, color=bar_colors, alpha=0.85,
                          label=f"Sphäre (S={strehl:.3f})")
            if verlust_s:
                ax.bar(x + w, verlust_s, width=w, color=GRN, alpha=0.75,
                       label=f"Verbessert (S={S_slide:.2f})")
                for i, v in enumerate(verlust_s):
                    ax.text(i+w, v+0.3, f"{v:.1f}%", ha="center", va="bottom",
                            fontsize=9, color=GRN, fontweight="bold")
            for i, v in enumerate(verlust_vals):
                ax.text(i, v+0.3, f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color=bar_colors[i])
            ymax = max(verlust_vals) * 1.35 + 3 if verlust_vals else 30
            ax.axhline(20, color="#BA7517", lw=2.0, ls="-", zorder=5)
            ax.text(0.01, 21.5, "20%-Schwelle", transform=ax.get_yaxis_transform(),
                    fontsize=9, fontweight="bold", color="#BA7517")
            ax.set_ylim(0, ymax)
            ax.set_ylabel("Kontrastverlust gg. Paraboloid [%]", fontsize=10)
            ax.set_title(f"Relativer Kontrastverlust — D={D:.0f}mm f/{f/D:.1f}  Strehl={strehl:.3f}",
                         fontsize=10)

        # ── CSF-Overlay: Augenempfindlichkeit bei Vergrößerung V ─────────────
        if detail_sizes_as:
            csf_vals = []
            for d_as in detail_sizes_as:
                # Detailgröße im Bild: d_as [arcsec] → Winkel für das Auge
                # Bei Vergrößerung V erscheint 1 arcsec als V arcsec = V/3600 Grad
                # Frequenz für das Auge: f_cpd = 1 / (2 * d_as * V/3600) [cpd]
                f_cpd_eye = 3600.0 / (2.0 * d_as * V)
                csf_vals.append(csf(f_cpd_eye))

            # CSF auf zweiter Y-Achse (0..1 rechts)
            ax2 = ax.twinx()
            ax2.set_facecolor("none")
            csf_color = "#1a7abf"
            ax2.plot(x, csf_vals, color=csf_color, lw=2.2, ls="-.",
                     marker="o", ms=6, zorder=8,
                     label=f"Augenempfindlichkeit CSF  ({V:.0f}×)")
            for i, cv in enumerate(csf_vals):
                ax2.text(i, cv + 0.03, f"{cv:.2f}",
                         ha="center", va="bottom", fontsize=8,
                         color=csf_color, fontweight="bold")
            ax2.set_ylim(0, 1.35)
            ax2.set_ylabel("Augenempfindlichkeit CSF  (0–1)", fontsize=9,
                           color=csf_color)
            ax2.tick_params(axis="y", labelcolor=csf_color)
            ax2.spines["right"].set_color(csf_color)
            ax2.legend(fontsize=9, loc="upper left",
                       bbox_to_anchor=(0.0, 0.88))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, linespacing=1.3)
        ax.legend(fontsize=9, loc="upper right")
        self._ax_fmt(ax)
        fig.tight_layout()
        canvas.draw_idle()



    def _diagramm_wahrnehmung(self, D, f, lam, V_slider,
                               S_slide=None, S_real=None):
        """
        Wahrnehmungs-Tab: CSF-gewichteter Qualitätsindex Q_perc vs. Vergrößerung.
        """
        fig, ax, canvas = self._tab_wahrnehmung
        ax.cla()

        V_arr = np.linspace(30, 250, 120)

        # Sphärischer Spiegel (theoretisch aus D/f)
        Qp_sph = perceived_quality_kurven(D, f, lam, V_arr, use_csf=True)

        # Paraboloid-Referenz
        f_para = D * 50.0
        Qp_para = perceived_quality_kurven(D, f_para, lam, V_arr, use_csf=True)

        # Strehl nach Schieber: eigene Kurve wenn abweichend
        strehl_label = f"Sphäre  f/{f/D:.1f}  S={S_real:.3f}" if S_real else f"Sphäre  f/{f/D:.1f}"
        has_slide = S_slide is not None and S_real is not None and S_slide > S_real + 0.01

        if has_slide:
            f_slide = strehl_to_f(S_slide, D, lam)
            Qp_slide = perceived_quality_kurven(D, f_slide, lam, V_arr, use_csf=True)
            ax.fill_between(V_arr, Qp_sph, Qp_para,
                            color=COR, alpha=0.10, label="Wahrnehmungsverlust (Sphäre)")
            ax.fill_between(V_arr, Qp_slide, Qp_para,
                            color=GRN, alpha=0.10, label="Wahrnehmungsverlust (Schieber)")
        else:
            ax.fill_between(V_arr, Qp_sph, Qp_para,
                            color=COR, alpha=0.13, label="Wahrnehmungsverlust")

        ax.plot(V_arr, Qp_para, color=GRN,  lw=2.0, ls="--",
                label="Paraboloid (Referenz)")
        ax.plot(V_arr, Qp_sph,  color=COR,  lw=2.0, ls=":",
                label=strehl_label)

        if has_slide:
            ax.plot(V_arr, Qp_slide, color=ACC, lw=2.5,
                    label=f"Strehl-Schieber  S={S_slide:.3f}")

        # Feste Markierungen — auf der Schieber-Kurve wenn vorhanden, sonst Sphäre
        q_curve = Qp_slide if has_slide else Qp_sph
        for V_mark, col, ls in [(30, "#c8a020", ":"), (80, "#888", "-."), (160, "#c62828", ":")]:
            idx = int(round((V_mark - 30) / (250 - 30) * (len(V_arr) - 1)))
            q_m = float(q_curve[idx])
            ax.axvline(V_mark, color=col, lw=1.2, ls=ls, alpha=0.6)
            ax.scatter([V_mark], [q_m], color=col, s=50, zorder=6)
            ax.annotate(f"{V_mark}×\n{q_m:.2f}",
                        xy=(V_mark, q_m),
                        xytext=(V_mark + 7, q_m - 0.05),
                        fontsize=8, color=col, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=col, alpha=0.80))

        # Slider-Vergrößerung
        q_v = perceived_quality(D, f_slide if has_slide else f, lam, V_slider, use_csf=True)
        ax.axvline(V_slider, color=ACC, lw=1.8, ls="-", alpha=0.9)
        ax.scatter([V_slider], [q_v], color=ACC, s=90, zorder=7)
        x_off = -65 if V_slider > 200 else 12
        ax.annotate(f"V={V_slider:.0f}×\nQ={q_v:.3f}",
                    xy=(V_slider, q_v),
                    xytext=(V_slider + x_off, q_v + 0.04),
                    fontsize=9, color=ACC, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=ACC, alpha=0.88))

        # Schwelllinien
        for q_thresh, col, label in [(0.9, GRN, "Q=0.90 (sehr gut)"),
                                      (0.7, "#BA7517", "Q=0.70 (spürbar)")]:
            ax.axhline(q_thresh, color=col, lw=1.0, ls=":", alpha=0.7)
            ax.text(32, q_thresh + 0.008, label,
                    fontsize=8, color=col, alpha=0.85)

        ax.set_xlim(30, 250)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("Vergrößerung V  [×]", fontsize=10)
        ax.set_ylabel("Wahrgenommener Qualitätsindex Q_perc", fontsize=10)
        ax.set_title(
            f"Wahrgenommener Kontrast (CSF-gewichtet)  —  D={D:.0f}mm  f/{f/D:.1f}",
            fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        self._ax_fmt(ax)
        fig.tight_layout()
        canvas.draw_idle()


if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        cli()
    else:
        app = App()
        app.mainloop()
