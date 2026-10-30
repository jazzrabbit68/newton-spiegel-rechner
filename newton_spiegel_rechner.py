"""
Newton-Teleskop: Kontrast- und Schärfeverlust sphärischer Hauptspiegel  v7.0.0-beta (2026-10-29)
============================================================================================
Änderungshistorie (jüngste zuerst):
  v7.0.0-beta (2026-10-29) Fangspiegel-Tab: Blende gegenüber dem OAZ (Gittertubus/Hutring) aus Hutring-Ø, Okular-f, AFOV und OAZ-Rohr (Innen-Ø, Länge) als Blendenrohr, Blende um den Hutring gebogen, Skizze (Draufsicht) mit Maßen und Strahlengang.
  v7.0.0-beta (2026-10-29) Vergleich: Augenpupille (Abblendung bei Austrittspupille > Augenpupille, mit Refokus),
                        Option „Spiegel sphärisch (aus D, f/D)“; Fehler behoben: Q_abs-Integrationsgitter war bei
                        großem D / kleinem V zu grob (nicht-monotones Q_abs(D))
  v7.0.0-beta (2026-10-28) Kern: mtf_sph_rel jetzt exakt (2D-Pupillen-Autokorrelation per FFT statt 1D-Näherung,
                        alte Funktion als mtf_sph_rel_1d erhalten) → alle Q_vis-/MTF-Tabs; Vergleich: Planetenbild
                        mit Augen-Filter bei der gewählten Vergrößerung (schaltbar)
  v7.0.0-beta (2026-10-28) Vergleich: Spiegel 1/2 wahlweise aus der Foucault-Wellenfront (MTF aus 2D-Pupillen-
                        Autokorrelation, mit Fangspiegel); S-Modell jetzt über exakte FFT-MTF statt 1D-Näherung.
                        Fehler behoben: _Wb_von_S war bei S<0,49 abgeschnitten (Suchbereich 0,44 → 0,90)
  v7.0.0-beta (2026-10-26) Neuer Tab "Vergleich (Beta)": absolutes Q_abs (fester Referenzspiegel), Seeing-Schalter,
                        D×S-Karte, Break-even-Strehl; Fangspiegel automatisch aus D und f/D
  v6.12.0 (2026-10-25)  Neuer Tab "Fangspiegel": Mindestgröße aus Okularauszug/Tubus/Bildfeld,
                        Obstruktion, Q_vis-Verlust
  v6.11.0 (2026-10-25)  Wahrnehmung: Fangspiegel-Ø-Feld + Zusatzkurve "Schieber + Fangspiegel";
                        Vergrößerungs- und Fangspiegelfeld nur ganzzahlig
  v6.10.1 (2026-10-25)  Fehler behoben: mtf_para_obstr (Obstruktions-MTF, Kreuzterm fehlte)
  v6.10.0 (2026-10-25)  Wahrnehmung: Ablese-Cursor (V) mit exakten Q_vis-Werten
  v6.9.1  (2026-10-24)  Foucault: Abtrag-Vorzeichen korrigiert, Werkstatt-Ansicht wie FigureXP;
                        Web-Version auf Desktop-Stand gebracht (inkl. Max. Vergrößerung,
                        Vmax vs. Ø/Seeing)

Berücksichtigt:
  - Wellenfrontfehler und Strehl-Quotient (Kontrast)
  - Geometrischen Unschärfefleck am besten Fokus (Schärfe)
  - Vergrößerungsabhängigen Schärfeverlust durch das Auge

Formeln:
  Kontrast (Strehl):
    W_PtV (paraxial)  = D⁴ / (512·f³·λ)          [Sacek §4.7.1, korrigiert 07/2026:
                                                      Reflexion verdoppelt Oberflaechen-
                                                      in Wellenfrontfehler]
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
    V_krit            = D[mm] / (13·Wb)            [empirisch gefittet, s. Chat-
                                                       Historie 08/2026: ersetzt alte,
                                                       nachweislich falsche Formel
                                                       D·0.7·√S — Wb monoton, S bei
                                                       starker Aberration nicht mehr]
    V_wendepunkt_naeh = V_krit / ³√(S²)             [nur für Sigmoid-Näherungskurve
                                                       Q̃(V), nicht V_krit selbst]

  MTF:
    Paraboloid:  analytisch (Kreisapertur)          [Sacek §4.1]
    Sphäre rel.: OTF-Faltungsintegral, numerisch    [Sacek §4.7]
      Wellenfrontform W(ρ) = Wp·(ρ⁴−ρ²)  [best focus]
                   oder Wp·ρ⁴             [paraxialer Fokus]
      Amplitude: Wp = 4·Wb
        (Wb = PtV am besten Fokus = Wp_paraxial/4
         → W(ρ) = Wp·(ρ⁴−ρ²) = 4·Wb·(ρ⁴−ρ²); konsistent mit Sacek §4.7)

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

try:
    from scipy.optimize import minimize_scalar as _minimize_scalar_scipy
    _SCIPY_OK = True
except ImportError:
    _minimize_scalar_scipy = None
    _SCIPY_OK = False


def _golden_section(f, a: float, b: float, tol: float = 1e-8) -> float:
    """Golden-Section-Suche fuer Minimum von f auf [a, b].
    Ersetzt scipy minimize_scalar -- keine scipy-Abhaengigkeit noetig.
    Konvergiert in <=200 Iterationen auf tol=1e-8.
    """
    import math as _math
    phi = (_math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(200):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c  = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d  = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


# ── Kernrechnung ──────────────────────────────────────────────────────────────

def _wellenfronten(D_mm: float, f_mm: float, lam_nm: float):
    """Gemeinsame Basis: Wp, Wb, Wrms, Strehl (Maréchal).

    Wp  = D⁴/(512·f³·λ)   — PtV-Wellenfrontfehler am paraxialen Fokus [Sacek §4.7.1]
                             (KORRIGIERT 07/2026: D^4/(1024 f^3) ist die reine
                             Oberflaechen-Sagittadifferenz Kugel-Parabel; die
                             Wellenfront ist bei Reflexion doppelt so gross,
                             s. Sacek's aberration-coefficient-Herleitung /
                             Zahlenbeispiel in lower_order_spherical.htm)
    Wb  = Wp/4             — PtV am besten Fokus [Sacek §4.7.2]
    Wrms= Wb/(1.5·√5)      — RMS für reine sphärische Aberration [Sacek §4.7.2]
    S   = exp(-(2π·Wrms)²) — Maréchal-Näherung [Maréchal 1947, zit. n. Sacek §4.8], gültig für S > ~0.6
    """
    lam_mm = lam_nm * 1e-6
    Wp   = D_mm**4 / (512.0 * f_mm**3 * lam_mm)   # 07/2026: 1024->512, s. Docstring oben
    Wb   = Wp / 4.0
    Wrms = Wb / (1.5 * math.sqrt(5))
    S    = math.exp(-(2 * math.pi * Wrms) ** 2)
    return lam_mm, Wp, Wb, Wrms, S


@lru_cache(maxsize=1024)
def _strehl_exakt(Wb: float, n: int = 2000) -> float:
    """Exakter Strehl via Pupillenintegral.

    S = |⟨exp(i·2π·W(ρ))⟩|²   mit  W(ρ) = 4·Wb·(ρ⁴−ρ²)
    Wb = PtV-Wellenfrontfehler am besten Fokus = Wp/4
    → W(ρ) = Wp·(ρ⁴−ρ²)  [konsistent mit Sacek §4.7, §4.8]
    Mittelwertsubtraktion: defokusfreie Auswertung am besten Fokus.
    Maréchal-Näherung versagt für Wb > 0.08λ (ca. f/N < 8).
    """
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    rho   = np.linspace(0.0, 1.0, n)
    W     = 4.0 * Wb * (rho**4 - rho**2)
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
    lo, hi = 0.0, 0.90   # S(Wb) ist bis ≈0,9 monoton fallend (S≈0,09); vorher 0,44 → S<0,49 wurde abgeschnitten
    for _ in range(n):
        mid = (lo + hi) / 2.0
        if _strehl_exakt(mid) > S:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def v_krit_naeh(D_mm: float, Wb: float) -> float:
    """Schnelle Näherungsformel für V_krit ohne MTF×CSF-Integration.

    V_krit ≈ D / (13 · Wb)

    KORRIGIERT 08/2026: ersetzt die alte Formel D·0,7·√S. Die behauptete
    Eigenschaft "= Wendepunkt von Q̃(V)" war nachweislich falsch (Faktor
    2-5 daneben, s. Chat-Historie) — u.a. weil die EXAKTE Strehl S für
    stark aberrierte (schnelle, sphärische) Spiegel ab Wb≈0,9λ nicht mehr
    monoton in Wb ist (Ringing des Pupillenintegrals bei großer Aberration),
    wodurch S als Eingabegröße für jede Formel strukturell ungeeignet ist.
    Auch die Maréchal-Näherung (glatt, aber nur für kleine Aberrationen
    gültig) taugt hier nicht — sie unterschätzt die Qualität bei den für
    dieses Tool zentralen schlechten Kugelspiegeln massiv (>1000% Fehler
    im Test).

    Stattdessen: direkter Potenzfit von V_krit gegen den ECHTEN, numerisch
    bestimmten 10%-Kontrastverlust-Punkt (s. v_krit_aus_qvis), mit Wb
    (Wellenfrontfehler, immer positiv und monoton mit der tatsächlichen
    Unschärfe) statt S als Eingabe. Getestet über D=80–400mm, f/4–f/10
    (61 gültige Datenpunkte nach Ausschluss von Randfällen, wo die
    10%-Schwelle im Suchbereich nicht erreicht wurde):
        mittlerer Fehler ≈ 12%, Median ≈ 7%, worst-case (kleine,
        bereits nahezu beugungsbegrenzte Öffnungen) bis ~85%.

    Geht für Wb→0 (perfekte Optik) korrekt gegen unendlich, statt bei
    einem willkürlichen endlichen Wert zu "deckeln" wie die alte Formel.
    """
    if Wb <= 1e-9:
        return 1e9
    return D_mm / (13.0 * Wb)


def v_wendepunkt_naeh(D_mm: float, Wb: float) -> float:
    """Separater 'Wendepunkt' NUR für die Sigmoid-Näherungskurve
    Q̃(V) = S + (1-S)·(1-w(V)), w(V)=1/(1+(Vk/V)²) — NICHT dasselbe wie
    V_krit (10%-Kontrastverlust-Schwelle, s. v_krit_naeh).

    Hintergrund (08/2026, s. Chat-Historie): mit V_krit selbst als Vk in
    der Sigmoid eingesetzt, trifft die Näherungskurve die exakte
    MTF×CSF-Kurve bei schlechten (schnellen) Kugelspiegeln nur grob
    (mittlerer Fehler bis ~0,27 bei f/6) — die Sigmoid-Form ist einfach zu
    schmal/steil für den lang auslaufenden echten Verlauf. Ein GRÖSSERES,
    strehlabhängiges Vk trifft die Kurve insgesamt besser (Fehler ~0,07-
    0,12), auch wenn's kein sauber herleitbarer Wert ist wie V_krit selbst.

    Faktor = 1 / ³√(S²) = S^(-2/3) — sauberer, fast identisch guter Ersatz
    für den ursprünglichen Rohfit (0,9074·S^-0,713, Koeffizient≈1 also
    einfach weggelassen). Deckel bei Faktor 20 gegen Extremfälle.

    NUR für die Darstellung der Näherungskurve verwenden — für alle
    anderen V_krit-Zwecke (Info-Panel, Vmax-Diagramme etc.) weiterhin
    v_krit_naeh() (die echte 10%-Schwelle) benutzen.
    """
    if Wb <= 1e-9:
        return 1e9
    S = _strehl_exakt(Wb)
    faktor = min(max(S, 1e-6) ** (-2.0 / 3.0), 20.0)
    return v_krit_naeh(D_mm, Wb) * faktor


def _kennzahlen_von_S(D_mm: float, S: float, Wb: float) -> dict:
    """Effektive Öffnungen und Vergrößerungsgrenzen aus S und Wb.

    D_eff_k = D · S^0.25  — äquivalente Öffnung bzgl. Kontrastübertragung
                             (eigene Näherung)
    V_krit  = D / (13·Wb) — Vergrößerung ab der Aberrationsblur sichtbar
                             wird (s. v_krit_naeh() für Herleitung/Genauigkeit)
    """
    sqrtS       = math.sqrt(max(S, 1e-9))
    Deff_k      = D_mm * S**0.25
    Deff_s      = D_mm * sqrtS
    theta_ideal = 116.0 / D_mm
    theta_eff   = theta_ideal / sqrtS
    V_krit_vis  = v_krit_naeh(D_mm, Wb)
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
    V_krit = v_krit_naeh(D_mm, _Wb_von_S(S))
    w      = 1.0 / (1.0 + (V_krit / V) ** 2)
    return float(Q_raw + (1.0 - Q_raw) * (1.0 - w))


# ── Öffentliche Rechenfunktionen ──────────────────────────────────────────────

# ── Geglättete Strehl-Kurve (nur für das Diagramm „D_eff vs. Öffnung“) ────────

_GLATT_TAB = None   # (ln Wb, ln S geglättet, ln S roh) — wird beim ersten Aufruf berechnet


def _lnS_geglaettet(Wb, roh: bool = False) -> np.ndarray:
    """ln(S) am besten Fokus, für Wb > ~0.4 λ über die Strehl-Oszillation geglättet.

    Der exakte Strehl (Pupillenintegral) ist für starke sphärische Aberration
    NICHT monoton: er pendelt zwischen ~0.03 und ~0.15 (Ringenergie wandert
    zwischen Kern und Halo). Das ist echte Physik, aber die Spitzenintensität
    sagt dort kaum noch etwas über die Bildqualität aus — und D·S^0.25 ist ohnehin
    eine Näherung für S > ~0.1. Daher: Gauß-Glättung in ln(Wb) (σ = 0.4), weich
    eingeblendet zwischen Wb = 0.4λ und 1.3λ. Darunter (S > ~0.5) bleibt die
    exakte Kurve unverändert.

    Alle Kurven f/N sind dieselbe Funktion S(Wb), denn Wb ∝ D/N³ — die Glättung
    wird daher nur einmal auf der universellen Kurve gemacht und per Interpolation
    auf beliebige (D, N) abgebildet.
    roh=True liefert stattdessen die ungeglättete, exakte Kurve (Vergleichslinie).
    """
    global _GLATT_TAB
    if _GLATT_TAB is None:
        lw = np.linspace(np.log(0.01), np.log(10.0), 1400)
        # __wrapped__: am lru_cache von _strehl_exakt vorbei (Cache nicht fluten)
        y = np.log(np.array([_strehl_exakt.__wrapped__(w) for w in np.exp(lw)]))
        sigma, W0, W1 = 0.4, 0.4, 1.3
        step = lw[1] - lw[0]
        h = int(4 * sigma / step)
        k = np.arange(-h, h + 1) * step
        ker = np.exp(-0.5 * (k / sigma) ** 2)
        ker /= ker.sum()
        ys = np.convolve(np.pad(y, h, mode="edge"), ker, mode="valid")
        t = np.clip((lw - np.log(W0)) / (np.log(W1) - np.log(W0)), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)                     # smoothstep
        _GLATT_TAB = (lw, (1.0 - t) * y + t * ys, y)
    lw, y_gl, y_roh = _GLATT_TAB
    return np.interp(np.log(np.maximum(Wb, 1e-6)), lw, y_roh if roh else y_gl)


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
    """Kritische Vergrößerung: V_krit = D_mm / (13·Wb)  (s. Chat-Historie 08/2026,
    ersetzt die alte, nachweislich falsche Formel D·0,7·√Strehl)."""
    r = berechne(D_mm, f_mm, lam_nm)
    return dict(Vk_sph=r["V_krit_vis"], Vk_para=1e9,
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
    Wb = _Wb_von_S(strehl)
    return dict(Vk_sph=v_krit_naeh(D_mm, Wb),
                Vk_para=v_krit_naeh(D_mm, 1e-9), Vmax=D_mm / 0.5, Vmin=D_mm / 7.0)


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
    """Exaktes V½ via MTF×CSF-Integral — Analogon zur Näherung V_krit/³√(S²).

    Gesucht: V wo Q_vis(V) = (1 + S_exakt) / 2
    Das ist der Halbwertspunkt zwischen Q=1 (V→0) und Q=S_exakt (V→∞),
    genau der Punkt, den die Sigmoid-Näherung Q̃(V) bei w(V)=0.5 (also
    V=V_wendepunkt_naeh) approximieren soll — hier aber exakt via
    Pupillenintegral statt der empirisch gefitteten Formel berechnet.

    Ermöglicht direkten Vergleich: V_wendepunkt_naeh (schnell, ~10-15%
    Fehler) vs. V½_exakt (hier, langsam aber exakt) — s. Chat-Historie
    08/2026.
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


# ═══════════════════════════════════════════════════════════════════════════
# "Sinnvolle Maximalvergrößerung" — Gegenprobe zum "2×Öffnung(mm)"-Mythos
# ═══════════════════════════════════════════════════════════════════════════
# Motivation (Forum-Thread 08/2026): die "2×D[mm]"-Faustregel behauptet eine
# einzige, aperturproportionale Grenze, unabhängig von Seeing und Objekt-
# helligkeit. Drei separate, unabhängig begründbare Grenzen statt einer
# einzelnen "V_optimal"-Zahl aus einem Integral (zwei Anläufe dazu sind an
# einem Jacobian-Artefakt bzw. trivialer Entartung gescheitert, s. Chat-
# Historie — das Objektspektrum selbst müsste mitmodelliert werden, was wir
# nicht seriös tun können ohne echte Objektdaten). Stattdessen: drei
# eigenständige Grenzen, jede für sich physikalisch klar begründet, analog
# zum bereits etablierten Muster von v_krit_blur_direkt() (rein geometrisch,
# keine Integration):
#
#   V_helligkeit_min — Austrittspupille = Augenpupille (ab hier kein
#                       Lichtverlust mehr, aber auch kein Gewinn durch noch
#                       groessere AP)
#   V_seeing_max      — Seeing-Unschaerfe (vergroessert) erreicht die
#                       Augenaufloesung (analog v_krit_blur_direkt, nur mit
#                       Seeing statt Spiegel-Blur)
#   V_helligkeit_max  — Netzhaut-Flaechenhelligkeit faellt unter die
#                       Wahrnehmungsschwelle (photopisch/mesopisch-Grenze)
#
# "Sinnvoller Bereich" = [V_helligkeit_min, min(V_seeing_max, V_helligkeit_max)]

def helligkeit_praxis(D_mm: float, V, L0: float, D_auge_mm: float = 6.0):
    """Netzhaut-Flächenhelligkeit L(V) für ausgedehnte Objekte (Mond, Planeten,
    Nebel — NICHT für Sterne, die dimmen mit V nicht, s. Modul-Docstring).

    Etendue-Erhaltung: kein Teleskop kann die Flächenhelligkeit über den
    Wert bei Austrittspupille=Augenpupille (=L0) hinaus steigern. Bei
    kleinerer Austrittspupille (höhere Vergrößerung) sinkt sie quadratisch:

        AP = D_mm / V                      (Austrittspupille, mm)
        L(V) = L0 · min(1, (AP / D_auge_mm)²)
    """
    V  = np.asarray(V, dtype=float)
    AP = D_mm / np.maximum(V, 1e-9)
    return L0 * np.minimum(1.0, (AP / D_auge_mm) ** 2)


def v_min_helligkeit(D_mm: float, D_auge_mm: float = 6.0) -> float:
    """V ab der Austrittspupille <= Augenpupille — kein Lichtverlust mehr
    durch eine für das Auge zu große Austrittspupille."""
    return D_mm / D_auge_mm


def v_max_seeing(fwhm_seeing_as: float, eye_res_as: float = 150.0) -> float:
    """Onset-Schwelle: V ab der die (vergrößerte) Seeing-Unschärfe die
    Augenauflösung übersteigt — analog v_krit_blur_direkt(), nur mit
    Seeing statt Spiegel-Blur. Seeing-FWHM ist bereits ein Himmelswinkel
    (arcsec), keine d_blur→α-Umrechnung nötig (gleiche Konvention wie
    dort: Blur-Winkel × V = am Auge erscheinender Winkel).

    KORRIGIERT 08/2026: eye_res_as war zunächst 60" (1 Bogenminute,
    übernommen von v_krit_blur_direkt) — das ergab bei typischem Seeing
    unrealistisch niedrige Praxiswerte (40×/80× bei 1,5"), deutlich unter
    publizierter Praxis-Konsens (~1×D bei gutem, ~2×D bei exzellentem
    Seeing, s. mehrere unabhängige Quellen: cosmicpursuits.com,
    telescopicwatch.com, nexstarsite.com — "50×/Zoll ≈ 2×D(mm)" als unter
    guten Bedingungen erreichbare Praxisgrenze, kein reiner Mythos).
    Rückwärts aus dieser Praxis-Erwartung gelöst ergibt sich konsistent
    ~150" (2,5 Bogenminuten) für BEIDE Testfälle (0,8" und 1,5" Seeing) —
    plausibel, da ein weiches, kontrastarmes Seeing-"Kochen" am Okular
    grober wahrgenommen wird als die scharfe 1-Bogenminuten-Sehschärfe-
    Testtafel-Schwelle. Getrennt von der 60"-Konvention für Spiegel-Blur
    (v_krit_blur_direkt) gehalten, da dort scharfe Beugungsringe eher
    auffallen als diffuses Seeing-Schimmern — bewusst zwei verschiedene
    Konstanten für zwei unterschiedlich wahrgenommene Unschärfe-Arten.

    WICHTIG: das ist der Onset-Punkt (Blur wird gerade eben sichtbar),
    NICHT die praktisch nutzbare Obergrenze — exakt wie V_krit für
    Spiegelfehler auch nur der Onset ist. Der Blenden-Tab verwendet dafür
    bereits die Konvention V_max_praktisch = 2×V_krit (s. _diagramm_blende,
    V_max_orig = min(V_max_AP, V_krit_orig*2.0)); dieselbe Verdopplung wird
    in _diagramm_vmax() für V_seeing_praktisch angewendet.
    """
    if fwhm_seeing_as <= 0:
        return 1e9
    return eye_res_as / fwhm_seeing_as


def v_max_diffraktion(D_mm: float, eye_res_as: float = 150.0) -> float:
    """Onset-Schwelle: V ab der die (vergrößerte) Beugungsscheibe (Dawes-
    Grenze) die Augenauflösung übersteigt — fundamentale, aperturabhängige
    Grenze, die auch für eine PERFEKTE Optik bestehen bleibt. Anders als
    v_max_seeing (aperturunabhängig, reines Seeing) und anders als
    v_krit_naeh (Onset durch Spiegelaberration, geht bei Wb→0 selbst
    schon gegen unendlich) — dies hier ist die Beugungsgrenze als
    eigener, dritter Effekt, unabhängig von Spiegelfehlern.

    dawes_as = 116/D_mm  (Standard-Dawes-Auflösungsgrenze, Bogensekunden)
    V = eye_res_as / dawes_as = eye_res_as · D_mm / 116

    Steigt LINEAR mit D — im Gegensatz zu v_max_seeing (aperturunabhängig,
    konstant). Für kleine Öffnungen dominiert i.d.R. diese Grenze, für
    große Öffnungen i.d.R. das Seeing — kombiniert über v_kombiniert_blur().
    Das ist der Kern der Gegenprobe zur "2×D"-Faustregel: die Regel trifft
    näherungsweise zu, solange die Beugungsgrenze bindend ist (kleine
    Öffnung), wird aber für große Öffnungen zunehmend zu optimistisch,
    sobald das (aperturunabhängige) Seeing die Grenze übernimmt.
    """
    if D_mm <= 0:
        return 0.0
    dawes_as = 116.0 / D_mm
    return eye_res_as * D_mm / 116.0


def v_kombiniert_blur(V_a: float, V_b: float) -> float:
    """Kombiniert zwei unabhängige Blur-Quellen (z.B. Seeing + Spiegelfehler),
    jede bereits als eigene V-Schwelle (eye_res/α) ausgedrückt.

    Herleitung: α_a=eye_res/V_a, α_b=eye_res/V_b. Unabhängige Unschärfen
    addieren sich quadratisch (Gauß-Näherung): α_komb=√(α_a²+α_b²).
    Daraus: V_komb = eye_res/α_komb = 1/√(1/V_a² + 1/V_b²).

    Ergebnis ist immer <= min(V_a, V_b) — zwei Unschärfequellen zusammen
    sind nie besser als die schlechtere allein.
    """
    if V_a <= 0 or V_b <= 0:
        return 0.0
    return 1.0 / math.sqrt(1.0 / V_a**2 + 1.0 / V_b**2)


def v_max_helligkeit(D_mm: float, L0: float, D_auge_mm: float = 6.0,
                     L_schwelle: float = 1.0) -> float:
    """V ab der die Netzhaut-Flächenhelligkeit unter die Wahrnehmungs-
    schwelle L_schwelle fällt (Default 1 cd/m², photopisch/mesopisch-
    Grenze — grobe Orientierung, kein scharfer physiologischer Wert).
    Gibt 0 zurück wenn das Objekt bereits bei minimaler Vergrößerung
    (volle Austrittspupille) zu schwach ist.
    """
    if L0 <= L_schwelle:
        return 0.0
    return (D_mm / D_auge_mm) * math.sqrt(L0 / L_schwelle)


# Referenz-Flächenhelligkeiten [cd/m²] — grobe Schätzwerte für typische
# Beobachtungsobjekte (manuell überschreibbar in der UI). Bereits vorhandene
# Werte aus csf_lum-Dokumentation übernommen, um Mars/Saturn getrennt und
# Mond ergänzt.
OBJEKT_HELLIGKEITEN = [
    ("Mond",           2500.0),
    ("Jupiter",        6000.0),
    ("Saturn",          500.0),
    ("Mars",            300.0),
    ("Helles DSO",        5.0),
    ("Schwaches DSO",     0.5),
]


def kurven_N(D_mm, lam_nm, N_min=3.0, N_max=15.0, schritte=150):
    """Strehl und eff. Öffnungen als Funktion von f/D.

    Rückgabe: ns, strehls_marechal, deff_ks, aufl_verluste, strehls_exakt

    strehls_marechal: Marechal-Naeherung  S = exp(-(2*pi*Wrms)^2)
    strehls_exakt:    exaktes Pupillenintegral (Born & Wolf)
    Beide am besten Fokus (Faktor 16 in Wellenfrontformel).
    """
    ns = np.linspace(N_min, N_max, schritte)
    strehls_mar, deff_ks, aufl_verluste, strehls_exakt = [], [], [], []
    for n in ns:
        r = berechne(D_mm, D_mm * n, lam_nm)
        strehls_mar.append(r["strehl_marechal"])   # echte Marechal-Naeherung
        deff_ks.append(r["Deff_k"])
        aufl_verluste.append(r["aufl_verlust_pct"])
        strehls_exakt.append(r["strehl"])           # exaktes Pupillenintegral
    return (ns, np.array(strehls_mar), np.array(deff_ks),
            np.array(aufl_verluste), np.array(strehls_exakt))


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

    KORRIGIERT 10/2026 (v6.10.1): Die frühere Formel
        [M(f) − ε²·M(f/ε)] / (1−ε²)   für f ≤ ε
    ließ den Kreuzterm zwischen Außenkreis und Fangspiegelscheibe weg und
    zeigte dadurch bei ε>0 fälschlich MTF-Werte ÜBER dem unobstruierten
    Teleskop (Verbesserung statt Verlust bei mittleren Ortsfrequenzen).
    Gegenprobe: numerische Autokorrelation der Ringpupille (FFT) — z.B.
    ε=0.25, f=0.2: alt 0.790, richtig 0.671 (unobstruiert 0.747).

    Autokorrelation der Ringpupille = Kreis(1) ⋆ Kreis(1)  +  Kreis(ε) ⋆ Kreis(ε)
                                      − 2 · Überlappfläche(Kreis 1, Kreis ε; Abstand s)
    mit s = 2f (Verschiebung in Pupillenradien), normiert auf die Ringfläche
    π(1−ε²), d.h. MTF(0)=1 (Kontrastübertrag relativ zum Mittelwert).

        MTF(f,ε) = [ M(f) + ε²·M(f/ε) − 2·A12(2f) / π ] / (1−ε²)
    mit M(x) = (2/π)·(arccos(x) − x·√(1−x²)), A12 = Kreis-Kreis-Überlappfläche.
    Quellen: Barakat (1962), O'Neill (1956); vgl. Sacek §4.2.
    """
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    if eps <= 0: return mtf_para(f)

    def M(x):
        if x >= 1: return 0.0
        if x <= 0: return 1.0
        return (2/math.pi) * (math.acos(x) - x * math.sqrt(1 - x**2))

    def _overlap(r1, r2, d):
        """Überlappfläche zweier Kreise (Radien r1, r2, Mittelpunktabstand d)."""
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return math.pi * min(r1, r2) ** 2
        a = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
        b = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
        c = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2)
                            * (d - r1 + r2) * (d + r1 + r2))
        return a + b - c

    norm = 1.0 - eps**2
    if norm <= 1e-6: return 0.0
    A12 = _overlap(1.0, eps, 2.0 * f) / math.pi
    return (M(f) + eps**2 * M(f / eps) - 2.0 * A12) / norm


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


# ── Foucault-Auswertung ───────────────────────────────────────────────────────

def foucault_zonen(D_mm: float, n: int = 4) -> np.ndarray:
    """Messradien gleichflächiger Zonen (Zonenmitte, flächengewichtet).

    r_i = R · √((2i−1) / (2n))   i = 1…n
    Entspricht der FigureXP-Konvention.
    """
    R = D_mm / 2.0
    return R * np.sqrt((2 * np.arange(1, n + 1) - 1) / (2 * n))


def foucault_zonengrenzen(D_mm: float, n: int = 4) -> np.ndarray:
    """Außengrenzen der n gleichflächigen Zonen (für Zonenmaske/Ringblende).

    r_aussen,i = R · √(i/n)   i = 1…n   (r_aussen,n = R, Spiegelrand)
    Innengrenze von Zone i ist r_aussen,i-1 (bzw. 0 für Zone 1).
    """
    R = D_mm / 2.0
    return R * np.sqrt(np.arange(1, n + 1) / n)


def _foucault_wellenfront_kern(r_zonen: np.ndarray, df_zonen: np.ndarray,
                               D_mm: float, R_c_mm: float,
                               lam_nm: float) -> np.ndarray:
    """Kernrechnung der Zonen-Wellenfront fuer gegebenen Kruemmungsradius R_c.

    Intern verwendet von foucault_wellenfront(); nicht direkt aufrufen
    (kein Best-Fit-ROC, s. dort).
    """
    lam_mm = lam_nm * 1e-6
    f_mm   = R_c_mm / 2.0
    n      = len(r_zonen)

    dl_para     = r_zonen**2 / (4.0 * f_mm)     # Parabolsollwert, moving source
    dl_para_rel = dl_para - dl_para[0]          # relativ zu Zone 1 (=0)
    residual    = df_zonen - dl_para_rel        # LA_gemessen - LA_ideal (moving source)

    # Zonenbreiten Z_i: Grenze zwischen Zone i und i+1 = Wurzel des Mittel-
    # werts der Radienquadrate (exakt fuer flaechengleiche Zonen, sinnvolle
    # Naeherung sonst). Aussengrenze der letzten Zone = Spiegelrand D/2.
    r_bound = np.zeros(n + 1)
    r_bound[-1] = D_mm / 2.0
    for i in range(1, n):
        r_bound[i] = math.sqrt((r_zonen[i-1]**2 + r_zonen[i]**2) / 2.0)
    Z = r_bound[1:] - r_bound[:-1]

    # TA = 2*(r/R_c)*residual: Faktor 2 konvertiert moving-source-Residuum
    # (unsere df_zonen-Konvention, = FigureXPs "Knife Reading") in die
    # fixed-source-Konvention, fuer die Sacek's TA/sigma-Formeln kalibriert
    # sind (s. Docstring foucault_wellenfront).
    TA    = 2.0 * (r_zonen / R_c_mm) * residual
    sigma = (TA / R_c_mm) * Z                   # Zonen-Steigungsbeitrag

    W_cum_mm = np.cumsum(sigma)                 # kumulative Integration
    return W_cum_mm / lam_mm


def foucault_wellenfront(r_zonen: np.ndarray, df_zonen: np.ndarray,
                         D_mm: float, f_mm: float, lam_nm: float,
                         best_fit_roc: bool = True,
                         roc_suchbereich_mm: float | None = None) -> np.ndarray:
    """Rekonstruiert die Wellenfront aus Foucault-Zonenmessungen (Maskentest).

    NEU IMPLEMENTIERT 08/2026 (loest Bug in Vorversion vollstaendig ab,
    nicht nur einen Faktor -- s. Chat-Historie "Foucault-Test / FigureXP-
    Vergleich" fuers Buch):

    Alte Version rechnete PUNKTWEISE pro Zone:
        W(r) = Δl(r) · r² / (8·f²·λ)
    Das ist nur fuer eine glatte, geschlossene Fehlerform (reine
    sphaerische Aberration) korrekt kalibriert (Kontrolle Kugel 200/f5:
    S=0.1232 traf zufaellig, weil dort dieselbe geschlossene Form vorliegt,
    fuer die die Punktformel algebraisch aequivalent zur korrekten Loesung
    ist). Fuer reale, unregelmaessige Zonenmessungen ist das falsch: der
    Foucault-Test misst die LOKALE STEIGUNG der Wellenfront (Ray-Winkel-
    fehler), nicht ihre Hoehe -- s. z.B. bbastrodesigns.com/JoyOfMirror-
    Making/Parabolizing.html: "Tests like the Foucault and Ronchi are
    slope tests". Am 14"-Realdatensatz (Maskentest, D=355mm) ergab die
    alte Formel W_ptv ≈ 0.51λ, FigureXP (identische Rohdaten) zeigte
    0.089λ -- Faktor 5.7 zu groß.

    Neue Methode nach Sacek (telescope-optics.net, Kap. 4.8.2 "Foucault
    test", Abschnitt "Residuals/Reduction method"):
      1. residual_i = LA_gemessen,i − LA_ideal,i     (Zone 1 als Referenz)
      2. TA_i  = 2·(r_i/R_c)·residual_i               (Faktor 2: s.u.)
      3. σ_i   = (TA_i/R_c)·Z_i                        (Z_i = Zonenbreite!)
      4. W(Zone n) = kumulative Summe σ_1..σ_n         (diskrete Integration
                                                          der Wellenfront-
                                                          steigung)
    Der Faktor 2 in Schritt 2 konvertiert unsere "moving source"-Messwerte
    (Quelle+Messer bewegen sich zusammen -- Standard beim Maskentest, ent-
    spricht FigureXPs "Knife Reading"-Spalte) in die "fixed source"-
    Konvention, fuer die Sacek's TA/σ-Formeln in seinem Zahlenbeispiel
    kalibriert sind.

    Verifiziert:
      a) Kugel-Kontrolle (df=0 ueberall), best_fit_roc=False:
         S_exakt konvergiert exakt auf 0.1232, W_ptv auf 1.4205 -- identisch
         zur unabhaengigen D⁴/(512·f³·λ)-Formel aus _wellenfronten().
      b) 14"-Realdaten (Maskentest) gegen FigureXP-Export: P-V roh 0.099λ
         vs. FigureXP 0.089λ (~10%, im Rahmen von Eingabe-Rundung und
         Optimierungskriterium, s.u.).

    best_fit_roc (Default True): FigureXP optimiert den Referenz-Kruemmungs-
    radius selbst ("Find Best Fit ROC"), weil der ROC typischerweise
    ungenauer bekannt ist als die Zonen-Differenzmessungen. Ohne diese
    Optimierung vergleicht man unsere Rohdaten mit FigureXPs bereits
    radius-optimiertem Ergebnis -- kein Apples-to-Apples-Vergleich. Bei
    True wird R_c=2·f_mm nur als Suchzentrum verwendet; der golden-section-
    optimierte R_c minimiert die RMS-Abweichung der Zonenwellenfront.
    Achtung: die Kugel-Kontrolle (a) gilt nur fuer best_fit_roc=False --
    mit aktiver Optimierung faellt S dort leicht anders aus (0.128 statt
    0.1232), weil die Best-Fit-Suche einen Teil des ρ⁴-Musters durch
    Radius-Anpassung kompensiert (Sacek beschreibt denselben Effekt in
    seinem eigenen Beispiel: 0.99λ roh → 0.25λ best-fit).
    """
    R_c0 = 2.0 * f_mm
    if not best_fit_roc:
        return _foucault_wellenfront_kern(r_zonen, df_zonen, D_mm, R_c0, lam_nm)

    if roc_suchbereich_mm is None:
        roc_suchbereich_mm = max(5.0, R_c0 * 0.01)   # +-1% Standard-Suchfenster

    def _rms_fuer_Rc(R_c_trial: float) -> float:
        W = _foucault_wellenfront_kern(r_zonen, df_zonen, D_mm, R_c_trial, lam_nm)
        return float(np.std(W))

    R_c_opt = _golden_section(_rms_fuer_Rc, R_c0 - roc_suchbereich_mm,
                              R_c0 + roc_suchbereich_mm)
    return _foucault_wellenfront_kern(r_zonen, df_zonen, D_mm, R_c_opt, lam_nm)



def _polyfit_bic(rho_z: np.ndarray, W_z: np.ndarray,
                 grad_modus: str = "bic") -> tuple:
    """Waehlt Polynomgrad fuer den Zonenfehler-Fit.

    Verwendet nur gerade Potenzen rho^2, rho^4, ... (Rotationssymmetrie).
    Achsenpunkt (0,0) wird als Stuetzpunkt erzwungen.
    Maximaler Grad = n//2 -- verhindert Ueberfit bei wenigen Zonen.

    grad_modus:
      "bic"  (Standard) -- Grad automatisch per BIC (Bayes Information
             Criterion) gewaehlt. Kann bei "rauschartigen" (nicht-glatten)
             Zonenabweichungen einen sehr niedrigen Grad waehlen (im
             Extremfall Grad 1 = reiner Defokus), der bei anschliessender
             Bestfokus-Korrektur komplett verschwindet -- s. "fest" unten
             und die Zuverlaessigkeits-Pruefung in foucault_kennzahlen().
      "fest" -- Grad IMMER auf das Maximum n//2 gesetzt (an Zonenzahl
             gekoppelt, z.B. 4 Zonen->Grad 2, 8 Zonen->Grad 4), unabhaengig
             vom BIC-Ergebnis. Empfohlen, wenn BIC einen verdaechtig
             niedrigen Grad waehlt (s. fit_residuum_rms-Warnung).
             WICHTIG: loest die Grad-1-Degenerierung NICHT automatisch --
             wenn die Zonenabweichungen genuin nicht polynomartig sind
             (echte lokale Textur statt glatter Formfehler), bleibt das
             Fit-Residuum auch bei Grad n//2 groß (s. Chat-Historie:
             14"-Beispiel, Residuum sinkt nur von 0.0342 auf 0.0339λ trotz
             Grad 1->4). In diesem Fall ist Hybrid-Modus (use_direct=True)
             die richtige Wahl, nicht ein hoeherer Polynomgrad.

    Rueckgabe: (grad, coeffs, potenzen)
    """
    rho_fit = np.concatenate([[0.0], rho_z])
    W_fit   = np.concatenate([[0.0], W_z])
    n_pts   = len(W_fit)
    max_grad = max(1, len(W_z) // 2)

    if grad_modus == "fest":
        grad = max_grad
        pot  = [2 * k for k in range(1, grad + 1)]
        A    = np.column_stack([rho_fit**p for p in pot])
        c, _, _, _ = np.linalg.lstsq(A, W_fit, rcond=None)
        return grad, c, pot

    best_bic    = np.inf
    best_grad   = 1
    best_coeffs = None
    best_pot    = None

    for grad in range(1, max_grad + 1):
        pot  = [2 * k for k in range(1, grad + 1)]
        A    = np.column_stack([rho_fit**p for p in pot])
        c, _, _, _ = np.linalg.lstsq(A, W_fit, rcond=None)
        rss  = float(np.sum((A @ c - W_fit)**2))
        bic  = n_pts * math.log(max(rss / n_pts, 1e-30)) + grad * math.log(n_pts)
        if bic < best_bic:
            best_bic    = bic
            best_grad   = grad
            best_coeffs = c
            best_pot    = pot

    return best_grad, best_coeffs, best_pot

# Schwelle, ab der ein Polynom-Fit als nicht mehr vertrauenswuerdig gilt
# (Zonenfehler werden weggemittelt statt erfasst -- s. foucault_kennzahlen).
FIT_RESIDUUM_SCHWELLE = 0.010  # lambda

def foucault_kennzahlen(W_zonen: np.ndarray, r_zonen: np.ndarray,
                        D_mm: float, lam_nm: float | None = None,
                        use_direct: bool = False,
                        grad_modus: str = "bic") -> dict:
    """PtV, RMS, Strehl aus Zonenwellenfronten.

    Zwei Modi (use_direct):

    False -- Polynom-Fit (Standard, glatte Flaeche):
        Passt W(rho) = a4*rho^4 + a2*rho^2 (oder hoehere Grade, s.
        grad_modus) an die Zonenwerte an. Korrekt fuer reine sphaerische
        Aberration. Zonenfehler (lokale Ringfehler), die sich nicht durch
        ein Polynom beschreiben lassen, werden weggemittelt: Strehl wird
        dann zu optimistisch -- im Extremfall exakt 1.0 (s. unten).

    True -- Hybrid-Modus (Zonenfehler):
        Gleiche Polynom-Grundform, aber Abweichungen der Messwerte
        vom Polynom (Residuen) werden als stueckweise konstante
        Ringzonen-Fehler auf die Wellenfront aufgesattelt.
        Erkennt echte Zonenfehler ohne die Grundform zu verfaelschen.
        Bei glattem Spiegel (Residuen=0): identisch mit Polynom-Modus.

    grad_modus: "bic" (Standard, s. _polyfit_bic) oder "fest" (Grad immer
        n//2, an Zonenzahl gekoppelt -- s. _polyfit_bic Docstring).

    SICHERHEITSNETZ (08/2026): Im Polynom-Modus (use_direct=False) kann der
    Fit (v.a. bei grad_modus="bic" mit sehr niedrigem gewaehltem Grad)
    durch die anschliessende Bestfokus-Korrektur degenerieren: waehlt BIC
    z.B. Grad 1 (= reiner Defokus-Term), hebt die Fokuskorrektur die
    gesamte Kurve exakt auf, und S_exakt kommt exakt 1.0 heraus --
    unabhaengig davon, wie groß die tatsaechliche Zonenstreuung
    (fit_residuum_rms) ist. Deshalb wird im Polynom-Modus geprueft, ob
    fit_residuum_rms > FIT_RESIDUUM_SCHWELLE ist; falls ja, wird
    kennzahlen_zuverlaessig=False zurueckgegeben. Die UI MUSS S_exakt/
    W_ptv/W_rms in diesem Fall sichtbar als unzuverlaessig kennzeichnen
    (s. Chat: "Strehl=1 sollte nie rauskommen koennen"). Im Hybrid-Modus
    (use_direct=True) ist kennzahlen_zuverlaessig immer True, da die
    Zonenresiduen dort bereits Teil der Wellenfront sind -- ein hohes
    fit_residuum_rms ist dort erwuenschte Information, kein Fehler.

    Rueckgabe-Dict -- alle Schluessel immer vorhanden:
        W_zonen, W_c_zonen, W_rms, W_ptv, Wb_fit
        a4, a2, fit_residuum_rms, kennzahlen_zuverlaessig
        S_mar, S_exakt, Deff_k
        Glas_f, Glas_zonen, Glas_ptv_nm, Glas_volumen_mm3, W_min, W_max, lam_nm
        rho_f, W_best, W_c
    """
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

    R     = D_mm / 2.0
    rho_z = r_zonen / R
    rho_f = np.linspace(0.0, 1.0, 2000)

    # Flaechengewichte der Zonen (Kreisringflaechen, normiert)
    r_inner    = np.concatenate([[0.0], r_zonen[:-1]])
    r_outer    = r_zonen.copy()
    zone_areas = r_outer**2 - r_inner**2
    zone_areas = zone_areas / zone_areas.sum()

    # Polynomgrad waehlen (BIC oder fest an Zonenzahl gekoppelt, s. grad_modus)
    poly_grad, poly_coeffs, poly_pot = _polyfit_bic(rho_z, W_zonen, grad_modus)

    # a4 fuer Wb-Berechnung (0 wenn rho^4 nicht im Modell)
    a4 = float(poly_coeffs[poly_pot.index(4)]) if 4 in poly_pot else 0.0
    a2 = float(poly_coeffs[poly_pot.index(2)]) if 2 in poly_pot else 0.0

    # Residuen = Zonenfehler (Abweichung vom glatten Polynom)
    W_poly_at_zones  = sum(float(c) * rho_z**p
                           for c, p in zip(poly_coeffs, poly_pot))
    residuen         = W_zonen - W_poly_at_zones
    fit_residuum_rms = float(np.sqrt(np.dot(residuen**2, zone_areas)))

    # Wellenfront auf feinem Gitter
    W_f = sum(float(c) * rho_f**p for c, p in zip(poly_coeffs, poly_pot))

    if use_direct:
        # Hybrid: Polynom (BIC-Grad) + stückweise konstante Residuen je Ringzone
        rho_inner_z = np.concatenate([[0.0], rho_z[:-1]])
        for ri, ro, res in zip(rho_inner_z, rho_z, residuen):
            mask = (rho_f >= ri) & (rho_f <= ro)
            W_f[mask] += res

    # Defokus-Optimierung (bester Fokus)
    def _wrms_sq(d):
        Wt = W_f + d * rho_f**2
        Wm = _trapz(Wt * rho_f, rho_f) / 0.5
        Wc = Wt - Wm
        return float(_trapz(Wc**2 * rho_f, rho_f) / 0.5)

    d_opt  = _golden_section(_wrms_sq, -200.0, 200.0)
    W_best = W_f + d_opt * rho_f**2
    Wm     = float(_trapz(W_best * rho_f, rho_f) / 0.5)
    W_c    = W_best - Wm

    # Zonenwerte im besten Fokus
    W_best_zonen = (W_poly_at_zones + d_opt * rho_z**2
                    + (residuen if use_direct else 0.0))
    Wm_zonen     = float(np.dot(W_best_zonen, zone_areas))
    W_c_zonen    = W_best_zonen - Wm_zonen

    # Kennzahlen
    phase   = 2.0 * math.pi * W_c
    A_re    = float(_trapz(np.cos(phase) * rho_f, rho_f) / 0.5)
    A_im    = float(_trapz(np.sin(phase) * rho_f, rho_f) / 0.5)
    S_exakt = A_re**2 + A_im**2

    W_ptv   = float(W_c.max() - W_c.min())
    W_rms   = math.sqrt(float(_trapz(W_c**2 * rho_f, rho_f) / 0.5))
    Wb_fit  = abs(a4) / 4.0 if a4 != 0.0 else W_ptv / 4.0
    S_mar   = math.exp(-(2 * math.pi * W_rms)**2)
    Deff_k  = D_mm * S_exakt**0.25

    # Materialabtrag (VORZEICHEN KORRIGIERT 10/2026, v6.9.1)
    # W ist eine "Tiefen"-Groesse (W_c gross = Oberflaeche relativ zu tief).
    # Abzutragen ist dort, wo die Oberflaeche relativ HOCH steht, also wo W
    # klein ist: Abtrag(rho) = (W_max - W(rho)) * lam/2, Null im Tal von
    # "Glass Remaining" (= Maximum von W). Fuer eine Kugel ergibt das Huegel
    # in Mitte und Rand, Null bei ~0.7R -- identisch zu FigureXP.
    # Vorher stand hier (W - W_min) -> exakt invertiert (Volumen 8.1 statt
    # ~4 mm3 bei 200/f5-Kugel).
    Glas_f = Glas_zonen = Glas_ptv_nm = Glas_volumen_mm3 = W_min = W_max = None
    if lam_nm is not None:
        lam_mm           = lam_nm * 1e-6
        W_min            = float(W_c.min())
        W_max            = float(W_c.max())
        Glas_f           = (W_max - W_c) * lam_mm / 2.0
        Glas_zonen       = (W_max - W_c_zonen) * lam_nm / 2.0
        Glas_ptv_nm      = float(W_ptv * lam_nm / 2.0)
        Glas_volumen_mm3 = float(
            2 * math.pi * R**2 * _trapz(Glas_f * rho_f, rho_f))

    # Zuverlaessigkeits-Sicherheitsnetz: NUR im Polynom-Modus relevant --
    # dort werden Zonenabweichungen vom Fit weggemittelt, koennen also
    # S_exakt/W_ptv/W_rms optimistisch verfaelschen (bis hin zur Grad-1-
    # Bestfokus-Degenerierung, S_exakt exakt 1.0). Im Hybrid-Modus sind die
    # Residuen bereits Teil der Wellenfront (W_f oben) -- ein hohes
    # fit_residuum_rms ist dort kein Zuverlaessigkeitsproblem.
    kennzahlen_zuverlaessig = use_direct or (fit_residuum_rms <= FIT_RESIDUUM_SCHWELLE)

    return dict(
        W_zonen=W_zonen,
        W_c_zonen=W_c_zonen,
        poly_grad=poly_grad,
        W_rms=W_rms,
        W_ptv=W_ptv,
        Wb_fit=Wb_fit,
        a4=a4,
        a2=a2,
        fit_residuum_rms=fit_residuum_rms,
        kennzahlen_zuverlaessig=kennzahlen_zuverlaessig,
        S_mar=S_mar,
        S_exakt=S_exakt,
        Deff_k=Deff_k,
        Glas_f=Glas_f,
        Glas_zonen=Glas_zonen,
        Glas_ptv_nm=Glas_ptv_nm,
        W_min=W_min,
        W_max=W_max,
        lam_nm=lam_nm,
        Glas_volumen_mm3=Glas_volumen_mm3,
        rho_f=rho_f,
        W_best=W_best,
        W_c=W_c,
    )


def mtf_sph_rel_1d(f: float, Wb: float, n: int = 80,
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
      KORRIGIERT 07/2026: Wp = Wb*4 (nicht *8). Die alte *8 kompensierte
      zufaellig den fehlenden Reflexionsfaktor 2 in _wellenfronten() -- die
      MTF-Kurven selbst waren dadurch bereits korrekt (Regressionstest nach
      dem Gesamt-Fix: MTF-Werte vor/nach Fix identisch bis auf Rundung).
      Jetzt, wo Wb von _wellenfronten() korrekt (verdoppelt) ist, muss der
      hiesige Faktor entsprechend halbiert werden, damit Wp_intern gleich bleibt.

    f  = normierte Ortsfrequenz (0..1)
    Wb = W_PtV_bestfocus  (= Wp_paraxial/4)
    """
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    lim = math.sqrt(max(0.0, 1.0 - (f/2)**2))
    re = im = norm = 0.0
    Wp = Wb * 4  # Wp_integral = 4·Wb  (Wb = PtV best focus = Wp_paraxial/4) [07/2026: war *8]
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


_MSR_CACHE = {}


def mtf_sph_rel(f: float, Wb: float, n: int = 80,
                paraxial: bool = False) -> float:
    """Relative MTF eines sphärischen Spiegels (MTF(0)=1) — EXAKT (v7.0.0-beta).

    Berechnet über die 2D-Pupillen-Autokorrelation (FFT) der Wellenfront
    W = 4·Wb·(ρ⁴−ρ²)  (bester Fokus)  bzw.  4·Wb·ρ⁴  (paraxial) und teilt
    durch die ideale MTF.  Ersetzt die frühere 1D-Linienintegral-Näherung
    (jetzt mtf_sph_rel_1d), die bei S < ~0,9 die MTF überschätzte; die
    FFT-Methode reproduziert den Strehl exakt (OTF-Summe = S).
    Für ν > 0,9 wird das Verhältnis konstant fortgesetzt (ideale MTF ≈ 0,
    Beitrag vernachlässigbar).  Parameter n bleibt aus Kompatibilitätsgründen."""
    if f <= 0:
        return 1.0
    if f >= 1:
        return 0.0
    key = (round(float(Wb), 5), bool(paraxial))
    tab = _MSR_CACHE.get(key)
    if tab is None:
        Np = 192
        rho = np.linspace(0.0, 1.0, 600)
        W = 4.0 * Wb * (rho ** 4 if paraxial else (rho ** 4 - rho ** 2))
        nu_g = np.arange(Np) / float(Np)
        m = mtf_aus_wellenfront(rho, W, 0.0, Np=Np, nu=nu_g)
        mp = np.array([mtf_para(x) for x in nu_g])
        ratio = np.where(mp > 1e-9, m / np.maximum(mp, 1e-9), 1.0)
        kmax = int(0.9 * Np)
        ratio[kmax:] = ratio[kmax]
        ratio[0] = 1.0
        tab = (nu_g, np.clip(ratio, 0.0, 1.5))
        _MSR_CACHE[key] = tab
    return float(np.interp(f, tab[0], tab[1]))



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

    KORRIGIERT 07/2026: strehl ist jetzt der EXAKTE Strehl (Pupillenintegral),
    nicht mehr die Maréchal-Näherung. Vorher nutzte _wellenfronten()'s S
    (=Maréchal) direkt für Titel/Legenden -- das fiel erst nach dem
    Wellenfront-Fix auf, weil Maréchal und exakt fuer schnelle Kugeln jetzt
    weit auseinanderliegen (0.0008 vs. 0.123 statt vorher 0.17 vs. 0.15).
    """
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    S = _strehl_exakt(Wb)
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


def Q_vis_blende_batch(D_mm: float, f_mm: float, lam_nm: float,
                       V_arr: np.ndarray, L_arr,
                       D_ref: float = 200.0, n: int = 60) -> np.ndarray:
    """Q_vis_blende vektorisiert über V_arr × L_arr — numpy-Broadcast.

    MTF-Arrays (frequenzabhängig, nicht V/L-abhängig) werden einmal berechnet.
    CSF-Gewichtung per 3D-Broadcast (nu × V × L) → ~1000× schneller als
    1500 Einzelaufrufe im Blende-Tab.

    Rückgabe: ndarray shape (len(V_arr), len(L_arr))
    """
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    d_fang  = d_fang_min(D_ref, f_mm)
    eps     = min(d_fang / D_mm, 0.99)
    fc_orig = D_ref / (lam_nm * 1e-6 * 206265)
    fc_bl   = D_mm  / (lam_nm * 1e-6 * 206265)
    f_ratio = fc_bl / fc_orig
    nu      = np.linspace(0.0, 1.0, n)
    nu_bl   = nu / max(f_ratio, 1e-6)
    # MTF einmal berechnen (nu-abhängig, nicht V/L-abhängig)
    mp  = np.array([mtf_para_obstr(min(x, 1.0), eps) for x in nu_bl])
    ms  = np.array([mtf_sph_rel(min(x, 1.0), Wb)     for x in nu_bl])
    mask = nu > f_ratio; mp[mask] = 0.0; ms[mask] = 0.0
    ms_mp = ms * mp   # (n,)
    # CSF per 3D-Broadcast: (n, nV, nL)
    V_arr = np.asarray(V_arr); L_arr = np.asarray(L_arr)
    L_ok  = (L_arr * (D_mm / D_ref) ** 2)[None, None, :]   # (1, 1, nL)
    nu_g  = nu[:, None, None]                               # (n, 1, 1)
    V_g   = V_arr[None, :, None]                            # (1, nV, 1)
    f_cpd = nu_g * fc_orig * 3600.0 / V_g                   # (n, nV, nL) via broadcast
    L_s   = np.maximum(L_ok, 1e-4)
    f_sc  = (L_s / 100.0) ** 0.1
    a_sc  = (L_s / 100.0) ** 0.5
    f_eff = np.maximum(f_cpd * f_sc, 1e-6)
    csf_m = 2.6 * (0.0192 + 0.114 * f_eff) * np.exp(-((0.114 * f_eff) ** 1.1)) * a_sc
    result = _trapz(ms_mp[:, None, None] * csf_m, nu, axis=0)  # (nV, nL)
    return result


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
                                    V_arr: np.ndarray, n_pts: int = 80) -> np.ndarray:
    """perceived_quality_exakt für ein Array von Vergrößerungen — vektorisiert.

    MTF-Arrays werden einmal berechnet, CSF-Gewichtung per numpy-Broadcast
    über alle V gleichzeitig → ~7× schneller als Schleife.
    """
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    lam_mm  = lam_nm * 1e-6
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    fc      = D_mm / (lam_mm * 206265)
    nu      = np.linspace(0.0, 1.0, n_pts)
    mp      = np.array([mtf_para(f)        for f in nu])
    msr     = np.array([mtf_sph_rel(f, Wb) for f in nu])
    # (n_pts, n_V) broadcast: CSF-Matrix
    V_arr   = np.asarray(V_arr)
    f_cpd   = np.maximum(nu[:, None] * fc * 3600.0 / V_arr[None, :], 1e-6)
    csf_mat = 2.6 * (0.0192 + 0.114 * f_cpd) * np.exp(-((0.114 * f_cpd) ** 1.1))
    num = _trapz(msr[:, None] * mp[:, None] * csf_mat, nu, axis=0)
    den = _trapz(mp[:, None] * csf_mat, nu, axis=0)
    return num / np.maximum(den, 1e-12)


def perceived_quality_exakt_von_Wb(D_mm: float, Wb: float, lam_nm: float,
                                    V: float, n_pts: int = 80) -> float:
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
                                           V_arr: np.ndarray, n_pts: int = 80) -> np.ndarray:
    """perceived_quality_exakt_von_Wb für ein Array von Vergrößerungen — vektorisiert."""
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
    """Strehl und eff. Öffnungen als Funktion von f/D.

    Rückgabe: ns, strehls_marechal, deff_ks, aufl_verluste, strehls_exakt

    strehls_marechal: Marechal-Naeherung  S = exp(-(2*pi*Wrms)^2)
    strehls_exakt:    exaktes Pupillenintegral (Born & Wolf)
    Beide am besten Fokus (Faktor 16 in Wellenfrontformel).
    """
    ns = np.linspace(N_min, N_max, schritte)
    strehls_mar, deff_ks, aufl_verluste, strehls_exakt = [], [], [], []
    for n in ns:
        r = berechne(D_mm, D_mm * n, lam_nm)
        strehls_mar.append(r["strehl_marechal"])   # echte Marechal-Naeherung
        deff_ks.append(r["Deff_k"])
        aufl_verluste.append(r["aufl_verlust_pct"])
        strehls_exakt.append(r["strehl"])           # exaktes Pupillenintegral
    return (ns, np.array(strehls_mar), np.array(deff_ks),
            np.array(aufl_verluste), np.array(strehls_exakt))


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
      → Wp_blende = D_blende^4 / (512·f^3·λ)  kleiner  [07/2026: 1024->512]
      → Strehl_blende höher
      → Beugungsgrenze schlechter (fc ∝ D_blende)

    Rückgabe: V_arr, Q_orig, Q_blend, S_orig, S_blend
    """
    V_arr = np.linspace(V_min, V_max, n_pts)

    def Q_naeh(V_a, S, D):
        Vk = v_wendepunkt_naeh(D, _Wb_von_S(S))
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


def qvis_einzelrechner(D_mm: float, S: float, V: float,
                       lam_nm: float = 550.0) -> dict:
    """Q_vis für EINEN einzelnen Spiegel: Öffnung D, Strehl S, Vergrößerung V
    (neu in v6.10.0).

    Für beliebige Kombinationen (z.B. den eigenen 400er mit gemessenem
    Strehl 0,96), unabhängig von den Hauptreglern. Brennweite wird nicht
    gebraucht: Q_vis hängt nur von D, vom Wellenfrontfehler Wb und von V ab;
    Wb folgt aus S per Inversion des exakten Pupillenintegrals (_Wb_von_S).

    Rückgabe:
        Wb        Best-Fokus-Wellenfrontfehler PtV [λ]
        Q         Q_vis exakt (MTF×CSF-Integral, rel. zum Paraboloid = 1)
        Q_naeh    Näherung Q̃ = S + (1−S)·(1−w(V))   (wie im Diagramm gestrichelt)
        verlust   Wahrnehmungsverlust ggü. Paraboloid [%]  = (1−Q)·100
        V_krit    10%-Kontrastverlust-Schwelle (Näherung D/(13·Wb)), 1e9 = ∞
        V_q90     Vergrößerung, ab der Q_vis exakt unter 0,90 fällt
                  (auf 30…600× begrenzt; 600.0 heißt „≥ 600×“, 30.0 „≤ 30×“)
    """
    S  = min(max(float(S), 1e-6), 1.0)
    Wb = _Wb_von_S(S)
    Q  = 1.0 if Wb <= 1e-9 else perceived_quality_exakt_von_Wb(D_mm, Wb, lam_nm, V)
    S_ph = _strehl_exakt(Wb)
    Vk_w = v_wendepunkt_naeh(D_mm, Wb)
    w    = 1.0 / (1.0 + (Vk_w / max(V, 1e-9)) ** 2)
    Q_n  = S_ph + (1.0 - S_ph) * (1.0 - w)
    return dict(D=D_mm, S=S, V=V, lam=lam_nm, Wb=Wb, Q=Q, Q_naeh=Q_n,
                verlust=(1.0 - Q) * 100.0,
                V_krit=v_krit_naeh(D_mm, Wb),
                V_q90=(600.0 if Wb <= 1e-9 else
                       v_krit_aus_qvis_von_Wb(D_mm, Wb, lam_nm, 0.90)))


def qvis_kurve_mit_fangspiegel(D_mm: float, S: float, d_fang_mm: float,
                               lam_nm: float, V_arr, n: int = 80) -> np.ndarray:
    """Q_vis(V) eines Spiegels mit Strehl S UND zentraler Fangspiegel-Abschattung
    (neu in v6.11.0).

    Bezug wie im Wahrnehmungsdiagramm: Paraboloid OHNE Fangspiegel = 1.
        Q = ∫ MTF_Aberration_rel(S) · MTF_Ring(ε) · CSF(V)  /  ∫ MTF_para · CSF(V)
    mit ε = d_fang/D (Ringblenden-MTF, s. mtf_para_obstr).

    NÄHERUNG: Die Aberrations-MTF (aus S bzw. Wb) wird unverändert mit der
    Ring-MTF multipliziert; die Abschattung ändert die Wellenfront-Statistik
    (Strehl) in Wirklichkeit geringfügig. d_fang_mm <= 0 → identisch zu
    perceived_quality_exakt_von_Wb().
    """
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    V_arr  = np.atleast_1d(np.asarray(V_arr, dtype=float))
    eps    = min(max(d_fang_mm / D_mm, 0.0), 0.95)
    Wb     = _Wb_von_S(min(max(float(S), 1e-6), 1.0))
    nu     = np.linspace(0.0, 1.0, n)
    mp     = np.array([mtf_para(x) for x in nu])
    mo     = np.array([mtf_para_obstr(x, eps) for x in nu])
    msr    = (np.ones_like(nu) if Wb <= 1e-9
              else np.array([mtf_sph_rel(x, Wb) for x in nu]))
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    out = []
    for V in V_arr:
        w = np.array([csf(x * 3600.0 / V) for x in nu * fc])
        out.append(float(_trapz(msr * mo * w, nu) / max(_trapz(mp * w, nu), 1e-12)))
    return np.array(out)


def fangspiegel_rechner(D_mm: float, f_mm: float, buet_mm: float = 50.0,
                        tubus_mm: float = 0.0, bild_mm: float = 10.0,
                        S: float = 1.0, V: float = 150.0,
                        lam_nm: float = 550.0, hut_mm: float = 0.0,
                        fe_mm: float = 21.0, afov_deg: float = 85.0,
                        oaz_d_mm: float = 50.0, oaz_l_mm: float = 80.0) -> dict:
    """Fangspiegel-Rechner (neu in v6.12.0).

    Minimaler Fangspiegel-Durchmesser (Strahlkegel-Geometrie, identisch zur
    Katalog-Formel  DF = A/f# + B  mit A = TAR + BüT):

        d_min = (TAR + BüT) · D/f + B

    TAR   Tubus-Außenradius  (tubus_mm/2; 0 = wie Hauptspiegel-Ø, D/2)
    BüT   Brennpunkt über Tubus (Abstand Fokalebene – Tubus-Außenwand)
    B     zu 100 % ausgeleuchtetes Bildfeld (visuell 10 mm reichen)

    Blende gegenüber dem OAZ (Gittertubus, Tubusersatz am Hutring, v7.0.0):
        Geometrie: Das OAZ-Rohr (Innen-Ø d_t, Länge L, Fokalebene = Feld-
        blende des Okulars am einen Ende) wirkt als Blendenrohr.  Die
        steilste Sichtlinie läuft vom Feldblendenrand (r_f = B_ok/2) über
        den gegenüberliegenden Innenrand am Rohrende (r_t = d_t/2, im
        Abstand L) und wird bis zur Blende im Abstand x = A + R_hut
        verlängert:
            r_blende = (r_t + r_f)·x/L − r_f
        Mindestens aber der Strahlkegel  x·D/f + B_ok.
        Die Blende ist um den Hutring gebogen (Radius R_hut): die Sichtlinie
        wird mit dem Kreis geschnitten (nicht mit einer Ebene im Abstand x),
        das macht die Blende kleiner.  Ausgabe: Sehne (Breite von der Seite)
        und abgewickelte Bogenlänge (Zuschnitt).
        B_ok = max(fe·AFOV/57,3; B)
    R_hut = hut_mm/2 (0 = Tubus-Außenradius bzw. D/2).

    Rückgabe (dict): A, d_min, d_pass (aufgerundet auf ganze mm), eps
    (=d/D), flaeche (Flächenverlust [%], ε²), D_kontrast (lineare
    Näherung D−d, Zmek/ICS), Q0 (Q_vis ohne Fangspiegel), Q (mit Fangspiegel
    d_pass, bei V), Q_verlust (Q0−Q).
    """
    tar = (tubus_mm if tubus_mm > 0 else D_mm) / 2.0
    A   = tar + max(buet_mm, 0.0)
    d   = A * D_mm / f_mm + max(bild_mm, 0.0)
    dp  = float(math.ceil(d - 1e-9))
    eps = min(dp / D_mm, 0.95)
    q0  = float(qvis_kurve_mit_fangspiegel(D_mm, S, 0.0, lam_nm, V)[0])
    q1  = float(qvis_kurve_mit_fangspiegel(D_mm, S, dp,  lam_nm, V)[0])
    r_hut = (hut_mm if hut_mm > 0 else (tubus_mm if tubus_mm > 0 else D_mm)) / 2.0
    b_ok = max(fe_mm * afov_deg / 57.2958, max(bild_mm, 0.0))
    x_bl = A + r_hut
    r_t = max(oaz_d_mm, 1.0) / 2.0
    r_f = b_ok / 2.0
    R_h = r_hut

    def _hit(m, v0):
        """Schnitt der Sichtlinie v = v0 + m·u (Start auf der OAZ-Achse bei
        u=0, Hutring-Mitte bei u=A, Radius R_h) mit dem Hutring (fernere
        Seite) -> halbe Sehne v (Blende ist um den Hutring gebogen)."""
        qa = 1.0 + m * m
        qb = -2.0 * (A - m * v0)
        qc = A * A + v0 * v0 - R_h * R_h
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0:
            return R_h, A + R_h
        u = (-qb + math.sqrt(disc)) / (2.0 * qa)
        return min(abs(v0 + m * u), R_h), u

    m_oaz = (r_t + r_f) / max(oaz_l_mm, 1.0)
    m_keg = 0.5 * D_mm / f_mm
    v_geo, u_geo = _hit(m_oaz, -r_f)
    v_keg, u_keg = _hit(m_keg, r_f)
    v_bl = max(v_geo, v_keg)
    hit = (u_geo, v_geo) if v_geo >= v_keg else (u_keg, v_keg)
    x_flach = A + R_h
    d_geo = 2.0 * (m_oaz * x_flach - r_f)        # flache Blende (zum Vergleich)
    d_kegel = x_flach * D_mm / f_mm + b_ok
    d_bl = 2.0 * v_bl                            # Sehne (Breite in Draufsicht)
    d_bogen = 2.0 * R_h * math.asin(min(v_bl / R_h, 1.0))   # abgewickelte Länge
    w_oaz = math.degrees(math.atan(m_oaz))
    return dict(A=A, d_min=d, d_pass=dp, eps=eps, flaeche=eps**2 * 100.0,
                D_kontrast=D_mm - dp, Q0=q0, Q=q1, Q_verlust=q0 - q1,
                x_blende=x_bl, B_ok=b_ok, d_geo=d_geo, d_kegel=d_kegel, w_oaz=w_oaz, d_bogen=d_bogen,
                R_hut=R_h, d_blende=d_bl, oaz_r=r_t, oaz_l=float(oaz_l_mm),
                m_oaz=m_oaz, m_keg=m_keg, hit=hit, hit_geo=(u_geo, v_geo),
                d_blende_pass=float(math.ceil(d_bl - 1e-9)),
                d_bogen_pass=float(math.ceil(d_bogen - 1e-9)))


def fangspiegel_zeichnen(ax, D, f, lam, S_u, S_txt, r, V):
    """Zeichnet Q_vis(V) für Mindest-Fangspiegel und +10/+20 mm in ax
    (gemeinsam für Desktop und Web, neu in v6.12.0)."""
    Vs = np.linspace(30, 400, 38)
    ax.plot(Vs, np.ones_like(Vs), color="#2e7d32", lw=1.3,
            label="Paraboloid ohne Fangspiegel = 1")
    ax.plot(Vs, qvis_kurve_mit_fangspiegel(D, S_u, 0.0, lam, Vs),
            color="#555", lw=1.5, ls="--", label=f"{S_txt} ohne Fangspiegel")
    for dd, col in [(r["d_pass"], "#6a1b9a"), (r["d_pass"] + 10, "#d84315"),
                    (r["d_pass"] + 20, "#b71c1c")]:
        lab = (f"d = {dd:.0f} mm (Minimum)" if dd == r["d_pass"]
               else f"d = {dd:.0f} mm (+{dd - r['d_pass']:.0f})")
        ax.plot(Vs, qvis_kurve_mit_fangspiegel(D, S_u, dd, lam, Vs),
                color=col, lw=1.6, label=lab)
    ax.axvline(V, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("Vergrößerung V [×]")
    ax.set_ylabel("Q_vis (Paraboloid ohne Fangspiegel = 1)")
    ax.set_title(f"Fangspiegel-Einfluss  —  D={D:.0f}mm  f/{f/D:.1f}", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    lo = min(float(np.min(l.get_ydata())) for l in ax.get_lines())
    ax.set_ylim(max(0.0, lo - 0.05), 1.03)


def fangspiegel_text(D, f, S_u, S_txt, r, V):
    return (
        f"D={D:.0f} mm  f={f:.0f} mm (f/{f/D:.1f})   Abstand Fangspiegel–Fokus A = {r['A']:.0f} mm\n"
        f"Mindest-Ø Fangspiegel: {r['d_min']:.1f} mm  →  nehmen: {r['d_pass']:.0f} mm   "
        f"(Obstruktion {r['eps']*100:.0f} %, Flächenverlust {r['flaeche']:.1f} %)\n"
        f"Q_vis bei {V}× ({S_txt} S={S_u:.3f}):  ohne Fangspiegel {r['Q0']:.4f}  →  "
        f"mit {r['d_pass']:.0f} mm: {r['Q']:.4f}   (Verlust {r['Q_verlust']*100:.1f} %-Punkte)\n"
        f"Effektiver Kontrast-Ø (D − d, Näherung): {r['D_kontrast']:.0f} mm\n"
        f"Blende gegenüber dem OAZ (Hutring-Ø {2*r['R_hut']:.0f} mm, um den Hutring gebogen): "
        f"Fokus–Hutring-Achse {r['x_blende']:.0f} mm, Feld {r['B_ok']:.0f} mm\n"
        f"   Sichtlinie durch das OAZ-Rohr (steilster Winkel {r['w_oaz']:.0f}°); ebene Blende wäre Ø {r['d_geo']:.0f} mm, "
        f"Strahlkegel allein {r['d_kegel']:.0f} mm\n"
        f"   → gebogene Blende: Breite (Sehne) mind. {r['d_blende']:.0f} mm, "
        f"Zuschnitt (Bogenlänge) {r['d_bogen_pass']:.0f} mm")


def fangspiegel_skizze(ax, D, f, r):
    """Draufsicht (Blick entlang der optischen Achse) auf Hutring, Fangspiegel,
    OAZ-Rohr und die gebogene Blende gegenüber dem OAZ, mit Maßen,
    Lichtkegel (Hauptspiegel → Okular) und der steilsten Sichtlinie durch das
    OAZ-Rohr (v7.0.0, gemeinsam für Desktop und Web).
    Koordinaten: u entlang der OAZ-Achse (Feldblende/Fokus bei u = 0),
    v quer dazu; optische Achse bei u = A."""
    A, R, rf, rt = r["A"], r["R_hut"], r["B_ok"] / 2.0, r["oaz_r"]
    L, dF, N = r["oaz_l"], r["d_pass"], f / D
    ax.set_aspect("equal")
    ax.set_title("Draufsicht: Blende gegenüber dem OAZ (Blick entlang der Achse)",
                 fontsize=9)
    # Hutring
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(A + R * np.cos(t), R * np.sin(t), color="#999", lw=1.2)
    ax.text(A, -R - 0.06 * R, f"Hutring Ø {2 * R:.0f}", ha="center", va="top",
            fontsize=7, color="#666")
    # Blende (gebogen) auf der OAZ-abgewandten Seite
    u_h, v_h = r["hit"]
    ph = math.atan2(v_h, u_h - A)
    tb = np.linspace(-ph, ph, 60)
    ax.plot(A + R * np.cos(tb), R * np.sin(tb), color="k", lw=4,
            solid_capstyle="butt")
    ax.text(A + R * 1.04, 0, f"Blende\nSehne {r['d_blende']:.0f}\nBogen {r['d_bogen_pass']:.0f}",
            fontsize=8, va="center", ha="left")
    # Fangspiegel von oben (Ellipse: d√2 längs u, d quer)
    from matplotlib.patches import Ellipse, Rectangle
    ax.add_patch(Ellipse((A, 0), dF * math.sqrt(2), dF, fc="#b3c7e6", ec="#1a4a8a",
                         lw=1.2, zorder=3))
    ax.text(A, 0, f"Fang-\nspiegel\nØ {dF:.0f}", ha="center", va="center", fontsize=7,
            zorder=4)
    # OAZ-Rohr (Wände) von u = -0.35 L (außen) bis u = L (Ende zum Fangspiegel)
    u0 = -0.35 * L
    for s in (-1, 1):
        ax.plot([u0, L], [s * rt, s * rt], color="#444", lw=2)
    ax.plot([L, L], [rt, rt + 0.0], color="#444")
    ax.add_patch(Rectangle((u0 - 0.5 * L, -rf), 0.5 * L, 2 * rf, fc="#ddd", ec="#888",
                           lw=1))
    ax.text(u0 - 0.25 * L, rf + 0.04 * R, "Okular", ha="center", va="bottom", fontsize=7)
    ax.plot([0, 0], [-rf, rf], color="#d84315", lw=3)
    ax.text(0, rf + 0.04 * R, "Feldblende", ha="center", va="bottom", fontsize=7,
            color="#d84315")
    # Lichtkegel: Feldblendenrand → Fangspiegel → (gestrichelt) bis zur Blende
    mk = r["m_keg"]
    for s in (-1, 1):
        ax.plot([0, A], [s * rf, s * (rf + mk * A)], color="#2e7d32", lw=1.4)
        uu = np.array([A, A + R])
        ax.plot(uu, s * (rf + mk * uu), color="#2e7d32", lw=1, ls="--")
    ax.fill([0, A, A, 0], [rf, rf + mk * A, -(rf + mk * A), -rf], color="#2e7d32",
            alpha=0.10, lw=0)
    # steilste Sichtlinie durch das OAZ-Rohr
    m = r["m_oaz"]
    ug = r["hit_geo"][0]
    for s in (-1, 1):
        ax.plot([0, ug], [-s * rf, -s * rf + s * m * ug], color="#c62828", lw=1.4)
    # Maße
    yb = -R * 0.80
    ax.annotate("", xy=(A, yb), xytext=(0, yb),
                arrowprops=dict(arrowstyle="<->", color="#1a4a8a", lw=0.9))
    ax.text(A / 2, yb - 0.03 * R, f"A = {A:.0f}", ha="center", va="top", fontsize=7,
            color="#1a4a8a")
    ya = -rt - 0.10 * R
    ax.annotate("", xy=(L, ya), xytext=(0, ya),
                arrowprops=dict(arrowstyle="<->", color="#6a1b9a", lw=0.9))
    ax.text(L / 2, ya - 0.02 * R, f"L = {L:.0f}", ha="center", va="top", fontsize=7,
            color="#6a1b9a")
    ax.annotate("", xy=(L + 0.04 * L, -rt), xytext=(L + 0.04 * L, rt),
                arrowprops=dict(arrowstyle="<->", color="#6a1b9a", lw=0.9))
    ax.text(L + 0.08 * L, rt * 0.0, f"Ø {2 * rt:.0f}", rotation=90, ha="left", va="center",
            fontsize=7, color="#6a1b9a")
    ax.plot([], [], color="#2e7d32", lw=1.4, label="Lichtkegel Hauptspiegel → Okular")
    ax.plot([], [], color="#c62828", lw=1.4, label="steilste Sichtlinie (Feldblendenrand\n"
            "über OAZ-Rohrende) → Blende")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    ax.set_xlim(u0 - 0.6 * L, A + R * 1.45)
    ax.set_ylim(-R * 1.18, R * 1.18)
    ax.axis("off")


# ── v7.0.0-beta: Vergleich verschiedener Spiegel (absolutes Q_abs) ────────────

def fangspiegel_d(D_mm: float, f_mm: float, buet_mm: float = 50.0,
                  tubus_mm: float = 0.0, bild_mm: float = 10.0) -> float:
    """Nur die Geometrie: Fangspiegel-Ø aufgerundet auf ganze mm (v7.0.0)."""
    tar = (tubus_mm if tubus_mm > 0 else D_mm) / 2.0
    d = (tar + max(buet_mm, 0.0)) * D_mm / f_mm + max(bild_mm, 0.0)
    return float(math.ceil(d - 1e-9))


def seeing_mtf(nu_cyc_rad, seeing_arcsec: float, lam_nm: float = 550.0):
    """Langzeit-Seeing-MTF (Fried):  exp(−3,44·(λν/r0)^(5/3)),
    r0 = 0,98·λ/FWHM.  seeing_arcsec <= 0 → 1 (aus).
    Langzeit-Belichtung = eher pessimistisch für den visuellen Eindruck."""
    nu = np.asarray(nu_cyc_rad, dtype=float)
    if seeing_arcsec <= 0:
        return np.ones_like(nu)
    lam_mm = lam_nm * 1e-6
    r0 = 0.98 * lam_mm / (seeing_arcsec / 206265.0)
    return np.exp(-3.44 * (lam_mm * nu / r0) ** (5.0 / 3.0))


_QABS_N = 100
_QABS_MSR = {}
_QABS_MO = {}


def _qabs_msr(S: float) -> np.ndarray:
    key = round(float(S), 4)
    if key not in _QABS_MSR:
        Wb = _Wb_von_S(min(max(key, 1e-6), 1.0))
        nu = np.linspace(0.0, 1.0, _QABS_N)
        _QABS_MSR[key] = (np.ones_like(nu) if Wb <= 1e-9
                          else np.array([mtf_sph_rel(x, Wb) for x in nu]))
    return _QABS_MSR[key]


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


def _qabs_basis(D_mm, d_fang_mm, V, lam_nm, seeing, mit_mo=True):
    """Strehl-unabhängiger Teil: (x_ang [cyc/″], Gewicht) für Spiegel-Ø D."""
    fc = D_mm / (lam_nm * 1e-6 * 206265.0)             # cyc/arcsec
    # Nur der vom Auge sichtbare Bereich (≤70 cyc/° am Auge) wird integriert: sonst wird bei
    # großem D / kleinem V das Gitter zu grob (CSF-Gewicht sitzt bei ν ≪ 1).
    nu_max = min(1.0, 70.0 * V / 3600.0 / fc)
    nu = np.linspace(0.0, nu_max, _QABS_N)
    x  = nu * fc
    w  = np.array([csf(v * 3600.0 / V) for v in x])
    see = seeing_mtf(x * 206265.0, seeing, lam_nm)
    return x, (_qabs_mo(d_fang_mm / D_mm) if mit_mo else 1.0) * w * see, nu


def mtf_aus_wellenfront(rho, W_c, eps=0.0, Np=128, nu=None) -> np.ndarray:
    """MTF aus einer rotationssymmetrischen Wellenfront W(ρ) [λ] (v7.0.0-beta).

    Pupille P = Maske(eps ≤ ρ ≤ 1)·exp(i·2π·W(ρ)); OTF = Autokorrelation von P
    (über FFT, Auffüllung ×2), normiert auf OTF(0); Rückgabe auf dem
    ν/ν_c-Gitter linspace(0,1,_QABS_N).  Enthält Fangspiegel (eps) und
    beliebige Fehlerform (Zonen, Kante) exakt — im Gegensatz zum S-Modell."""
    rho = np.asarray(rho, float); W_c = np.asarray(W_c, float)
    ax = (np.arange(Np) - Np / 2.0 + 0.5) / (Np / 2.0)
    X, Y = np.meshgrid(ax, ax)
    R = np.hypot(X, Y)
    Wg = np.interp(R, rho, W_c)
    P = ((R <= 1.0) & (R >= eps)) * np.exp(2j * np.pi * Wg)
    M = 2 * Np
    F = np.fft.fft2(P, s=(M, M))
    ac = np.fft.ifft2(np.abs(F) ** 2)
    otf = np.real(ac[0, :Np]) / max(float(np.real(ac[0, 0])), 1e-12)
    nu_k = np.arange(Np) / float(Np)
    if nu is None:
        nu = np.linspace(0.0, 1.0, _QABS_N)
    return np.clip(np.interp(nu, nu_k, otf, right=0.0), 0.0, 1.0)


def mtf_aus_wellenfront_pupille(rho, W_c, eps=0.0, rho_s=1.0) -> np.ndarray:
    """Wie mtf_aus_wellenfront, aber das Auge nutzt nur die zentrale Teilöffnung
    ρ ≤ rho_s (Austrittspupille größer als Augenpupille ⇒ das Auge blendet das
    Teleskop ab).  Das Auge fokussiert neu: Defokus a·ρ² wird für die Teilöffnung
    (Ring eps…rho_s) varianzminimal nachgeführt.  MTF-Gitter normiert auf die
    Grenzfrequenz der Teilöffnung (v7.0.0-beta)."""
    if rho_s >= 0.999:
        return mtf_aus_wellenfront(rho, W_c, eps)
    rp = np.linspace(0.0, 1.0, 400)
    Ws = np.interp(rp * rho_s, np.asarray(rho, float), np.asarray(W_c, float))
    e2 = min(eps / max(rho_s, 1e-6), 0.95)
    w = rp * (rp >= e2)
    x = rp ** 2
    sw = max(float(np.sum(w)), 1e-12)
    mx = float(np.sum(w * x)) / sw
    mW = float(np.sum(w * Ws)) / sw
    a = -float(np.sum(w * (x - mx) * (Ws - mW))) / max(float(np.sum(w * (x - mx) ** 2)), 1e-12)
    return mtf_aus_wellenfront(rp, Ws + a * x, e2)


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
                 mtf_nu=None, d_pupil=0.0, wf=None) -> float:
    """∫ MTF_ges(ν) · CSF(ν/V) dν  über die WINKELfrequenz (absolut).
    d_pupil: Augenpupille [mm] (0 = aus): bei Austrittspupille D/V > d_pupil
    nutzt das Auge nur die Teilöffnung V·d_pupil (blendet den Spiegel ab).
    wf: optionale Foucault-Wellenfront; mtf_nu: fertige Gesamt-MTF (nur ohne Pupille)."""
    _tr = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if mtf_nu is not None and wf is None and _rho_s(D_mm, V, d_pupil) >= 0.999:
        x, base, nu = _qabs_basis(D_mm, d_fang_mm, V, lam_nm, seeing, mit_mo=False)
        return float(_tr(np.interp(nu, _QABS_NUG, mtf_nu) * base, x))
    m, D_use = _mtf_eff(D_mm, S, d_fang_mm, V, d_pupil, wf, lam_nm)
    x, base, nu = _qabs_basis(D_use, d_fang_mm, V, lam_nm, seeing, mit_mo=False)
    return float(_tr(np.interp(nu, _QABS_NUG, m) * base, x))


def qabs(D_mm, S, d_fang_mm, V, ref_D=200.0, ref_S=1.0, ref_d=0.0,
         lam_nm=550.0, seeing=0.0, mtf_nu=None, d_pupil=0.0, wf=None) -> float:
    """Absolutes Q_abs: sichtbare Detailübertragung relativ zu einem festen
    Referenzspiegel (Standard: idealer 200-mm ohne Fangspiegel = 1; der
    Referenzspiegel wird von derselben Augenpupille abgeblendet)."""
    return (qabs_zaehler(D_mm, S, d_fang_mm, V, lam_nm, seeing, mtf_nu, d_pupil, wf) /
            max(qabs_zaehler(ref_D, ref_S, ref_d, V, lam_nm, seeing, None, d_pupil), 1e-12))


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
    """Augen-Tiefpass bei Vergrößerung V (v7.0.0-beta): Barten-CSF normiert auf ihr
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

VERSION  = "7.0.0-beta (2026-10-29)"
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
    "sl_lam":    {"de": "Wellenlänge λ",   "en": "Wavelength λ"},
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
    "rVkrit_naeh":  {"de": "V_krit  Näherung  = D/(13·Wb)", "en": "V_crit  approx.  = D/(13·Wb)"},
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
    "tab_blende":      {"de": "  Blende  ",                 "en": "  Stop-down  "},
    "tab_foucault":    {"de": "  Foucault  ",               "en": "  Foucault  "},
    "tab_vmax":        {"de": "  Max. Vergrößerung  ",       "en": "  Max. magnification  "},
    "tab_vmax_D":      {"de": "  Vmax vs. Ø/Seeing  ",        "en": "  Vmax vs. D/seeing  "},
    "tab_fangspiegel": {"de": "  Fangspiegel  ",              "en": "  Secondary mirror  "},
    "tab_vergleich":   {"de": "  Vergleich (Beta)  ",         "en": "  Comparison (beta)  "},
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
        self._maximiere_fenster()
        self.configure(bg=BG)
        self._update_job      = None
        self._compute_cancel  = False
        self._last_params     = None   # zuletzt berechnete Parameter
        self._dirty_tabs      = set()  # Tabs die beim nächsten Wechsel neu gezeichnet werden
        self._build_ui()
        # Tab-Wechsel: bei Bedarf das neue Tab nachzeichnen
        self._nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._aktualisieren()

    def _maximiere_fenster(self):
        """Startet das Fenster maximiert (Bildschirmgröße) statt in Default-Breite.

        '-zoomed' funktioniert unter Linux/X11 (Ubuntu) und Windows; falls das
        Attribut nicht unterstützt wird (z.B. manche macOS/Wayland-Setups),
        wird ersatzweise die Geometrie manuell auf die Bildschirmgröße gesetzt.
        """
        try:
            self.attributes("-zoomed", True)
        except tk.TclError:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"{sw}x{sh}+0+0")

    def _sprache_wechseln(self, lang):
        global _LANG
        _LANG = lang
        self.title(T("win_title"))
        self._texte_aktualisieren()
        self._aktualisieren()

    def _texte_aktualisieren(self):
        """Alle statischen Label-Texte nach Sprachwechsel neu setzen."""
        # Slider-Namen
        for key, widget in self._param_name_widgets.items():
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
        self._param_name_widgets = {}

        self.var_D      = tk.DoubleVar(value=200)
        self.var_f      = tk.DoubleVar(value=1000)
        self.var_strehl = tk.DoubleVar(value=0)
        self.var_blende = tk.DoubleVar(value=150)
        self.var_lam    = tk.DoubleVar(value=550)   # fix 550 nm, kein Eingabefeld

        # (key, var, min, max, unit, info-label-breite)
        param_defs = [
            ("sl_D",      self.var_D,        50,   500,  "mm",  14),
            ("sl_f",      self.var_f,        200,  4000, "mm",  14),
            ("sl_strehl", self.var_strehl,   0,    100,  "%",   22),
        ]
        self._param_labels = {}   # key → (lbl_val, unit)
        self._param_vars   = {}   # key → (var, mn, mx)
        for i, (key, var, mn, mx, unit, info_w) in enumerate(param_defs, 2):
            lbl_name = tk.Label(left, text=T(key), bg=BG, width=16, anchor="w")
            lbl_name.grid(row=i, column=0, sticky="w", **pad)
            self._param_name_widgets[key] = lbl_name

            # Eingabefeld + Einheit in einem kleinen Frame
            unit_frm = tk.Frame(left, bg=BG)
            unit_frm.grid(row=i, column=1, sticky="w", **pad)

            ent = tk.Entry(unit_frm, textvariable=var, width=9,
                           font=("Helvetica", 10), justify="right",
                           relief="solid", bd=1)
            ent.grid(row=0, column=0)
            ent.bind("<Return>",   self._param_changed)
            ent.bind("<KP_Enter>", self._param_changed)

            tk.Label(unit_frm, text=f" {unit}", bg=BG, fg="#555",
                     font=("Helvetica", 10)).grid(row=0, column=1)

            lbl_val = tk.Label(left, text="", bg=BG, width=info_w, anchor="w",
                               font=("Helvetica", 9), fg="#555")
            lbl_val.grid(row=i, column=2, sticky="w", **pad)
            self._param_labels[key] = (lbl_val, unit)
            self._param_vars[key]   = (var, mn, mx)

        row = len(param_defs) + 2   # 5 Parameter + Titelzeile (row=1) + Startoffset

        # Hinweis + Berechnen-Button
        hint_frm = tk.Frame(left, bg=BG)
        hint_frm.grid(row=row, column=0, columnspan=3, sticky="w",
                      padx=10, pady=(2, 6))
        tk.Label(hint_frm, text="⏎  Wert eingeben und Enter drücken",
                 bg=BG, fg="#888", font=("Helvetica", 8, "italic")).pack(side="left")
        tk.Button(hint_frm, text="  Berechnen  ",
                  command=self._param_changed,
                  font=("Helvetica", 9), bg=ACC, fg="white",
                  relief="flat", padx=6, pady=2).pack(side="left", padx=(16, 0))
        row += 1

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
        self._mtf_label_widgets = {}

        tab_attrs = [
            ("_tab_strehl",       "tab_strehl"),
            ("_tab_oeff",         "tab_oeff"),
            ("_tab_deff_D",       "tab_deff_D"),
            ("_tab_beugung",      "tab_beugung"),
            ("_tab_mtf",          "tab_mtf"),
            ("_tab_wahrnehmung",  "tab_wahrnehmung"),
            ("_tab_blende",       "tab_blende"),
            ("_tab_foucault",     "tab_foucault"),
            ("_tab_vmax",         "tab_vmax"),
            ("_tab_vmax_D",       "tab_vmax_D"),
            ("_tab_fangspiegel",  "tab_fangspiegel"),
            ("_tab_vergleich",    "tab_vergleich"),
        ]
        self._tab_keys = [k for _, k in tab_attrs]
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
                # ── Ablese-Leiste: Cursor-Vergrößerung ──
                wn = tk.Frame(frm, bg=BG)
                wn.pack(side="top", fill="x", padx=8, pady=(6, 0))
                tk.Label(wn, text="Ablesen bei V =", bg=BG, fg="#111",
                         font=("Helvetica", 9, "bold")).pack(side="left")
                self._wn_V_var = tk.StringVar(value="150")
                e_v = tk.Entry(wn, textvariable=self._wn_V_var, width=6,
                               font=("Helvetica", 9))
                e_v.pack(side="left", padx=(3, 0))
                tk.Label(wn, text="×   (oder ins Diagramm klicken;  Spiegel = Regler Öffnung / Brennweite / Strehl-Schieber)",
                         bg=BG, fg="#666", font=("Helvetica", 9)).pack(side="left", padx=(2, 14))

                wn2 = tk.Frame(frm, bg=BG)
                wn2.pack(side="top", fill="x", padx=8, pady=(2, 0))
                tk.Label(wn2, text="Fangspiegel Ø d =", bg=BG, fg="#6a1b9a",
                         font=("Helvetica", 9, "bold")).pack(side="left")
                self._wn_fang_var = tk.StringVar(value="0")
                e_fs = tk.Entry(wn2, textvariable=self._wn_fang_var, width=6,
                                font=("Helvetica", 9))
                e_fs.pack(side="left", padx=(3, 0))
                self._wn_fang_hint = tk.Label(wn2, text="mm   (0 = aus)", bg=BG,
                                              fg="#666", font=("Helvetica", 9))
                self._wn_fang_hint.pack(side="left", padx=(2, 0))
                e_fs.bind("<Return>",   self._wahrnehmung_neu)
                e_fs.bind("<FocusOut>", self._wahrnehmung_neu)
                for _e in (e_v,):
                    _e.bind("<Return>",   self._wahrnehmung_neu)
                    _e.bind("<FocusOut>", self._wahrnehmung_neu)

                self._wn_readout = tk.Label(frm, text="", bg=BG, fg="#222",
                                            font=("Helvetica", 9), justify="left",
                                            anchor="w")
                self._wn_readout.pack(side="top", fill="x", padx=8, pady=(2, 0))

                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)

                def _wn_klick(ev, _ax=ax, _self=self):
                    # Klick ins Diagramm setzt die Ablese-Vergrößerung
                    if ev.inaxes is _ax and ev.xdata is not None and ev.button == 1:
                        _self._wn_V_var.set(str(int(round(float(ev.xdata)))))
                        _self._wahrnehmung_neu()
                canvas.mpl_connect("button_press_event", _wn_klick)
                setattr(self, attr, (fig, ax, canvas))
            elif attr == "_tab_blende":
                fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.0), dpi=96)
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG); ax2.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, ax2, canvas))

            elif attr == "_tab_foucault":
                # ── Steuerleiste oben ──────────────────────────────────────
                ctrl = tk.Frame(frm, bg=BG)
                ctrl.pack(side="top", fill="x", padx=8, pady=(6, 2))

                tk.Label(ctrl, text="Zonen:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._foucault_n = tk.IntVar(value=4)
                for n in [4, 6, 8]:
                    tk.Radiobutton(ctrl, text=str(n), variable=self._foucault_n,
                                   value=n, bg=BG, activebackground=BG,
                                   selectcolor=BG2, font=("Helvetica", 9),
                                   command=self._foucault_eingabe_neu
                                   ).pack(side="left", padx=4)

                tk.Label(ctrl, text="    λ = 550 nm (fix)", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left", padx=8)
                self._foucault_lam = tk.DoubleVar(value=550.0)

                # Zeile 2: Auswertungs-Modus + Status
                ctrl2 = tk.Frame(frm, bg=BG)
                ctrl2.pack(side="top", fill="x", padx=8, pady=(0, 2))

                tk.Label(ctrl2, text="Auswertung:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._foucault_use_direct = tk.BooleanVar(value=False)
                tk.Radiobutton(ctrl2,
                               text="Polynom-Fit  (glatte Flaeche)",
                               variable=self._foucault_use_direct,
                               value=False, bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#1565c0",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=6)
                tk.Radiobutton(ctrl2,
                               text="Hybrid  (Zonenfehler)",
                               variable=self._foucault_use_direct,
                               value=True, bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#6a1b9a",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=6)

                self._foucault_werkstatt = tk.BooleanVar(value=True)
                tk.Checkbutton(ctrl2,
                               text="🔧 Werkstatt-Ansicht (wie FigureXP)",
                               variable=self._foucault_werkstatt,
                               bg=BG, activebackground=BG, selectcolor=BG2,
                               fg="#E65100", font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=6)

                tk.Label(ctrl2, text="  Grad:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left", padx=(8, 0))
                self._foucault_grad_modus = tk.StringVar(value="bic")
                tk.Radiobutton(ctrl2,
                               text="BIC (auto)",
                               variable=self._foucault_grad_modus,
                               value="bic", bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#555",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=4)
                tk.Radiobutton(ctrl2,
                               text="Fest (n//2)",
                               variable=self._foucault_grad_modus,
                               value="fest", bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#555",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=4)

                self._foucault_status = tk.Label(ctrl2, text="", bg=BG,
                                                  font=("Helvetica", 9, "bold"))
                self._foucault_status.pack(side="left", padx=12)

                # ── Zonenradien-Info ──────────────────────────────────────
                self._foucault_info = tk.Label(frm, text="", bg=BG, fg="#555",
                                               font=("Helvetica", 8), anchor="w")
                self._foucault_info.pack(side="top", fill="x", padx=8)

                # ── ΔF-Eingabefelder (dynamisch, max 8 Zonen) ─────────────
                eingabe_frm = tk.Frame(frm, bg=BG)
                eingabe_frm.pack(side="top", fill="x", padx=8, pady=(4, 2))
                self._foucault_eingabe_frm = eingabe_frm
                self._foucault_df_vars = []   # wird in _foucault_eingabe_neu befüllt

                # ── Diagramm ──────────────────────────────────────────────
                fig, (ax_wf, ax_info) = plt.subplots(
                    1, 2, figsize=(9.0, 4.2), dpi=96,
                    gridspec_kw={"width_ratios": [3, 2]})
                fig.patch.set_facecolor(BG)
                ax_wf.set_facecolor(BG); ax_info.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax_wf, ax_info, canvas))

                # Eingabefelder initial aufbauen
                self._foucault_eingabe_neu()

            elif attr == "_tab_vmax":
                # ── Steuerleiste ──────────────────────────────────────────
                ctrlv = tk.Frame(frm, bg=BG)
                ctrlv.pack(side="top", fill="x", padx=8, pady=(6, 2))

                tk.Label(ctrlv, text="Objekt:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_objekt = tk.StringVar(value=OBJEKT_HELLIGKEITEN[1][0])
                objekt_namen = [n for n, _ in OBJEKT_HELLIGKEITEN]
                om = tk.OptionMenu(ctrlv, self._vmax_objekt, *objekt_namen,
                                   command=lambda *_: self._vmax_objekt_gewaehlt())
                om.config(bg=BG, fg="#222", font=("Helvetica", 9),
                          highlightthickness=0)
                om.pack(side="left", padx=(4, 12))

                tk.Label(ctrlv, text="L₀ [cd/m²]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_L0 = tk.DoubleVar(value=6000.0)
                tk.Entry(ctrlv, textvariable=self._vmax_L0, width=8,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 12))
                tk.Button(ctrlv, text="↻ Update", font=("Helvetica", 8),
                          command=self._aktualisieren
                          ).pack(side="left", padx=(0, 4))

                ctrlv2 = tk.Frame(frm, bg=BG)
                ctrlv2.pack(side="top", fill="x", padx=8, pady=(0, 4))

                tk.Label(ctrlv2, text="Seeing FWHM [\"]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_seeing = tk.DoubleVar(value=1.5)
                tk.Entry(ctrlv2, textvariable=self._vmax_seeing, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 12))

                tk.Label(ctrlv2, text="Augenpupille [mm]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_D_auge = tk.DoubleVar(value=6.0)
                tk.Entry(ctrlv2, textvariable=self._vmax_D_auge, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 12))

                tk.Label(ctrlv2, text="Helligkeitsschwelle [cd/m²]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_L_schwelle = tk.DoubleVar(value=1.0)
                tk.Entry(ctrlv2, textvariable=self._vmax_L_schwelle, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 4))

                # ── Diagramm ──────────────────────────────────────────────
                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

            elif attr == "_tab_vmax_D":
                # ── Steuerleiste ──────────────────────────────────────────
                ctrld = tk.Frame(frm, bg=BG)
                ctrld.pack(side="top", fill="x", padx=8, pady=(6, 4))

                tk.Label(ctrld, text="Seeing-Werte [\"] (kommagetrennt):",
                         bg=BG, fg="#555", font=("Helvetica", 9)
                         ).pack(side="left")
                self._vmaxD_seeings = tk.StringVar(value="0.8, 1.5, 2.5, 4.0")
                tk.Entry(ctrld, textvariable=self._vmaxD_seeings, width=24,
                         font=("Helvetica", 9)).pack(side="left", padx=(4, 12))

                tk.Label(ctrld, text="D-Bereich [mm]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmaxD_Dmin = tk.DoubleVar(value=50.0)
                tk.Entry(ctrld, textvariable=self._vmaxD_Dmin, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 2))
                tk.Label(ctrld, text="–", bg=BG).pack(side="left")
                self._vmaxD_Dmax = tk.DoubleVar(value=500.0)
                tk.Entry(ctrld, textvariable=self._vmaxD_Dmax, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 4))

                # ── Diagramm ──────────────────────────────────────────────
                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

            elif attr == "_tab_vergleich":
                self._vg_vars = {}
                def _zeile(felder):
                    fr = tk.Frame(frm, bg=BG)
                    fr.pack(side="top", fill="x", padx=8, pady=(4, 0))
                    for lbl, key, val, hint in felder:
                        tk.Label(fr, text=lbl, bg=BG, fg="#1565c0",
                                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 3))
                        v = tk.StringVar(value=val); self._vg_vars[key] = v
                        e = tk.Entry(fr, textvariable=v, width=6, font=("Helvetica", 9))
                        e.pack(side="left")
                        e.bind("<Return>",   self._vergleich_neu)
                        e.bind("<FocusOut>", self._vergleich_neu)
                        tk.Label(fr, text=hint, bg=BG, fg="#666",
                                 font=("Helvetica", 8)).pack(side="left", padx=(2, 12))
                _zeile([("Spiegel 1  D =", "D1", "200", "mm"), ("S =", "S1", "1.00", ""),
                        ("Spiegel 2  D =", "D2", "300", "mm"), ("S =", "S2", "0.80", ""),
                        ("f/D =", "N", "5", "(Fangspiegel automatisch)")])
                _zeile([("V =", "V", "200", "×"), ("Seeing", "seeing", "1.5", "″  (0 = aus)"),
                        ("Referenz: idealer", "refD", "200", "mm"),
                        ("Okularauszug", "buet", "50", "mm"),
                        ("Bildfeld", "bild", "10", "mm"),
                        ("Augenpupille", "pupille", "0", "mm (0 = aus; Mond 2–3, Planet 3–5, Deep-Sky 6–7)")])
                fr_wf = tk.Frame(frm, bg=BG); fr_wf.pack(side="top", fill="x", padx=8, pady=(4, 0))
                self._vg_wf1 = tk.BooleanVar(value=False)
                self._vg_wf2 = tk.BooleanVar(value=False)
                for _v, _t in [(self._vg_wf1, "Spiegel 1 aus Foucault-Test (Wellenfront)"),
                               (self._vg_wf2, "Spiegel 2 aus Foucault-Test (Wellenfront)")]:
                    tk.Checkbutton(fr_wf, text=_t, variable=_v, bg=BG, fg="#6a1b9a",
                                   activebackground=BG, selectcolor=BG2,
                                   font=("Helvetica", 9, "bold"),
                                   command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_sph1 = tk.BooleanVar(value=False)
                self._vg_sph2 = tk.BooleanVar(value=False)
                for _v, _t in [(self._vg_sph1, "Spiegel 1 sphärisch (aus D, f/D)"),
                               (self._vg_sph2, "Spiegel 2 sphärisch (aus D, f/D)")]:
                    tk.Checkbutton(fr_wf, text=_t, variable=_v, bg=BG, fg="#6a1b9a",
                                   activebackground=BG, selectcolor=BG2,
                                   font=("Helvetica", 9, "bold"),
                                   command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_auge = tk.BooleanVar(value=True)
                tk.Checkbutton(fr_wf, text="Bild mit Augen-Filter bei V", variable=self._vg_auge,
                               bg=BG, fg="#1565c0", activebackground=BG, selectcolor=BG2,
                               font=("Helvetica", 9, "bold"),
                               command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_mass = tk.BooleanVar(value=True)
                tk.Checkbutton(fr_wf, text="Augenmaßstab (Planet wächst mit V)", variable=self._vg_mass,
                               bg=BG, fg="#1565c0", activebackground=BG, selectcolor=BG2,
                               font=("Helvetica", 9, "bold"),
                               command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_readout = tk.Label(frm, text="", bg=BG, fg="#222",
                                            font=("Helvetica", 10), justify="left",
                                            anchor="w")
                self._vg_readout.pack(side="top", fill="x", padx=8, pady=(4, 0))
                fig = plt.figure(figsize=(11.0, 7.4), dpi=96)
                fig.patch.set_facecolor(BG)
                ax1, ax2, _axb, _axw = vergleich_achsen(fig)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax1, ax2, canvas))

            elif attr == "_tab_fangspiegel":
                fsr = tk.Frame(frm, bg=BG)
                fsr.pack(side="top", fill="x", padx=8, pady=(6, 0))
                self._fs_vars = {}
                for lbl, key, val, hint in [
                        ("Okularauszug (Fokus über Tubus)", "buet", "50", "mm"),
                        ("Tubus-Außen-Ø (0 = wie Spiegel)", "tubus", "0", "mm"),
                        ("100%-Bildfeld", "bild", "10", "mm  (visuell 10, Foto 20–25)"),
                        ("Q_vis bei V =", "V", "150", "×"),
                        ("Hutring-Ø (0 = Tubus-Ø)", "hut", "0", "mm"),
                        ("Übersichtsokular f", "fe", "21", "mm"),
                        ("AFOV", "afov", "85", "°"),
                        ("OAZ-Rohr Innen-Ø", "oazd", "50", "mm"),
                        ("OAZ-Rohr Länge", "oazl", "80", "mm")]:
                    if key == "hut":
                        fsr = tk.Frame(frm, bg=BG)
                        fsr.pack(side="top", fill="x", padx=8, pady=(2, 0))
                    tk.Label(fsr, text=lbl, bg=BG, fg="#6a1b9a",
                             font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 3))
                    v = tk.StringVar(value=val)
                    self._fs_vars[key] = v
                    e = tk.Entry(fsr, textvariable=v, width=5, font=("Helvetica", 9))
                    e.pack(side="left")
                    e.bind("<Return>",   self._fangspiegel_neu)
                    e.bind("<FocusOut>", self._fangspiegel_neu)
                    tk.Label(fsr, text=hint, bg=BG, fg="#666",
                             font=("Helvetica", 8)).pack(side="left", padx=(2, 10))
                self._fs_readout = tk.Label(frm, text="", bg=BG, fg="#222",
                                            font=("Helvetica", 10), justify="left",
                                            anchor="w")
                self._fs_readout.pack(side="top", fill="x", padx=8, pady=(4, 0))
                fig = plt.figure(figsize=(11.0, 5.0), dpi=96)
                fig.patch.set_facecolor(BG)
                ax = fig.add_subplot(1, 2, 1); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

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

    def _param_changed(self, _=None):
        """Werte klemmen und Neuberechnung starten (nur auf Enter)."""
        for key, (var, mn, mx) in self._param_vars.items():
            try:
                v = var.get()
                if v < mn: var.set(mn)
                elif v > mx: var.set(mx)
            except tk.TclError:
                var.set(mn)
        # laufenden Job abbrechen und sofort neu starten
        if getattr(self, "_update_job", None):
            self.after_cancel(self._update_job)
            self._update_job = None
        self._start_update()

    def _start_update(self):
        """Startet Berechnung im Hintergrund-Thread."""
        self._compute_cancel = True          # laufenden Thread abbrechen
        params = (self.var_D.get(), self.var_f.get(), self.var_lam.get(),
                  self.var_blende.get(), self.var_strehl.get())
        self._compute_cancel = False
        import threading
        threading.Thread(target=self._compute_thread,
                         args=(params,), daemon=True).start()

    def _compute_thread(self, params):
        """Alle schweren Berechnungen im Hintergrund (kein tkinter)."""
        try:
            D, f, lam, D_bl_raw, S_pct = params
            D_bl = min(D_bl_raw, D - 10)

            r_real       = berechne(D, f, lam)
            S_real       = r_real["strehl"]
            S_slide      = S_real + (S_pct / 100.0) * (1.0 - S_real)
            S_slide      = min(max(S_slide, S_real), 1.0)
            Deff_k_slide = D * S_slide**0.25
            Vk           = v_kritisch(D, f, lam, EYE_RES)
            Vk_slide     = v_kritisch_strehl(D, S_slide)
            rv           = berechne_vergr(D, f, lam, 150.0, EYE_RES)
            Vk_naeh      = Vk["Vk_sph"]
            Vk_blur      = v_krit_blur_direkt(D, f, EYE_RES)
            Vk_exakt     = v_krit_aus_qvis(D, f, lam, rel_schwelle=0.90)
            Vk_halb      = v_halb_exakt(D, f, lam)
            Wb_s         = _Wb_von_S(S_slide)
            Vk_exakt_s   = v_krit_aus_qvis_von_Wb(D, Wb_s, lam, rel_schwelle=0.90)
            Qp_80        = perceived_quality(D, f, lam, 30.0)
            Qp_200       = perceived_quality(D, f, lam, 80.0)
            Qp_350       = perceived_quality(D, f, lam, 160.0)
            Qp_V         = perceived_quality(D, f, lam, 150.0)

            if self._compute_cancel:
                return

            res = dict(D=D, f=f, lam=lam, D_bl=D_bl, S_pct=S_pct,
                       r_real=r_real, S_real=S_real, S_slide=S_slide,
                       Deff_k_slide=Deff_k_slide, Vk=Vk, Vk_slide=Vk_slide,
                       rv=rv, Vk_naeh=Vk_naeh, Vk_blur=Vk_blur,
                       Vk_exakt=Vk_exakt, Vk_halb=Vk_halb,
                       Vk_exakt_s=Vk_exakt_s,
                       Qp_80=Qp_80, Qp_200=Qp_200, Qp_350=Qp_350, Qp_V=Qp_V)
            self.after(0, lambda r=res: self._aktualisieren(r))
        except Exception as exc:
            import traceback
            msg = traceback.format_exc()
            self.after(0, lambda m=msg: self._zeige_fehler(m))

    def _on_tab_changed(self, _=None):
        """Zeichnet den neu sichtbaren Tab (immer, nicht nur wenn dirty)."""
        if not self._last_params:
            return
        active = self._aktiver_tab()
        if active:
            self._dirty_tabs.discard(active)
            self._diagramm_fuer_tab(active, self._last_params)

    def _aktiver_tab(self) -> str:
        """Gibt den Tab-Key des aktuell sichtbaren Tabs zurück."""
        try:
            selected = self._nb.select()
            if not selected:
                return self._tab_keys[0]
            idx = self._nb.index(selected)
            return self._tab_keys[idx]
        except Exception:
            return self._tab_keys[0] if self._tab_keys else ""

    def _diagramm_fuer_tab(self, tab_key: str, r: dict):
        """Zeichnet genau das Diagramm für tab_key."""
        D, f, lam   = r["D"], r["f"], r["lam"]
        S_real      = r["S_real"]
        S_slide     = r["S_slide"]
        Deff_k_slide= r["Deff_k_slide"]
        D_bl        = r["D_bl"]
        r_real      = r["r_real"]
        if tab_key == "tab_strehl":
            self._diagramm_strehl(D, f, lam, r_real["N"], S_real, S_slide)
        elif tab_key == "tab_oeff":
            self._diagramm_oeffnung(D, f, lam, r_real["N"], r_real["Deff_k"],
                                    S_sph=S_real, Dk_slide=Deff_k_slide)
        elif tab_key == "tab_deff_D":
            self._diagramm_deff_D(D, r_real["N"], S_slide)
        elif tab_key == "tab_beugung":
            self._diagramm_beugung(D, f)
        elif tab_key == "tab_mtf":
            self._diagramm_mtf(D, f, lam, S_slide)
        elif tab_key == "tab_wahrnehmung":
            self._diagramm_wahrnehmung(D, f, lam, S_slide=S_slide, S_real=S_real)
        elif tab_key == "tab_blende":
            self._diagramm_blende(D, f, lam, D_bl)
        elif tab_key == "tab_foucault":
            if hasattr(self, "_foucault_df_vars"):
                D_prev = getattr(self, "_foucault_D_prev", None)
                if D_prev != D:
                    self._foucault_D_prev = D
                    self._foucault_eingabe_neu()
                else:
                    self._foucault_aktualisieren()
        elif tab_key == "tab_vmax":
            self._diagramm_vmax(D, f, S_slide)
        elif tab_key == "tab_vmax_D":
            self._diagramm_vmax_D()
        elif tab_key == "tab_fangspiegel":
            self._diagramm_fangspiegel(D, f, lam, S_slide, S_real)
        elif tab_key == "tab_vergleich":
            self._diagramm_vergleich(lam)

    def _vg_eingaben(self):
        """Eingaben des Vergleich-Tabs (Komma/Punkt, Fallback bei Fehler)."""
        std = dict(D1=200, S1=1.0, D2=300, S2=0.8, N=5, V=200, seeing=1.5,
                   refD=200, buet=50, bild=10, pupille=0.0)
        out = {}
        for k, v in self._vg_vars.items():
            try:
                x = float(v.get().replace(",", ".").strip())
                if k in ("D1", "D2", "refD", "N", "V") and x <= 0: raise ValueError
                if k in ("S1", "S2"): x = min(max(x, 0.05), 1.0)
                if x < 0: raise ValueError
            except ValueError:
                x = std[k]
            if k not in ("S1", "S2", "N", "seeing", "pupille"):
                x = int(round(x))
            out[k] = x
            v.set(f"{x:.2f}" if k in ("S1", "S2") else f"{x:g}")
        return out

    def _vergleich_neu(self, *_):
        try:
            self._diagramm_vergleich(getattr(self, "_last_params", {}).get("lam", 550.0))
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _diagramm_vergleich(self, lam):
        """v7.0.0-beta: Q_abs-Vergleich zweier Spiegel + D×S-Karte; optional
        mit der Wellenfront aus dem Foucault-Test (Spiegel 1 und/oder 2)."""
        fig, ax1, ax2, canvas = self._tab_vergleich
        e = self._vg_eingaben()
        notiz = []
        wf = {}
        kz = getattr(self, "_foucault_letzte_kz", None)
        if (self._vg_wf1.get() or self._vg_wf2.get()) and kz is None \
                and hasattr(self, "_foucault_df_vars"):
            try:
                self._foucault_aktualisieren()
                kz = getattr(self, "_foucault_letzte_kz", None)
            except Exception:
                kz = None
        if kz is None:
            if self._vg_wf1.get() or self._vg_wf2.get():
                notiz.append("Keine Foucault-Daten vorhanden — erst im Foucault-Tab Messwerte eingeben.")
            self._vg_wf1.set(False); self._vg_wf2.set(False)
        else:
            Df = float(getattr(self, "_foucault_letzte_D", e["D1"]))
            wfd = dict(rho=kz["rho_f"], W=kz["W_c"], lam=kz.get("lam_nm") or 550.0)
            for k, var in (("1", self._vg_wf1), ("2", self._vg_wf2)):
                if var.get():
                    wf[k] = wfd
                    e["D" + k] = Df
                    e["S" + k] = float(kz["S_exakt"])
                    self._vg_vars["D" + k].set(f"{Df:g}")
                    self._vg_vars["S" + k].set(f"{kz['S_exakt']:.2f}")
            if wf:
                try:                      # f/D des gemessenen Spiegels aus den Hauptreglern
                    e["N"] = float(self.var_f.get()) / float(self.var_D.get())
                    self._vg_vars["N"].set(f"{e['N']:.2f}")
                except Exception:
                    pass
                if not kz.get("kennzahlen_zuverlaessig", True):
                    notiz.append("⚠ Foucault-Fit unzuverlässig (Polynom-Modus, Residuum hoch) — "
                                 "Wellenfront ist geglättet, Ergebnis optimistisch! Hybrid-Modus (n≥6) verwenden.")
        fig.clf()
        ax1, ax2, axb, axw = vergleich_achsen(fig, extra=bool(wf) or self._vg_sph1.get() or self._vg_sph2.get())
        for a in (ax1, ax2): a.set_facecolor(BG)
        self._tab_vergleich = (fig, ax1, ax2, canvas)
        p = dict(e, refS=1.0, tubus=0.0, lam=lam,
                 wf1=wf.get("1"), wf2=wf.get("2"), auge=self._vg_auge.get(),
                 sph1=self._vg_sph1.get(), sph2=self._vg_sph2.get(),
                 augenmass=self._vg_mass.get())
        zeilen = vergleich_zeichnen(ax1, ax2, p, axb, axw)
        self._vg_readout.config(text="\n".join(notiz + zeilen))
        fig.tight_layout()
        canvas.draw_idle()

    def _fs_eingaben(self):
        """Ganzzahlige Eingaben des Fangspiegel-Tabs (Komma/Punkt, Fallback)."""
        std = dict(buet=50, tubus=0, bild=10, V=150, hut=0, fe=21, afov=85, oazd=50, oazl=80)
        out = {}
        for k, v in self._fs_vars.items():
            try:
                x = int(round(float(v.get().replace(",", ".").strip())))
                if x < 0 or (k in ("V", "fe", "afov", "oazd", "oazl") and x < 1):
                    raise ValueError
            except ValueError:
                x = std[k]
            out[k] = x
            v.set(str(x))
        return out

    def _fangspiegel_neu(self, *_):
        res = getattr(self, "_last_params", None)
        if res is None:
            return
        try:
            self._diagramm_fuer_tab("tab_fangspiegel", res)
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _diagramm_fangspiegel(self, D, f, lam, S_slide=None, S_real=None):
        """Fangspiegel-Rechner: Mindestgröße + Q_vis(V) für d_min, +10, +20 mm
        (neu in v6.12.0).  Spiegel: Strehl-Schieber, sonst Sphäre."""
        fig, ax, canvas = self._tab_fangspiegel
        fig.clf()
        ax = fig.add_subplot(1, 2, 1); ax.set_facecolor(BG)
        ax2 = fig.add_subplot(1, 2, 2); ax2.set_facecolor(BG)
        self._tab_fangspiegel = (fig, ax, canvas)
        e = self._fs_eingaben()
        if S_slide is not None:
            S_u, S_txt = S_slide, "Schieber"
        else:
            _, _, Wb_sph, _, _ = _wellenfronten(D, f, lam)
            S_u, S_txt = _strehl_exakt(Wb_sph), "Sphäre"
        r = fangspiegel_rechner(D, f, e["buet"], e["tubus"], e["bild"],
                                S_u, e["V"], lam, e["hut"], e["fe"], e["afov"], e["oazd"], e["oazl"])
        self._fs_readout.config(text=fangspiegel_text(D, f, S_u, S_txt, r, e["V"]))
        fangspiegel_zeichnen(ax, D, f, lam, S_u, S_txt, r, e["V"])
        fangspiegel_skizze(ax2, D, f, r)
        fig.tight_layout()
        canvas.draw_idle()

    def _diagramm_vmax_D(self):
        """Zeigt V_max(D) für PERFEKTE Optik (S=1): Beugungsgrenze (linear
        in D) kombiniert mit Seeing (aperturunabhängig, konstant) für
        mehrere Seeing-Werte, plus "2×D"-Mythos zum Vergleich.

        Vereinfachung ggü. _diagramm_vmax() (08/2026, auf Wunsch): keine
        Verknüpfung mit Spiegel-Strehl — zeigt nur die FUNDAMENTALE Grenze,
        die auch bei perfekter Politur bleibt: Beugung + Seeing. Die
        Kombination mit tatsächlichem Spiegel-Strehl macht _diagramm_vmax()
        (inzwischen korrekt, seit v_krit_naeh bei Wb→0 sauber gegen
        unendlich geht statt bei der alten D·0,7·√S-Formel zu deckeln).

        Kernaussage: die "2×D"-Regel trifft näherungsweise zu, solange die
        (aperturproportionale) Beugungsgrenze bindend ist (kleine
        Öffnungen) — wird aber zunehmend zu optimistisch, sobald das
        (aperturunabhängige) Seeing übernimmt (große Öffnungen). Der
        Übergang ("Knick") liegt ungefähr dort, wo beide Kurven sich
        schneiden.
        """
        fig, ax, canvas = self._tab_vmax_D
        ax.cla()
        ax.set_facecolor(BG)

        try:
            seeings = [float(s.strip()) for s in self._vmaxD_seeings.get().split(",") if s.strip()]
        except ValueError:
            seeings = [0.8, 1.5, 2.5, 4.0]
        Dmin = max(self._vmaxD_Dmin.get(), 10.0)
        Dmax = max(self._vmaxD_Dmax.get(), Dmin + 10.0)

        D_arr = np.linspace(Dmin, Dmax, 300)
        farben = ["#1565c0", "#2e7d32", "#E65100", "#8e44ad", "#00838f", "#ad1457"]

        for i, seeing in enumerate(seeings):
            V_see = v_max_seeing(seeing)
            V_diff = np.array([v_max_diffraktion(D) for D in D_arr])
            V_komb = np.array([v_kombiniert_blur(v, V_see) for v in V_diff])
            farbe = farben[i % len(farben)]
            ax.plot(D_arr, V_komb * 2.0, color=farbe, lw=2.0,
                   label=f"Seeing {seeing:.1f}\"  (praktisch)")

        ax.plot(D_arr, 2.0 * D_arr, color="#c62828", lw=1.8, ls=":",
               label='"2×D"-Mythos')

        ax.set_xlabel("Öffnung D [mm]", fontsize=9)
        ax.set_ylabel("V_max praktisch  [×]  (perfekte Optik, S=1)", fontsize=9)
        ax.set_title("Sinnvolle Maximalvergrößerung vs. Öffnung — "
                     "Beugung + Seeing, perfekte Optik",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(D_arr[0], D_arr[-1])
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        canvas.draw()

    def _vmax_objekt_gewaehlt(self):
        """Füllt L₀ mit dem Referenzwert des gewählten Objekt-Presets;
        bleibt danach frei überschreibbar (Preset + manuell, s. Chat)."""
        name = self._vmax_objekt.get()
        for n, L0 in OBJEKT_HELLIGKEITEN:
            if n == name:
                self._vmax_L0.set(L0)
                break
        self._aktualisieren()

    def _diagramm_vmax(self, D: float, f: float, S_slide: float):
        """'Sinnvolle Maximalvergrößerung' — Gegenprobe zum 2×D-Mythos.

        Zeigt L(V) (Netzhaut-Flächenhelligkeit) sowie die unabhängigen
        Grenzen V_helligkeit_min, V_seeing (Onset/praktisch, jetzt mit
        Spiegelfehler kombiniert), V_helligkeit_max, dazu die klassische
        2×D-Faustregel zum Vergleich. S. Modul-Docstring bei
        v_min_helligkeit() für die Herleitung und warum keine einzelne
        'V_optimal'-Zahl aus einem Integral verwendet wird.

        WICHTIG (08/2026): V_seeing allein ignoriert die Spiegelqualität.
        Für einen schlechten (v.a. schnellen) Kugelspiegel kann der
        Spiegel selbst mehr Unschärfe beitragen als die Atmosphäre — daher
        wird die Seeing-Grenze jetzt mit dem geometrischen Spiegel-Blur des
        AKTUELLEN Spiegels (S_slide, wie im Rest des Tools) über
        v_kombiniert_blur() kombiniert. Für eine sehr gute Optik (S→1)
        reduziert sich das automatisch auf reines Seeing.
        """
        fig, ax, canvas = self._tab_vmax
        ax.cla()
        ax.set_facecolor(BG)

        L0         = max(self._vmax_L0.get(), 1e-6)
        seeing     = max(self._vmax_seeing.get(), 0.0)
        D_auge     = max(self._vmax_D_auge.get(), 0.5)
        L_schwelle = max(self._vmax_L_schwelle.get(), 1e-6)
        objekt     = self._vmax_objekt.get()

        V_hmin   = v_min_helligkeit(D, D_auge)
        V_seeM   = v_max_seeing(seeing)                 # Onset: Seeing allein
        V_mirror = v_krit_naeh(D, _Wb_von_S(S_slide))    # Onset: Spiegel allein
        V_komb   = v_kombiniert_blur(V_seeM, V_mirror)   # Onset: beide zusammen
        V_komb_prakt = V_komb * 2.0                      # praktisch, s. Konvention V_max=2×V_krit
        V_hmax   = v_max_helligkeit(D, L0, D_auge, L_schwelle)
        V_mythos = 2.0 * D

        V_obergrenze = max(V_mythos, V_komb_prakt, V_hmax if V_hmax > 0 else 0, V_hmin) * 1.15
        V_arr = np.logspace(math.log10(3.0), math.log10(V_obergrenze), 400)
        L_V   = helligkeit_praxis(D, V_arr, L0, D_auge)

        ACC = "#1565c0"; COR = "#E65100"; GRN = "#2e7d32"; VIO = "#8e44ad"; GRA = "#888888"

        ax.plot(V_arr, L_V, color=ACC, lw=2.0, label=f"L(V) — {objekt} (L₀={L0:.1f} cd/m²)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(L_schwelle, color="#999", lw=1.0, ls=":",
                   label=f"Wahrnehmungsschwelle ({L_schwelle:.2g} cd/m²)")

        ax.axvline(V_hmin, color=GRN, lw=1.6, ls="--",
                   label=f"V_helligkeit_min={V_hmin:.0f}×  (AP=Augenpupille)")
        if V_seeM < V_obergrenze:
            ax.axvline(V_seeM, color=GRA, lw=1.0, ls=":",
                       label=f"V_seeing allein={V_seeM:.0f}×  (Seeing {seeing:.1f}\")")
        if V_mirror < V_obergrenze:
            ax.axvline(V_mirror, color=GRA, lw=1.0, ls="-.",
                       label=f"V_spiegel allein={V_mirror:.0f}×  (S={S_slide:.3f})")
        if V_komb < V_obergrenze:
            ax.axvline(V_komb, color=COR, lw=1.2, ls=":",
                       label=f"V_kombiniert Onset={V_komb:.0f}×")
            ax.axvline(V_komb_prakt, color=COR, lw=1.6, ls="--",
                       label=f"V_kombiniert praktisch={V_komb_prakt:.0f}×  (=2×Onset)")
        if 0 < V_hmax < V_obergrenze:
            ax.axvline(V_hmax, color=VIO, lw=1.6, ls="--",
                       label=f"V_helligkeit_max={V_hmax:.0f}×")
        ax.axvline(V_mythos, color="#c62828", lw=1.6, ls=":",
                   label=f"\"2×D\"-Mythos={V_mythos:.0f}×")

        V_ober_sinn = min(x for x in [V_komb_prakt, V_hmax if V_hmax > 0 else 1e9] if x > 0)
        if V_ober_sinn > V_hmin:
            ax.axvspan(V_hmin, V_ober_sinn, color=GRN, alpha=0.08,
                       label=f"sinnvoller Bereich ≈ {V_hmin:.0f}–{V_ober_sinn:.0f}×")

        ax.set_xlim(V_arr[0], V_arr[-1])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Vergrößerung V [×]  (log)", fontsize=9)
        ax.set_ylabel("Netzhaut-Flächenhelligkeit L(V) [cd/m²]", fontsize=9)
        ax.set_title(f"Sinnvolle Maximalvergrößerung  —  D={D:.0f}mm  f/{f/D:.1f}  S={S_slide:.3f}  "
                     f"Seeing={seeing:.1f}\"  Augenpupille={D_auge:.1f}mm",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        canvas.draw()

    def _zeige_fehler(self, msg: str):
        """Zeigt Berechnungsfehler im Statusbereich an."""
        print("BERECHNUNGSFEHLER:\n", msg)   # immer in Terminal
        try:
            self.lbl_verdict.config(
                text=f"⚠ Berechnungsfehler (Details in Terminal)", fg="#c62828")
        except Exception:
            pass

    def _aktualisieren(self, res: dict | None = None):
        """UI-Update auf dem Hauptthread. Ohne res: Erststart → Thread starten."""
        if res is None:
            D        = self.var_D.get()
            f        = self.var_f.get()
            lam      = self.var_lam.get()
            D_bl_raw = self.var_blende.get()
            S_pct    = self.var_strehl.get()
            params   = (D, f, lam, D_bl_raw, S_pct)
            import threading
            threading.Thread(target=self._compute_thread,
                             args=(params,), daemon=True).start()
            return
        try:
            self._aktualisieren_ui(res)
        except Exception as exc:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _aktualisieren_ui(self, res: dict):

        # ── Parameter aus Ergebnis-Dict ────────────────────────────────────
        D, f, lam   = res["D"], res["f"], res["lam"]
        D_bl        = res["D_bl"]
        r_real      = res["r_real"]
        S_real      = res["S_real"]
        S_slide     = res["S_slide"]
        Deff_k_slide= res["Deff_k_slide"]
        Vk          = res["Vk"]
        Vk_slide    = res["Vk_slide"]
        rv          = res["rv"]
        Vk_naeh     = res["Vk_naeh"]
        Vk_blur     = res["Vk_blur"]
        Vk_exakt    = res["Vk_exakt"]
        Vk_halb     = res["Vk_halb"]
        Vk_exakt_s  = res["Vk_exakt_s"]
        Qp_80, Qp_200, Qp_350, Qp_V = res["Qp_80"], res["Qp_200"], res["Qp_350"], res["Qp_V"]
        self._last_params = res
        # ── Info-Labels neben den Eingabefeldern ──────────────────────────
        r = r_real
        Vk_naeh_s = Vk_slide["Vk_sph"]
        for key, (lbl, unit) in self._param_labels.items():
            if key == "sl_D":
                lbl.config(text=f"f/{f/D:.1f}")
            elif key == "sl_f":
                lbl.config(text=f"f/{f/D:.1f}")
            elif key == "sl_lam":
                lbl.config(text="sichtbares Licht" if 380 <= lam <= 700 else "außerhalb VIS")
            elif key == "sl_strehl":
                lbl.config(text=f"→ S={S_slide:.3f}  "
                                 f"({'Paraboloid' if S_slide >= 0.9999 else 'Sphäre'})")
            elif key == "sl_blende":
                lbl.config(text=f"f/{f/D_bl:.1f}")

        # ── Ergebniskarten ─────────────────────────────────────────────────
        rv2 = self._res
        r   = r_real
        S_mar      = r["strehl_marechal"]
        _col_ex    = "#2e7d32" if S_real >= 0.95 else "#e65100" if S_real >= 0.80 else "#c62828"
        Vk_naeh_s  = Vk_slide["Vk_sph"]
        delta_halb = Vk_halb - Vk_naeh
        delta_ex   = Vk_exakt - Vk_naeh
        col_naeh   = "#534AB7"
        col_blur   = "#2e7d32" if Vk_blur > 50 else "#e65100" if Vk_blur > 20 else "#c62828"
        col_ex     = "#2e7d32" if abs(delta_ex) < 30 else "#e65100"
        avp        = r["aufl_verlust_pct"]

        def _vk_ex_text(Ve):
            return f"{Ve:.0f}×" if Ve > 30.5 else "< 30×"

        rv2["rN"].config(    text=f"f/{r['N']:.2f}")
        rv2["rWp"].config(   text=f"{r['Wp']:.4f} λ")
        rv2["rWb"].config(   text=f"{r['Wb']:.4f} λ")
        rv2["rWrms"].config( text=f"{r['Wrms']:.5f} λ")
        rv2["rS"].config(    text=f"{S_real:.4f}  →  {S_slide:.4f}")
        rv2["rS_exakt"].config(
            text=f"{S_real:.4f}  (Δ vs. Mar. {S_real - S_mar:+.4f})", fg=_col_ex)
        rv2["rDeff_k"].config(
            text=f"{r['Deff_k']:.1f} → {Deff_k_slide:.1f} mm")
        rv2["rVkrit_naeh"].config(
            text=f"{Vk_naeh:.0f}×  →  {Vk_naeh_s:.0f}×"
                 f"  |  exakt: {Vk_halb:.0f}×  (Δ {delta_halb:+.0f}×)",
            fg=col_naeh)
        rv2["rVkrit_blur"].config(
            text=(f"{Vk_blur:.0f}×  (α_blur={D**3/(64*f**2)/f*206265:.1f}\"  — unabh. von S)"
                  if Vk_blur > 1 else "< 1×  — Blur dominiert immer"),
            fg=col_blur)
        rv2["rVkrit_exakt"].config(
            text=f"{_vk_ex_text(Vk_exakt)}  →  {_vk_ex_text(Vk_exakt_s)}",
            fg=col_ex)
        rv2["rAuflVerlust"].config(
            text=f"{avp:.1f}%  (Blur {r['d_blur_as']:.1f}\")",
            fg="#c62828" if avp > 80 else "#e65100" if avp > 40 else "#2e7d32")
        rv2["rVerlustV"].config(
            text=f"{rv['verlust']:.1f} mm",
            fg="#c62828" if rv["verlust"] > 5 else "#333")

        text, farbe = beurteilung(S_real)
        self.lbl_verdict.config(text=text, fg=farbe)
        self.lbl_vkrit.config(
            text=f"⚡ Real: V_krit={Vk['Vk_sph']:.0f}×  →  "
                 f"Verbessert (S={S_slide:.2f}): V_krit={Vk_slide['Vk_sph']:.0f}×  |  "
                 f"Paraboloid: V_krit={Vk['Vk_para']:.0f}×  |  V_max={Vk['Vmax']:.0f}×",
            fg="#534AB7")

        def _qcolor(q):
            return "#2e7d32" if q >= 0.90 else "#e65100" if q >= 0.70 else "#c62828"
        rv2["rQp80"].config( text=f"{Qp_80:.3f}",  fg=_qcolor(Qp_80))
        rv2["rQp200"].config(text=f"{Qp_200:.3f}", fg=_qcolor(Qp_200))
        rv2["rQp350"].config(text=f"{Qp_350:.3f}", fg=_qcolor(Qp_350))
        rv2["rQpV"].config(  text=f"{Qp_V:.3f}  (bei 150×)", fg=_qcolor(Qp_V))

        # ── Diagramme ──────────────────────────────────────────────────────
        # Beim ersten Durchlauf (startup) alle Tabs zeichnen, danach nur
        # das sichtbare Tab — Rest wird beim Tab-Wechsel nachgeholt.
        active = self._aktiver_tab()
        alle_tab_keys = [k for _, k in [
            ("_tab_strehl",      "tab_strehl"),
            ("_tab_oeff",        "tab_oeff"),
            ("_tab_deff_D",      "tab_deff_D"),
            ("_tab_beugung",     "tab_beugung"),
            ("_tab_mtf",         "tab_mtf"),
            ("_tab_wahrnehmung", "tab_wahrnehmung"),
            ("_tab_blende",      "tab_blende"),
            ("_tab_foucault",    "tab_foucault"),
            ("_tab_vmax",        "tab_vmax"),
            ("_tab_vmax_D",      "tab_vmax_D"),
            ("_tab_fangspiegel", "tab_fangspiegel"),
            ("_tab_vergleich",   "tab_vergleich"),
        ]]
        startup = not getattr(self, "_first_draw_done", False)
        if startup:
            self._first_draw_done = True
            # Beim ersten Durchlauf nur das sichtbare Tab zeichnen;
            # alle anderen werden beim Tab-Wechsel nachgeholt (lazy).
            self._dirty_tabs = set(alle_tab_keys) - {active}
            self._diagramm_fuer_tab(active, res)
        else:
            self._dirty_tabs = set(alle_tab_keys) - {active}
            self._diagramm_fuer_tab(active, res)

    # ── Foucault-Tab Hilfsmethoden ─────────────────────────────────────────────

    def _foucault_eingabe_neu(self):
        """Eingabefelder neu aufbauen wenn Zonenanzahl geändert."""
        n = self._foucault_n.get()
        # Alte Widgets entfernen
        for w in self._foucault_eingabe_frm.winfo_children():
            w.destroy()
        self._foucault_df_vars = []

        D = self.var_D.get()
        f = self.var_f.get()
        Rc    = 2.0 * f
        r_zon   = foucault_zonen(D, n)
        r_grenz = foucault_zonengrenzen(D, n)
        df_para_abs = r_zon**2 / (2.0 * Rc)        # absoluter Parabolsollwert (+r²/4f)
        df_para     = df_para_abs - df_para_abs[0]  # relativ zu Zone 1 (Referenz = 0)

        # Info-Text mit Zonenradien und Parabolsollwerten
        info = ("Gleichflächige Zonen — Parabolspiegel-Sollwerte, relativ zu Zone 1 "
                "(+ = Messerschneide weiter vom Spiegel weg, FigureXP-Konvention):  " +
                "  |  ".join(
                    f"Z{i+1}: r={rv:.1f}mm  ΔF_soll={dp:.3f}mm"
                    for i, (rv, dp) in enumerate(zip(r_zon, df_para))) +
                "      [ΔF=0 → Kugel  |  ΔF=ΔF_soll → Parabel  |  Zone 1 = Referenz, fest 0]")
        self._foucault_info.config(text=info)

        # Kopfzeile
        headers = ["Zone", "Zonenradius [mm]", "Zonengrenze außen [mm]",
                   "ΔF_soll [mm]  (rel. Zone 1)",
                   "ΔF gemessen [mm]  (rel. Zone 1; = ΔF_soll → Parabel | = 0 → Kugel)", "Abw. [mm]"]
        widths  = [6, 16, 16, 14, 38, 10]
        for col, (hdr, w) in enumerate(zip(headers, widths)):
            tk.Label(self._foucault_eingabe_frm,
                     text=hdr, bg=BG, fg="#555",
                     font=("Helvetica", 9, "bold"), width=w).grid(
                     row=0, column=col, padx=4, pady=2)

        for i, rv in enumerate(r_zon):
            dp = df_para[i]           # Parabolsollwert für diese Zone (rel. Zone 1)
            r_innen  = r_grenz[i-1] if i > 0 else 0.0
            r_aussen = r_grenz[i]
            ist_referenz = (i == 0)
            var = tk.DoubleVar(value=0.0 if ist_referenz else round(dp, 3))
            var.trace_add("write", lambda *_: self._foucault_aktualisieren())
            self._foucault_df_vars.append(var)

            # Zone
            tk.Label(self._foucault_eingabe_frm,
                     text=f"Z{i+1}", bg=BG, fg=ACC,
                     font=("Helvetica", 9, "bold"), width=6).grid(
                     row=i+1, column=0, padx=4, pady=2)
            # Zonenradius
            tk.Label(self._foucault_eingabe_frm,
                     text=f"{rv:.2f}", bg=BG, fg="#333",
                     font=("Helvetica", 9), width=16).grid(
                     row=i+1, column=1, padx=4)
            # Zonengrenze (innen–außen, für Zonenmaske)
            tk.Label(self._foucault_eingabe_frm,
                     text=f"{r_innen:.2f} – {r_aussen:.2f}", bg=BG, fg="#333",
                     font=("Helvetica", 9), width=16).grid(
                     row=i+1, column=2, padx=4)
            # Sollwert (klickbar → Wert übernehmen, außer bei der Referenzzone)
            lbl_soll = tk.Label(self._foucault_eingabe_frm,
                                text=f"{dp:.3f}", bg=BG2, fg="#1565c0",
                                font=("Helvetica", 9, "bold"), width=14,
                                cursor="hand2" if not ist_referenz else "arrow",
                                relief="flat")
            lbl_soll.grid(row=i+1, column=3, padx=4)
            if not ist_referenz:
                lbl_soll.bind("<Button-1>",
                              lambda e, v=var, d=dp: (v.set(round(d, 3)), None))
            # Messwert-Eingabe (Zone 1 = Referenz, fest 0, nicht editierbar)
            if ist_referenz:
                tk.Label(self._foucault_eingabe_frm,
                         text="0.000  (Referenz)", bg=BG, fg="#777",
                         font=("Helvetica", 9, "italic"), width=12).grid(
                         row=i+1, column=4, padx=4, sticky="w")
            else:
                sp = tk.Spinbox(self._foucault_eingabe_frm,
                                from_=-50.0, to=50.0, increment=0.001,
                                textvariable=var, width=12, format="%.3f",
                                font=("Helvetica", 10))
                sp.grid(row=i+1, column=4, padx=4, sticky="w")
            # Abweichung vom Sollwert (wird in _foucault_aktualisieren befüllt)
            lbl_abw = tk.Label(self._foucault_eingabe_frm,
                               text="0.000", bg=BG, fg="#333",
                               font=("Helvetica", 9), width=10)
            lbl_abw.grid(row=i+1, column=5, padx=4)

        # Knopf: alle Zonen auf Parabolsollwerte zurücksetzen
        btn_frame = tk.Frame(self._foucault_eingabe_frm, bg=BG)
        btn_frame.grid(row=n+1, column=0, columnspan=6, pady=6, sticky="w")
        tk.Button(btn_frame, text="↺  Alle auf Parabel-Sollwerte",
                  font=("Helvetica", 9), bg=BG2,
                  command=lambda: [v.set(round(d,3))
                                   for v,d in zip(self._foucault_df_vars, df_para)]
                  ).pack(side="left", padx=4)
        tk.Button(btn_frame, text="✕  Alle auf Null (Kugel)",
                  font=("Helvetica", 9), bg=BG2,
                  command=lambda: [v.set(0.0) for v in self._foucault_df_vars]
                  ).pack(side="left", padx=4)

        self._foucault_aktualisieren()

    def _foucault_aktualisieren(self):
        """Neuberechnung und Diagramm-Update fuer Foucault-Tab."""
        try:
            D          = self.var_D.get()
            f          = self.var_f.get()
            lam        = self._foucault_lam.get()
            n          = self._foucault_n.get()
            # _foucault_use_direct wird erst beim Tab-Aufbau erstellt;
            # Default False (Polynom) wenn Tab noch nicht geoeffnet wurde.
            if not hasattr(self, "_foucault_use_direct"):
                self._foucault_use_direct = tk.BooleanVar(value=False)
            use_direct = self._foucault_use_direct.get()
            if not hasattr(self, "_foucault_werkstatt"):
                self._foucault_werkstatt = tk.BooleanVar(value=True)
            werkstatt_ansicht = self._foucault_werkstatt.get()
            if not hasattr(self, "_foucault_grad_modus"):
                self._foucault_grad_modus = tk.StringVar(value="bic")
            grad_modus = self._foucault_grad_modus.get()

            r_zon  = foucault_zonen(D, n)
            df_arr = np.array([v.get() for v in self._foucault_df_vars])
            df_para_abs = r_zon**2 / (2.0 * 2.0 * f)
            df_para     = df_para_abs - df_para_abs[0]
            abw         = df_arr - df_para

            for i, dv in enumerate(abw):
                for w in self._foucault_eingabe_frm.grid_slaves(row=i+1, column=5):
                    col = "#2e7d32" if abs(dv) < 0.01 else                           "#e65100" if abs(dv) < 0.1  else "#c62828"
                    w.config(text=f"{dv:+.3f}", fg=col)

            W_z = foucault_wellenfront(r_zon, df_arr, D, f, lam)

            # Hybrid-Modus braucht mindestens 6 Zonen um zuverlaessig
            # zu sein: bei n=4 hat das BIC-Polynom zu viel Spielraum
            # und schluckt Zonenfehler (n Punkte, nur 2 Koeffizienten
            # -> bei n=4 fast Ueberfit, Residuen zu klein).
            ud_effektiv = use_direct and (n >= 6)
            kz  = foucault_kennzahlen(W_z, r_zon, D, lam,
                                      use_direct=ud_effektiv,
                                      grad_modus=grad_modus)

            rayleigh_ok  = kz["S_exakt"] >= 0.80
            zuverlaessig = kz.get("kennzahlen_zuverlaessig", True)
            s_col = ("#c62828" if not zuverlaessig else
                     "#2e7d32" if kz["S_exakt"] >= 0.95 else
                     "#e65100" if rayleigh_ok else "#c62828")
            modus = "Hybrid" if use_direct else "Polynom"
            warn  = ""
            if use_direct and n < 6:
                warn = "  INFO: Hybrid-Modus benoetigt n>=6 Zonen -- Polynom-Modus verwendet"
            elif not zuverlaessig:
                warn = (f"  ⚠ UNZUVERLAESSIG: Fit-Res={kz['fit_residuum_rms']:.3f}lambda"
                        f" > 0.010 -- Hybrid-Modus (n>=6) verwenden!")

            # Warnung wenn Zonen noch nahe Kugelfehler (weniger als 20% korrigiert)
            dl_soll_arr = (r_zon**2 - r_zon[0]**2) / (4.0 * f)
            fehlend = np.where(dl_soll_arr > 0.001,
                               (dl_soll_arr - df_arr) / dl_soll_arr, 0.0)
            kugel_zonen = [f"Z{i+1}" for i, feh in enumerate(fehlend)
                           if feh > 0.80 and i > 0]
            if kugel_zonen:
                warn += (f"  !! {", ".join(kugel_zonen)} noch"
                         f" auf Kugelnaehe -- S nicht repraesentativ!")
                s_col = "#c62828"
            status = (
                f"[{modus}]  "
                f"S={kz['S_exakt']:.4f}   "
                f"Wrms={kz['W_rms']:.4f}lambda   "
                f"WPtV={kz['W_ptv']:.4f}lambda   "
                f"Abtrag={kz['Glas_ptv_nm']:.1f}nm   "
                f"V={kz['Glas_volumen_mm3']:.1f}mm3   "
                f"Deff={kz['Deff_k']:.1f}mm   "
                f"{'OK Rayleigh' if rayleigh_ok else 'FAIL Rayleigh'}"
                f"{warn}"
            )
            self._foucault_status.config(text=status, fg=s_col)
            self._foucault_letzte_kz = kz          # 07/2026: fuer MTF-Tab Vergleichskurve
            self._foucault_letzte_D  = D
            self._diagramm_foucault(D, f, lam, r_zon, df_arr, kz,
                                    use_direct=use_direct,
                                    werkstatt_ansicht=werkstatt_ansicht)
        except Exception as e:
            if hasattr(self, "_foucault_status"):
                self._foucault_status.config(
                    text=f"Fehler: {e}", fg="#c62828")

    def _diagramm_foucault(self, D, f, lam, r_zon, df_arr, kz, use_direct=False,
                           werkstatt_ansicht=False):
        """Foucault-Diagramm: Wellenfrontkurve + Kennzahlentafel."""
        fig, ax_wf, ax_info, canvas = self._tab_foucault
        ax_wf.cla(); ax_info.cla()
        ax_wf.set_facecolor("#f5f5f5")
        ax_info.set_facecolor("#f5f5f5")

        COR = "#E65100"; ACC = "#1565c0"; GRN = "#2e7d32"
        R     = D / 2.0
        rho_z = r_zon / R

        # ── Wellenfrontkurve (aus Polynom-Fit, am besten Fokus) ───────────
        rho_fine = kz["rho_f"]
        W_fine   = kz["W_c"]    # pistonbereinigt, bester Fokus

        # Werkstatt-Ansicht (FigureXP-Darstellung): Kurve vertikal gespiegelt,
        # Null im Tal (bei ~0.7R), Huegel in Mitte und Rand -- genau wie
        # FigureXPs "Surface Error Analysis" / "Glass Remaining". Reine
        # Anzeigetransformation y -> W_max - y: Kennzahlen in kz (S_exakt/
        # W_rms/W_ptv) bleiben unveraendert. ACHTUNG Vorzeichen: gegenueber
        # der Standardansicht (Abtrag-Sicht, Maximum im Tal bei 0.7R) ist das
        # exakt die gespiegelte Kurve (Summe beider = PtV).
        W_ref   = float(np.max(W_fine))
        W_fine_disp = (W_ref - W_fine) if werkstatt_ansicht else W_fine

        _grad = kz.get("poly_grad", 1)
        _res  = kz.get("fit_residuum_rms", 0.0)
        _a4   = kz.get("a4", 0.0)
        if use_direct:
            lbl = f"Wellenfront (Hybrid Grad {_grad}, res={_res:.3f}lambda)"
        elif _res > 0.010:
            lbl = f"Wellenfront (Polynom Grad {_grad}, a4={_a4:.3f}) WARN res={_res:.3f}"
        else:
            lbl = f"Wellenfront (Polynom Grad {_grad}, a4={_a4:.3f})"
        ax_wf.plot(rho_fine, W_fine_disp, color=ACC, lw=2.0, label=lbl)

        # Zonenpunkte: y-Position auf rechter Achse (nm Glas)
        # Da matplotlib nur eine Daten-y-Achse kennt, rechnen wir die nm-Werte
        # über die Inverse der secondary_yaxis-Transformation in λ-Einheiten zurück.
        # zu_lam(g) = g * 2/lam + W_min  →  Punkt sitzt visuell auf der nm-Skala.
        W_z    = kz["W_zonen"]
        Glas_z = kz.get("Glas_zonen")
        W_min  = kz.get("W_min")
        if Glas_z is not None and W_min is not None:
            # Punkte liegen auf der Kurve (gleiche Referenz W_max wie Abtrag).
            y_punkte = (W_ref - np.asarray(kz["W_c_zonen"])) if werkstatt_ansicht \
                       else np.asarray(kz["W_c_zonen"])
            Glas_lbl = Glas_z
            Wm_disc  = float(np.mean(W_z))
            ax_wf.scatter(rho_z, y_punkte, color=COR, s=50, zorder=5,
                          label=f"Messzonen (n={len(r_zon)})")
            for i, (rho, yp) in enumerate(zip(rho_z, y_punkte)):
                wc  = W_z[i] - Wm_disc   # unverschoben, fuer λ-Angabe im Label
                txt = f"Z{i+1}: {Glas_lbl[i]:.0f} nm\n({wc:+.3f}λ)"
                ax_wf.annotate(txt,
                               xy=(rho, yp), xytext=(0, 10),
                               textcoords="offset points",
                               fontsize=7, ha="center", color=COR)
        else:
            # Fallback ohne lam_nm: nur λ-Skala
            Wm_disc  = float(np.mean(W_z))
            W_c_disc = (W_ref - (W_z - Wm_disc)) if werkstatt_ansicht else (W_z - Wm_disc)
            ax_wf.scatter(rho_z, W_c_disc, color=COR, s=50, zorder=5,
                          label=f"Messzonen (n={len(r_zon)})")
            for i, (rho, wc) in enumerate(zip(rho_z, W_c_disc)):
                ax_wf.annotate(f"Z{i+1}: {wc:+.3f}λ",
                               xy=(rho, wc), xytext=(0, 10),
                               textcoords="offset points",
                               fontsize=7, ha="center", color=COR)

        ax_wf.axhline(0,      color="#999",    lw=0.8)
        if not werkstatt_ansicht:
            ax_wf.axhline(+0.125, color="#1565c0", lw=0.7, ls=":", alpha=0.7,
                          label="±λ/8 (Rayleigh)")
            ax_wf.axhline(-0.125, color="#1565c0", lw=0.7, ls=":", alpha=0.7)
        else:
            ax_wf.axhline(+0.25, color="#1565c0", lw=0.7, ls=":", alpha=0.7,
                          label="λ/4 (Rayleigh, PtV)")
        ax_wf.set_xlim(-0.02, 1.05)
        ax_wf.set_xlabel("Normierter Zonenradius ρ = r/R", fontsize=8)
        ax_wf.set_ylabel("Oberflächenfehler, Wellenfront-Skala  [Einheiten von λ]"
                         if werkstatt_ansicht else
                         "Wellenfrontfehler W(ρ)  [Einheiten von λ]", fontsize=8)
        ax_wf.set_title("Werkstatt-Ansicht (wie FigureXP: Surface Error / Glass Remaining)"
                        if werkstatt_ansicht else
                        "Gemessene Wellenfront (bester Fokus, pistonbereinigt)",
                        fontsize=8, fontweight="bold")
        ax_wf.legend(fontsize=7,
                     loc="upper center" if werkstatt_ansicht else "lower center")
        ax_wf.grid(True, alpha=0.25)

        # ── Zweite y-Achse: Materialabtrag [nm Glas], referenziert auf den
        #    am wenigsten bearbeiteten Punkt (Minimum der Wellenfrontkurve) ──
        W_min = kz.get("W_min")
        if W_min is not None:
            if werkstatt_ansicht:
                zu_glas = lambda w: np.asarray(w) * lam / 2.0
                zu_lam  = lambda g: np.asarray(g) * 2.0 / lam
                _glas_lbl = "Glass Remaining [nm]  (0 = Tal: dort nichts abtragen, wie FigureXP)"
            else:
                zu_glas = lambda w: (W_ref - np.asarray(w)) * lam / 2.0
                zu_lam  = lambda g: W_ref - np.asarray(g) * 2.0 / lam
                _glas_lbl = "Materialabtrag [nm Glas]  (0 = Tal: dort nichts abtragen)"
            ax_glas = ax_wf.secondary_yaxis("right", functions=(zu_glas, zu_lam))
            ax_glas.set_ylabel(_glas_lbl, fontsize=8, color="#6a1b9a")
            ax_glas.tick_params(colors="#6a1b9a", labelsize=7)

        # ── Kennzahlentafel ───────────────────────────────────────────────
        ax_info.axis("off")
        zuverlaessig = kz.get("kennzahlen_zuverlaessig", True)
        rayleigh_ok  = kz["S_exakt"] >= 0.80
        if not zuverlaessig:
            s_col = "#c62828"   # unzuverlaessig -> immer rot, unabh. vom Zahlenwert
        else:
            s_col = GRN if kz["S_exakt"] >= 0.95 else \
                    "#e65100" if rayleigh_ok else "#c62828"

        _s_exakt_str = f"{kz['S_exakt']:.4f}"
        _wptv_str    = f"{kz['W_ptv']:.4f} λ"
        _wrms_str    = f"{kz['W_rms']:.4f} λ"
        if not zuverlaessig:
            _s_exakt_str += "  ⚠ unzuverlässig"
            _wptv_str    += "  ⚠"
            _wrms_str    += "  ⚠"

        zeilen = [
            ("Öffnung / f-Zahl",  f"D = {D:.0f} mm  f/{f/D:.1f}"),
            ("Wellenlänge",        f"λ = {lam:.0f} nm"),
            ("", ""),
            ("W_rms",              _wrms_str),
            ("W_PtV",              _wptv_str),
            ("Wb (aus Fit)",       f"{kz['Wb_fit']:.4f} λ"),
            ("Fit-Residuum RMS",   f"{kz['fit_residuum_rms']:.4f} λ"
                                   + ("  ⚠ > 0.010 (Grenze)" if not zuverlaessig else "")),
            ("", ""),
            ("Abtrag RMS (Glas)",  f"{kz['W_rms']*lam/2.0:.1f} nm"
                                   if kz.get("Glas_ptv_nm") is not None else "–"),
            ("Abtrag PtV (Glas)",  f"{kz['Glas_ptv_nm']:.1f} nm"
                                   if kz.get("Glas_ptv_nm") is not None else "–"),
            ("Abtrag V (Glas)",  f"{kz['Glas_volumen_mm3']:.1f} mm³"
                                   if kz.get("Glas_volumen_mm3") is not None else "–"),
            ("", ""),
            ("Strehl  Maréchal",   f"{kz['S_mar']:.4f}"),
            ("Strehl  exakt",      _s_exakt_str),
            ("D_eff  (Kontrast)",  f"{kz['Deff_k']:.1f} mm"),
            ("", ""),
            ("Beurteilung",
             "⚠ Zonenfehler nicht erfasst — Hybrid-Modus nutzen!" if not zuverlaessig
             else ("✓ Rayleigh erfüllt (S ≥ 0.80)" if rayleigh_ok
                   else "✗ Rayleigh nicht erfüllt")),
        ]

        y = 0.97
        for lbl, val in zeilen:
            if not lbl and not val:
                y -= 0.04; continue
            is_strehl  = lbl.startswith("Strehl") or lbl in ("W_rms", "W_PtV")
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
        _titel_zuv = "  ⚠ UNZUVERLÄSSIG" if not zuverlaessig else ""
        fig.suptitle(
            f"Foucault-Auswertung  —  D={D:.0f}mm  f/{f/D:.1f}  "
            f"S={kz['S_exakt']:.4f}  W_rms={kz['W_rms']:.4f}λ{_titel_zuv}",
            fontsize=8, fontweight="bold", color=(s_col if not zuverlaessig else "black"))
        fig.tight_layout()
        canvas.draw()

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
        # Maréchal: gestrichelt, deutlich dicker, andere Farbe
        ax.plot(ns, sts,       color="#c07000", lw=2.2, ls="--",
                label="Strehl Maréchal (Näherung, gueltig S>0.5)")
        # Exakt: durchgezogen, Hauptkurve
        ax.plot(ns, sts_exakt, color="#534AB7", lw=2.0, ls="-",
                label="Strehl exakt (Pupillenintegral)")
        ax.axhline(0.80, color="#BA7517", lw=1.2, ls="--", label="Rayleigh S=0.80")
        ax.axhline(0.95, color="#888",    lw=1.0, ls=":",  label="S=0.95")
        ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
        # Hinweistext: Marechal ungueltig unter f/8
        ax.axvline(8.0, color="#c07000", lw=0.7, ls=":", alpha=0.5)
        ax.text(8.1, 0.05, "Marechal\n< f/8 ungenau", fontsize=7,
                color="#c07000", alpha=0.8, va="bottom")
        ax.scatter([N_akt], [S_akt_mar], color="#c07000", s=60, zorder=5,
                   marker="s")
        ax.scatter([N_akt], [S_akt],    color="#534AB7", s=50, zorder=5,
                   marker="D")
        ax.annotate(f"f/{N_akt:.1f}  S={S_akt:.3f} (exakt) / "
                    f"S={S_akt_mar:.3f} (Maréchal)",
                    xy=(N_akt, S_akt), xytext=(8, 10),
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
            Wb_arr = np.array([_wellenfronten(D, D * N, 550.0)[2] for D in D_arr])
            loss_pct = (1.0 - np.exp(0.25 * _lnS_geglaettet(Wb_arr))) * 100
            col = colors_N[N]
            loss_roh = (1.0 - np.exp(0.25 * _lnS_geglaettet(Wb_arr, roh=True))) * 100
            ax.plot(D_arr, loss_roh, color=col, lw=1.0, ls="--", alpha=0.35, zorder=1)
            ax.plot(D_arr, loss_pct, color=col, lw=2, label=f"f/{N}", zorder=3)
            ax.text(D_arr[-1] + 3, loss_pct[-1], f"f/{N}",
                    color=col, fontsize=9, va="center", fontweight="bold")

        Wb_akt   = _wellenfronten(D_akt, D_akt * N_akt, 550.0)[2]
        S_akt    = _strehl_exakt(Wb_akt)          # exakt, für Vergleich mit Schieber
        loss_akt = (1.0 - np.exp(0.25 * float(_lnS_geglaettet(Wb_akt)))) * 100  # auf der Kurve
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
            Line2D([0],[0], color="#555", lw=2,   label="D × S^0.25  (geglättet)"),
            Line2D([0],[0], color="#555", lw=1.0, ls="--", alpha=0.5,
                   label="D × S^0.25  (exakter Strehl, ungeglättet)"),
        ]
        ax.legend(handles=proxy, fontsize=9, loc="upper left")
        ax.set_title("Öffnungsverlust vs. Öffnung — D × S^0.25  (Kontrast)", fontsize=9)
        ax.text(0.5, -0.13,
                "Gilt für V→∞ (reine Kontrast-/MTF-Form, ohne Augenauflösung/CSF) — "
                "bei kleiner Vergrößerung besser: Tab „Q_vis vs. Vergrößerung“\n"
                "Durchgezogen: Oszillation des exakten Strehl ab S < ~0.5 geglättet; "
                "gestrichelt: exakte Kurve. Dort ist S^0.25 nur noch qualitativ.",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#777", style="italic")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_beugung(self, D_akt, f_akt):
        fig, ax, canvas = self._tab_beugung
        ax.cla()

        lam_mm = 550e-6
        f_arr  = np.linspace(200, 2000, 500)

        def D_grenz(S):
            Wb = _Wb_von_S(S)
            return (Wb * 4.0 * 512 * lam_mm) ** 0.25 * f_arr ** 0.75  # 07/2026: 1024->512

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
        Wb_k        = D**4 / (512.0 * f**3 * lam_mm) / 4.0  # 07/2026: 1024->512
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

        # Vergleichskurve: bevorzugt die tatsaechlich am Foucault-Tab
        # gemessene Wellenfront (aussagekraeftiger als ein Prozent-Schieber
        # auf einen frei gewaehlten Ziel-Strehl). Faellt zurueck auf den
        # alten S_slide-Mechanismus, falls noch keine Foucault-Messung fuer
        # diese Oeffnung D vorliegt.  [07/2026]
        kz_fc = getattr(self, '_foucault_letzte_kz', None)
        D_fc  = getattr(self, '_foucault_letzte_D', None)
        if kz_fc is not None and D_fc == D:
            Wb_cmp    = kz_fc['Wb_fit']
            S_cmp     = kz_fc['S_exakt']
            cmp_label = f"Foucault gemessen  (S={S_cmp:.3f})"
            cmp_ok    = S_cmp > strehl + 0.01
        elif S_slide is not None and S_slide > strehl + 0.01:
            Wb_cmp    = _Wb_von_S(S_slide)
            cmp_label = f"Schieber absolut  (S={S_slide:.2f})"
            cmp_ok    = True
        else:
            Wb_cmp = None
            cmp_ok = False

        if cmp_ok and Wb_cmp is not None:
            msr_s_k = np.array([mtf_sph_rel(nu, Wb_cmp, paraxial=paraxial)
                                 for nu in nu_k2])
            ax.plot(nu_k2, mp_k2 * msr_s_k, color="#1a7abf", lw=1.8, ls="-.",
                    label=cmp_label)

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

    def _wahrnehmung_lesen(self):
        """Liest Cursor-V und Fangspiegel-Ø als GANZE Zahlen aus den Feldern
        (Nachkommastellen werden gerundet; Komma oder Punkt sind erlaubt).
        Ungültige Eingaben (leer/Text/V≤0/d<0) lassen den letzten Wert stehen;
        das Feld wird danach auf die ganze Zahl zurückgeschrieben."""
        def _ganz(var, fallback, minimum):
            try:
                x = int(round(float(str(var.get()).strip().replace(",", "."))))
                if x >= minimum:
                    return x
            except (tk.TclError, ValueError):
                pass
            return fallback
        V = _ganz(self._wn_V_var, int(getattr(self, "_wn_V", 150)), 1)
        d = _ganz(self._wn_fang_var, int(getattr(self, "_wn_fang", 0)), 0)
        self._wn_V, self._wn_fang = float(V), float(d)   # d: Fangspiegel-Ø [mm], 0 = aus
        self._wn_V_var.set(str(V))
        self._wn_fang_var.set(str(d))
        return V

    def _wahrnehmung_readout(self, D, f, lam, S_slide, S_real):
        """Exakte Q_vis-Werte bei der Cursor-Vergrößerung für die Spiegel der
        Hauptregler (Öffnung, Brennweite, Strehl-Schieber)."""
        V = self._wn_V

        def _vtxt(v):
            return "≥ 600×" if v >= 599.5 else "≤ 30×" if v <= 30.5 else f"{v:.0f}×"
        def _vk(v):
            return "∞" if v >= 1e8 else f"{v:.0f}×"
        q_sph = perceived_quality_exakt(D, f, lam, V)
        _, _, Wb_sph, _, _ = _wellenfronten(D, f, lam)
        S_sph = _strehl_exakt(Wb_sph)
        zeilen = [
            f"V = {V:.0f}×  (λ={lam:.0f} nm)   —   D={D:.0f} mm  f={f:.0f} mm  (f/{f/D:.1f})",
            f"Sphäre   S={S_sph:.3f}:   Q_vis = {q_sph:.4f}   Verlust {(1-q_sph)*100:.1f} %",
        ]
        if S_slide is not None and S_real is not None and S_slide > S_real + 0.01:
            e = qvis_einzelrechner(D, S_slide, V, lam)
            zeilen.append(
                f"Schieber S={S_slide:.3f}:   Q_vis = {e['Q']:.4f}   Q̃ = {e['Q_naeh']:.4f}   "
                f"Verlust {e['verlust']:.1f} %   Q≥0.90 bis {_vtxt(e['V_q90'])}   V_krit {_vk(e['V_krit'])}")
        d_fs = getattr(self, "_wn_fang", 0.0)
        d_min = d_fang_min(D, f)
        self._wn_fang_hint.config(
            text=f"mm   (0 = aus;  Mindestgröße für D/f: {d_min:.0f} mm)")
        if d_fs > 0:
            S_u = S_slide if S_slide is not None else S_sph
            q_fs = float(qvis_kurve_mit_fangspiegel(D, S_u, d_fs, lam, V)[0])
            zeilen.append(f"Schieber + Fangspiegel  d={d_fs:.0f} mm (ε={min(d_fs/D,0.95):.2f}):   "
                          f"Q_vis = {q_fs:.4f}   (Näherung)")
        zeilen.append("Paraboloid:   Q_vis = 1.0000")
        self._wn_readout.config(text="\n".join(zeilen))

    def _wahrnehmung_neu(self, *_):
        """Nur das Wahrnehmungs-Diagramm neu zeichnen (Cursor)."""
        res = getattr(self, "_last_params", None)
        if res is None:
            return
        try:
            self._diagramm_fuer_tab("tab_wahrnehmung", res)
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _diagramm_wahrnehmung(self, D, f, lam,
                               S_slide=None, S_real=None):
        """Visueller Qualitätsindex Q_vis vs. Vergrößerung.

        Zwei Kurven pro Spiegel:
          · Q_vis exakt  = MTF×CSF-Integral ohne w(V)         (durchgezogen)
          · Q̃ Näherung  = S + (1−S)·(1−w(V))  [w(V) mit V_wendepunkt_naeh,
                           NICHT V_krit — s. Chat-Historie 08/2026:
                           V_krit selbst passt die Kurvenform schlecht,
                           separater gröberer Korrekturfaktor für S<1]
                           (gestrichelt)
        """
        fig, ax, canvas = self._tab_wahrnehmung
        ax.cla()

        # Ablese-Cursor (10/2026, v6.10.0): V_cur=None -> kein Cursor.
        # Desktop: Werte aus den Eingabefeldern lesen + Textzeile aktualisieren;
        # Web: Attribute werden von plot_wahrnehmung() gesetzt.
        if hasattr(self, "_wn_V_var"):
            self._wahrnehmung_lesen()
            self._wahrnehmung_readout(D, f, lam, S_slide, S_real)
        V_cur = getattr(self, "_wn_V", None)

        _, _, Wb, _, S_mar = _wellenfronten(D, f, lam)
        S_exakt = _strehl_exakt(Wb)

        V_krit     = v_krit_naeh(D, Wb)
        V_wende    = v_wendepunkt_naeh(D, Wb)
        V_halb     = v_halb_exakt(D, f, lam)
        V_max_plot = 400.0
        V_min_plot = 30.0
        if V_cur is not None:                 # Achse so erweitern, dass V sichtbar bleibt
            V_max_plot = max(V_max_plot, V_cur * 1.15)
            V_min_plot = min(V_min_plot, V_cur * 0.85)
        V_arr      = np.linspace(V_min_plot, V_max_plot, 150)

        Qe_sph  = perceived_quality_exakt_kurven(D, f, lam, V_arr)
        f_para  = D * 50.0
        Qe_para = perceived_quality_exakt_kurven(D, f_para, lam, V_arr)

        def Q_naeh(V_a, S, Wb_val):
            Vk = v_wendepunkt_naeh(D, Wb_val)
            w  = 1.0 / (1.0 + (Vk / np.maximum(V_a, 1e-9))**2)
            return S + (1 - S) * (1 - w)

        Qn_sph  = Q_naeh(V_arr, S_exakt, Wb)
        Qn_para = Q_naeh(V_arr, 1.0, 1e-9)

        has_slide = (S_slide is not None and S_real is not None
                     and S_slide > S_mar + 0.01)
        if has_slide:
            Wb_s      = _Wb_von_S(S_slide)
            S_ex_s    = S_slide
            S_ex_s_ph = _strehl_exakt(Wb_s)
            Qe_slide  = perceived_quality_exakt_kurven_von_Wb(D, Wb_s, lam, V_arr)
            Qn_slide  = Q_naeh(V_arr, S_ex_s_ph, Wb_s)

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
                (V_krit,  "#534AB7", "--", f"V_krit={V_krit:.0f}×  (10%-Schwelle)"),
                (V_wende, "#c62828", "-.", f"V_wendepunkt≈{V_wende:.0f}×  (Näherung, =V_krit/³√S²)"),
                (V_halb,  "#00838f", ":",  f"V½_exakt={V_halb:.0f}×  (echter Halbwertspunkt)")]:
            if V_min_plot <= V_m <= V_max_plot:
                ax.axvline(V_m, color=col, lw=1.0, ls=ls, alpha=0.65)
                ax.text(V_m + 2, 0.03, lbl,
                        fontsize=7, color=col, rotation=90, va="bottom", alpha=0.8)

        for q_thresh, col, lbl in [(0.9, GRN, "Q=0.90 (sehr gut)"),
                                    (0.7, "#BA7517", "Q=0.70 (spürbar)")]:
            ax.axhline(q_thresh, color=col, lw=1.0, ls=":", alpha=0.7)
            ax.text(V_min_plot * 1.03, q_thresh + 0.008, lbl, fontsize=8, color=col, alpha=0.85)

        # ── Fangspiegel-Effekt auf den Schieber-Spiegel (zusätzliche Kurve) ──
        d_fs = getattr(self, "_wn_fang", 0.0) or 0.0
        S_fs = S_slide if S_slide is not None else S_exakt
        if d_fs > 0:
            Qe_fs = qvis_kurve_mit_fangspiegel(D, S_fs, d_fs, lam, V_arr)
            ax.plot(V_arr, Qe_fs, color="#6a1b9a", lw=2.4, ls="-",
                    label=f"Schieber + Fangspiegel  d={d_fs:.0f}mm "
                          f"(ε={min(d_fs/D, 0.95):.2f}, S={S_fs:.3f})  Näherung")

        # ── Ablese-Cursor: exakter Wert bei der gewählten Vergrößerung ───────
        if V_cur is not None:
            ax.axvline(V_cur, color="#111", lw=1.3, ls="-", alpha=0.8, zorder=4)
            pts = [(perceived_quality_exakt(D, f, lam, V_cur), COR,
                    f"Sphäre {{q:.3f}}"),
                   (1.0, GRN, "Parab. 1.000")]
            if has_slide:
                pts.append((perceived_quality_exakt_von_Wb(D, Wb_s, lam, V_cur),
                            ACC, "Schieber {q:.3f}"))
            if d_fs > 0:
                pts.append((float(qvis_kurve_mit_fangspiegel(D, S_fs, d_fs, lam, V_cur)[0]),
                            "#6a1b9a", "Schieber+FS {q:.3f}"))
            # Beschriftungen von oben nach unten, mit Mindestabstand (Punkte
            # liegen oft dicht beieinander, z.B. Parabol 1.000 / Schieber 0.983)
            axh_pt = fig.get_size_inches()[1] * 72.0 * 0.72      # grobe Achsenhöhe [pt]
            y_top  = 1.28
            vorher_q, vorher_off = None, None
            for q, col, txt in sorted(pts, key=lambda t: -t[0]):
                ax.scatter([V_cur], [q], color=col, s=46, zorder=7,
                           edgecolor="white", linewidth=0.8)
                off = 2.0
                if vorher_q is not None:
                    gap = (vorher_q - q) / y_top * axh_pt        # Abstand der Punkte [pt]
                    off = vorher_off - 11.0 + gap if gap < 11.0 else 2.0
                ax.annotate(txt.format(q=q), xy=(V_cur, q), xytext=(7, off - 5),
                            textcoords="offset points", fontsize=8,
                            fontweight="bold", color=col, zorder=8,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.8))
                vorher_q, vorher_off = q, off

        max_abw = float(np.max(np.abs(Qe_sph - Qn_sph)))
        ax.text(0.015, 0.985,
                f"V_krit (10%-Schwelle, exakt)   = {V_krit:.0f}×   [D/(13·Wb)]\n"
                f"V_wendepunkt (Näherungsformel) = {V_wende:.0f}×   [V_krit/³√(S²)]\n"
                f"V½ (echter Halbwertspunkt)     = {V_halb:.0f}×   [nur via Integral]",
                transform=ax.transAxes, fontsize=7, family="monospace",
                va="top", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.85))
        ax.set_xlim(V_min_plot, V_max_plot)
        ax.set_ylim(0, 1.28 if V_cur is not None else 1.12)   # Kopfraum fuer Info-Kasten + Cursor-Werte
        ax.set_xlabel(T("ax_vergr"), fontsize=10)
        ax.set_ylabel(T("ax_qvis"), fontsize=10)
        ax.set_title(
            f"Visuelle Wahrnehmung  —  D={D:.0f}mm  f/{f/D:.1f}"
            f"  |  S_exakt={S_exakt:.3f}  max.Abw.={max_abw:.3f}"
            + (f"  |  Cursor V={V_cur:.0f}×" if V_cur is not None else ""),
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
        _, _, Wb_orig, _, _ = _wellenfronten(D, f, lam)
        V_krit_orig = v_krit_naeh(D, Wb_orig)
        V_max_orig  = min(V_max_AP, V_krit_orig * 2.0)
        V_min_blend = D_blend / 7.0
        V_max_AP_bl = D_blend / 0.5
        _, _, Wb_bl, _, _ = _wellenfronten(D_blend, f, lam)
        V_krit_bl   = v_krit_naeh(D_blend, Wb_bl)
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
            Vk      = v_krit_naeh(Db, Wb)
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
        L_arr_lum = [L for _, L, _ in objekte_lum]
        # MTF einmal pro Öffnung berechnen, dann über V×L vektorisieren
        Q_bl_mat  = Q_vis_blende_batch(D_blend, f, lam, V_arr, L_arr_lum, D_ref=D)
        Q_ori_mat = Q_vis_blende_batch(D,       f, lam, V_arr, L_arr_lum, D_ref=D)
        gain_mat  = Q_bl_mat / np.maximum(Q_ori_mat, 1e-12)   # (nV, nL)

        for li, (name, L_obj, col) in enumerate(objekte_lum):
            gain = gain_mat[:, li]
            ls = "-" if L_obj >= 500 else ("--" if L_obj >= 5 else ":")
            ax2.plot(V_arr, gain, color=col, lw=1.8, ls=ls,
                     label=f"{name}  (L≈{L_obj:.0f} cd/m²)")

        ax2.axhline(1.0, color="#888", lw=1.2, ls="-")
        ax2.text(V_max_plot * 0.98, 1.02, "Abblenden lohnt sich ↑",
                 fontsize=8, color="#2e7d32", ha="right")
        ax2.text(V_max_plot * 0.98, 0.96, "Abblenden schadet ↓",
                 fontsize=8, color="#c62828", ha="right")

        # Y-Limits dynamisch — gain_mat schon berechnet
        y_max = min(float(gain_mat.max()) * 1.1, 4.0)
        y_min = max(float(gain_mat.min()) * 0.95, 0.3)
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
