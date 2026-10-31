"""
Newton-Teleskop: Kontrast- und Schärfeverlust sphärischer Hauptspiegel
=======================================================================
Streamlit-Version v7.2.5 (2026-10-31)
Entspricht newton_spiegel_rechner.py v7.2.5

Änderungshistorie (jüngste zuerst):
  v7.2.5 (2026-10-31)  Web: kompaktere Seitenleiste (ohne Überschrift "Strehl-Schieber"/"Ansicht" und ohne λ-Angabe, engere Abstände) und
                        einzeiliger Kopf statt großem Titel.
  v7.2.4 (2026-10-31)  Web: Ansichtswahl als Knöpfe in der Seitenleiste (wie Desktop), nur die gewählte Ansicht wird gezeichnet (Eingaben bleiben beim
                        Wechsel erhalten); Kernwerte/Beurteilung nur noch in "Kennwerte"; Blendeneingabe nur im Tab "Blende" (Web) bzw. neu
                        als Eingabefeld im Tab "Blende" (Desktop).
  v7.2.3 (2026-10-31)  MTF-Tab: Kurve für den verbesserten Spiegel (Strehl-Schieber) ergänzt – wurde bisher von der Foucault-Kurve verdrängt;
                        Foucault-Kurve getrennt davon; relative Kurve entfernt.
  v7.2.2 (2026-10-31)  Schieber: Eingabe ist jetzt direkt der Ziel-Strehl in % (statt "Verbesserung" 0-100 %); Werte bis zum Strehl der Kugel
                        (Voreinstellung 0) = Kugel, 100 = Paraboloid. Berechnen-Knopf unter den Eingabefeldern, linke Spalte schmaler.
                        ACHTUNG: gleicher Zahlenwert im Schieber bedeutet jetzt etwas anderes als in v7.2.1 (z. B. 50 = Strehl 0,50).
  v7.2.1 (2026-10-31)  Fehler: Schieber 0 zeigte bei tiefem Strehl (Wb > 0,9, S nicht monoton) eine andere Kurve/V_krit/Q_perc als die Kugel
                        (falscher Ast der S->Wb-Inversion); jetzt gilt bei Schieber 0 die echte Kugel-Wellenfront (_Wb_schieber).
                        Oberfläche: Kernwerte/Beurteilung in Grundlagen > Kennwerte, Tab-Auswahl als Knöpfe unter den Parametern (nur Desktop)
                        (schmale linke Spalte); Vergleich: Spiegel 1 = Hauptspiegel (D, f/D, S vom Schieber), nur Vergleichsspiegel 2
                        wird eingegeben (Voreinstellung 200 mm, S=1). Okular-Kasten aus der Blenden-Skizze entfernt.
  v7.2.0 (2026-10-31)  Wahrnehmung: zusätzliche Q_vis-Kurven MIT Abblenden durch die Augenpupille (Eingabe d_p, Voreinstellung
                        3 mm, 0 = aus; Abblendgrenze V = D/d_p markiert, Cursor- und Tabellenwerte); Kennwerte, V_krit, Q_perc und
                        die bisherigen Kurven unverändert (volle Öffnung). Doku: documentation_v7_2.html
  v7.1.0 (2026-10-30)  Oberfläche neu geordnet: Tab-Gruppen (Grundlagen / Bildqualität / Messen / Grenzen (Max. Vergrößerung + Vmax vs. Ø/Seeing in einem Tab) / Bauen /
                        Vergleich), Grundlagen als 2×2-Übersicht (Ansicht umschaltbar), oben nur Kernwerte (volle
                        Tabelle im Unter-Reiter „Kennwerte“); Paraboloid-V_krit als ∞. Rechenkerne unverändert.
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

Der Rechenkern (alles zwischen "Kernrechnung" und "Plot-Funktionen") ist 1:1
aus newton_spiegel_rechner.py übernommen (ohne tkinter/Sprachumschaltung);
bei Änderungen am Desktop-Kern diesen Block erneut synchronisieren.

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

VERSION = "7.2.5 (2026-10-31)"
EYE_RES = 60.0   # Augenauflösung [arcsec]
BG  = "#f5f5f5"
ACC = "#534AB7"
COR = "#D85A30"
GRN = "#0F6E56"

# ══════════════════════════════════════════════════════════════════════════════
# Kernrechnung (identisch mit newton_spiegel_rechner.py v7.0.0-beta)
# ══════════════════════════════════════════════════════════════════════════════


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


def _Wb_schieber(D_mm: float, f_mm: float, lam_nm: float, S_slide: float) -> float:
    """Wb für die Schieberstellung.  Ohne Verbesserung (S_slide = S der Kugel)
    gilt die echte Kugel-Wellenfront; die Inversion _Wb_von_S würde bei tiefem
    Strehl (Wb > 0,9, S nicht mehr monoton) auf einen falschen Ast fallen und
    bei Schieber 0 eine andere Kurve liefern als die Kugel selbst."""
    Wb_r = _wellenfronten(D_mm, f_mm, lam_nm)[2]
    if S_slide <= _strehl_exakt(Wb_r) + 1e-6:
        return Wb_r
    return _Wb_von_S(S_slide)


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


def perceived_quality_von_S(D_mm: float, S: float, lam_nm: float, V: float,
                            n_pts: int = 120) -> float:
    """Q_perc(V) für einen Spiegel mit vorgegebenem Strehl S (Schieber)."""
    freqs_as, fc, mp, msa, _, S_ = mtf_kurven_von_S(D_mm, S, lam_nm, n_pts)
    return _perceived_quality_kern(freqs_as, fc, mp, msa, S_, V, D_mm, True)


def _qperc_schieber(D_mm, f_mm, lam_nm, S_slide, V):
    """Q_perc für die Schieberstellung; ohne Verbesserung = echte Kugel."""
    Wb_r = _wellenfronten(D_mm, f_mm, lam_nm)[2]
    if S_slide <= _strehl_exakt(Wb_r) + 1e-6:
        return perceived_quality(D_mm, f_mm, lam_nm, V)
    return perceived_quality_von_S(D_mm, S_slide, lam_nm, V)


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
    ax.set_xlim(u0 - 0.25 * L, A + R * 1.45)
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


def _wf_kugel(Wb: float, lam_nm: float = 550.0) -> dict:
    """Wellenfront der Kugel am besten Fokus als Eingabe für qabs(wf=...)."""
    rho = np.linspace(0.0, 1.0, 600)
    return dict(rho=rho, W=4.0 * Wb * (rho ** 4 - rho ** 2), lam=lam_nm)


def qvis_pupille_kurve(D_mm: float, S, V_arr, d_pupil: float,
                       lam_nm: float = 550.0, wf=None) -> np.ndarray:
    """Q_vis(V) MIT Abblenden durch die Augenpupille (v7.2.0).

    Relativ zu einem idealen Spiegel gleicher Öffnung, der von derselben
    Pupille abgeblendet wird (Paraboloid = 1).  Ist die Austrittspupille D/V
    größer als d_pupil, nutzt das Auge nur die Mittelzone ρ_s = V·d_p/D; die
    Aberration wird nur über diese Teilöffnung gerechnet (mit Refokus).
    wf: Wellenfront der Kugel (dict) oder None -> S-Modell (Kugelwellenfront
    zu S, wie beim Schieber).  Ohne Seeing, ohne Fangspiegel.
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
                       d_pupil=d_pupil, wf=wf) for v in gitter])
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
        return "✓ Sehr gut — nahezu gleichwertig mit Parabolspiegel (Strehl ≥ 0.95)", "#2e7d32"
    elif strehl >= 0.80:
        return "⚠  Noch beugungsbegrenzt (Rayleigh), spürbarer Kontrastverlust bei Planeten  (0.80 ≤ Strehl < 0.95)", "#e65100"
    else:
        return "✗  Nicht beugungsbegrenzt — erheblicher Kontrast- und Schärfeverlust, Parabolspiegel dringend empfohlen (Strehl < 0.80)", "#c62828"


# ══════════════════════════════════════════════════════════════════════════════
# Texte / Übersetzungstabelle (für T() in den Diagrammmethoden)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Diagramm-Schicht: 1:1 aus dem Desktop-Programm übernommen
# ══════════════════════════════════════════════════════════════════════════════
# Die Diagramm-Methoden unten sind wörtliche Kopien der _diagramm_*-Methoden
# aus newton_spiegel_rechner.py (v7.0.0-beta). Dadurch sehen Web- und Desktop-
# Version identisch aus; bei Änderungen am Desktop einfach erneut kopieren.
# Nur die Tk-Variablen (StringVar/DoubleVar) werden durch _Var ersetzt.

import types as _types


class _Var:
    """Minimaler Ersatz für tk.StringVar / tk.DoubleVar / tk.BooleanVar."""
    def __init__(self, value=None):
        self._v = value
    def get(self):
        return self._v
    def set(self, v):
        self._v = v


tk = _types.SimpleNamespace(StringVar=_Var, DoubleVar=_Var, BooleanVar=_Var)


class _FakeCanvas:
    """Ersatz für FigureCanvasTkAgg — Streamlit zeigt die Figure selbst an."""
    def draw(self): pass
    def draw_idle(self): pass




class _Diagramme:
    """Hält die Desktop-Diagrammmethoden; Zustand (Optionen) als _Var-Attribute."""

    def __init__(self):
        self._mtf_objekte = _Var("planeten")
        self._mtf_fokus   = _Var("bestfocus")
        self._vmax_objekt     = _Var(OBJEKT_HELLIGKEITEN[1][0])
        self._vmax_L0         = _Var(6000.0)
        self._vmax_seeing     = _Var(1.5)
        self._vmax_D_auge     = _Var(6.0)
        self._vmax_L_schwelle = _Var(1.0)
        self._vmaxD_seeings   = _Var("0.8, 1.5, 2.5, 4.0")
        self._vmaxD_Dmin      = _Var(50.0)
        self._vmaxD_Dmax      = _Var(500.0)
        self._foucault_letzte_kz = None
        self._foucault_letzte_D  = None
        self._wn_V     = None    # Ablese-Cursor (Wahrnehmung)
        self._wn_fang  = 0.0     # Fangspiegel-Ø [mm], 0 = aus

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

        # v7.2.3: Schieber-Kurve (verbesserter Spiegel) UND – falls für dieses D
        # vorhanden – die Foucault-Messung als getrennte Kurven (relative Kurve entfällt).
        kz_fc = getattr(self, '_foucault_letzte_kz', None)
        D_fc  = getattr(self, '_foucault_letzte_D', None)
        if S_slide is not None and S_slide > strehl + 0.002:
            Wb_sl = _Wb_schieber(D, f, lam, S_slide)
            msr_sl = np.array([mtf_sph_rel(nu, Wb_sl, paraxial=paraxial) for nu in nu_k2])
            ax.plot(nu_k2, mp_k2 * msr_sl, color=ACC, lw=2.0, ls="--",
                    label=f"Verbessert (Schieber)  S={S_slide:.3f}")
        if kz_fc is not None and D_fc == D and kz_fc['S_exakt'] > strehl + 0.01:
            msr_fc = np.array([mtf_sph_rel(nu, kz_fc['Wb_fit'], paraxial=paraxial)
                               for nu in nu_k2])
            ax.plot(nu_k2, mp_k2 * msr_fc, color="#1a7abf", lw=1.8, ls="-.",
                    label=f"Foucault gemessen  (S={kz_fc['S_exakt']:.3f})")

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
                     and S_slide > S_exakt + 0.002)
        if has_slide:
            Wb_s      = _Wb_schieber(D, f, lam, S_slide)
            S_ex_s    = S_slide
            S_ex_s_ph = _strehl_exakt(Wb_s)
            Qe_slide  = perceived_quality_exakt_kurven_von_Wb(D, Wb_s, lam, V_arr)
            Qn_slide  = Q_naeh(V_arr, S_ex_s_ph, Wb_s)

        strehl_label = f"Sphäre f/{f/D:.1f}  S={S_exakt:.3f} (exakt)"

        # v7.2.0: Kurven MIT Abblenden durch die Augenpupille (zusätzlich)
        d_pup = getattr(self, "_wn_pup", 0.0) or 0.0
        if d_pup > 0:
            Qp_sph = qvis_pupille_kurve(D, 1.0, V_arr, d_pup, lam, wf=_wf_kugel(Wb, lam))
            if has_slide:
                Qp_slide = qvis_pupille_kurve(D, S_slide, V_arr, d_pup, lam)

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

        if d_pup > 0:
            ax.plot(V_arr, Qp_sph, color=COR, lw=2.4, ls="-.",
                    label=f"Sphäre MIT Augenpupille {d_pup:g} mm  (exakt)")
            if has_slide:
                ax.plot(V_arr, Qp_slide, color=ACC, lw=2.2, ls="-.",
                        label=f"Schieber MIT Augenpupille {d_pup:g} mm  (exakt)")
            V_stop = D / d_pup
            if V_min_plot <= V_stop <= V_max_plot:
                ax.axvline(V_stop, color="#ef6c00", lw=1.2, ls="-.", alpha=0.8)
                ax.text(V_stop + 2, 0.03, f"Abblendgrenze D/d_p = {V_stop:.0f}×",
                        fontsize=7, color="#ef6c00", rotation=90, va="bottom", alpha=0.9)

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
            if d_pup > 0:
                pts.append((float(qvis_pupille_kurve(D, 1.0, [V_cur], d_pup, lam,
                                                     wf=_wf_kugel(Wb, lam))[0]),
                            COR, "Sphäre+Pupille {q:.3f}"))
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
                  title="— exakt (MTF×CSF)   - - Näherung Q̃(V)   -· mit Augenpupille",
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
        V_mirror = v_krit_naeh(D, _Wb_schieber(D, f, 550.0, S_slide))    # Onset: Spiegel allein
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


# ══════════════════════════════════════════════════════════════════════════════
# Plot-Funktionen (Web): dünne Hüllen um die Desktop-Diagrammmethoden
# ══════════════════════════════════════════════════════════════════════════════

def _ax_fmt(ax):
    ax.grid(True, color="#ddd", lw=0.5)
    ax.set_facecolor(BG)


def _neue_figur(figsize, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(BG)
    for a in np.atleast_1d(axes).ravel():
        a.set_facecolor(BG)
    return fig, axes


def plot_strehl(D, f, lam, S_slide):
    r = berechne(D, f, lam)
    fig, ax = _neue_figur((8, 4.5))
    d = _Diagramme(); d._tab_strehl = (fig, ax, _FakeCanvas())
    d._diagramm_strehl(D, f, lam, r["N"], r["strehl"], S_slide)
    return fig


def plot_oeffnung(D, f, lam, S_slide):
    r = berechne(D, f, lam)
    fig, ax = _neue_figur((8, 4.5))
    d = _Diagramme(); d._tab_oeff = (fig, ax, _FakeCanvas())
    d._diagramm_oeffnung(D, f, lam, r["N"], r["Deff_k"],
                         S_sph=r["strehl"], Dk_slide=D * S_slide**0.25)
    return fig


def plot_deff_D(D, N, S_slide):
    fig, ax = _neue_figur((8, 4.5))
    d = _Diagramme(); d._tab_deff_D = (fig, ax, _FakeCanvas())
    d._diagramm_deff_D(D, N, S_slide)
    return fig


def plot_beugung(D, f):
    fig, ax = _neue_figur((8, 5))
    d = _Diagramme(); d._tab_beugung = (fig, ax, _FakeCanvas())
    d._diagramm_beugung(D, f)
    return fig


def plot_mtf(D, f, lam, S_slide, paraxial=False, objekte="planeten",
             kz_fc=None, D_fc=None):
    fig, ax = _neue_figur((8, 4.5))
    d = _Diagramme(); d._tab_mtf = (fig, ax, _FakeCanvas())
    d._mtf_objekte = _Var(objekte)
    d._mtf_fokus   = _Var("paraxial" if paraxial else "bestfocus")
    d._foucault_letzte_kz = kz_fc       # Vergleichskurve aus dem Foucault-Tab
    d._foucault_letzte_D  = D_fc
    d._diagramm_mtf(D, f, lam, S_slide)
    return fig


def plot_wahrnehmung(D, f, lam, S_slide, S_real, V_cur=None, d_fang=0.0, d_pup=0.0):
    fig, ax = _neue_figur((8, 5))
    d = _Diagramme(); d._tab_wahrnehmung = (fig, ax, _FakeCanvas())
    d._wn_V = V_cur
    d._wn_fang = d_fang
    d._wn_pup = d_pup
    d._diagramm_wahrnehmung(D, f, lam, S_slide=S_slide, S_real=S_real)
    return fig


def plot_blende(D, f, lam, D_blend):
    D_bl = min(D_blend, D - 10)
    fig, (ax, ax2) = _neue_figur((9, 8), nrows=2)
    d = _Diagramme(); d._tab_blende = (fig, ax, ax2, _FakeCanvas())
    d._diagramm_blende(D, f, lam, D_bl)
    return fig


def plot_foucault(D, f, lam, r_zon, df_arr, kz, use_direct=False,
                  werkstatt_ansicht=True):
    fig, (ax_wf, ax_info) = _neue_figur((11, 4.8), ncols=2,
                                        gridspec_kw={"width_ratios": [3, 2]})
    d = _Diagramme(); d._tab_foucault = (fig, ax_wf, ax_info, _FakeCanvas())
    d._diagramm_foucault(D, f, lam, r_zon, df_arr, kz, use_direct=use_direct,
                         werkstatt_ansicht=werkstatt_ansicht)
    return fig


def plot_vmax(D, f, S_slide, objekt, L0, seeing, D_auge, L_schwelle):
    fig, ax = _neue_figur((9, 5))
    d = _Diagramme(); d._tab_vmax = (fig, ax, _FakeCanvas())
    d._vmax_objekt.set(objekt);   d._vmax_L0.set(L0)
    d._vmax_seeing.set(seeing);   d._vmax_D_auge.set(D_auge)
    d._vmax_L_schwelle.set(L_schwelle)
    d._diagramm_vmax(D, f, S_slide)
    return fig


def plot_vmax_D(seeings_text, Dmin, Dmax):
    fig, ax = _neue_figur((8, 4.8))
    d = _Diagramme(); d._tab_vmax_D = (fig, ax, _FakeCanvas())
    d._vmaxD_seeings.set(seeings_text)
    d._vmaxD_Dmin.set(Dmin); d._vmaxD_Dmax.set(Dmax)
    d._diagramm_vmax_D()
    return fig


def plot_fangspiegel(D, f, lam, S_u, S_txt, r, V):
    fig, (ax, ax2) = _neue_figur((13, 5.4), 1, 2, gridspec_kw={"width_ratios": [1, 1.15]})
    fangspiegel_zeichnen(ax, D, f, lam, S_u, S_txt, r, V)
    fangspiegel_skizze(ax2, D, f, r)
    fig.tight_layout()
    return fig


def plot_vergleich(p):
    extra = bool(p.get("wf1") or p.get("wf2") or p.get("sph1") or p.get("sph2"))
    fig = plt.figure(figsize=(11, 10.6 if extra else 7.4))
    ax1, ax2, axb, axw = vergleich_achsen(fig, extra=extra)
    zeilen = vergleich_zeichnen(ax1, ax2, p, axb, axw)
    fig.tight_layout()
    return fig, zeilen


def main():
    st.set_page_config(
        page_title=f"Newton-Spiegel  v{VERSION}",
        page_icon="🔭",
        layout="wide",
    )

    # v7.2.4: Ansichten werden nur noch bei Auswahl gerendert (Knöpfe in der Seitenleiste).
    # Streamlit verwirft den Zustand von Widgets, die in einem Lauf nicht gezeichnet werden;
    # deshalb alle bestehenden Werte jedes Mal neu zuweisen (Foucault-Eingaben usw. bleiben erhalten).
    for _k in list(st.session_state.keys()):
        if _k == "nav" or str(_k).startswith(("navbtn_", "$$")):
            continue
        try:
            st.session_state[_k] = st.session_state[_k]
        except Exception:
            pass

    # v7.2.5: kompakter Kopf (eine kleine Zeile) und engere Seitenleiste
    st.markdown(
        """<style>
        .block-container {padding-top: 3rem; padding-bottom: 1rem;}
        section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {gap: 0.35rem;}
        section[data-testid="stSidebar"] button {min-height: 1.9rem; padding-top: 0.1rem; padding-bottom: 0.1rem;}
        </style>""", unsafe_allow_html=True)
    st.markdown(f"**🔭 Newton-Spiegel** <span style='color:#888;font-size:0.85em'>"
                f"Kontrast- und Schärfeverlust sphärischer Hauptspiegel · v{VERSION}</span>",
                unsafe_allow_html=True)

    # ── Sidebar: Teleskop-Parameter ──────────────────────────────────────────
    st.sidebar.subheader("Teleskop-Parameter")

    D   = st.sidebar.slider("Öffnung D [mm]",    min_value=50,  max_value=600,  value=200, step=5)
    f   = st.sidebar.slider("Brennweite f [mm]", min_value=200, max_value=6000, value=1000, step=50)
    lam = 550.0  # fix, λ=550nm

    N = f / D
    st.sidebar.markdown(f"**f/D = {N:.2f}**")

    S_pct = st.sidebar.slider(
        "Strehl [%]", 0, 100, 0,
        help="0 = reale Kugel; alles darüber ist der Ziel-Strehl in Prozent (simuliert "
             "Figurierungsfortschritt), 100 = Paraboloid. Liegt der Wert unter dem Strehl der Kugel, "
             "gilt die Kugel.")

    st.sidebar.markdown("---")
    _NAV = [("Grundlagen", [("grund", "Diagramme"), ("kenn", "Kennwerte")]),
            ("Bildqualität", [("wahr", "Wahrnehmung"), ("mtf", "MTF"), ("blende", "Blende")]),
            ("", [("messen", "Messen (Foucault)"), ("grenzen", "Grenzen"), ("bauen", "Bauen"),
                  ("vergleich", "Vergleich (Beta)"), ("doku", "ℹ️ Dokumentation")])]
    if "nav" not in st.session_state:
        st.session_state["nav"] = "grund"

    def _set_nav(k):
        st.session_state["nav"] = k

    for _gname, _items in _NAV:
        if _gname:
            st.sidebar.caption(_gname)
        for _k, _lbl in _items:
            st.sidebar.button(_lbl, key=f"navbtn_{_k}", use_container_width=True,
                              type="primary" if st.session_state["nav"] == _k else "secondary",
                              on_click=_set_nav, args=(_k,))
    _nav = st.session_state["nav"]

    # ── Rechnung ─────────────────────────────────────────────────────────────
    r_real     = berechne(D, f, lam)
    S_real     = r_real["strehl"]
    S_mar      = r_real["strehl_marechal"]
    S_slide    = S_pct / 100.0                    # v7.2.2: Eingabe = Ziel-Strehl in %
    S_slide    = min(max(S_slide, S_real), 1.0)
    Deff_k_sl  = D * S_slide**0.25
    Vk         = v_kritisch(D, f, lam, EYE_RES)
    Vk_slide   = v_kritisch_strehl(D, S_slide)
    Vk_slide["Vk_sph"] = v_krit_naeh(D, _Wb_schieber(D, f, lam, S_slide))
    rv         = berechne_vergr(D, f, lam, 150.0, EYE_RES)
    Vk_naeh    = Vk["Vk_sph"]
    Vk_naeh_s  = Vk_slide["Vk_sph"]
    Vk_blur    = v_krit_blur_direkt(D, f, EYE_RES)
    Vk_exakt   = v_krit_aus_qvis(D, f, lam, rel_schwelle=0.90)
    Wb_s       = _Wb_schieber(D, f, lam, S_slide)
    Vk_exakt_s = v_krit_aus_qvis_von_Wb(D, Wb_s, lam, rel_schwelle=0.90)
    Vk_halb    = v_halb_exakt(D, f, lam)
    Qp_80      = perceived_quality(D, f, lam, 30.0)
    Qp_200     = perceived_quality(D, f, lam, 80.0)
    Qp_350     = perceived_quality(D, f, lam, 160.0)
    Qp_V       = perceived_quality(D, f, lam, 150.0)
    Qs_80, Qs_200, Qs_350, Qs_V = (_qperc_schieber(D, f, lam, S_slide, v)
                                   for v in (30.0, 80.0, 160.0, 150.0))

    verdict_text, verdict_color = beurteilung(S_slide)
    if S_slide > S_real + 1e-6:
        verdict_text = f"Mit Verbesserung (S={S_slide:.2f}):  {verdict_text}"

    # ── Kennzahlen-Übersicht (kompakt) ──────────────────────────────────────
    def _qc(q):
        return "#2e7d32" if q >= 0.90 else "#e65100" if q >= 0.70 else "#c62828"

    def _vk_ex_text(v):
        return f"{v:.0f}x" if v > 30.5 else "< 30x"

    delta_s    = S_real - S_mar
    delta_halb = Vk_halb - Vk_naeh
    avp        = r_real["aufl_verlust_pct"]
    alpha_blur = D**3 / (64 * f**2) / f * 206265

    s_col = "#2e7d32" if S_real >= 0.95 else "#e65100" if S_real >= 0.80 else "#c62828"

    _verdict_html = (
        f"<div style='background:{verdict_color};color:white;padding:5px 10px;"
        f"border-radius:5px;font-size:0.82em;font-weight:bold;margin-bottom:5px'>"
        f"{verdict_text}</div>")

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
                 f"{S_real:.4f} &#8594; {S_slide:.4f}  (&#916; Mar. {delta_s:+.4f})", s_col)
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
                 f"{Vk['Vk_sph']:.0f}x &#8594; {Vk_slide['Vk_sph']:.0f}x  | Para: {'&#8734;' if Vk['Vk_para'] >= 1e8 else format(Vk['Vk_para'], '.0f') + 'x'}  | Vmax: {Vk['Vmax']:.0f}x",
                 "#534AB7")
    rows += _hdr("Q_perc (CSF-gewichtet)")
    for lbl_q, q, qs in [("Q bei  30x", Qp_80, Qs_80), ("Q bei  80x", Qp_200, Qs_200),
                         ("Q bei 160x", Qp_350, Qs_350), ("Q bei 150x", Qp_V, Qs_V)]:
        rows += _row(lbl_q, f"{q:.3f} &#8594; {qs:.3f}", _qc(qs))

    _kenn_html = f"<table style='border-collapse:collapse;width:100%;margin-bottom:4px'>{rows}</table>"

    # v7.2.4: Kernwerte, Beurteilung und Tabelle erscheinen nur in der Ansicht "Kennwerte".
    if _nav == "kenn":
        st.markdown(_verdict_html, unsafe_allow_html=True)
        _k1, _k2, _k3, _k4 = st.columns(4)
        _k1.metric("Strehl exakt", f"{S_slide:.3f}",
                   delta=f"Sphäre {S_real:.3f}" if S_slide > S_real + 1e-6 else None,
                   delta_color="off")
        _k2.metric("V_krit (MTF×CSF >10 %)", _vk_ex_text(Vk_exakt_s),
                   delta=f"Sphäre {_vk_ex_text(Vk_exakt)}" if S_slide > S_real + 1e-6 else None,
                   delta_color="off")
        _k3.metric("D_eff Kontrast", f"{Deff_k_sl:.0f} mm",
                   delta=f"Sphäre {r_real['Deff_k']:.0f} mm" if S_slide > S_real + 1e-6 else None,
                   delta_color="off")
        _k4.metric("Q_perc bei 150×", f"{Qs_V:.3f}",
                   delta=f"Sphäre {Qp_V:.3f}" if S_slide > S_real + 1e-6 else None,
                   delta_color="off")

        st.markdown(_kenn_html, unsafe_allow_html=True)

    # ── Lazy Tab-Rendering: Figure als PNG-Bytes im session_state ─────────────
    # Figures werden nur neu gezeichnet wenn sich die relevanten Parameter
    # geändert haben. Matplotlib-Figures sind nicht pickle-bar → PNG-Bytes.
    import io as _io

    def _fig_to_png(fig):
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _show_cached(key, build_fn, param_hash):
        """Zeigt gecachten Plot oder zeichnet neu wenn param_hash sich geändert hat."""
        hash_key = key + "_hash"
        if st.session_state.get(hash_key) != param_hash or key not in st.session_state:
            with st.spinner("Berechne…"):
                fig = build_fn()
                st.session_state[key]      = _fig_to_png(fig)
                st.session_state[hash_key] = param_hash
        st.image(st.session_state[key], use_container_width=True)

    _h = (D, f, lam, round(S_slide, 4))   # Basis-Hash ohne Blende/MTF-Optionen

    if _nav == "grund":
        _ans = st.radio("Ansicht", ["Übersicht (2×2)", "Strehl vs. f/D",
                                    "Eff. Öffnung vs. f/D", "D_eff vs. Öffnung",
                                    "Beugungsgrenze"], horizontal=True, key="grund_ansicht")
        _g = {"Strehl vs. f/D":       lambda: _show_cached("fig_strehl",   lambda: plot_strehl(D, f, lam, S_slide),   _h),
              "Eff. Öffnung vs. f/D": lambda: _show_cached("fig_oeffnung", lambda: plot_oeffnung(D, f, lam, S_slide), _h),
              "D_eff vs. Öffnung":    lambda: _show_cached("fig_deff_D",   lambda: plot_deff_D(D, N, S_slide),        _h),
              "Beugungsgrenze":       lambda: _show_cached("fig_beugung",  lambda: plot_beugung(D, f),                (D, f))}
        if _ans == "Übersicht (2×2)":
            _ga, _gb = st.columns(2)
            with _ga:
                _g["Strehl vs. f/D"]()
                _g["D_eff vs. Öffnung"]()
            with _gb:
                _g["Eff. Öffnung vs. f/D"]()
                _g["Beugungsgrenze"]()
            st.caption("Ansicht oben umschalten = einzelnes Diagramm groß.")
        else:
            _g[_ans]()

    if _nav == "mtf":
        c1, c2 = st.columns(2)
        with c1:
            fokus_mode = st.radio("Fokusebene:",
                                  ["Best Focus  (ρ⁴−ρ²)", "Paraxialer Fokus  (ρ⁴)"],
                                  horizontal=True, key="mtf_fokus")
        with c2:
            objekte_mode = st.radio("Objekte:", ["Planeten", "Deep-Sky (M42)", "Alle"],
                                    horizontal=True, key="mtf_objekte")
        paraxial  = "Paraxial" in fokus_mode
        objekte_k = {"Planeten": "planeten", "Deep-Sky (M42)": "deepsky", "Alle": "alle"}[objekte_mode]
        # Vergleichskurve: zuletzt im Foucault-Tab gemessene Wellenfront (falls
        # für diese Öffnung vorhanden), sonst Strehl-Schieber — wie im Desktop.
        _kz_fc = st.session_state.get("fc_last_kz")
        _D_fc  = st.session_state.get("fc_last_D")
        _fc_key = (round(float(_kz_fc["S_exakt"]), 4), _D_fc) if _kz_fc is not None else None
        _show_cached("fig_mtf", lambda: plot_mtf(D, f, lam, S_slide,
                     paraxial=paraxial, objekte=objekte_k,
                     kz_fc=_kz_fc, D_fc=_D_fc),
                     _h + (paraxial, objekte_k, _fc_key))

    if _nav == "wahr":
        # ── Ablesen: exakte Q_vis-Werte bei gewählter Vergrößerung ───────────
        # Spiegel = Regler in der Seitenleiste (Öffnung, Brennweite, Strehl-Schieber)
        wc1, wc2, wc3 = st.columns([3, 1.3, 1.3])
        d_pup = wc3.number_input(
            "Augenpupille d_p [mm]  (0 = aus)", min_value=0.0, max_value=9.0,
            value=3.0, step=0.5, key="wn_pup",
            help="Zusätzliche Kurven MIT Abblenden durch die Augenpupille: Ist die "
                 "Austrittspupille D/V größer als d_p, nutzt das Auge nur die Mitte "
                 "des Spiegels. Richtwerte: Mond 2–3 mm, Planeten 3–5 mm, Deep-Sky 6–7 mm.")
        d_fs = wc2.number_input(
            "Fangspiegel Ø d [mm]  (0 = aus)", min_value=0, max_value=int(D * 0.9),
            value=0, step=5, key="wn_fang",
            help=f"Zusätzliche Kurve: Schieber-Spiegel MIT Fangspiegel-Abschattung "
                 f"(ε = d/D). Mindestgröße für diese Öffnung/Brennweite: "
                 f"{d_fang_min(D, f):.0f} mm. Näherung: Aberration bleibt unverändert.")
        V_cur = wc1.slider("Ablesen bei Vergrößerung V [×]", 20, 800, 150, 1,
                          key="wn_V",
                          help="Senkrechte Linie im Diagramm; die exakten Q_vis-Werte "
                               "stehen darunter und als Punkte im Diagramm. Der Spiegel "
                               "ergibt sich aus Öffnung, Brennweite und Strehl-Schieber.")
        _vt = lambda v: ("≥ 600×" if v >= 599.5 else "≤ 30×" if v <= 30.5 else f"{v:.0f}×")
        _vk = lambda v: ("∞" if v >= 1e8 else f"{v:.0f}×")
        _q_sph = perceived_quality_exakt(D, f, lam, V_cur)
        _S_sph = _strehl_exakt(_wellenfronten(D, f, lam)[2])
        _zeilen = [
            "| Spiegel | Q_vis exakt | Q̃ Näherung | Verlust ggü. Parabol | Q ≥ 0.90 bis | V_krit |",
            "|---|---|---|---|---|---|",
            f"| Sphäre  D={D} mm, f/{f/D:.1f}, S={_S_sph:.3f} "
            f"| **{_q_sph:.4f}** | – | {(1-_q_sph)*100:.1f} % | – | – |",
        ]
        if S_slide > S_real + 0.01:
            _es = qvis_einzelrechner(D, S_slide, V_cur, lam)
            _zeilen.append(f"| Schieber  S={S_slide:.3f} | **{_es['Q']:.4f}** | {_es['Q_naeh']:.4f} "
                           f"| {_es['verlust']:.1f} % | {_vt(_es['V_q90'])} | {_vk(_es['V_krit'])} |")
        if d_fs > 0:
            _S_u = S_slide
            _q_fs = float(qvis_kurve_mit_fangspiegel(D, _S_u, d_fs, lam, V_cur)[0])
            _zeilen.append(f"| Schieber + Fangspiegel d={d_fs:.0f} mm (ε={min(d_fs/D, 0.95):.2f}) "
                           f"| **{_q_fs:.4f}** | – | {(1-_q_fs)*100:.1f} % | – | – |")
        if d_pup > 0:
            _q_pk = float(qvis_pupille_kurve(D, 1.0, [V_cur], d_pup, lam,
                                             wf=_wf_kugel(_wellenfronten(D, f, lam)[2], lam))[0])
            _blend = "Auge blendet ab" if D / V_cur > d_pup else "volle Öffnung"
            _zeilen.append(f"| Sphäre MIT Augenpupille {d_pup:g} mm (Austrittspupille {D/V_cur:.1f} mm, {_blend}) "
                           f"| **{_q_pk:.4f}** | – | {(1-_q_pk)*100:.1f} % | – | – |")
            if S_slide > S_real + 0.01:
                _q_ps = float(qvis_pupille_kurve(D, S_slide, [V_cur], d_pup, lam)[0])
                _zeilen.append(f"| Schieber MIT Augenpupille {d_pup:g} mm | **{_q_ps:.4f}** | – "
                               f"| {(1-_q_ps)*100:.1f} % | – | – |")
        _zeilen.append("| Paraboloid | **1.0000** | – | 0.0 % | – | – |")
        st.markdown(f"**V = {V_cur}×** (λ = {lam:.0f} nm)\n\n" + "\n".join(_zeilen))

        _show_cached("fig_wahrnehmung", lambda: plot_wahrnehmung(D, f, lam,
                     S_slide, S_real, V_cur=V_cur, d_fang=d_fs, d_pup=d_pup),
                     _h + (V_cur, d_fs, d_pup))

    if _nav == "blende":
        # v7.2.4: Blendeneingabe nur hier (vorher verwirrend in der Seitenleiste)
        if st.session_state.get("blende_D", 0) > D:
            st.session_state["blende_D"] = D
        D_blend = st.number_input("Blende D [mm]  (Abblenden des Spiegels)", min_value=50, max_value=int(D),
                                  value=int(min(max(50, D // 2), D)), step=5, key="blende_D")
        _show_cached("fig_blende", lambda: plot_blende(D, f, lam, D_blend), _h + (D_blend,))

    if _nav == "messen":
        # Foucault-Tab
        st.subheader("Foucault-Auswertung")
        lam_fc = 550.0   # fix
        fc1, fc2 = st.columns([1, 3])
        with fc1:
            n_zonen = st.radio("Zonen:", [4, 6, 8], horizontal=True, index=0, key="fc_n_zonen")
            use_direct = st.radio(
                "Auswertungs-Modus:",
                ["Polynom-Fit  (glatte Fläche)", "Hybrid  (Zonenfehler)"],
                index=0, horizontal=False, key="fc_modus",
                help="Polynom-Fit: glatte Fläche aus geraden ρ-Potenzen — dämpft "
                     "Messrauschen, mittelt aber lokale Zonenfehler weg.\n"
                     "Hybrid: Polynom + stückweise konstante Zonenresiduen — erfasst "
                     "Ringfehler. Braucht mindestens 6 Zonen; empfohlen, wenn das "
                     "Fit-Residuum > 0.010 λ ist.")
            use_direct = use_direct.startswith("Hybrid")
            grad_label = st.radio(
                "Polynomgrad:", ["BIC (auto)", "Fest (n//2)"],
                index=0, horizontal=True, key="fc_grad",
                help="BIC wählt den Grad automatisch (kann bei rauschigen Daten sehr "
                     "niedrig ausfallen). Fest koppelt den Grad an die Zonenzahl.")
            grad_modus = "fest" if grad_label.startswith("Fest") else "bic"
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

        # Berechnen — expliziter Button verhindert Neurechnung bei jeder Eingabe
        st.markdown("---")
        col_btn, col_hint = st.columns([1, 4])
        do_calc = col_btn.button("▶  Berechnen", type="primary", use_container_width=True)
        col_hint.caption(
            "Alle Zonenwerte eingeben, dann **Berechnen** klicken. "
            "Die übrigen Diagramm-Tabs werden durch Zoneneingaben nicht neu berechnet.")

        # Letztes Ergebnis im session_state zwischenspeichern
        _fc_res_key = f"fc_result_{D}_{f}_{n_zonen}"
        if do_calc:
            df_arr = np.array(df_vals)
            try:
                W_z = foucault_wellenfront(r_zon, df_arr, D, f, lam_fc)
                # Hybrid braucht >= 6 Zonen (wie Desktop); sonst Polynom-Modus
                _ud_eff = use_direct and (n_zonen >= 6)
                kz  = foucault_kennzahlen(W_z, r_zon, D, lam_fc,
                                          use_direct=_ud_eff, grad_modus=grad_modus)
                st.session_state[_fc_res_key] = (df_arr, kz, use_direct, _ud_eff)
                # für die MTF-Vergleichskurve (wie _foucault_letzte_kz im Desktop)
                st.session_state["fc_last_kz"] = kz
                st.session_state["fc_last_D"]  = D
            except Exception as _e:
                st.error(f"Foucault-Berechnungsfehler: {_e}")
                kz = None
        elif _fc_res_key in st.session_state:
            _stored_df, kz, use_direct, _ud_eff = st.session_state[_fc_res_key]
            df_arr = _stored_df
        else:
            kz = None

        if kz is not None:
         try:
            W_z = foucault_wellenfront(r_zon, df_arr, D, f, lam_fc)  # für plot_foucault

            rayleigh_ok  = kz["S_exakt"] >= 0.80
            zuverlaessig = kz.get("kennzahlen_zuverlaessig", True)
            s_color      = ("red" if not zuverlaessig else
                            "green" if kz["S_exakt"] >= 0.95 else
                            "orange" if rayleigh_ok else "red")
            status_icon  = "✓" if (rayleigh_ok and zuverlaessig) else "✗"
            modus_label  = "Hybrid" if _ud_eff else "Polynom-Fit"
            modus_color  = "#6a1b9a" if _ud_eff else "#1565c0"
            res_rms      = kz["fit_residuum_rms"]
            warn_txt = ""
            if use_direct and n_zonen < 6:
                warn_txt = " &nbsp;INFO: Hybrid braucht n≥6 Zonen — Polynom-Modus verwendet"
            elif not zuverlaessig:
                warn_txt = (f" &nbsp;⚠ UNZUVERLÄSSIG: Fit-Residuum={res_rms:.4f}λ &gt; 0.010λ "
                            f"→ Hybrid-Modus (n≥6) verwenden!")
            elif not _ud_eff:
                warn_txt = f" &nbsp;Fit-Residuum={res_rms:.4f}λ"
            # Zonen noch nahe Kugelfehler? (wie Desktop: >80 % der Parabelkorrektur fehlen)
            _dl_soll = (r_zon**2 - r_zon[0]**2) / (4.0 * f)
            _fehlend = np.zeros_like(_dl_soll)
            np.divide(_dl_soll - df_arr, _dl_soll, out=_fehlend, where=_dl_soll > 0.001)
            _kugel = [f"Z{i+1}" for i, fe in enumerate(_fehlend) if fe > 0.80 and i > 0]
            if _kugel:
                warn_txt += (f" &nbsp;!! {', '.join(_kugel)} noch auf Kugelnähe — "
                             f"S nicht repräsentativ!")
                s_color = "red"
            st.markdown(
                f"<div style='background-color:{'#e8f5e9' if (rayleigh_ok and zuverlaessig) else '#ffebee'};"
                f"padding:8px;border-radius:6px;margin:8px 0'>"
                f"<b style='color:{s_color}'>{status_icon} "
                f"S_exakt = {kz['S_exakt']:.4f} &nbsp;|&nbsp; "
                f"W_rms = {kz['W_rms']:.4f}λ &nbsp;|&nbsp; "
                f"W_PtV = {kz['W_ptv']:.4f}λ &nbsp;|&nbsp; "
                f"Abtrag PtV = {kz['Glas_ptv_nm']:.1f} nm &nbsp;|&nbsp; "
                f"Abtrag V = {kz['Glas_volumen_mm3']:.1f} mm³ &nbsp;|&nbsp; "
                f"D_eff = {kz['Deff_k']:.1f} mm &nbsp;|&nbsp; "
                f"{'✓ Rayleigh OK' if rayleigh_ok else '✗ Rayleigh nicht erfüllt'}"
                f"</b>"
                f"<span style='color:{modus_color};font-size:0.85em;font-weight:normal'>"
                f" &nbsp;|&nbsp; Modus: {modus_label} (Grad {kz.get('poly_grad', '?')}){warn_txt}</span>"
                f"</div>",
                unsafe_allow_html=True)

            werkstatt_ansicht = st.toggle(
                "🔧 Werkstatt-Ansicht (wie FigureXP)",
                value=True, key="fc_werkstatt_ansicht",
                help="Zeigt die Fläche wie FigureXP („Surface Error / Glass Remaining“): "
                     "Null im Tal (bei ca. 0,7 R), Hügel = dort muss noch Glas runter. "
                     "Nur die Darstellung ändert sich — S_exakt/W_rms/W_PtV bleiben "
                     "unverändert. Aus: Wellenfront W(ρ) im besten Fokus.")
            st.pyplot(plot_foucault(D, f, lam_fc, r_zon, df_arr, kz,
                                    use_direct=_ud_eff,
                                    werkstatt_ansicht=werkstatt_ansicht))

         except Exception as _ex:
            st.error(f"Foucault-Berechnungsfehler: {_ex}")

    if _nav == "grenzen":
        # ── Sinnvolle Maximalvergrößerung (Helligkeit / Seeing / Spiegel) ───
        st.markdown(
            "Gegenprobe zur „2×D“-Faustregel: Netzhaut-Flächenhelligkeit L(V) sowie die "
            "Grenzen durch Helligkeit, Seeing und Spiegelfehler (aktueller Strehl-Schieber).")
        _namen = [n for n, _ in OBJEKT_HELLIGKEITEN]
        _preset = dict(OBJEKT_HELLIGKEITEN)

        def _objekt_gewaehlt():
            st.session_state["vmax_L0"] = float(_preset[st.session_state["vmax_objekt"]])

        if "vmax_L0" not in st.session_state:
            st.session_state["vmax_L0"] = float(_preset[OBJEKT_HELLIGKEITEN[1][0]])
        vc1, vc2, vc3, vc4, vc5 = st.columns([2, 2, 1.5, 1.5, 2])
        objekt_v = vc1.selectbox("Objekt:", _namen, index=1, key="vmax_objekt",
                                 on_change=_objekt_gewaehlt,
                                 help="Füllt L₀ mit dem Referenzwert des Objekts; "
                                      "danach frei überschreibbar.")
        L0_v = vc2.number_input("L₀ [cd/m²]", min_value=0.0, step=10.0,
                                format="%.1f", key="vmax_L0")
        seeing_v = vc3.number_input("Seeing [\"]", min_value=0.0, value=1.5,
                                    step=0.1, format="%.1f", key="vmax_seeing")
        D_auge_v = vc4.number_input("Augenpupille [mm]", min_value=0.5, value=6.0,
                                    step=0.5, format="%.1f", key="vmax_D_auge")
        L_schw_v = vc5.number_input("Schwelle [cd/m²]", min_value=0.0, value=1.0,
                                    step=0.1, format="%.2f", key="vmax_L_schwelle")
        _show_cached("fig_vmax",
                     lambda: plot_vmax(D, f, S_slide, objekt_v, L0_v, seeing_v,
                                       D_auge_v, L_schw_v),
                     _h + (objekt_v, L0_v, seeing_v, D_auge_v, L_schw_v))

    if _nav == "grenzen":
        st.markdown("---")
        # ── Vmax vs. Öffnung für verschiedene Seeing-Werte (perfekte Optik) ──
        st.markdown(
            "Fundamentale Grenze auch bei perfekter Politur (S=1): Beugung + Seeing, "
            "im Vergleich zur „2×D“-Regel.")
        wc1, wc2, wc3 = st.columns([3, 1, 1])
        seeings_txt = wc1.text_input("Seeing-Werte [\"], kommagetrennt",
                                     value="0.8, 1.5, 2.5, 4.0", key="vmaxD_seeings")
        Dmin_v = wc2.number_input("D min [mm]", min_value=10.0, value=50.0,
                                  step=10.0, key="vmaxD_Dmin")
        Dmax_v = wc3.number_input("D max [mm]", min_value=20.0, value=500.0,
                                  step=10.0, key="vmaxD_Dmax")
        _show_cached("fig_vmax_D",
                     lambda: plot_vmax_D(seeings_txt, Dmin_v, Dmax_v),
                     (seeings_txt, Dmin_v, Dmax_v))

    if _nav == "bauen":
        st.markdown(
            "Mindest-Ø nach Strahlkegel-Geometrie:  **d = (Tubusradius + Okularauszug) · D/f + Bildfeld**.")
        fc1, fc2, fc3, fc4 = st.columns(4)
        buet_v  = fc1.number_input("Okularauszug (Fokus über Tubus) [mm]", min_value=0,
                                   value=50, step=5, key="fs_buet")
        tubus_v = fc2.number_input("Tubus-Außen-Ø [mm] (0 = wie Spiegel)", min_value=0,
                                   value=0, step=10, key="fs_tubus")
        bild_v  = fc3.number_input("100%-Bildfeld [mm] (visuell 10, Foto 20–25)", min_value=0,
                                   value=10, step=1, key="fs_bild")
        V_fs    = fc4.number_input("Q_vis bei V [×]", min_value=1, value=150,
                                   step=10, key="fs_V")
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        hut_v  = gc1.number_input("Hutring-Ø für Blende [mm] (0 = Tubus-Ø)", min_value=0,
                                  value=0, step=10, key="fs_hut")
        fe_v   = gc2.number_input("Übersichtsokular f [mm]", min_value=1,
                                  value=21, step=1, key="fs_fe")
        afov_v = gc3.number_input("Okular-AFOV [°]", min_value=1,
                                  value=85, step=1, key="fs_afov")
        oazd_v = gc4.number_input("OAZ-Rohr Innen-Ø [mm]", min_value=1,
                                  value=50, step=1, key="fs_oazd")
        oazl_v = gc5.number_input("OAZ-Rohr Länge [mm]", min_value=1,
                                  value=80, step=5, key="fs_oazl")
        if S_slide is not None:
            S_fs, S_fs_txt = S_slide, "Schieber"
        else:
            _, _, _Wb_s, _, _ = _wellenfronten(D, f, lam)
            S_fs, S_fs_txt = _strehl_exakt(_Wb_s), "Sphäre"
        r_fs = fangspiegel_rechner(D, f, buet_v, tubus_v, bild_v, S_fs, V_fs, lam,
                                   hut_v, fe_v, afov_v, oazd_v, oazl_v)
        st.markdown(fangspiegel_text(D, f, S_fs, S_fs_txt, r_fs, V_fs).replace("\n", "  \n"))
        _show_cached("fig_fangspiegel",
                     lambda: plot_fangspiegel(D, f, lam, S_fs, S_fs_txt, r_fs, V_fs),
                     (D, f, lam, S_fs, buet_v, tubus_v, bild_v, V_fs, hut_v, fe_v, afov_v, oazd_v, oazl_v))

    if _nav == "vergleich":
        st.markdown("**Beta:** absoluter Vergleich zweier Spiegel gegen einen festen Referenzspiegel "
                    "(Q_abs). Fangspiegel automatisch aus D und f/D.")
        # v7.2.1: Spiegel 1 = Hauptspiegel (Seitenleiste, S = Schieberwert), f/D vom Hauptspiegel;
        # nur der Vergleichsspiegel 2 wird hier eingegeben.
        vD1 = float(D)
        if S_slide is not None:
            vS1 = float(S_slide)
        else:
            _, _, _Wb1, _, _ = _wellenfronten(D, f, lam)
            vS1 = float(_strehl_exakt(_Wb1))
        vS1 = min(max(vS1, 0.05), 1.0)
        vN = f / D
        st.info(f"**Spiegel 1 = Hauptspiegel:** D = {vD1:.0f} mm, f/D = {vN:.2f}, S = {vS1:.3f}"
                f"{' (Schieber)' if S_slide is not None else ''}  ·  Spiegel 2 hat dasselbe f/D.")
        c3, c4, _c5 = st.columns([1, 1, 3])
        vD2 = c3.number_input("Vergleichsspiegel 2: D [mm]", 50, 1500, 200, 10, key="vg_D2")
        vS2 = c4.number_input("Vergleichsspiegel 2: S", 0.05, 1.0, 1.00, 0.01, format="%.2f", key="vg_S2")
        d1c, d2c, d3c, d4c, d5c, d6c = st.columns(6)
        vV  = d1c.number_input("V [×]", 10, 1000, 200, 10, key="vg_V")
        vSe = d2c.number_input("Seeing [″] (0 = aus)", 0.0, 6.0, 1.5, 0.1, key="vg_see")
        vRD = d3c.number_input("Referenz: idealer D [mm]", 50, 1500, 200, 10, key="vg_refD")
        vBu = d4c.number_input("Okularauszug [mm]", 0, 300, 50, 5, key="vg_buet")
        vBi = d5c.number_input("Bildfeld [mm]", 0, 60, 10, 1, key="vg_bild")
        vPu = d6c.number_input("Augenpupille [mm] (0 = aus)", 0.0, 9.0, 0.0, 0.5, key="vg_pupille",
                               help="Ist die Austrittspupille D/V größer als die Augenpupille, nutzt das Auge nur "
                                    "die Teilöffnung V·Pupille (blendet das Teleskop ab). Mond 2–3 mm, Planeten "
                                    "3–5 mm, Deep-Sky 6–7 mm.")
        _kz = st.session_state.get("fc_last_kz")
        _Df = st.session_state.get("fc_last_D")
        w1c, w2c = st.columns(2)
        use_w1 = w1c.checkbox("Spiegel 1 aus Foucault-Test (Wellenfront)", key="vg_wf1")
        use_w2 = w2c.checkbox("Spiegel 2 aus Foucault-Test (Wellenfront)", key="vg_wf2")
        s1c, s2c = st.columns(2)
        sph1 = s1c.checkbox("Spiegel 1 sphärisch (aus D, f/D)", key="vg_sph1")
        sph2 = s2c.checkbox("Spiegel 2 sphärisch (aus D, f/D)", key="vg_sph2")
        v_auge = st.checkbox("Bild mit Augen-Filter bei V (Barten-Tiefpass)", value=True, key="vg_auge")
        v_mass = st.checkbox("Augenmaßstab (Planet wächst mit V)", value=True, key="vg_mass")
        _notiz = []
        _wf = {}
        if (use_w1 or use_w2) and (_kz is None or _Df is None):
            _notiz.append("Keine Foucault-Daten vorhanden — erst im Foucault-Tab Messwerte eingeben.")
        elif use_w1 or use_w2:
            _wfd = dict(rho=_kz["rho_f"], W=_kz["W_c"], lam=_kz.get("lam_nm") or 550.0)
            if use_w1:
                _wf["1"] = _wfd; vD1 = float(_Df); vS1 = float(_kz["S_exakt"])
            if use_w2:
                _wf["2"] = _wfd; vD2 = float(_Df); vS2 = float(_kz["S_exakt"])
            vN = f / D                       # f/D des gemessenen Spiegels (Hauptregler)
            _notiz.append(f"Foucault-Wellenfront: D={_Df:.0f} mm, S={_kz['S_exakt']:.3f}, f/D={vN:.2f}")
            if not _kz.get("kennzahlen_zuverlaessig", True):
                _notiz.append("⚠ Foucault-Fit unzuverlässig (Polynom-Modus, Residuum hoch) — "
                              "Ergebnis optimistisch! Hybrid-Modus (n≥6) verwenden.")
        _p = dict(V=vV, seeing=vSe, refD=vRD, refS=1.0, N=vN, D1=vD1, S1=vS1,
                  D2=vD2, S2=vS2, lam=lam, buet=vBu, tubus=0.0, bild=vBi, pupille=vPu,
                  wf1=_wf.get("1"), wf2=_wf.get("2"), auge=v_auge, augenmass=v_mass,
                  sph1=sph1, sph2=sph2)
        _sig = {k: v for k, v in _p.items() if k not in ("wf1", "wf2")}
        _sig["wf"] = tuple(None if _wf.get(k) is None else
                           (round(float(np.sum(_wf[k]["W"])), 5), round(float(_wf[k]["lam"]), 1))
                           for k in ("1", "2"))
        _key = tuple(sorted(_sig.items()))
        if st.session_state.get("vg_key") != _key:
            _fig, _z = plot_vergleich(_p)
            st.session_state["vg_png"] = _fig_to_png(_fig)
            st.session_state["vg_txt"] = _z
            st.session_state["vg_key"] = _key
            plt.close(_fig)
        for _t in _notiz:
            st.warning(_t) if _t.startswith(("⚠", "Keine")) else st.caption(_t)
        st.markdown("  \n".join(st.session_state["vg_txt"]))
        st.image(st.session_state["vg_png"])

    if _nav == "doku":
        # Doku-HTML aus Datei lesen (liegt neben app.py) oder eingebettet anzeigen
        import os, pathlib
        _doc_candidates = [
            pathlib.Path(__file__).parent / "documentation_v7_2.html",
            pathlib.Path("documentation_v7_2.html"),
            pathlib.Path(__file__).parent / "documentation_v7_1.html",
            pathlib.Path("documentation_v7_1.html"),
            pathlib.Path(__file__).parent / "documentation_v6_1.html",
            pathlib.Path("documentation_v6_1.html"),
        ]
        _doc_html = None
        for _p in _doc_candidates:
            if _p.exists():
                _doc_html = _p.read_text(encoding="utf-8")
                break

        if _doc_html is not None:
            # <body>...</body> extrahieren, Banner-Margins auf 0 setzen
            import re as _re
            _body = _re.search(r"<body[^>]*>(.*)</body>", _doc_html, _re.DOTALL)
            _style = _re.search(r"<style[^>]*>(.*?)</style>", _doc_html, _re.DOTALL)
            _inner = _body.group(1).strip() if _body else _doc_html
            _css   = _style.group(1) if _style else ""
            # Banner-Margin für Web-Darstellung neutralisieren
            _css = _css.replace("margin: -20mm -15mm 25px -15mm;", "margin: 0 0 25px 0;")
            _css = _css.replace("padding: 30px 15mm 25px 15mm;", "padding: 20px 24px 16px 24px;")
            st.markdown(
                f"<style>{_css}</style><div style='max-width:860px'>{_inner}</div>",
                unsafe_allow_html=True)
        else:
            st.info(
                "Dokumentationsdatei **documentation_v7_2.html** nicht gefunden. "
                "Bitte die Datei ins gleiche Verzeichnis wie app.py legen.",
                icon="ℹ️")
            st.markdown("""
### Kurzreferenz Formeln

| Größe | Formel |
|---|---|
| W_PtV paraxial | D⁴ / (512·f³·λ) |
| W_PtV best focus | W_p / 4 |
| W_RMS | W_b / (1.5·√5) |
| Strehl Maréchal | exp(−(2π·W_rms)²) |
| Strehl exakt | Pupillenintegral |
| D_eff Kontrast | D · S^0.25 |
| V_krit | D · 0.7 · √S |
| Foucault W_i | ΔF·r²/(8·f²·λ) |
| Parabel-Soll δl | r²/(4f), rel. Zone 1 |
""")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Newton-Spiegel v{VERSION} · "
        "Formeln: Sacek (telescope-optics.net) · Suiter (Star Testing) · Barten 1999 (CSF) · "
        "Strehl: exaktes Pupillenintegral"
    )


if __name__ == "__main__" or True:
    main()
