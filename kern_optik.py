"""Kern: Optik und Wahrnehmung – Wellenfront, Strehl, MTF, CSF, Q_vis, Étendue, Grenzen (V_krit, V_max), Seeing.
(v8.0.0: aus newton_spiegel_rechner.py v7.11.0 ausgegliedert, Code unverändert)"""
import math
import numpy as np
from functools import lru_cache
from texte import T




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

    Wp  = D⁴/(512·f³·λ)   — PtV-Wellenfrontfehler am paraxialen Fokus [Sacek §4.1.2]
                             (KORRIGIERT 07/2026: D^4/(1024 f^3) ist die reine
                             Oberflaechen-Sagittadifferenz Kugel-Parabel; die
                             Wellenfront ist bei Reflexion doppelt so gross,
                             s. Sacek's aberration-coefficient-Herleitung /
                             Zahlenbeispiel in lower_order_spherical.htm)
    Wb  = Wp/4             — PtV am besten Fokus [Sacek §4.1.2]
    Wrms= Wb/(1.5·√5)      — RMS für reine sphärische Aberration [Sacek]
    S   = exp(-(2π·Wrms)²) — Maréchal-Näherung [Maréchal 1947, zit. n. Sacek], gültig für S > ~0.6
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
    → W(ρ) = Wp·(ρ⁴−ρ²)  [konsistent mit Sacek]
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
    # v7.5.1: KEIN zusätzlicher Faktor S mehr (mtf_sph_rel enthält den Strehl bereits; Doppelzählung)
    Qvis    = Qshape
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
    msa = msr * mp      # v7.3.0: KEIN zusätzlicher Faktor S (der Strehl steckt bereits in msr; MTF(0)=1)
    return freqs_as, mp, msa, msr


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
    # Geometrischer Unschärfefleck [Sacek]
    d_blur_mm = D_mm**3 / (128.0 * f_mm**2)
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

    d_blur [mm] = D³/(128·f²)  — Durchmesser des kleinsten Zerstreuungskreises am best focus
    (1/4 des Blur-Durchmessers D³/(32·f²) am paraxialen Fokus; Strahlverfolgung 200/1000: 0,0628 mm) [Sacek]
    α_blur ["]  = d_blur / f · 206265  — Winkelgröße im Teleskop-Bildfeld
    Am Auge erscheint der Blur vergrößert: α_auge = V · α_blur
    Blur sichtbar wenn α_auge >= eye_res_as:
      V_krit = eye_res_as / α_blur

    Keine Näherung, keine Strehl-Formel — rein geometrisch.
    Interpretation: unterhalb dieser V ist der Blur kleiner als das Auflösungsvermögen
    des Auges und damit nicht wahrnehmbar.
    """
    d_blur_mm      = D_mm**3 / (128.0 * f_mm**2)
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
                    V_min: float = 30.0, V_max: float = None,
                    n_pts: int = 60) -> float:
    """V_krit aus MTF×CSF-Integral: Vergrößerung ab der Q_vis < rel_schwelle · Q_para.

    rel_schwelle=0.90: ab hier ist der Sphäre-Spiegel um >10% schlechter
    als ein gleichgroßer Paraboloid bei derselben Vergrößerung.
    Gibt V_max zurück wenn der Verlust überall unter der Schwelle bleibt.
    """
    if V_max is None:          # v7.3.1: praktische Obergrenze 2·D [mm] (hier endet jede sinnvolle Vergrößerung)
        V_max = max(2.0 * D_mm, 2.0 * V_min)
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
                            V_min: float = 30.0, V_max: float = None,
                            n_pts: int = 60) -> float:
    """Wie v_krit_aus_qvis(), aber direkt aus Wb — für Schieber-Wert."""
    if V_max is None:          # v7.3.1: praktische Obergrenze 2·D [mm] (hier endet jede sinnvolle Vergrößerung)
        V_max = max(2.0 * D_mm, 2.0 * V_min)
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
    ("Mond",           3500.0),
    ("Jupiter",         600.0),
    ("Saturn",          200.0),
    ("Mars",           2000.0),
    ("Helles DSO",        5.0),
    ("Schwaches DSO",     0.5),
]

# Augenpupille d_p [mm] je Objekt (v7.5.0, Richtwerte aus der Praxis)
OBJEKT_PUPILLE = {"Mond": 2.5, "Jupiter": 3.0, "Saturn": 4.0, "Mars": 4.0,
                  "Helles DSO": 6.0, "Schwaches DSO": 7.0}


def kurven_N(D_mm, lam_nm, N_min=3.0, N_max=15.0, schritte=150):
    """Strehl und eff. Öffnungen als Funktion von f/D.

    Rückgabe: ns, strehls_marechal, deff_ks, aufl_verluste, strehls_exakt

    strehls_marechal: Marechal-Naeherung  S = exp(-(2*pi*Wrms)^2)
    strehls_exakt:    exaktes Pupillenintegral (Born & Wolf)
    Beide am besten Fokus (Wb = D^4/(2048 f^3 lam)).
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

    M(f) = (2/π)·(arccos(f) − f·√(1−f²))   [Sacek, Standardformel Kreisapertur]
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
    Quellen: O'Neill, JOSA 46, 285 (1956); Barakat, JOSA 55, 538 (1965); vgl. Sacek.
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


def mtf_sph_rel_1d(f: float, Wb: float, n: int = 80,
                   paraxial: bool = False) -> float:
    """Relative MTF eines sphärischen Spiegels (normiert: MTF(0)=1).

    OTF-Faltungsintegral [Sacek]:
      MTF(f) = |∫ exp(i·2π·ΔW(x)) h(x) dx| / ∫ h(x) dx
    mit ΔW(x) = W(x+f/2) − W(x−f/2)

    Wellenfrontform:
      best focus   (paraxial=False): W(ρ) = Wp·(ρ⁴−ρ²)
      parax. Fokus (paraxial=True):  W(ρ) = Wp·ρ⁴

    Amplitude Wp = 4·Wb  [Wb = Wp/4; konsistent mit Sacek]:
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
    """Contrast Sensitivity Function nach Mannos & Sakrison (1974).

    csf(f) = 2.6·(0.0192 + 0.114·f)·exp(−(0.114·f)^1.1)
    Gültig für photopisches Sehen bei mittlerer Leuchtdichte.  [Mannos/Sakrison 1974]
    """
    f = max(f_cpd, 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))


# Leuchtdichte-Referenz für csf_lum
_L_REF = 100.0   # cd/m²  — mittlere photopische Leuchtdichte


def csf_lum(f_cpd: float, L: float) -> float:
    """Luminanzabhängige CSF (angelehnt an Barten 1999) — vereinfachte Skalierung.

    Bei niedriger Leuchtdichte L [cd/m²]:
      - Amplitude sinkt ∝ √(L/L_ref)
      - Spitze wandert zu NIEDRIGEREN Frequenzen ∝ (L/L_ref)^0,25  (s. csf_L)
        (bis v7.3.3 war die Richtung falsch: Spitze wanderte zu höheren Frequenzen)

    L_ref = 100 cd/m² (mittlere photopische Leuchtdichte).
    Typische Werte am Okular (geschätzt):
      Mond             ~3500 cd/m²   Mars (Opposition) ~2000 cd/m²
      Jupiter-Scheibe   ~600 cd/m²   Saturn            ~200 cd/m²
      Schwacher Planet   ~50 cd/m²
      Heller Nebel        ~5 cd/m²
      Schwaches Deep-Sky ~0.5 cd/m²
    """
    L = max(L, 1e-4)
    a_scale = (L / _L_REF) ** 0.5
    return float(csf_L(f_cpd, L)) * a_scale


# ── v7.4.0: Leuchtdichte-abhängige CSF für Q_vis ─────────────────────────
# Spitze der Kontrastempfindlichkeit f_Spitze ≈ 7,9 cpd · (L/100)^0,25
# (nach oben bei L ≥ 100 cd/m² gedeckelt). Exponent: Näherung, angelehnt an
# Barten (1999); bei 0,5 / 5 / 100 cd/m² liegt die Spitze bei 2,1 / 3,7 / 7,9 cpd.
# Die Amplitude kürzt sich in Q_vis (Verhältnis zum Paraboloid) heraus, nur die
# Verschiebung der Spitze wirkt.
_CSF_L_EXPO = 0.25


def _csf_shift(L=None):
    """Frequenz-Skalierung der CSF-Spitze; L None oder ≥ L_ref -> 1 (photopisch).
    L darf Skalar oder Array sein (v7.6.0)."""
    if L is None:
        return 1.0
    s = np.minimum((np.maximum(np.asarray(L, dtype=float), 1e-4) / _L_REF) ** _CSF_L_EXPO, 1.0)
    return float(s) if s.ndim == 0 else s


def csf_L(f_cpd, L=None):
    """CSF (Mannos/Sakrison-Form) mit zu niedrigeren Frequenzen verschobener
    Spitze bei wenig Licht; akzeptiert Skalar oder Array. L=None -> identisch
    zu csf()."""
    f = np.maximum(np.asarray(f_cpd, dtype=float) / _csf_shift(L), 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))


def _L_roh(L):
    """Rohe Objekt-Leuchtdichte [cd/m²] als float > 0, sonst None (= Aus / photopisch)."""
    try:
        L = float(L or 0.0)
    except (TypeError, ValueError):
        return None
    return L if L > 0.0 else None


def _L_eff(L, D_mm, V, d_p=None):
    """Leuchtdichte auf der Netzhaut (Étendue, v7.6.0).

    L_eff = L · min(1, (D/(V·d_p))²): erst wenn die Austrittspupille D/V kleiner als
    die Augenpupille d_p wird, sinkt die Netzhautbeleuchtung.  V darf Array sein
    (Rückgabe dann Array).  None, wenn L nicht gesetzt oder überall ≥ 100 cd/m²
    (= photopisch, CSF unverändert)."""
    if L is None:
        return None
    L = float(L)
    if L <= 0.0:
        return None
    if d_p is not None and d_p > 0.0:
        L = L * np.minimum(1.0, (D_mm / (np.maximum(np.asarray(V, dtype=float), 1e-9) * d_p)) ** 2)
    if np.min(L) >= _L_REF:
        return None
    return float(L) if np.ndim(L) == 0 else L


def csf_spitze_cpd(L=None) -> float:
    """Lage der CSF-Spitze [cpd] (nur zur Anzeige)."""
    return 7.9 * _csf_shift(L)


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
    f_sc  = np.minimum((L_s / 100.0) ** _CSF_L_EXPO, 1.0)
    a_sc  = (L_s / 100.0) ** 0.5
    f_eff = np.maximum(f_cpd / f_sc, 1e-6)
    csf_m = 2.6 * (0.0192 + 0.114 * f_eff) * np.exp(-((0.114 * f_eff) ** 1.1)) * a_sc
    result = _trapz(ms_mp[:, None, None] * csf_m, nu, axis=0)  # (nV, nL)
    return result


def _qvis_schieber(D_mm, f_mm, lam_nm, S_slide, V, L=None, d_p=None):
    """Q_vis(V) exakt (MTF×CSF, Paraboloid = 1) für die Schieberstellung; ohne Verbesserung = echte Kugel.
    L: Leuchtdichte [cd/m²] oder None (= photopisch, wie bisher)."""
    Wb_r = _wellenfronten(D_mm, f_mm, lam_nm)[2]
    if S_slide <= _strehl_exakt(Wb_r) + 1e-6:
        return perceived_quality_exakt(D_mm, f_mm, lam_nm, V, L=L, d_p=d_p)
    return perceived_quality_exakt_von_Wb(D_mm, _Wb_schieber(D_mm, f_mm, lam_nm, S_slide), lam_nm, V, L=L, d_p=d_p)


def perceived_quality_exakt(D_mm: float, f_mm: float, lam_nm: float,
                             V: float, n_pts: int = 200, L=None, d_p=None) -> float:
    """Q_vis exakt: MTF×CSF-Integral ohne Dämpfungsfunktion w(V).

    Q_vis(V) = ∫ msr·mp·CSF dν / ∫ mp·CSF dν
    Grenzwerte:  Q_vis(V→V_min) → 1   ·   Q_vis(V→∞) → S_form
    L (v7.4.0): Leuchtdichte [cd/m²]; verschiebt die CSF-Spitze zu niedrigeren
    Frequenzen (s. csf_L).  None = photopisch (Mannos/Sakrison, wie bisher).
    """
    L = _L_eff(L, D_mm, V, d_p)   # v7.6.0: Étendue
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu     = np.linspace(0.0, 1.0, n_pts)
    mp     = np.array([mtf_para(f)          for f in nu])
    msr    = np.array([mtf_sph_rel(f, Wb)   for f in nu])
    w_csf  = csf_L(nu * fc * 3600.0 / V, L)
    return float(_trapz(msr * mp * w_csf, nu) /
                 max(_trapz(mp * w_csf, nu), 1e-12))


def perceived_quality_exakt_kurven(D_mm: float, f_mm: float, lam_nm: float,
                                    V_arr: np.ndarray, n_pts: int = 200, L=None, d_p=None) -> np.ndarray:
    """perceived_quality_exakt für ein Array von Vergrößerungen — vektorisiert.

    MTF-Arrays werden einmal berechnet, CSF-Gewichtung per numpy-Broadcast
    über alle V gleichzeitig → ~7× schneller als Schleife.
    L: Leuchtdichte [cd/m²] oder None (= photopisch).
    """
    V_arr   = np.asarray(V_arr, dtype=float)
    L = _L_eff(L, D_mm, V_arr, d_p)   # v7.6.0: Étendue, pro V
    if L is not None:
        n_pts = max(n_pts, 200)   # v7.5.1: bei wenig Licht liegt die CSF-Spitze bei sehr kleinem ν -> feineres Gitter
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    lam_mm  = lam_nm * 1e-6
    _, _, Wb, _, _ = _wellenfronten(D_mm, f_mm, lam_nm)
    fc      = D_mm / (lam_mm * 206265)
    nu      = np.linspace(0.0, 1.0, n_pts)
    mp      = np.array([mtf_para(f)        for f in nu])
    msr     = np.array([mtf_sph_rel(f, Wb) for f in nu])
    # (n_pts, n_V) broadcast: CSF-Matrix
    f_cpd   = np.maximum(nu[:, None] * fc * 3600.0 / V_arr[None, :], 1e-6)
    csf_mat = csf_L(f_cpd, None if L is None else np.atleast_1d(L)[None, :])
    num = _trapz(msr[:, None] * mp[:, None] * csf_mat, nu, axis=0)
    den = _trapz(mp[:, None] * csf_mat, nu, axis=0)
    return num / np.maximum(den, 1e-12)


def perceived_quality_exakt_von_Wb(D_mm: float, Wb: float, lam_nm: float,
                                    V: float, n_pts: int = 200, L=None, d_p=None) -> float:
    """Q_vis exakt direkt aus Wb — für Schieber-Kurve mit beliebigem Strehl.
    L: Leuchtdichte [cd/m²] oder None (= photopisch)."""
    L = _L_eff(L, D_mm, V, d_p)   # v7.6.0: Étendue
    if L is not None:
        n_pts = max(n_pts, 200)   # v7.5.1: feineres Gitter bei wenig Licht
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    nu     = np.linspace(0.0, 1.0, n_pts)
    mp     = np.array([mtf_para(f)          for f in nu])
    msr    = np.array([mtf_sph_rel(f, Wb)   for f in nu])
    w_csf  = csf_L(nu * fc * 3600.0 / V, L)
    return float(_trapz(msr * mp * w_csf, nu) /
                 max(_trapz(mp * w_csf, nu), 1e-12))


def perceived_quality_exakt_kurven_von_Wb(D_mm: float, Wb: float, lam_nm: float,
                                           V_arr: np.ndarray, n_pts: int = 200, L=None, d_p=None) -> np.ndarray:
    """perceived_quality_exakt_von_Wb für ein Array von Vergrößerungen — vektorisiert.
    L: Leuchtdichte [cd/m²] oder None (= photopisch)."""
    V_arr   = np.asarray(V_arr, dtype=float)
    L = _L_eff(L, D_mm, V_arr, d_p)   # v7.6.0: Étendue, pro V
    if L is not None:
        n_pts = max(n_pts, 200)   # v7.5.1: feineres Gitter bei wenig Licht
    _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
    fc      = D_mm / (lam_nm * 1e-6 * 206265)
    nu      = np.linspace(0.0, 1.0, n_pts)
    mp      = np.array([mtf_para(f)        for f in nu])
    msr     = np.array([mtf_sph_rel(f, Wb) for f in nu])
    f_cpd   = np.maximum(nu[:, None] * fc * 3600.0 / V_arr[None, :], 1e-6)
    csf_mat = csf_L(f_cpd, None if L is None else np.atleast_1d(L)[None, :])
    num = _trapz(msr[:, None] * mp[:, None] * csf_mat, nu, axis=0)
    den = _trapz(mp[:, None] * csf_mat, nu, axis=0)
    return num / np.maximum(den, 1e-12)


def qvis_einzelrechner(D_mm: float, S: float, V: float,
                       lam_nm: float = 550.0, L=None, d_p=None) -> dict:
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
                  (auf 30…2·D× begrenzt; 2·D heißt „> 2·D×“, 30.0 „≤ 30×“)
    """
    S  = min(max(float(S), 1e-6), 1.0)
    Wb = _Wb_von_S(S)
    Q  = 1.0 if Wb <= 1e-9 else perceived_quality_exakt_von_Wb(D_mm, Wb, lam_nm, V, L=L, d_p=d_p)   # L: v7.4.0, None = photopisch
    S_ph = _strehl_exakt(Wb)
    Vk_w = v_wendepunkt_naeh(D_mm, Wb)
    w    = 1.0 / (1.0 + (Vk_w / max(V, 1e-9)) ** 2)
    Q_n  = S_ph + (1.0 - S_ph) * (1.0 - w)
    return dict(D=D_mm, S=S, V=V, lam=lam_nm, Wb=Wb, Q=Q, Q_naeh=Q_n,
                verlust=(1.0 - Q) * 100.0,
                V_krit=v_krit_naeh(D_mm, Wb),
                V_q90=(max(2.0 * D_mm, 60.0) if Wb <= 1e-9 else
                       v_krit_aus_qvis_von_Wb(D_mm, Wb, lam_nm, 0.90)))


# ── v7.0.0-beta: Vergleich verschiedener Spiegel (absolutes Q_abs) ────────────

ANTONIADI_HINWEIS = ("Antoniadi (grobe Richtwerte, keine Norm): I perfekt ≈ <0,5″ · II gut ≈ 0,5–1″ · "
                     "III mäßig ≈ 1–2″ · IV schlecht ≈ 2–3″ · V sehr schlecht ≈ >3″")


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
    # v7.3.1: MTF = |OTF| (Betrag). Der Imaginärteil ist bei rotationssymmetrischer Pupille null (≈1e-16);
    # negative OTF-Werte (Kontrastumkehr) wurden früher bei 0 abgeschnitten, jetzt als Betrag gezählt.
    otf = np.abs(np.real(ac[0, :Np])) / max(float(np.real(ac[0, 0])), 1e-12)
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


def beurteilung(strehl):
    if strehl >= 0.95:
        return T("verdict_good"), "#2e7d32"
    elif strehl >= 0.80:
        return T("verdict_ok"), "#e65100"
    else:
        return T("verdict_bad"), "#c62828"
