"""Kern: Foucault-Test – Zonen, Wellenfront, Best-Fit-ROC, Fehlerkennzahlen.
(v8.0.0: aus newton_spiegel_rechner.py v7.11.0 ausgegliedert, Code unverändert)"""
import math
import numpy as np
from kern_optik import _golden_section



try:
    from scipy.optimize import minimize_scalar as _minimize_scalar_scipy
    _SCIPY_OK = True
except ImportError:
    _minimize_scalar_scipy = None
    _SCIPY_OK = False


# ── Foucault-Auswertung ───────────────────────────────────────────────────────

def _foucault_kanten_r2(D_mm: float, n: int, rand_geteilt: bool) -> np.ndarray:
    """Zonenkanten in r² (gleichflächig). rand_geteilt: die äußerste Zone wird in zwei
    flächengleiche Hälften geteilt (n+1 Messzonen, v7.3.1)."""
    R2 = (D_mm / 2.0) ** 2
    k = R2 * np.arange(0, n + 1) / n
    if rand_geteilt:
        k = np.concatenate([k[:-1], [R2 * (n - 0.5) / n], [R2]])
    return k


def foucault_zonen(D_mm: float, n: int = 4, rand_geteilt: bool = False) -> np.ndarray:
    """Messradien gleichflächiger Zonen (Zonenmitte, flächengewichtet).

    r_i = R · √((2i−1) / (2n))   i = 1…n      (FigureXP-Konvention)
    rand_geteilt=True: äußerste Zone in zwei flächengleiche Hälften geteilt
    → n+1 Messradien (innere und äußere Randhälfte getrennt messen).
    """
    k = _foucault_kanten_r2(D_mm, n, rand_geteilt)
    return np.sqrt((k[:-1] + k[1:]) / 2.0)


def foucault_zonengrenzen(D_mm: float, n: int = 4, rand_geteilt: bool = False) -> np.ndarray:
    """Außengrenzen der Zonen (für Zonenmaske/Ringblende).

    r_aussen,i = R · √(i/n)   i = 1…n   (r_aussen,n = R, Spiegelrand)
    Innengrenze von Zone i ist r_aussen,i-1 (bzw. 0 für Zone 1).
    rand_geteilt=True: n+1 Zonen, die letzte Zone ist halbiert.
    """
    return np.sqrt(_foucault_kanten_r2(D_mm, n, rand_geteilt)[1:])


def _foucault_wellenfront_kern(r_zonen: np.ndarray, df_zonen: np.ndarray,
                               D_mm: float, R_c_mm: float,
                               lam_nm: float, r_grenz=None) -> np.ndarray:
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
    if r_grenz is not None:                     # v7.3.1: echte Zonengrenzen (z.B. geteilte Randzone)
        r_bound[1:] = np.asarray(r_grenz, float)
    else:
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
                         roc_suchbereich_mm: float | None = None,
                         r_grenz=None) -> np.ndarray:
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
        return _foucault_wellenfront_kern(r_zonen, df_zonen, D_mm, R_c0, lam_nm, r_grenz)

    if roc_suchbereich_mm is None:
        roc_suchbereich_mm = max(5.0, R_c0 * 0.01)   # +-1% Standard-Suchfenster

    def _rms_fuer_Rc(R_c_trial: float) -> float:
        W = _foucault_wellenfront_kern(r_zonen, df_zonen, D_mm, R_c_trial, lam_nm, r_grenz)
        return float(np.std(W))

    R_c_opt = _golden_section(_rms_fuer_Rc, R_c0 - roc_suchbereich_mm,
                              R_c0 + roc_suchbereich_mm)
    return _foucault_wellenfront_kern(r_zonen, df_zonen, D_mm, R_c_opt, lam_nm, r_grenz)



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
                        grad_modus: str = "bic", r_grenz=None) -> dict:
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
    rho_f = np.linspace(0.0, 1.0, 2000)

    # Flaechengewichte der Zonen (Kreisringflaechen, normiert)
    # v7.3.0: echte Zonengrenzen (gleichflächig), nicht die Zonenmitten
    r_outer    = (np.asarray(r_grenz, float) if r_grenz is not None
                  else foucault_zonengrenzen(D_mm, len(r_zonen)))
    # v7.3.1: W_zonen[i] ist das kumulative Integral der Steigung BIS ZUM AUSSENRAND der Zone i
    # (foucault_wellenfront) – gehört also zum Radius r_outer[i], nicht zur Zonenmitte. Der Fit
    # wird deshalb an den Zonengrenzen ausgewertet; dadurch hängt das Ergebnis kaum noch von der
    # Zonenzahl ab (vorher z.B. Kugel 200/1000: n=4 -> 0,117, n=6 -> 0,127; jetzt ~0,128 für alle n).
    rho_z = r_outer / R
    r_inner    = np.concatenate([[0.0], r_outer[:-1]])
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
        rho_aussen  = r_outer / R
        rho_inner_z = np.concatenate([[0.0], rho_aussen[:-1]])
        for ri, ro, res in zip(rho_inner_z, rho_aussen, residuen):
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
        rho_zonen=rho_z,
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
