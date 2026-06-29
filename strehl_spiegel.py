"""
Strehl-Berechnung für sphärischen Spiegel – ab initio
======================================================

Physikalisches Modell:
  1. Kugelgeometrie      – exakter Formspiegel z = R - sqrt(R² - r²)
  2. Wellenfrontfehler   – Abweichung von idealer Parabolform (exakt):
                           W(ρ) = 2 * [z_sph(r) - z_par(r)]
                           Faktor 2 wegen doppeltem Weg (Reflexion)
  3. Seidel-Näherung     – W040 = (D/2)^4 / (128·f³)  [nur zum Vergleich]
  4. Defokus-Term        – W = W040·ρ⁴ + B·ρ²  (B in Wellenlängen)
  5. Pupillenintegral    – S = |2∫₀¹ exp(i·2π·W(ρ))·ρ dρ|²
  6. Best-Focus-Suche    – Maximiert Strehl über B

Konventionen:
  - Wellenfrontfehler in Einheiten der Wellenlänge λ
  - Normierte Pupillenkoordinate ρ ∈ [0, 1]
  - Seidel-Koeffizient W040 = PV-Aberration am Rand (bei ρ=1, kein Defokus)

Wichtige Erkenntnisse:
  - Die klassische Seidel-Näherung gilt NUR paraxial (y ≪ R)
  - Bei f/5 (NA=0.1) versagt die Paraxialnäherung – exakte Geometrie gibt
    größere Wellenfrontfehler (Faktor ~4 beim Randwert)
  - Die drei relevanten Fokuspositionen sind:
      Paraxialfokus (B=0), Best-PV-Fokus (B=-A/2), Marginalfokus (B=-A)
  - Der Strehl ist MAXIMAL im Marginalfokus (Fokus des Randstrahls)
"""

import numpy as np
from scipy.optimize import minimize_scalar
import argparse


# ══════════════════════════════════════════════════════════════
# 1. EXAKTE WELLENFRONTBERECHNUNG
# ══════════════════════════════════════════════════════════════

def wellenfrontfehler(D_mm, f_mm, lam_mm, N=5000):
    """
    Exakter Wellenfrontfehler eines sphärischen gegenüber einem parabolischen
    Spiegel, als Funktion der normierten Pupillenkoordinate ρ.

    W(ρ) = 2 · [z_sph(r) - z_par(r)]   [in mm]

    mit r = ρ · D/2,
         z_sph(r) = R - √(R² - r²)     (Sphäre, Scheitel bei z=0)
         z_par(r) = r² / (2R)           (Paraboloid gleicher Brennweite)
         Faktor 2: doppelter Weg beim Spiegel (Reflexion)

    Rückgabe: (rho_arr, W_mm, W_lam)
    """
    R    = 2.0 * f_mm
    rho  = np.linspace(0.0, 1.0, N)
    r    = rho * (D_mm / 2.0)

    z_sph = R - np.sqrt(R**2 - r**2)   # exakte Sphäre
    z_par = r**2 / (2.0 * R)            # Paraboloid (Referenz)

    W_mm  = 2.0 * (z_sph - z_par)       # Wellenfrontfehler in mm
    W_lam = W_mm / lam_mm               # in Wellenlängen

    return rho, W_mm, W_lam


def seidel_koeffizient(D_mm, f_mm, lam_mm):
    """
    Paraxiale Seidel-Näherung:  W040 = (D/2)^4 / (128·f³)  [in λ]
    """
    return (D_mm / 2.0)**4 / (128.0 * f_mm**3 * lam_mm)


# ══════════════════════════════════════════════════════════════
# 2. DEFOKUS-KORREKTUR
# ══════════════════════════════════════════════════════════════

def mit_defokus(rho, W_lam, B_lam):
    """
    Addiert einen Defokus-Term: W_total = W_lam(ρ) + B · ρ²
    Subtrahiert anschließend die flächengewichtete Piston.
    B_lam:  Defokus-Koeffizient in Wellenlängen
    """
    W  = W_lam + B_lam * rho**2
    piston = 2.0 * np.trapezoid(W * rho, rho)
    return W - piston


# ══════════════════════════════════════════════════════════════
# 3. STREHL-BERECHNUNG (Pupillenintegral)
# ══════════════════════════════════════════════════════════════

def strehl_integral(rho, W_lam_korr):
    """
    S = |2 ∫₀¹ exp(i·2π·W(ρ)) · ρ dρ|²

    W_lam_korr: Wellenfrontfehler in Wellenlängen (Piston bereits entfernt)
    """
    phase     = 2.0 * np.pi * W_lam_korr
    integrand = np.exp(1j * phase) * rho
    I         = 2.0 * np.trapezoid(integrand, rho)
    return float(abs(I)**2)


# ══════════════════════════════════════════════════════════════
# 4. BEST-FOCUS-SUCHE
# ══════════════════════════════════════════════════════════════

def best_focus_suche(rho, W_lam, suchbereich_lam=None):
    """
    Findet den Defokus-Koeffizient B, der den Strehl maximiert.

    Suchbereich: [-3·|W040|, 0.5·|W040|]  (Standard)
    """
    A = W_lam.max()   # Schätzung des PV als Suchgrenze

    if suchbereich_lam is None:
        suchbereich_lam = (-3.5 * abs(A), 1.0 * abs(A))

    def neg_strehl(B):
        Wk = mit_defokus(rho, W_lam, B)
        return -strehl_integral(rho, Wk)

    res = minimize_scalar(
        neg_strehl,
        bounds=suchbereich_lam,
        method='bounded',
        options={'xatol': 1e-4}
    )
    return res.x, -res.fun


# ══════════════════════════════════════════════════════════════
# 5. HAUPTBERICHT
# ══════════════════════════════════════════════════════════════

def report(D_mm, fVerhältnis, lam_nm=550.0):
    f_mm   = D_mm * fVerhältnis
    lam_mm = lam_nm * 1e-6

    print("=" * 62)
    print("  STREHL-ANALYSE – SPHÄRISCHER SPIEGEL (ab initio)")
    print("=" * 62)
    print(f"  Durchmesser          D  = {D_mm:.1f} mm")
    print(f"  Öffnungsverhältnis   f/ = {fVerhältnis}")
    print(f"  Brennweite (parax.)  f  = {f_mm:.1f} mm")
    print(f"  Krümmungsradius      R  = {2*f_mm:.1f} mm")
    print(f"  Numerische Apertur   NA = {1/(2*fVerhältnis):.4f}")
    print(f"  Wellenlänge          λ  = {lam_nm:.0f} nm")
    print("-" * 62)

    # ── Wellenfront berechnen ──────────────────────────────
    rho, W_mm, W_lam = wellenfrontfehler(D_mm, f_mm, lam_mm)
    A_seidel = seidel_koeffizient(D_mm, f_mm, lam_mm)

    print(f"\n  WELLENFRONT (exakte Kugelgeometrie):")
    print(f"    W040 exakt am Rand  = {W_lam[-1]:.4f} λ")
    print(f"    W040 Seidel-Näherung= {A_seidel:.4f} λ  "
          f"(Faktor {W_lam[-1]/A_seidel:.1f}× größer exakt)")

    # ── Paraxialfokus ──────────────────────────────────────
    Wk_par = mit_defokus(rho, W_lam, 0.0)
    PV_par = Wk_par.max() - Wk_par.min()
    RMS_par = np.sqrt(2.0 * np.trapezoid((Wk_par**2) * rho, rho))
    S_par  = strehl_integral(rho, Wk_par)

    print(f"\n  ① PARAXIALFOKUS  (B = 0):")
    print(f"     PV-Wellenfrontfehler  = {PV_par:.4f} λ")
    print(f"     RMS-Wellenfrontfehler = {RMS_par:.5f} λ")
    print(f"     Strehl                = {S_par:.5f}")
    print(f"     Maréchal-Näherung     = {np.exp(-(2*np.pi*RMS_par)**2):.5f}"
          f"{'  (UNGÜLTIG bei W_rms≫λ!)' if RMS_par > 0.07 else ''}")

    # ── Best-PV-Fokus (klassisch B=-A/2) ──────────────────
    B_bpv = -W_lam[-1] / 2.0
    Wk_bpv = mit_defokus(rho, W_lam, B_bpv)
    PV_bpv = Wk_bpv.max() - Wk_bpv.min()
    RMS_bpv = np.sqrt(2.0 * np.trapezoid((Wk_bpv**2) * rho, rho))
    S_bpv  = strehl_integral(rho, Wk_bpv)

    print(f"\n  ② BEST-PV-FOKUS  (B = −A/2 = {B_bpv:.4f} λ):")
    print(f"     PV-Wellenfrontfehler  = {PV_bpv:.4f} λ")
    print(f"     RMS-Wellenfrontfehler = {RMS_bpv:.5f} λ")
    print(f"     Strehl                = {S_bpv:.5f}")

    # ── Marginalfokus (B=-A, max. Strehl für reine W040) ──
    B_marg = -W_lam[-1]
    Wk_marg = mit_defokus(rho, W_lam, B_marg)
    PV_marg = Wk_marg.max() - Wk_marg.min()
    RMS_marg = np.sqrt(2.0 * np.trapezoid((Wk_marg**2) * rho, rho))
    S_marg  = strehl_integral(rho, Wk_marg)

    print(f"\n  ③ MARGINALFOKUS  (B = −A = {B_marg:.4f} λ):")
    print(f"     [Fokus des Randstrahls, minimale PV, max. Strehl]")
    print(f"     PV-Wellenfrontfehler  = {PV_marg:.4f} λ")
    print(f"     RMS-Wellenfrontfehler = {RMS_marg:.5f} λ")
    print(f"     Strehl                = {S_marg:.5f}")
    print(f"     Maréchal-Näherung     = {np.exp(-(2*np.pi*RMS_marg)**2):.5f}")

    # ── Numerisch optimierter Best-Focus ─────────────────
    print(f"\n  ④ NUMERISCH OPTIMIERTER BEST-FOCUS:")
    B_opt, S_opt = best_focus_suche(rho, W_lam)
    Wk_opt  = mit_defokus(rho, W_lam, B_opt)
    PV_opt  = Wk_opt.max() - Wk_opt.min()
    RMS_opt = np.sqrt(2.0 * np.trapezoid((Wk_opt**2) * rho, rho))

    print(f"     Defokus B             = {B_opt:.4f} λ  (={B_opt/W_lam[-1]:.3f}·A)")
    print(f"     PV-Wellenfrontfehler  = {PV_opt:.4f} λ")
    print(f"     RMS-Wellenfrontfehler = {RMS_opt:.5f} λ")
    print(f"     ► STREHL (Maximum)    = {S_opt:.5f}")
    print(f"     Maréchal-Näherung     = {np.exp(-(2*np.pi*RMS_opt)**2):.5f}")

    print("=" * 62)
    print(f"\n  SEIDEL-NÄHERUNG (zum Vergleich, gilt streng nur für NA≪1):")
    A_s = A_seidel
    rho_s = rho
    B_s_opt, S_s_opt = best_focus_suche(rho_s, A_s * rho_s**4)
    Wk_s = mit_defokus(rho_s, A_s * rho_s**4, B_s_opt)
    RMS_s = np.sqrt(2.0 * np.trapezoid((Wk_s**2)*rho_s, rho_s))
    print(f"    W040 (Seidel)         = {A_s:.4f} λ")
    print(f"    Strehl (Seidel, opt.) = {S_s_opt:.5f}")
    print(f"    RMS  (Seidel, opt.)   = {RMS_s:.5f} λ")
    print("=" * 62)

    return {
        'paraxial': S_par,
        'best_pv':  S_bpv,
        'marginal': S_marg,
        'optimal':  S_opt,
        'B_opt_lam': B_opt,
    }


# ══════════════════════════════════════════════════════════════
# 6. EINSTIEGSPUNKT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strehl-Berechnung für sphärischen Spiegel (ab initio)"
    )
    parser.add_argument("-D",  type=float, default=200.0,
                        help="Durchmesser in mm (Standard: 200)")
    parser.add_argument("-f",  type=float, default=5.0,
                        help="Öffnungsverhältnis z.B. 5 für f/5 (Standard: 5)")
    parser.add_argument("-l",  type=float, default=550.0,
                        help="Wellenlänge in nm (Standard: 550)")
    args = parser.parse_args()

    report(args.D, args.f, args.l)
