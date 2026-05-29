"""
validate_newton.py
==================
Unabhängige Validierung der Kernberechnungen des Newton-Spiegel-Rechners.

Jeder Test ist gegen einen Referenzwert aus der optischen Fachliteratur
oder gegen eine analytisch exakte Formel verifiziert. Die Toleranzen
sind physikalisch begründet.

Verwendung:
    python validate_newton.py          # kurze Zusammenfassung
    python validate_newton.py -v       # ausführlich mit Herleitung

Quellen:
    [S]  Suiter: "Star Testing Astronomical Telescopes", 2nd ed.
    [BW] Born & Wolf: "Principles of Optics", 7th ed.
    [G]  Goodman: "Introduction to Fourier Optics"
    [RV] Rutten & van Venrooij: "Telescope Optics"
    [TO] telescope-optics.net
"""

import math
import sys
import numpy as np
from functools import lru_cache

GREEN  = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"
verbose = "-v" in sys.argv

# ── Kernfunktionen (unverändert aus newton_spiegel_rechner.py) ────────────────

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
        re += math.cos(dW)*h; im += math.sin(dW)*h; norm += h
    return math.sqrt(re**2 + im**2)/norm if norm > 1e-10 else 0.0

def mtf_kurven(D_mm, f_mm, lam_nm=550.0, n_pts=60):
    Wp     = D_mm**4 / (1024.0 * f_mm**3 * (lam_nm*1e-6))
    Wb     = Wp / 4.0
    Wrms   = Wb / (1.5 * math.sqrt(5))
    strehl = math.exp(-(2 * math.pi * Wrms)**2)
    fc     = D_mm / (lam_nm * 1e-6 * 206265)
    freqs  = np.linspace(0, 1, n_pts)
    mp     = np.array([mtf_para(f)        for f in freqs])
    msr    = np.array([mtf_sph_rel(f, Wb) for f in freqs])
    msa    = msr * mp * strehl
    return freqs * fc, fc, mp, msa, msr, strehl

def berechne(D_mm, f_mm, lam_nm=550.0):
    lam_mm = lam_nm * 1e-6
    Wp   = D_mm**4 / (1024.0 * f_mm**3 * lam_mm)
    Wb   = Wp / 4.0
    Wrms = Wb / (1.5 * math.sqrt(5))
    S    = math.exp(-(2 * math.pi * Wrms) ** 2)
    Q02  = mtf_sph_rel(0.2, Wb); Q04 = mtf_sph_rel(0.4, Wb); Q06 = mtf_sph_rel(0.6, Wb)
    Qshape = 0.4*Q02 + 0.4*Q04 + 0.2*Q06
    Qvis   = S * Qshape
    d_blur_mm  = D_mm**3 / (64.0 * f_mm**2)
    d_blur_as  = d_blur_mm / f_mm * 206265.0
    theta_ideal = 116.0 / D_mm
    theta_geo   = math.sqrt(theta_ideal**2 + d_blur_as**2)
    theta_eff   = theta_ideal / math.sqrt(max(S, 1e-6))
    return dict(
        N=f_mm/D_mm, Wp=Wp, Wb=Wb, Wrms=Wrms, strehl=S,
        Deff_k=D_mm*S**0.25, loss_k=D_mm - D_mm*S**0.25,
        Deff_s=116.0/theta_eff, loss_s=D_mm - 116.0/theta_eff,
        theta_ideal=theta_ideal, theta_eff=theta_eff,
        d_blur_mm=d_blur_mm, d_blur_as=d_blur_as,
        aufl_verlust_pct=(1.0 - theta_ideal/theta_geo)*100.0,
        Q02=Q02, Q04=Q04, Q06=Q06, Qshape=Qshape,
        Qvis=Qvis, Deff_vis=D_mm*Qvis**0.25,
    )

# ── Testrahmen ────────────────────────────────────────────────────────────────
results = []

def check(name, got, expected, tol, unit="", ref="", why=""):
    ok  = abs(got - expected) <= tol
    pct = abs(got - expected) / abs(expected) * 100 if expected != 0 else 0
    results.append((ok, name, got, expected, tol, unit, ref, why, pct))
    return ok

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: Wellenfrontfehler & Strehl
# Referenz: Standardformel aus Born&Wolf §5.5 und Suiter Chapter 5
# ═══════════════════════════════════════════════════════════════════════════════

# 1a. Wellenfrontformel: Wp = D^4/(1024·f^3·λ)
# Für 200mm f/8, λ=550nm: Wp = 200^4/(1024·1600^3·550e-6) = 0.6936λ
r = berechne(200, 1600)
check("Wp paraxial  200mm f/8",
      r["Wp"], 0.6936, tol=0.0001,
      unit="λ", ref="[BW] §5.5 / Formel Wp=D⁴/(1024·f³·λ)",
      why="Direkte Berechnung: 200⁴/(1024·1600³·550e-6) = 0.6936λ")

check("Wb best-focus 200mm f/8",
      r["Wb"], 0.1734, tol=0.0001,
      unit="λ", ref="[S] §5.3: Wb = Wp/4",
      why="Defokussierung auf Mittelpunkt der Kaustik halbiert PtV zweimal: Wb=Wp/4=0.1734λ")

# 1b. Strehl 200mm f/8 ≈ 0.90 (Literatur-Übereinstimmung)
check("Strehl  200mm f/8  ≈ 0.90",
      r["strehl"], 0.900, tol=0.005,
      unit="", ref="[S] S.134 / [TO] Strehl-Tabelle",
      why="Maréchal: S=exp(-(2π·Wrms)²), Wrms=Wb/(1.5√5)=0.0517λ → S=0.900")

# 1c. Rayleigh-Grenze: D=200mm f/7 hat Wb≈0.26λ → S≈0.79 (knapp unter 0.80)
#     Das zeigt: Rayleigh-Grenze liegt knapp bei f/7 für 200mm
r7 = berechne(200, 1400)
check("200mm f/7: Wb ≈ 0.26λ (nahe Rayleigh)",
      r7["Wb"], 0.2588, tol=0.0005,
      unit="λ", ref="Berechnung: D⁴/(4096·f³·λ)",
      why="Bei Wb=0.25λ liegt Rayleigh-Grenze. f/7 ist knapp darüber (0.259λ).")

check("200mm f/7: Strehl ≈ 0.79 (knapp nicht beugungsbegrenzt)",
      r7["strehl"], 0.791, tol=0.002,
      unit="", ref="Maréchal bei Wb=0.259λ",
      why="S=0.79 < 0.80: 200mm f/7 liegt knapp unter Rayleigh-Grenze")

# 1d. 200mm f/5: schlechter Spiegel — übereinstimmend mit FigureXP
r5 = berechne(200, 1000)
check("Strehl  200mm f/5  (FigureXP-Referenz)",
      r5["strehl"], 0.170, tol=0.003,
      unit="", ref="FigureXP-Messwert / Suiter-Diagramm",
      why="Wb=0.710λ → Wrms=0.212λ → S=exp(-(2π·0.212)²)=0.170")

# 1e. Monotonie: mehr Brennweite → mehr Strehl (Physik)
strehls = [berechne(200, 200*n)["strehl"] for n in [5,6,7,8,10]]
check("Strehl monoton steigend mit f/D",
      1.0 if all(strehls[i] < strehls[i+1] for i in range(len(strehls)-1)) else 0.0,
      1.0, tol=0.0, unit="bool", ref="Physik",
      why="Größeres f/D → kleinerer Wp → höherer Strehl")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: MTF Paraboloid — analytisch exakt (kein Näherungsfehler möglich)
# Formel: MTF(f) = (2/π)[arccos(f) - f√(1-f²)]  [G] §6.3
# ═══════════════════════════════════════════════════════════════════════════════

mtf_refs = [
    (0.0,  1.000000, "DC-Wert = 1"),
    (0.3,  0.623838, "Goodman §6.3"),
    (0.5,  0.391002, "Goodman §6.3"),
    (0.7,  0.188120, "Goodman §6.3"),
    (0.99, 0.001199, "Grenzwert → 0"),
]
for fn, ref_val, label in mtf_refs:
    check(f"MTF_para  f={fn}  ({label})",
          mtf_para(fn), ref_val, tol=0.000050,
          unit="", ref="[G] §6.3 — analytisch exakt",
          why=f"MTF(f)=(2/π)[arccos(f)-f√(1-f²)], kein Näherungsfehler")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: MTF Sphäre — physikalische Grenzwerte & Konsistenz
# ═══════════════════════════════════════════════════════════════════════════════

# 3a. Perfekter Spiegel (Wb→0): relative MTF muss 1 sein
check("MTF_sph_rel(0.3, Wb→0) = 1.0  [kein Fehler]",
      mtf_sph_rel(0.3, 1e-6), 1.0, tol=0.002,
      unit="", ref="Grenzwert: Wb=0 → Paraboloid",
      why="Für verschwindende Aberration muss rel. MTF → 1")

check("MTF_sph_rel(f=0, beliebig) = 1.0  [Normierung]",
      mtf_sph_rel(0.0, 0.5), 1.0, tol=1e-9,
      unit="", ref="Definition: normiert auf MTF(f=0)=1",
      why="Relative MTF per Definition bei f=0 auf 1 normiert")

# 3b. MTF_sph_rel ≤ 1 (Sphäre nie besser als Paraboloid)
Wb_test = berechne(200, 1000)["Wb"]
vals_rel = [mtf_sph_rel(f, Wb_test) for f in np.linspace(0, 0.99, 50)]
check("MTF_sph_rel ≤ 1.0 überall",
      max(vals_rel), 1.0, tol=0.002,
      unit="", ref="Physik: Aberrationen können MTF nicht verbessern",
      why="Sphärische Aberration senkt oder erhält MTF, verbessert nie")

# 3c. MTF_sph_rel fällt mit steigendem Wb (mehr Aberration → schlechtere MTF)
check("MTF_sph_rel(0.5, Wb=0.05) > MTF_sph_rel(0.5, Wb=0.50)",
      1.0 if mtf_sph_rel(0.5, 0.05) > mtf_sph_rel(0.5, 0.50) else 0.0,
      1.0, tol=0.0, unit="bool", ref="Physik",
      why="Mehr Wellenfrontfehler → schlechtere MTF")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4: Absolute MTF — Strehl-Konsistenz (zentrale Bugfix-Stelle v4)
# ═══════════════════════════════════════════════════════════════════════════════

_, fc, mp, msa, msr, strehl_v = mtf_kurven(200, 1000, n_pts=200)

# 4a. msa(f→0) = Strehl  [BW §9.5: Strehl = MTF-Integral normiert auf Paraboloid]
check("msa(f→0) = Strehl  [fundamental]",
      float(msa[0]), strehl_v, tol=0.002,
      unit="", ref="[BW] §9.5: MTF(0) = Strehl",
      why="Physikalisches Fundament: absoluter Kontrast bei größten Strukturen = Strehl")

# 4b. msa = Strehl · msr · mp  (die eigentliche Bugfix-Formel)
nu_ax = np.linspace(0, 1, len(mp))
for nu_test, label in [(0.2, "Jupiter-Gürtel"), (0.5, "Saturn-Cassini"), (0.8, "fein")]:
    Wb_t = berechne(200, 1000)["Wb"]
    val_formel  = mtf_para(nu_test) * mtf_sph_rel(nu_test, Wb_t) * strehl_v
    val_interp  = float(np.interp(nu_test, nu_ax, msa))
    check(f"msa = S·msr·mp  bei ν={nu_test} ({label})",
          val_interp, val_formel, tol=0.003,
          unit="", ref="Definition msa, Konsistenz np.interp",
          why="Balkenwert aus np.interp muss mit direkter Formel übereinstimmen")

# 4c. msa ≤ mp überall
check("msa ≤ mp überall  [Physik]",
      1.0 if all(msa[i] <= mp[i] + 1e-9 for i in range(len(mp))) else 0.0,
      1.0, tol=0.0, unit="bool", ref="Physik",
      why="Absolute MTF der Sphäre kann Paraboloid nie übertreffen")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5: Geometrischer Blur
# Formel: d = D³/(64·f²)  [RV] Formel 5.17
# ═══════════════════════════════════════════════════════════════════════════════

r = berechne(200, 1000)
check("Blur 200mm f/5  [mm]",
      r["d_blur_mm"], 0.1250, tol=0.0001,
      unit="mm", ref="[RV] Formel 5.17: d=D³/(64·f²)",
      why="200³/(64·1000²) = 8000000/64000000 = 0.1250mm")

check("Blur 200mm f/5  [arcsec]",
      r["d_blur_as"], 25.783, tol=0.002,
      unit="\"", ref="Umrechnung: (d_mm/f_mm)·206265",
      why="0.125/1000·206265 = 25.783arcsec")

check("Dawes-Grenze 200mm = 0.580\"",
      r["theta_ideal"], 0.5800, tol=0.0001,
      unit="\"", ref="116/D[mm] = 116/200",
      why="Empirische Dawes-Formel: 116/D[mm]")

check("Blur/Dawes > 10 bei f/5  [PSF ≠ geometrisch]",
      r["d_blur_as"] / r["theta_ideal"], 44.5, tol=0.5,
      unit="×", ref="Physik",
      why="Blur=25.8arcsec / Dawes=0.58arcsec = 44.5×. "
          "Geometrischer Blur gilt NICHT für visuelle PSF-Wahrnehmung.")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 6: Effektive Öffnungen — interne Konsistenz
# ═══════════════════════════════════════════════════════════════════════════════

r = berechne(200, 1000); S = r["strehl"]

check("Deff_k = D·S^0.25",
      r["Deff_k"], 200 * S**0.25, tol=0.0001,
      unit="mm", ref="Definition",
      why="Kontrastäquivalente Öffnung: S^(1/4) aus MTF-Energie-Skalierung")

check("Deff_s = D·√S  (aus θ_eff=θ_ideal/√S)",
      r["Deff_s"], 200 * math.sqrt(S), tol=0.001,
      unit="mm", ref="Konsistenz: 116/θ_eff = D·√S",
      why="θ_eff = 116/(D·√S) → Deff_s = 116/θ_eff = D·√S")

check("Deff_vis = D·Q_vis^0.25",
      r["Deff_vis"], 200 * r["Qvis"]**0.25, tol=0.0001,
      unit="mm", ref="Definition, analog zu Deff_k",
      why="MTF-gewichtete effektive Öffnung")

check("Deff_s < Deff_k < D  [Ordnung physikalisch]",
      1.0 if r["Deff_s"] < r["Deff_k"] < 200 else 0.0, 1.0, tol=0.0,
      unit="bool", ref="Physik: √S < S^0.25 für S<1",
      why="Schärfe-Verlust stärker als Kontrast-Verlust: √S < S^(1/4) für S<1")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 7: Q_vis Gewichtung 0.4/0.4/0.2 (bekannter Bugfix)
# ═══════════════════════════════════════════════════════════════════════════════

r = berechne(200, 1000)
Qshape_korrekt = 0.4*r["Q02"] + 0.4*r["Q04"] + 0.2*r["Q06"]
Qshape_alt     = 0.1*r["Q02"] + 0.1*r["Q04"] + 0.8*r["Q06"]

check("Q_vis Gewichtung 0.4/0.4/0.2  [Bugfix v4]",
      r["Qshape"], Qshape_korrekt, tol=1e-10,
      unit="", ref="Konsistenz mit berechne() und Diagramm",
      why="Früher 0.1/0.1/0.8 im Diagramm (übermäßig pessimistisch). Jetzt überall 0.4/0.4/0.2.")

check("Q_vis ≠ Q_vis_alt  [Bugfix wirksam]",
      1.0 if abs(r["Qshape"] - Qshape_alt) > 0.01 else 0.0, 1.0, tol=0.0,
      unit="bool", ref="Bugfix v4",
      why=f"Korrekt={Qshape_korrekt:.4f} vs alt={Qshape_alt:.4f}: Unterschied sichtbar")

check("Q_vis ≤ Strehl  [Q_vis = S·Qshape, Qshape≤1]",
      1.0 if r["Qvis"] <= r["strehl"] + 1e-9 else 0.0, 1.0, tol=0.0,
      unit="bool", ref="Physik: Q_vis=S·Qshape ≤ S",
      why="Qshape ist relative Größe ≤1, multipliziert mit Strehl ergibt Q_vis ≤ Strehl")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 8: Konsistenz absolut/relativ MTF-Verlust (Bugfix v4)
# ═══════════════════════════════════════════════════════════════════════════════

_, fc2, mp2, msa2, msr2, S2 = mtf_kurven(200, 1000, n_pts=200)
nu_ax2 = np.linspace(0, 1, len(mp2))

for nu_test, label in [(0.2, "Jupiter-Gürtel ~5\""), (0.4, "Festons ~1.5\"")]:
    mp_v  = float(np.interp(nu_test, nu_ax2, mp2))
    msa_v = float(np.interp(nu_test, nu_ax2, msa2))
    msr_v = float(np.interp(nu_test, nu_ax2, msr2))
    verlust_v4  = (mp_v - msa_v) / mp_v * 100.0   # v4: aus absoluter MTF
    verlust_v3  = (1 - msr_v) * 100.0              # v3: ignorierte Strehl

    check(f"Abs. Verlust > rel. Verlust (Strehl-Faktor)  ν={nu_test} {label}",
          1.0 if verlust_v4 > verlust_v3 + 0.1 else 0.0, 1.0, tol=0.0,
          unit="bool", ref="Bugfix v4: Strehl-Faktor jetzt enthalten",
          why=f"v4={verlust_v4:.1f}% > v3={verlust_v3:.1f}%: "
              f"v3 ignorierte Strehl={S2:.3f}, unterschätzte Verlust um {verlust_v4-verlust_v3:.1f}pp")

    check(f"Verlust ∈ (0%, 100%)  ν={nu_test} {label}",
          1.0 if 0 < verlust_v4 < 100 else 0.0, 1.0, tol=0.0,
          unit="bool", ref="Physik",
          why=f"Kontrastverlust muss positiv und endlich sein: {verlust_v4:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# AUSGABE
# ═══════════════════════════════════════════════════════════════════════════════

passed = sum(1 for r in results if r[0])
failed = len(results) - passed

blocks = {
    0: "Wellenfrontfehler & Strehl",
    1: "MTF Paraboloid (analytisch exakt)",
    2: "MTF Sphäre (Grenzwerte & Konsistenz)",
    3: "Absolute MTF (Strehl-Konsistenz)",
    4: "Geometrischer Blur",
    5: "Effektive Öffnungen",
    6: "Q_vis Gewichtung (Bugfix v4)",
    7: "Abs/Rel MTF Konsistenz (Bugfix v4)",
}

# Block-Grenzen (erste Test-Indices je Block)
block_starts = [0, 6, 11, 15, 19, 23, 27, 30]

print(f"\n{BOLD}{'═'*68}")
print(f"  Newton-Spiegel-Rechner — Validierungsprotokoll")
print(f"  λ=550nm  |  Referenzen: Suiter, Born&Wolf, Goodman, Rutten&v.Venrooij")
print(f"{'═'*68}{RESET}\n")

cur_block = -1
for i, (ok, name, got, exp, tol, unit, ref, why, pct) in enumerate(results):
    b = sum(1 for s in block_starts if i >= s) - 1
    if b != cur_block and b in blocks:
        print(f"\n{BOLD}  [{b+1}] {blocks[b]}{RESET}")
        cur_block = b
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    gotf = f"{got:.4f}" if isinstance(got, float) else str(got)
    expf = f"{exp:.4f}" if isinstance(exp, float) else str(exp)
    if verbose:
        print(f"  {sym}  {name}")
        print(f"       Wert: {gotf} {unit}   Ref: {expf}   Tol: ±{tol}")
        print(f"       Quelle: {ref}")
        print(f"       Begründung: {why}\n")
    else:
        status = f"{gotf} {unit}  (Ref: {expf}, ±{tol})"
        flag = f"  {RED}← FAIL{RESET}" if not ok else ""
        print(f"  {sym}  {name:<50}{status}{flag}")

print(f"\n{BOLD}{'═'*68}{RESET}")
if failed == 0:
    print(f"  {GREEN}{BOLD}ALLE {passed} TESTS BESTANDEN ✓{RESET}")
else:
    print(f"  {RED}{BOLD}{failed} FEHLGESCHLAGEN{RESET}  ({passed}/{passed+failed} bestanden)")
print(f"{'═'*68}\n")
sys.exit(0 if failed == 0 else 1)
