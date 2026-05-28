import numpy as np
import matplotlib.pyplot as plt
import math
from functools import lru_cache

# --- EXAKTE PHYSIK AUS DEINEM ORIGINAL-RECHNER (V19) ---

def mtf_para(f: float) -> float:
    """Analytische MTF eines beugungsbegrenzten Kreisteleskops (Paraboloid)."""
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    return (2/math.pi) * (math.acos(f) - f * math.sqrt(1 - f**2))

@lru_cache(maxsize=2048)
def mtf_sph_rel(f: float, Wb: float, n: int = 80) -> float:
    """Relative MTF eines sphärischen Spiegels."""
    if f <= 0: return 1.0
    if f >= 1: return 0.0
    lim = math.sqrt(max(0.0, 1.0 - (f/2)**2))
    re = im = norm = 0.0
    for i in range(n):
        x  = -lim + (2*i + 1) / (2*n) * 2*lim
        x1 = x - f/2
        x2 = x + f/2
        if x1**2 >= 1 or x2**2 >= 1: continue
        h  = min(math.sqrt(max(0.0, 1 - x1**2)), math.sqrt(max(0.0, 1 - x2**2)))
        dW = 2*math.pi * (Wb*(x2**4 - x2**2) - Wb*(x1**4 - x1**2))
        re   += math.cos(dW) * h
        im   += math.sin(dW) * h
        norm += h
    if norm < 1e-10: return 0.0
    return math.sqrt(re**2 + im**2) / norm

def mtf_kurven(D_mm: float, f_mm: float, lam_nm: float = 550.0, n_pts: int = 60) -> tuple:
    """Berechnet Ortsfrequenzen und MTF-Verläufe unter Berücksichtigung von Wb."""
    Wp     = D_mm**4 / (1024.0 * f_mm**3 * (lam_nm*1e-6))     # W_PtV paraxial [λ]
    Wb     = Wp / 4.0                                          # W_PtV best focus [λ]
    Wrms   = Wb / (1.5 * math.sqrt(5))                        # W_RMS best focus [λ]
    strehl = math.exp(-(2 * math.pi * Wrms)**2)

    fc = D_mm / (lam_nm * 1e-6 * 206265)                      # Grenzfrequenz in L/arcsec

    freqs_norm = np.linspace(0, 1, n_pts)
    freqs_as   = freqs_norm * fc

    mp  = np.array([mtf_para(f)        for f in freqs_norm])
    msr = np.array([mtf_sph_rel(f, Wb) for f in freqs_norm])  # FEHLER BEHOBEN: Wb übergeben!
    msa = msr * mp * strehl

    return freqs_as, fc, mp, msa, msr, strehl

def csf(f_cpd: float) -> float:
    """Contrast Sensitivity Function (CSF) des menschlichen Auges."""
    f = max(f_cpd, 1e-6)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))

def perceived_quality(D_mm: float, f_mm: float, lam_nm: float, V: float, n_pts: int = 120) -> float:
    """Berechnet den vergrößerungsabhängigen Qualitätsindex Q_perc."""
    freqs_as, fc_scope, mp_arr, msa_arr, msr_arr, strehl = mtf_kurven(D_mm, f_mm, lam_nm, n_pts)
    nu = freqs_as / fc_scope

    # Frequenzen im Auge (Cycles per Degree)
    f_cpd_arr = freqs_as * 3600.0 / V
    weight = np.array([csf(fi) for fi in f_cpd_arr])

    msa_scope = msr_arr * mp_arr * strehl
    
    # FEHLER BEHOBEN: Kompatibilität für alte und neue NumPy-Versionen (trapz vs trapezoid)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    num   = _trapz(msa_scope * weight, nu)
    denom = _trapz(mp_arr    * weight, nu)
    Q_raw = float(num / denom) if denom > 1e-12 else 0.0

    # Kritische Vergrößerung (Suiter-basiert aus deinem Code)
    V_krit = D_mm * 0.7 * math.sqrt(max(strehl, 1e-6))
    
    # Das mathematische Übergangsgewicht w(V)
    w = 1.0 / (1.0 + (V_krit / V) ** 2)
    return float(Q_raw + (1.0 - Q_raw) * (1.0 - w))

# --- PLOT GENERIEREN ---

teleskope = [
    {"D": 114, "f": 900, "name": "114/900 mm (Der Klassiker)", "color": "#2ca02c"},
    {"D": 130, "f": 650, "name": "130/650 mm (Bresser Spica)", "color": "#ff7f0e"},
    {"D": 200, "f": 1000, "name": "200/1000 mm (Seben 8'er)", "color": "#d62728"}
]

lam_nm = 550.0
V_range = np.linspace(30, 250, 100)

plt.figure(figsize=(10, 6), dpi=150)

for t in teleskope:
    q_visuell = []
    for V in V_range:
        Q = perceived_quality(t["D"], t["f"], lam_nm, V)
        q_visuell.append(Q)
        
    # Strehlwert für die Legende holen
    _, _, _, _, _, s_wert = mtf_kurven(t["D"], t["f"], lam_nm)
    plt.plot(V_range, q_visuell, label=f"{t['name']} (Strehl: {s_wert:.3f})", color=t["color"], lw=2.5)

# Schwellwertlinien aus deinem GUI-Design
plt.axhline(0.90, color="#0F6E56", linestyle="--", alpha=0.6, label="Q = 0.90 (Sehr gut)")
plt.axhline(0.70, color="#BA7517", linestyle=":", alpha=0.6, label="Q = 0.70 (Spürbarer Verlust)")

plt.title("Wahrgenommene Bildqualität $Q_{perc}$ im Vergleich (CSF-gewichtet)\n Martin's Auswertung sphärischer Hauptspiegel", fontsize=11, fontweight="bold", pad=12)
plt.xlabel("Vergrößerung (V-fach)", fontsize=10)
plt.ylabel("Wahrnehmbare Bildqualität $Q_{perc}$ (1.0 = Perfekt)", fontsize=10)
plt.xlim(30, 250)
plt.ylim(0.0, 1.05)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.show()