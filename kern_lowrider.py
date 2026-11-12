# -*- coding: utf-8 -*-
"""
kern_lowrider.py  –  v8.5.0

Low-Rider-Newton: Der Strahl wird am Fangspiegel nicht rechtwinklig (45°), sondern um den Winkel δ
nach unten abgelenkt. Der Okularauszug sitzt dadurch tiefer (bequemerer Einblick bei langen
Brennweiten), der Weg Fangspiegel -> Fokus wird länger, der Fangspiegel wächst.

Modell (Geometrie, keine Strahlrechnung):
  A      = Tubusradius + Okularauszug  (seitlicher Fokusabstand von der Tubusachse, bleibt konstant)
  Weg Fangspiegel -> Fokus        A' = A / cos δ
  Fokushöhe über Spiegelmitte     z_f = f − A·(1/cos δ + tan δ)      (45°: f − A)
  Absenkung des Einblicks         Δh  = A·(1/cos δ + tan δ − 1)
  Strahlkegel am Fangspiegel      d_K = D·A'/f + Bildfeld          = kleine Achse = Schattenbreite
  Einfallswinkel am Fangspiegel   i   = 45° − δ/2                  -> große Achse = d_K / cos i
  Obstruktion                     ε   = d_K / D
  Parabolisierungstiefe           s   = D⁴ / (1024 f³)             (Kugel minus Paraboloid)
Nicht enthalten: Fangspiegel-Versatz bei schnellen Systemen, Ausleuchtung am Feldrand, Abschattung
des schrägen Strahls durch Tubus/Okularauszug.
"""
import math
import numpy as np
from scipy.optimize import brentq

from kern_bau import tubus_aus_okular

ZOLL = 25.4
GEO_STD = dict(buet=50.0, tubus=0.0, bild=10.0, fe=30.0, afov=70.0)


def _geo(geo):
    g = dict(GEO_STD)
    if geo:
        g.update({k: float(v) for k, v in geo.items() if k in g})
    return g


def _A(D, f, geo):
    """Seitlicher Fokusabstand A und Tubus-Ø. tubus <= 0: aus Übersichtsokular (wie im Fangspiegel-Tab)."""
    t = geo["tubus"]
    if t <= 0:
        t = tubus_aus_okular(D, f, geo["buet"], geo["fe"], geo["afov"])["d_innen"]
    return t / 2.0 + geo["buet"], t


def lr_punkt(D, f, winkel, geo=None):
    """Alle Größen für Spiegel D, Brennweite f und Neigung winkel [°] (0 = normaler 45°-Newton)."""
    geo = _geo(geo)
    A, t = _A(D, f, geo)
    i = math.radians(winkel)
    sec = 1.0 / math.cos(i)
    dmin = D * A * sec / f + geo["bild"]
    einfall = 45.0 - winkel / 2.0
    return dict(winkel=winkel, A=A, tubus=t, weg=A * sec, zf=f - A * (sec + math.tan(i)),
                dh=A * (sec + math.tan(i) - 1.0), dmin=dmin,
                dmaj=dmin / math.cos(math.radians(einfall)), einfall=einfall,
                obs=dmin / D, zs=f - A * sec, asph_um=D ** 4 / (1024.0 * f ** 3) * 1000.0)


def lr_f_fuer_hoehe(D, z, winkel, geo=None):
    """Brennweite, bei der Spiegel D mit Neigung winkel die Fokushöhe z hat (None wenn nicht lösbar)."""
    try:
        return brentq(lambda f: lr_punkt(D, f, winkel, geo)["zf"] - z, 1.2 * D, 14.0 * D, xtol=0.05)
    except ValueError:
        return None


def lr_winkel_fuer_hoehe(D, f, z, geo=None, gmax=40.0):
    """Nötige Neigung [°], damit Spiegel D, f die Fokushöhe z erreicht (0 wenn schon tiefer, None wenn > gmax)."""
    if lr_punkt(D, f, 0.0, geo)["zf"] <= z:
        return 0.0
    if lr_punkt(D, f, gmax, geo)["zf"] > z:
        return None
    return brentq(lambda g: lr_punkt(D, f, g, geo)["zf"] - z, 0.0, gmax, xtol=0.01)


def lr_rechner(D, f, winkel, geo=None):
    """Ergebnis für die Textausgabe: gewählte Neigung gegen den normalen Newton."""
    n = lr_punkt(D, f, 0.0, geo)
    l = lr_punkt(D, f, winkel, geo)
    feq = lr_f_fuer_hoehe(D, l["zf"], 0.0, geo)           # normaler Newton mit gleicher Fokushöhe
    d = dict(norm=n, lr=l, N_eq=(feq / D) if feq else None)
    # Winkel mit kleinster großer Achse (Bauteilgröße)
    gs = np.linspace(0.0, 35.0, 141)
    mj = [lr_punkt(D, f, g, geo)["dmaj"] for g in gs]
    d["g_minmaj"] = float(gs[int(np.argmin(mj))])
    d["dmaj_min"] = float(min(mj))
    return d


# ----------------------------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------------------------
def lowrider_text(D, f, winkel, geo=None):
    d = lr_rechner(D, f, winkel, geo)
    n, l = d["norm"], d["lr"]
    z = [f"Low-Rider · Ø {D:.0f} mm · f = {f:.0f} mm (f/{f / D:.2f}) · Strahl {winkel:.0f}° nach unten",
         "",
         f"{'':34s}{'45°-Newton':>12s}{'Low-Rider':>12s}",
         f"{'Fokushöhe über Spiegelmitte [mm]':34s}{n['zf']:12.0f}{l['zf']:12.0f}",
         f"{'Einblick tiefer um [mm]':34s}{0:12.0f}{l['dh']:12.0f}",
         f"{'Einfallswinkel Fangspiegel [°]':34s}{n['einfall']:12.1f}{l['einfall']:12.1f}",
         f"{'Weg Fangspiegel → Fokus [mm]':34s}{n['weg']:12.0f}{l['weg']:12.0f}",
         f"{'Fangspiegel kleine Achse [mm]':34s}{n['dmin']:12.1f}{l['dmin']:12.1f}",
         f"{'Fangspiegel große Achse [mm]':34s}{n['dmaj']:12.1f}{l['dmaj']:12.1f}",
         f"{'Obstruktion [% von D]':34s}{n['obs'] * 100:12.1f}{l['obs'] * 100:12.1f}",
         "",
         ]
    if d["N_eq"]:
        z.append(f"Fokushöhe wie ein normaler Newton mit f/{d['N_eq']:.2f} (also "
                 f"{abs(d['N_eq'] - f / D):.2f} Blendenstufen schneller); Parabolisierungstiefe "
                 f"{D ** 4 / (1024 * (d['N_eq'] * D) ** 3) * 1000:.1f} µm statt {l['asph_um']:.1f} µm.")
    z.append(f"Kleinste große Achse (Bauteil) bei etwa {d['g_minmaj']:.0f}°: {d['dmaj_min']:.1f} mm "
             f"(45°: {n['dmaj']:.1f} mm). Die Schattenbreite (kleine Achse) wächst dabei immer.")
    z.append("Annahmen: gleicher seitlicher Fokusabstand, kein Fangspiegel-Versatz, keine Randausleuchtung, "
             "keine Abschattung des schrägen Strahls durch Tubus oder Okularauszug.")
    return "\n".join(z)


# ----------------------------------------------------------------------------------------------
# Zeichnung
# ----------------------------------------------------------------------------------------------
_COL = {0: "#78909c", 15: "#f57c00", 20: "#f57c00", 30: "#c62828"}


def _skizze(ax, D, f, winkel, geo):
    for g, c in ((0.0, "#78909c"), (winkel, "#d84315")):
        if g == 0.0 and winkel == 0.0 and c == "#d84315":
            continue
        p = lr_punkt(D, f, g, geo)
        i = math.radians(g)
        uo = np.array([math.cos(i), -math.sin(i)])
        ui = np.array([0.0, 1.0])
        nrm = uo - ui
        nrm /= np.linalg.norm(nrm)
        t = np.array([-nrm[1], nrm[0]])
        ax.plot([0, 0], [-0.15 * p["A"], 1.1 * p["A"]], color=c, lw=1, ls=":")
        ax.plot([0, p["A"]], [0, -p["A"] * math.tan(i)], color=c, lw=2.4)
        ax.plot([-p["dmaj"] / 2 * t[0], p["dmaj"] / 2 * t[0]], [-p["dmaj"] / 2 * t[1], p["dmaj"] / 2 * t[1]],
                color=c, lw=5, solid_capstyle="butt")
        ax.plot([p["A"]], [-p["A"] * math.tan(i)], "o", color=c)
        ax.text(p["A"] + 0.03 * p["A"], -p["A"] * math.tan(i),
                "45°-Newton" if g == 0.0 else f"Low-Rider {g:.0f}°\nFokus {p['dh']:.0f} mm tiefer",
                color=c, va="center", fontsize=9)
    A = lr_punkt(D, f, 0.0, geo)["A"]
    ax.set_aspect("equal")
    ax.set_xlim(-0.35 * A, 1.9 * A)
    ax.set_ylim(-0.95 * A * max(math.tan(math.radians(max(winkel, 10))), 0.2) - 0.3 * A, 0.9 * A)
    ax.axis("off")
    ax.set_title("Prinzip: Fangspiegel flacher, Strahl geht nach unten", fontsize=10.5)


def _ansicht_fangspiegel(fig, D, f, winkel, geo):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.25]))
    _skizze(a1, D, f, winkel, geo)
    gs = np.linspace(0.0, 35.0, 141)
    P = [lr_punkt(D, f, g, geo) for g in gs]
    dh = np.array([p["dh"] for p in P])
    mn = np.array([p["dmin"] for p in P])
    mj = np.array([p["dmaj"] for p in P])
    a2.plot(dh, mn, color="#1565c0", lw=2.6, label="kleine Achse = Schattenbreite")
    a2.plot(dh, mj, color="#9e9e9e", lw=1.6, ls="--", label="große Achse (Bauteil)")
    k = int(np.argmin(mj))
    a2.plot([dh[k]], [mj[k]], "v", color="#757575")
    a2.annotate(f"kleinste große Achse\n{mj[k]:.0f} mm bei {gs[k]:.0f}°", (dh[k], mj[k]),
                xytext=(dh[k] * 0.55, mj[k] + 0.07 * (mj.max() - mn.min())), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="#777"))
    s = lr_punkt(D, f, winkel, geo)
    a2.plot([s["dh"], s["dh"]], [s["dmin"], s["dmaj"]], color="#d84315", lw=1, ls=":")
    a2.plot([s["dh"]], [s["dmin"]], "o", color="#d84315", ms=8)
    a2.plot([s["dh"]], [s["dmaj"]], "o", color="#d84315", ms=6, mfc="none")
    a2.annotate(f"{winkel:.0f}°: −{s['dh']:.0f} mm\nkleine Achse {s['dmin']:.0f} mm\nObstruktion {s['obs'] * 100:.1f} %",
                (s["dh"], s["dmin"]), xytext=(s["dh"] + 0.12 * dh.max(), 0.5 * (s["dmin"] + mj.min()) - 0.04 * (mj.max() - mn.min())),
                fontsize=8.5, color="#d84315", arrowprops=dict(arrowstyle="-", color="#d84315", lw=.8))
    a2.set_xlabel("Einblickhöhe tiefer als beim 45°-Newton [mm]")
    a2.set_ylabel("Fangspiegel [mm]")
    a2.grid(alpha=.3)
    ob = a2.secondary_yaxis("right", functions=(lambda y: y / D * 100, lambda q: q * D / 100))
    ob.set_ylabel("Obstruktion [% von D]")
    wk = a2.secondary_xaxis("top", functions=(lambda x: np.interp(x, dh, gs), lambda a: np.interp(a, gs, dh)))
    wk.set_xlabel("Neigung des Strahls nach unten [°]")
    a2.set_title(f"Ø {D:.0f} mm, f/{f / D:.1f}, Okularauszug {_geo(geo)['buet']:.0f} mm, Bildfeld {_geo(geo)['bild']:.0f} mm",
                 fontsize=10.5)
    a2.legend(loc="upper left", fontsize=9)


def _ansicht_schar(fig, D, f, geo, gmax=30.0):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.25, 1]))
    Nu = f / D
    ziele = [n for n in (Nu - 1.0, Nu - 0.5, Nu) if n >= 2.5]
    cols = ["#3b1f78", "#2a788e", "#5ec962"][-len(ziele):]
    Ns = np.linspace(max(2.5, Nu - 1.0), Nu + 1.6, 260)
    for N0, c in zip(ziele, cols):
        z = lr_punkt(D, N0 * D, 0.0, geo)["zf"]
        X, O, G = [], [], []
        for N in Ns:
            if N < N0 - 1e-9:
                continue
            g = lr_winkel_fuer_hoehe(D, N * D, z, geo, gmax)
            if g is None:
                continue
            X.append(N)
            G.append(g)
            O.append(lr_punkt(D, N * D, g, geo)["obs"] * 100)
        a1.plot(X, O, color=c, lw=2.5, label=f"Einblickhöhe wie f/{N0:.1f} ohne Low-Rider ({z:.0f} mm)")
        a1.plot([N0], [lr_punkt(D, N0 * D, 0.0, geo)["obs"] * 100], "s", color=c, ms=8)
        a2.plot(X, G, color=c, lw=2.5)
        for g_mark in (10, 20, 30):
            if G and max(G) >= g_mark > 0:
                idx = int(np.argmin(np.abs(np.array(G) - g_mark)))
                a1.plot([X[idx]], [O[idx]], "o", color=c, ms=4)
                a1.annotate(f"{g_mark}°", (X[idx], O[idx]), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8, color=c)
    a1.set_xlabel("Öffnungsverhältnis f/D")
    a1.set_ylabel("Obstruktion [% von D]")
    a1.grid(alpha=.3)
    a1.legend(fontsize=8.5, loc="best")
    a1.set_title(f"Bei gleicher Einblickhöhe: Obstruktion (D = {D:.0f} mm; ■ = normaler Newton)", fontsize=10.5)
    a2.set_xlabel("Öffnungsverhältnis f/D")
    a2.set_ylabel("nötiger Low-Rider-Winkel [°]")
    a2.grid(alpha=.3)
    ax = a2.twinx()
    Nn = np.linspace(Ns[0], Ns[-1], 100)
    ax.plot(Nn, [D ** 4 / (1024 * (N * D) ** 3) * 1000 for N in Nn], color="#d84315", lw=2, ls="--")
    ax.set_ylabel("Parabolisierungstiefe [µm] (gestrichelt)", color="#d84315", fontsize=9)
    ax.tick_params(axis="y", colors="#d84315")
    a2.set_title("Preis: Winkel · Gewinn: weniger Asphärik", fontsize=10.5)


def _ansicht_durchmesser(fig, D, f, D_max, geo, winkel_liste=(0.0, 15.0, 30.0)):
    axs = fig.subplots(1, 4)
    N0 = f / D
    z0 = lr_punkt(D, f, 0.0, geo)["zf"]
    Ds = np.linspace(D, max(D_max, D * 1.05), 4)
    cols = {0.0: "#78909c", 15.0: "#f57c00", 30.0: "#c62828"}
    for g in winkel_liste:
        X, N_, E_, S_, K_ = [], [], [], [], []
        for Dx in Ds:
            fx = lr_f_fuer_hoehe(Dx, z0, g, geo)
            if fx is None:
                continue
            p = lr_punkt(Dx, fx, g, geo)
            X.append(Dx / ZOLL)
            N_.append(fx / Dx)
            E_.append(p["obs"] * 100)
            S_.append(p["asph_um"])
            K_.append((N0 / (fx / Dx)) ** 2)
        lab = "45° (normaler Newton)" if g == 0 else f"Low-Rider {g:.0f}°"
        for ax, y in zip(axs, (N_, E_, S_, K_)):
            ax.plot(X, y, "o-", color=cols.get(g, "#555"), lw=2.2, label=lab)
    for ax, yl, tt in zip(axs, ("nötiges f/D", "Obstruktion [% von D]", "Parabolisierungstiefe [µm]",
                                "Koma relativ (1 = Referenz)"),
                          ("Öffnungsverhältnis bei gleicher\nEinblickhöhe", "Obstruktion",
                           "Schleif-/Polier-\nschwierigkeit", "Koma ∝ 1/N²")):
        ax.set_xlabel("Spiegeldurchmesser [Zoll]")
        ax.set_ylabel(yl)
        ax.set_title(tt, fontsize=10.5)
        ax.grid(alpha=.3)
    axs[0].legend(fontsize=8.5)
    fig.suptitle(f"Einblickhöhe wie Ø {D / ZOLL:.1f}″ f/{N0:.1f} ({z0:.0f} mm über Spiegelmitte): "
                 f"was kostet der größere Spiegel?", fontsize=11.5)


ANSICHTEN = ["Fangspiegel", "Gleiche Einblickhöhe", "Größerer Spiegel"]


def lowrider_zeichnen(fig, D, f, winkel, ansicht, geo=None, D_max=None):
    """Zeichnet die gewählte Ansicht in fig und gibt lr_rechner() zurück."""
    fig.clf()
    geo = _geo(geo)
    if ansicht == ANSICHTEN[1]:
        _ansicht_schar(fig, D, f, geo)
    elif ansicht == ANSICHTEN[2]:
        _ansicht_durchmesser(fig, D, f, D_max if D_max else 20.0 * ZOLL, geo)
    else:
        _ansicht_fangspiegel(fig, D, f, winkel, geo)
    fig.tight_layout()
    return lr_rechner(D, f, winkel, geo)
