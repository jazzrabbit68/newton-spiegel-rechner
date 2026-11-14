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
  Absenkung der Brennebene        Δh  = A·(1/cos δ + tan δ − 1)
  Augenpunkt: L hinter der Brennebene entlang des schrägen Strahls (Okularlänge + Augenabstand)
     Augenhöhe                    z_a = z_f − L·sin δ
     Absenkung des Auges          Δh_a = Δh + L·sin δ                 (45°: Strahl waagerecht, Auge = Fokushöhe)
  Strahlkegel am Fangspiegel      d_K = D·A'/f + Bildfeld          = nötige kleine Achse
  Einfallswinkel am Fangspiegel   i   = 45° − δ/2
  Fangspiegelform                 Standard-Ellipse (große Achse = 1,41·kleine) oder passend (= kleine/cos i)
  Schatten in Achsrichtung        große Achse·cos i  (+ Randseite t·sin|45° − i| bei Standard-Ellipse), quer: kleine Achse
  Obstruktion                     ε = √(Schattenlänge · Schattenbreite) / D
  Parabolisierungstiefe           s = D⁴ / (1024 f³)             (Kugel minus Paraboloid)
Nicht enthalten: Fangspiegel-Versatz bei schnellen Systemen, Ausleuchtung am Feldrand, Abschattung
des schrägen Strahls durch Tubus/Okularauszug, Fangspiegelhalter und Spinne.
"""
import math
import numpy as np
from scipy.optimize import brentq

from kern_bau import tubus_aus_okular

ZOLL = 25.4
FORMEN = ["Standard-Ellipse (1,41)", "passend geformt"]
GEO_STD = dict(buet=50.0, tubus=0.0, bild=10.0, fe=30.0, afov=70.0,
               form=0.0,      # 0 = Standard-Ellipse (Achsenverhältnis 1,41), 1 = passend zum Winkel geformt
               dicke=10.0,    # Randlänge des Fangspiegels [mm]; wirkt nur bei Abweichung von 45°
               auge=80.0)     # Auge hinter der Brennebene [mm] (Okularlänge + Augenabstand)
ANSICHTEN = ["Absenkung", "Größerer Spiegel"]


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
    dmin = D * A * sec / f + geo["bild"]                     # Strahlkegel + Bildfeld = nötige kleine Achse
    einfall = 45.0 - winkel / 2.0
    ci = math.cos(math.radians(einfall))
    dmaj = dmin / ci if geo["form"] >= 0.5 else math.sqrt(2.0) * dmin   # passend / Standard-Ellipse
    # Standard-Fangspiegel sind bei 45° gebohrt: Die Randseite steht bei 45° parallel zur Achse (kein Schatten) und
    # ragt erst bei Abweichung von 45° herein. Eine passend gebohrte Ellipse hat stets eine achsparallele Randseite.
    rand = geo["dicke"] * math.sin(math.radians(abs(45.0 - einfall))) if geo["form"] < 0.5 else 0.0
    sch_l = dmaj * ci + rand
    zf = f - A * (sec + math.tan(i))
    dh = A * (sec + math.tan(i) - 1.0)
    dauge = geo["auge"] * math.sin(i)
    return dict(winkel=winkel, A=A, tubus=t, weg=A * sec, zf=zf, dh=dh,
                z_auge=zf - dauge, dh_auge=dh + dauge,
                dmin=dmin, dmaj=dmaj, einfall=einfall, sch_l=sch_l, sch_q=dmin,
                obs=math.sqrt(sch_l * dmin) / D, zs=f - A * sec,
                asph_um=D ** 4 / (1024.0 * f ** 3) * 1000.0)


def lr_f_fuer_hoehe(D, z, winkel, geo=None):
    """Brennweite, bei der Spiegel D mit Neigung winkel die Augenhöhe z hat (None wenn nicht lösbar)."""
    try:
        return brentq(lambda f: lr_punkt(D, f, winkel, geo)["z_auge"] - z, 1.2 * D, 14.0 * D, xtol=0.05)
    except ValueError:
        return None


def lr_rechner(D, f, winkel, geo=None):
    """Ergebnis für die Textausgabe: gewählte Neigung gegen den normalen Newton."""
    n = lr_punkt(D, f, 0.0, geo)
    l = lr_punkt(D, f, winkel, geo)
    feq = lr_f_fuer_hoehe(D, l["z_auge"], 0.0, geo)       # normaler Newton mit gleicher Augenhöhe
    return dict(norm=n, lr=l, N_eq=(feq / D) if feq else None)


# ----------------------------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------------------------
def lowrider_text(D, f, winkel, geo=None, ansicht=None, D_max=None):
    d = lr_rechner(D, f, winkel, geo)
    n, l = d["norm"], d["lr"]
    g_ = _geo(geo)
    z = [f"Low-Rider · Ø {D:.0f} mm · f = {f:.0f} mm (f/{f / D:.2f}) · Strahl {winkel:.0f}° nach unten",
         f"Fangspiegel: {FORMEN[1] if g_['form'] >= 0.5 else FORMEN[0]}, Randlänge {g_['dicke']:.0f} mm "
         f"(wirkt bei Abweichung von 45°); Auge {g_['auge']:.0f} mm hinter der Brennebene",
         "",
         f"{'':38s}{'45°-Newton':>12s}{'Low-Rider':>12s}",
         f"{'Brennebene über Spiegelmitte [mm]':38s}{n['zf']:12.0f}{l['zf']:12.0f}",
         f"{'Auge über Spiegelmitte [mm]':38s}{n['z_auge']:12.0f}{l['z_auge']:12.0f}",
         f"{'Absenkung Brennebene [mm]':38s}{0:12.0f}{l['dh']:12.0f}",
         f"{'Absenkung Auge [mm]':38s}{0:12.0f}{l['dh_auge']:12.0f}",
         f"{'Einfallswinkel Fangspiegel [°]':38s}{n['einfall']:12.1f}{l['einfall']:12.1f}",
         f"{'Weg Fangspiegel → Fokus [mm]':38s}{n['weg']:12.0f}{l['weg']:12.0f}",
         f"{'Fangspiegel kleine Achse [mm]':38s}{n['dmin']:12.1f}{l['dmin']:12.1f}",
         f"{'Fangspiegel große Achse [mm]':38s}{n['dmaj']:12.1f}{l['dmaj']:12.1f}",
         f"{'Schatten in Kipprichtung [mm]':38s}{n['sch_l']:12.1f}{l['sch_l']:12.1f}",
         f"{'Schatten quer [mm]':38s}{n['sch_q']:12.1f}{l['sch_q']:12.1f}",
         f"{'Obstruktion (flächengleich) [%]':38s}{n['obs'] * 100:12.2f}{l['obs'] * 100:12.2f}",
         ""]
    if d["N_eq"]:
        z.append(f"Augenhöhe wie ein normaler Newton mit f/{d['N_eq']:.2f} ({abs(d['N_eq'] - f / D):.2f} Blendenstufen "
                 f"schneller); Parabolisierungstiefe {D ** 4 / (1024 * (d['N_eq'] * D) ** 3) * 1000:.1f} µm statt "
                 f"{l['asph_um']:.1f} µm.")
    if g_["form"] < 0.5:
        pf = lr_punkt(D, f, winkel, dict(geo or {}, form=1.0))
        z.append(f"Mit passend geformter Ellipse (Achsenverhältnis {pf['dmaj'] / pf['dmin']:.2f}) läge die Obstruktion "
                 f"bei {pf['obs'] * 100:.2f} % statt {l['obs'] * 100:.2f} %.")
    if D < 13.5 * ZOLL:
        z.append(f"Hinweis: Bei Spiegeln unter etwa 14″ lohnt der Low-Rider kaum (hier {l['dh_auge'] / 10:.1f} cm tiefer "
                 f"bei {winkel:.0f}°).")
    if ansicht == ANSICHTEN[1]:
        z += ["", _vergleich_tabelle(D, f, D_max if D_max else 20.0 * ZOLL, geo)]
    z.append("")
    z.append("Annahmen: gleicher seitlicher Fokusabstand, kein Fangspiegel-Versatz, keine Randausleuchtung, keine Abschattung "
             "des schrägen Strahls durch Tubus oder Okularauszug, ohne Fangspiegelhalter und Spinne.")
    return "\n".join(z)


# ----------------------------------------------------------------------------------------------
# Vergleich größerer Spiegel
# ----------------------------------------------------------------------------------------------
_WINKEL_VGL = (0.0, 15.0, 30.0)


def _vergleich_daten(D, f, D_max, geo, n=4):
    """Je Winkel und Durchmesser: nötiges f/D, Obstruktion, Parabolisierungstiefe, Koma relativ – bei gleicher Augenhöhe."""
    N0 = f / D
    z0 = lr_punkt(D, f, 0.0, geo)["z_auge"]
    Ds = np.linspace(D, max(D_max, D * 1.05), n)
    daten = {}
    for g in _WINKEL_VGL:
        zeilen = []
        for Dx in Ds:
            fx = lr_f_fuer_hoehe(Dx, z0, g, geo)
            if fx is None:
                continue
            p = lr_punkt(Dx, fx, g, geo)
            zeilen.append(dict(D=Dx, N=fx / Dx, obs=p["obs"] * 100, asph=p["asph_um"], koma=(N0 / (fx / Dx)) ** 2))
        daten[g] = zeilen
    return daten, N0, z0


def _vergleich_tabelle(D, f, D_max, geo):
    daten, N0, z0 = _vergleich_daten(D, f, D_max, geo)
    z = [f"Gleiche Augenhöhe ({z0:.0f} mm über Spiegelmitte) wie Ø {D / ZOLL:.1f}″ f/{N0:.1f} im 45°-Newton:",
         f"{'':14s}{'Ø [″]':>7s}{'f/D':>7s}{'Obstr. [%]':>12s}{'Parab. [µm]':>13s}{'Koma rel.':>11s}"]
    for g, zeilen in daten.items():
        for k, r in enumerate(zeilen):
            lab = ("45°" if g == 0 else f"Low-Rider {g:.0f}°") if k == 0 else ""
            z.append(f"{lab:14s}{r['D'] / ZOLL:7.1f}{r['N']:7.2f}{r['obs']:12.1f}{r['asph']:13.1f}{r['koma']:11.2f}")
    return "\n".join(z)


# ----------------------------------------------------------------------------------------------
# Zeichnung
# ----------------------------------------------------------------------------------------------
def plt_cmap(n, name="viridis"):
    import matplotlib
    cmap = matplotlib.colormaps[name]
    return [cmap(v) for v in np.linspace(0.05, 0.85, n)]


def _skizze(ax, D, f, winkel, geo):
    geo = _geo(geo)
    L = geo["auge"]
    A = lr_punkt(D, f, 0.0, geo)["A"]
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
        fok = p["weg"] * uo
        aug = fok + L * uo
        ax.plot([0, 0], [-0.1 * A, 0.9 * A], color=c, lw=1, ls=":")
        ax.plot([0, aug[0]], [0, aug[1]], color=c, lw=2.2)
        ax.plot([-p["dmaj"] / 2 * t[0], p["dmaj"] / 2 * t[0]], [-p["dmaj"] / 2 * t[1], p["dmaj"] / 2 * t[1]],
                color=c, lw=5, solid_capstyle="butt")
        ax.plot([fok[0]], [fok[1]], "o", color=c, ms=7)                 # Brennebene
        ax.plot([aug[0]], [aug[1]], "D", color=c, ms=7, mfc="white")    # Auge
        if g == 0.0:
            ax.text(aug[0] + 0.04 * A, aug[1] + 0.06 * A, "45°-Newton: Strahl waagerecht,\nBrennebene und Auge gleich hoch",
                    color=c, va="bottom", fontsize=8.5)
        else:
            ax.plot([fok[0], fok[0]], [0, fok[1]], color=c, lw=1.2)
            ax.plot([aug[0], aug[0]], [0, aug[1]], color=c, lw=1.2, ls="--")
            ax.annotate("", xy=(fok[0] - 0.06 * A, fok[1]), xytext=(fok[0] - 0.06 * A, 0),
                        arrowprops=dict(arrowstyle="<->", color=c, lw=1.2))
            ax.annotate("", xy=(aug[0] + 0.06 * A, aug[1]), xytext=(aug[0] + 0.06 * A, 0),
                        arrowprops=dict(arrowstyle="<->", color=c, lw=1.2))
            ax.text(fok[0] - 0.09 * A, fok[1] / 2, f"Brennebene\n−{p['dh']:.0f} mm", color=c, ha="right", va="center", fontsize=8.5)
            ax.text(aug[0] + 0.09 * A, aug[1] / 2, f"Auge\n−{p['dh_auge']:.0f} mm", color=c, ha="left", va="center", fontsize=8.5)
    ax.plot([], [], "o", color="#555", label="Brennebene")
    ax.plot([], [], "D", color="#555", mfc="white", label="Auge")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_aspect("equal")
    pw = lr_punkt(D, f, max(winkel, 8.0), geo)
    ax.set_xlim(-0.3 * A, pw["weg"] + L + 0.9 * A)
    ax.set_ylim(-1.15 * pw["dh_auge"] - 0.2 * A, 0.6 * A)
    ax.axis("off")
    ax.set_title("Prinzip: Strahl geht nach unten, das Auge liegt noch tiefer", fontsize=10.5)


def _ansicht_absenkung(fig, D, f, winkel, geo):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.15]))
    _skizze(a1, D, f, winkel, geo)
    gs = np.linspace(0.0, 35.0, 71)
    zolls = [14, 16, 18, 20, 22]
    N = f / D
    for z, c in zip(zolls, plt_cmap(len(zolls))):
        Dx = z * ZOLL
        a2.plot(gs, [lr_punkt(Dx, N * Dx, g, geo)["dh_auge"] / 10.0 for g in gs], color=c, lw=1.8, label=f"{z}″")
    a2.plot(gs, [lr_punkt(D, f, g, geo)["dh_auge"] / 10.0 for g in gs], color="k", lw=2.6, label=f"dein Spiegel (Auge)")
    a2.plot(gs, [lr_punkt(D, f, g, geo)["dh"] / 10.0 for g in gs], color="k", lw=1.4, ls="--", label="dein Spiegel (nur Brennebene)")
    s = lr_punkt(D, f, winkel, geo)
    a2.plot([winkel], [s["dh_auge"] / 10.0], "o", color="#d84315", ms=8)
    a2.plot([winkel], [s["dh"] / 10.0], "o", color="#d84315", ms=6, mfc="white")
    a2.axvline(winkel, color="#999", lw=.8, ls=":")
    a2.set_xlabel("Strahl nach unten δ [°]")
    a2.set_ylabel("Einblick tiefer um [cm]")
    a2.set_title("Absenkung des Auges (durchgezogen) für 14–22″", fontsize=10.5)
    a2.grid(alpha=.3)
    a2.legend(fontsize=8.5, loc="upper left")


def _ansicht_durchmesser(fig, D, f, D_max, geo):
    a1, a2 = fig.subplots(1, 2)
    cols = {0.0: "#78909c", 15.0: "#f57c00", 30.0: "#c62828"}
    geo_s = dict(geo, form=0.0)       # Standard-Ellipse (im Handel erhältlich)
    geo_p = dict(geo, form=1.0)       # passend geformt (Sonderanfertigung, "optimiert")
    ds, N0, z0 = _vergleich_daten(D, f, D_max, geo_s)
    dp, _, _ = _vergleich_daten(D, f, D_max, geo_p)
    for g in _WINKEL_VGL:
        if not ds.get(g):
            continue
        X = [r["D"] / ZOLL for r in ds[g]]
        lab = "45° (normaler Newton)" if g == 0 else f"Low-Rider {g:.0f}°"
        a1.plot(X, [r["N"] for r in ds[g]], "o-", color=cols[g], lw=2.2, label=lab)
        a2.plot(X, [r["obs"] for r in ds[g]], "o-", color=cols[g], lw=2.2,
                label=lab + ("" if g == 0 else ", Standard-Ellipse"))
        if g > 0 and dp.get(g):
            a2.plot([r["D"] / ZOLL for r in dp[g]], [r["obs"] for r in dp[g]], "s--", color=cols[g], lw=1.8,
                    mfc="white", label=f"Low-Rider {g:.0f}°, passend geformt")
    a1.set_ylabel("nötiges f/D")
    a1.set_title("Öffnungsverhältnis bei gleicher Augenhöhe", fontsize=10.5)
    a2.set_ylabel("Obstruktion [% von D]")
    a2.set_title("Obstruktion bei gleicher Augenhöhe", fontsize=10.5)
    for ax in (a1, a2):
        ax.set_xlabel("Spiegeldurchmesser [Zoll]")
        ax.grid(alpha=.3)
    a1.legend(fontsize=8.5)
    a2.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Augenhöhe wie Ø {D / ZOLL:.1f}″ f/{N0:.1f} im 45°-Newton ({z0:.0f} mm über Spiegelmitte): "
                 f"was kostet der größere Spiegel?", fontsize=11)


def lowrider_zeichnen(fig, D, f, winkel, ansicht, geo=None, D_max=None):
    """Zeichnet die gewählte Ansicht in fig und gibt lr_rechner() zurück."""
    fig.clf()
    geo = _geo(geo)
    if ansicht == ANSICHTEN[1]:
        _ansicht_durchmesser(fig, D, f, D_max if D_max else 20.0 * ZOLL, geo)
    else:
        _ansicht_absenkung(fig, D, f, winkel, geo)
    fig.tight_layout()
    return lr_rechner(D, f, winkel, geo)
