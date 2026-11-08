"""3D-Ansicht des Gittertubus (neu in v8.2.0).

Maßstäblich aus den Rechenwerten von gitterrohr_auto / gitterrohr_stangen / fangspiegel_rechner.
Keine Tk-/Streamlit-Abhängigkeit; zeichnet in eine übergebene matplotlib-Figure.
Geschätzte (nicht berechnete) Teile: Spiegeldicke (D/8), Spinnenbleche, OAZ-Winkel, Höhe der Streulichtblende.
Orientierung: OAZ bei Azimut 0°, Spinnenbeine bei 60°/180°/300° (OAZ zwischen zwei Beinen, 3. Bein in Verlängerung gegenüber).
"""
import math
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from kern_bau import fangspiegel_rechner, fangspiegel_versatz

_WOOD = "#d9b98a"; _ALU = "#c9ced4"; _RED = "#b02a2a"; _BLK = "#2b2b2b"; _ROD = "#b9bec5"
_LIGHT = np.array([-0.4, -0.5, 0.8]); _LIGHT /= np.linalg.norm(_LIGHT)


class _Szene:
    def __init__(self):
        self.polys = []; self.cols = []

    def add(self, pts, col, amb=0.45, alpha=1.0):
        pts = np.array(pts, float)
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0]); nn = np.linalg.norm(n)
        k = amb + (1 - amb) * abs(np.dot(n / nn, _LIGHT)) if nn > 0 else 1
        c = np.array(to_rgb(col)) * k
        self.polys.append(pts); self.cols.append((*np.clip(c, 0, 1), alpha))

    def prism(self, c0, u, v, w, col):
        c0, u, v, w = (np.array(x, float) for x in (c0, u, v, w))
        P = lambda a, b, c: c0 + a * u + b * v + c * w
        for q in (((0,0,0),(1,0,0),(1,1,0),(0,1,0)), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)),
                  ((0,0,0),(1,0,0),(1,0,1),(0,0,1)), ((0,1,0),(1,1,0),(1,1,1),(0,1,1)),
                  ((0,0,0),(0,1,0),(0,1,1),(0,0,1)), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))):
            self.add([P(*p) for p in q], col)

    def cyl(self, p0, p1, rad, col, n=14):
        p0 = np.array(p0, float); p1 = np.array(p1, float)
        d = p1 - p0; d /= np.linalg.norm(d)
        a = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.array([1., 0, 0])
        a /= np.linalg.norm(a); b = np.cross(d, a); t = np.linspace(0, 2 * np.pi, n + 1)
        for s, e in zip(t[:-1], t[1:]):
            self.add([p0 + rad * (math.cos(s) * a + math.sin(s) * b), p0 + rad * (math.cos(e) * a + math.sin(e) * b),
                      p1 + rad * (math.cos(e) * a + math.sin(e) * b), p1 + rad * (math.cos(s) * a + math.sin(s) * b)], col)

    def ringplate(self, ri, ro, z0, z1, col, n=96):
        t = np.linspace(0, 2 * np.pi, n + 1)
        c = math.cos; s_ = math.sin
        for s, e in zip(t[:-1], t[1:]):
            self.add([(ro*c(s), ro*s_(s), z1), (ro*c(e), ro*s_(e), z1), (ri*c(e), ri*s_(e), z1), (ri*c(s), ri*s_(s), z1)], col, 0.75)
            self.add([(ro*c(s), ro*s_(s), z0), (ro*c(e), ro*s_(e), z0), (ro*c(e), ro*s_(e), z1), (ro*c(s), ro*s_(s), z1)], col, 0.6)
            self.add([(ri*c(s), ri*s_(s), z0), (ri*c(e), ri*s_(e), z0), (ri*c(e), ri*s_(e), z1), (ri*c(s), ri*s_(s), z1)], col, 0.6)


def gitter3d_szene(D, f, ga, g, h, r, ring_b, ring_dick, loch_ring, bauart, fs=None):
    """Baut Polygone. ga = gitterrohr_auto-Dict, g = gitterrohr_stangen-Dict, r = Teilkreisradius.
    fs = dict(buet, fe, afov) für die Blende; Rückgabe (Szene, Kegel, Maße-Dict)."""
    S = _Szene()
    ri = ga["d_innen"] / 2.0; ro = ri + ring_b
    z0 = ga["z0"]; zt = ga["tief"]; zs = ga["z_sec"]; A = ga["A"]; dfs = ga["d_fang"]
    ru = z0 + h + loch_ring; zo = ru + ring_dick          # Unterkante / Oberkante oberer Ring
    S.ringplate(ri, ro, zt - ring_dick, zt, _WOOD); S.ringplate(ri, ro, ru, zo, _WOOD)
    # Spiegel
    dick = D / 8.0; t = np.linspace(0, 2 * np.pi, 80)
    S.add([(D/2*math.cos(a), D/2*math.sin(a), 0) for a in t], "#8fc1d9", 0.9)
    for s, e in zip(t[:-1], t[1:]):
        S.add([(D/2*math.cos(s), D/2*math.sin(s), -dick), (D/2*math.cos(e), D/2*math.sin(e), -dick),
               (D/2*math.cos(e), D/2*math.sin(e), 0), (D/2*math.cos(s), D/2*math.sin(s), 0)], "#6fa3bd")
    # Anker, Stangen
    n = g["n"]; off = math.radians(30.0)
    ph = [off + 2 * math.pi * i / n for i in range(n)]
    dphi_r = math.radians(g["dphi"]); tx = g.get("bauart") == "triplex"; zg = g.get("bauart") == "zigzag"
    P = lambda a, z: np.array([r * math.cos(a), r * math.sin(a), z])

    def block(a, zf, direction):
        c = P(a, zf); rad = np.array([math.cos(a), math.sin(a), 0]); tg = np.array([-math.sin(a), math.cos(a), 0])
        S.prism(c - rad * 12 - tg * 14, rad * 24, tg * 28, np.array([0, 0, 1.0]) * direction * (loch_ring + 10), _ALU)
    rods = []; lo = []; up = []
    for i in range(n):
        for sgn in (((1 if i % 2 == 0 else -1),) if tx else ((1, -1) if (g["kreuzt"] or zg) else (1,))):
            a0 = ph[i]; a1 = ph[i] + sgn * dphi_r
            rods.append((P(a0, z0), P(a1, z0 + h))); lo.append(a0); up.append(a1)
    for a in sorted(set(round(x, 6) for x in lo)): block(a, zt, 1)
    for a in sorted(set(round(x, 6) for x in up)): block(a, ru, -1)
    kop = max(4.0, 0.4 * 25)
    ab = 25.0
    for p, q in rods:
        d = (q - p); L = np.linalg.norm(d)
        if L < 3 * ab:
            S.cyl(p, q, 7.5, _ROD); continue
        d /= L
        S.cyl(p, p + d * ab, 10, _ALU); S.cyl(p + d * ab, q - d * ab, 7.5, _ROD); S.cyl(q - d * ab, q, 10, _ALU)
    if tx:   # Verspannung in der Kreuzungsmitte
        for i in range(0, n - 1, 2):
            m = (P(ph[i], z0) + P(ph[i + 1], z0 + h)) / 2; m2 = (P(ph[i + 1], z0) + P(ph[i], z0 + h)) / 2
            mid = (m + m2) / 2; S.cyl(mid - np.array([0, 0, 10]), mid + np.array([0, 0, 10]), 11, "#f2f2f2")
    # Fangspiegel (mit Versatz) + Spinne
    vs = fangspiegel_versatz(A, D, f); dl = vs["quer"]
    c = np.array([-dl, 0, zs - dl]); n45 = np.array([1, 0, -1.0]) / math.sqrt(2); u = np.array([0, 1.0, 0])
    v = np.cross(n45, u); v /= np.linalg.norm(v)
    S.add([c + dfs/2*math.cos(a)*u + dfs/2*math.sqrt(2)*math.sin(a)*v for a in t], "#7b3fa0", 0.9)
    for az in (math.radians(60), math.radians(180), math.radians(300)):
        rad = np.array([math.cos(az), math.sin(az), 0]); tg = np.array([-math.sin(az), math.cos(az), 0]); e = 3.0
        out = rad * ri
        quad = [out + np.array([0, 0, zo]), out + np.array([0, 0, zo + 50]),
                rad * 14 + np.array([-dl, 0, zs + 56]), rad * 14 + np.array([-dl, 0, zs + 40])]
        for s in (-e, e):
            S.add([q + tg * s for q in quad], "#aab0b8", 0.8)
        S.prism(out - rad * 20 - tg * 12 + np.array([0, 0, zo]), rad * 40, tg * 24, np.array([0, 0, 18]), _ALU)
    S.prism(np.array([-dl - 9, -9, zs + 36]), np.array([18, 0, 0]), np.array([0, 18, 0]), np.array([0, 0, 24]), _ALU)
    S.cyl((-dl, 0, zs + 36), (-dl, 0, zs - dl + 24), 4, "#888", 10)
    # Okularauszug auf Alu-Winkel (Azimut 0°)
    S.prism(np.array([ri + 25, -45, zo]), np.array([20, 0, 0]), np.array([0, 90, 0]), np.array([0, 0, 80]), _ALU)
    S.prism(np.array([ri - 5, -45, zo]), np.array([85, 0, 0]), np.array([0, 90, 0]), np.array([0, 0, 4]), _ALU)
    S.cyl((ri - 30, 0, zs), (ri + 120, 0, zs), 25, _BLK, 24)
    S.cyl((ri + 45, 0, zs), (ri + 100, 0, zs), 40, _RED, 28)
    S.cyl((A - 12, 0, zs), (A + 60, 0, zs), 20, _BLK, 20)
    # Streulichtblende gegenüber dem OAZ: Bogenbreite aus dem Fangspiegel-Rechner, Höhe geschätzt
    fs = fs or {}
    try:
        fr = fangspiegel_rechner(D, f, buet_mm=fs.get("buet", 50.0), tubus_mm=ga["d_innen"], fe_mm=fs.get("fe", 30.0),
                                 afov_deg=fs.get("afov", 70.0))
        bogen = float(fr["d_bogen_pass"])
    except Exception:
        bogen = 1.6 * ro
    half = min(bogen / ro / 2.0, math.radians(100)); hu = 0.74 * D; hd = 0.45 * D
    tt = np.linspace(math.pi - half, math.pi + half, 30); rr = ro + 2
    hh = lambda a: hu * (1 - 0.35 * ((a - math.pi) / half) ** 2)
    hdn = lambda a: hd * (1 - 0.35 * ((a - math.pi) / half) ** 2)
    for s, e in zip(tt[:-1], tt[1:]):
        S.add([(rr*math.cos(s), rr*math.sin(s), zo - hdn(s)), (rr*math.cos(e), rr*math.sin(e), zo - hdn(e)),
               (rr*math.cos(e), rr*math.sin(e), zo + hh(e)), (rr*math.cos(s), rr*math.sin(s), zo + hh(s))], "#c9a46c", 0.6)
    # Lichtkegel
    zz = np.array([0, zs]); R = D / 2 * (1 - zz / f); tq = np.linspace(0, 2 * np.pi, 40)
    kegel = [[(R[0]*math.cos(a), R[0]*math.sin(a), 0), (R[0]*math.cos(b), R[0]*math.sin(b), 0),
              (R[1]*math.cos(b), R[1]*math.sin(b), zs), (R[1]*math.cos(a), R[1]*math.sin(a), zs)] for a, b in zip(tq[:-1], tq[1:])]
    mass = dict(ro=ro, ri=ri, zo=zo, zs=zs, A=A, hu=hu, hd=hd, bogen=bogen, dick=dick)
    return S, kegel, mass


def gitter3d_zeichnen(fig, D, f, ga, g, h, r, ring_b, ring_dick, loch_ring, bauart, fs=None):
    """Zeichnet drei Ansichten (Seite, Seite um 90° gedreht, Draufsicht) in fig."""
    fig.clf()
    S, kegel, m = gitter3d_szene(D, f, ga, g, h, r, ring_b, ring_dick, loch_ring, bauart, fs)
    ro, zo, zs, A = m["ro"], m["zo"], m["zs"], m["A"]
    ext = max(ro, A + 60) * 1.05
    zlo = -m["dick"] - ring_dick - m["hd"] * 0.0; zhi = zo + m["hu"] * 1.05
    ansichten = [("Seitenansicht", 16, -62, (1, 1, 2.1, 1.1), (zlo, zhi)),
                 ("von der Seite (OAZ links)", 16, 28, (1, 1, 2.1, 1.1), (zlo, zhi)),
                 ("Draufsicht", 58, -90, (1, 1, 0.9, 1.4), (0, zo + m["hu"] * 0.9))]
    for i, (titel, el, az, box, zl) in enumerate(ansichten):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(S.polys, facecolors=S.cols, edgecolors=(0, 0, 0, 0.12), linewidths=0.2))
        ax.add_collection3d(Poly3DCollection(kegel, facecolors=[(1, 0.88, 0.3, 0.14)] * len(kegel), edgecolors="none"))
        ax.plot([0, A], [0, 0], [zs, zs], "--", color="#e6a700", lw=1.2); ax.plot([A], [0], [zs], "x", color="r", ms=7)
        ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_zlim(*zl)
        ax.set_box_aspect(box[:3], zoom=box[3]); ax.view_init(elev=el, azim=az)
        ax.set_axis_off(); ax.set_title(titel, fontsize=10)
    fig.suptitle(f"Gittertubus {D/25.4:.0f}\" f/{f/D:.1f} ({D:.0f}/{f:.0f}), Ring {2*m['ri']:.0f}/{2*m['ro']:.0f} mm, "
                 f"Stangen {g['zuschnitt']:.0f} mm – maßstäblich; Spinne, Blendenhöhe, OAZ-Winkel geschätzt", fontsize=10)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=-0.12)
    return fig
