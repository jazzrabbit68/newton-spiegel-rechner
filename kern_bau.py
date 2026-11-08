"""Kern: Bauen – Fangspiegel (Größe, Ausleuchtung, Blende), Tubus aus Okular, Gitterrohr, Pfeiltiefe.
(v8.0.0: aus newton_spiegel_rechner.py v7.11.0 ausgegliedert, Code unverändert)"""
import math
import numpy as np
from kern_optik import _Wb_von_S, csf, mtf_para, mtf_para_obstr, mtf_sph_rel




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


def _kreis_overlap(R1: float, R2: float, d: float) -> float:
    """Überlappfläche zweier Kreise (Radien R1, R2, Mittenabstand d)."""
    if d >= R1 + R2:
        return 0.0
    if d <= abs(R1 - R2):
        return math.pi * min(R1, R2) ** 2
    a1 = math.acos((d * d + R1 * R1 - R2 * R2) / (2 * d * R1))
    a2 = math.acos((d * d + R2 * R2 - R1 * R1) / (2 * d * R2))
    return (R1 * R1 * (a1 - math.sin(2 * a1) / 2) + R2 * R2 * (a2 - math.sin(2 * a2) / 2))


def tubus_aus_okular(D_mm: float, f_mm: float, buet_mm: float = 50.0, fe_mm: float = 30.0,
                     afov_deg: float = 70.0, ueberstand_mm: float = 0.0) -> dict:
    """Erforderlicher Tubus-Innendurchmesser aus den Daten des Übersichtsokulars (v7.6.1/7.6.2).

    Gittertubus: Wandstärke 0, Innen-Ø = Außen-Ø (Hutring-Innenkante).
    Das Okular sieht bis zum Feldblendenrand B = fe·AFOV/57,3 [mm]; das entspricht dem
    halben wahren Feld θ = atan(B/2f).  Das Strahlenbündel (Ø D) eines Feldrandpunkts
    ist in der Höhe z über dem Hauptspiegel um z·tanθ zur Achse versetzt.  Am
    vorderen Tubusende (z = f − A + Überstand, A = Tubusradius + Okularauszug) muss der
    Tubus das Bündel bei voller Ausleuchtung durchlassen:
        d_innen = D + 2·z·tanθ.
    A hängt vom Tubus ab -> Iteration.  Rückgabe: d_innen, B, wahres Feld [°], z, Versatz."""
    B = fe_mm * afov_deg / 57.2958
    th = math.atan(B / 2.0 / f_mm)
    ri = D_mm / 2.0
    for _ in range(60):
        A = ri + max(buet_mm, 0.0)
        z = max(f_mm - A + ueberstand_mm, 0.0)
        s = z * math.tan(th)
        neu = D_mm / 2.0 + s
        if abs(neu - ri) < 1e-4:
            ri = neu
            break
        ri = neu
    return dict(B=B, feld_deg=2.0 * math.degrees(th), z=z, versatz=s, d_innen=2.0 * ri)


def fangspiegel_ausleuchtung(r_mm: float, D_mm: float, f_mm: float, A_mm: float, d_s_mm: float) -> float:
    """Relative Helligkeit (0..1) eines Feldpunkts im Abstand r von der Achse in der Fokalebene (v7.7.1).

    Der Lichtkegel (Radius R_K = D·A/(2f) am Fangspiegel) ist gegenüber dem Fangspiegel
    (Radius R_s = d_s/2, kleine Achse) um s = r·(1 − A/f) versetzt; die Helligkeit ist der
    Flächenanteil des Kegels, der noch auf den Fangspiegel fällt (Überlapp zweier Kreise).
    A = Abstand Fangspiegelmitte–Fokalebene.  Nur Haupt- und Fangspiegel; Okularauszug,
    Rohrwand und Blenden sowie die Kippung der Fokalebene sind nicht berücksichtigt."""
    Rk = D_mm * A_mm / (2.0 * f_mm)
    Rs = d_s_mm / 2.0
    if Rk <= 0.0:
        return 1.0
    s = abs(r_mm) * (1.0 - A_mm / f_mm)
    if s <= Rs - Rk:
        return 1.0
    if s >= Rs + Rk:
        return 0.0
    return min(1.0, _kreis_overlap(Rk, Rs, s) / (math.pi * Rk * Rk))


def fangspiegel_feld(p: float, D_mm: float, f_mm: float, A_mm: float, d_s_mm: float) -> float:
    """Durchmesser des Bildfelds [mm], bis zu dem die Helligkeit mindestens p (0..1) beträgt."""
    lo, hi = 0.0, 4.0 * d_s_mm
    if fangspiegel_ausleuchtung(hi, D_mm, f_mm, A_mm, d_s_mm) >= p:
        return 2.0 * hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fangspiegel_ausleuchtung(mid, D_mm, f_mm, A_mm, d_s_mm) >= p:
            lo = mid
        else:
            hi = mid
    return 2.0 * lo


def fangspiegel_d_fuer_rand(p: float, r_rand_mm: float, D_mm: float, f_mm: float, A_mm: float) -> float:
    """Kleinster Fangspiegel [mm, kleine Achse], der am Feldradius r_rand mindestens die Helligkeit p liefert."""
    lo, hi = 1.0, 2.0 * D_mm
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fangspiegel_ausleuchtung(r_rand_mm, D_mm, f_mm, A_mm, mid) >= p:
            hi = mid
        else:
            lo = mid
    return hi


def fangspiegel_rechner(D_mm: float, f_mm: float, buet_mm: float = 50.0,
                        tubus_mm: float = 0.0, bild_mm: float = 10.0,
                        S: float = 1.0, V: float = 150.0,
                        lam_nm: float = 550.0, hut_mm: float = 0.0,
                        fe_mm: float = 30.0, afov_deg: float = 70.0,
                        oaz_d_mm: float = 50.0, oaz_l_mm: float = 80.0) -> dict:
    """Fangspiegel-Rechner (neu in v6.12.0).

    Minimaler Fangspiegel-Durchmesser (Strahlkegel-Geometrie, identisch zur
    Katalog-Formel  DF = A/f# + B  mit A = TAR + BüT):

        d_min = (TAR + BüT) · D/f + B

    TAR   Tubus-Außenradius  (tubus_mm/2; 0 = Innen-Ø aus dem Übersichtsokular, tubus_aus_okular)
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
    R_hut = hut_mm/2 (0 = Tubusradius, bei Tubus 0 also der Innen-Ø aus dem Übersichtsokular).

    Ausleuchtung (v7.7.1): rand = Helligkeit am Rand der Okular-Feldblende mit d_pass, feld100/feld75 =
    Felddurchmesser mit 100 %/75 %, d_rand90/d_rand75 = Fangspiegel für 90 %/75 % am Rand
    (fangspiegel_ausleuchtung: Kreisüberlapp Lichtkegel ∩ Fangspiegel).

    Rückgabe (dict): A, d_min, d_pass (aufgerundet auf ganze mm), eps
    (=d/D), flaeche (Flächenverlust [%], ε²), D_kontrast (lineare
    Näherung D−d, Zmek/ICS), Q0 (Q_vis ohne Fangspiegel), Q (mit Fangspiegel
    d_pass, bei V), Q_verlust (Q0−Q).
    """
    tub = tubus_aus_okular(D_mm, f_mm, buet_mm, fe_mm, afov_deg)
    tubus_mm = tubus_mm if tubus_mm > 0 else tub['d_innen']   # 0 = Innen-Ø aus Übersichtsokular (v7.6.2)
    tar = tubus_mm / 2.0
    A   = tar + max(buet_mm, 0.0)
    d   = A * D_mm / f_mm + max(bild_mm, 0.0)
    dp  = float(math.ceil(d - 1e-9))
    eps = min(dp / D_mm, 0.95)
    q0  = float(qvis_kurve_mit_fangspiegel(D_mm, S, 0.0, lam_nm, V)[0])
    q1  = float(qvis_kurve_mit_fangspiegel(D_mm, S, dp,  lam_nm, V)[0])
    r_hut = (hut_mm if hut_mm > 0 else tubus_mm) / 2.0
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
                tub=tub,
                rand=fangspiegel_ausleuchtung(b_ok / 2.0, D_mm, f_mm, A, dp),
                feld100=fangspiegel_feld(0.9999, D_mm, f_mm, A, dp),
                feld75=fangspiegel_feld(0.75, D_mm, f_mm, A, dp),
                d_rand90=fangspiegel_d_fuer_rand(0.90, b_ok / 2.0, D_mm, f_mm, A),
                d_rand75=fangspiegel_d_fuer_rand(0.75, b_ok / 2.0, D_mm, f_mm, A),
                d_blende_pass=float(math.ceil(d_bl - 1e-9)),
                d_bogen_pass=float(math.ceil(d_bogen - 1e-9)))


def pfeiltiefe(D_mm: float, f_mm: float) -> dict:
    """Pfeiltiefe (v7.10.0): Tiefe der Spiegelmitte gegenüber dem Rand.
    Kugel (R = 2f):  s = R − √(R² − (D/2)²);   Paraboloid:  s = D²/(16 f).
    Rückgabe: R, s_kugel, s_parabel, diff (Paraboloid − Kugel, µm: der Unterschied, den die Parabolisierung abträgt)."""
    R = 2.0 * f_mm
    s_k = R - math.sqrt(max(R * R - (D_mm / 2.0) ** 2, 0.0))
    s_p = D_mm * D_mm / (16.0 * f_mm)
    return dict(R=R, s_kugel=s_k, s_parabel=s_p, diff_um=(s_k - s_p) * 1000.0)


def gitterrohr_stangen(h_mm: float, r_mm: float, n: int = 6, k: int = 1,
                       zugabe_mm: float = 0.0, kreuz: bool = True, bauart: str = "frei") -> dict:
    """Stangenlänge eines Gitterrohrs (v7.8.0).

    Zwei Ringe im Abstand h, je n Anschlusspunkte gleichmäßig auf dem Teilkreis (Radius r).
    Jede Stange verbindet einen Punkt des unteren Rings mit dem um k Teilungen versetzten
    Punkt des oberen Rings (Versatzwinkel Δφ = k·360°/n).  Die Stange ist die Hypotenuse aus
    der Höhe h und der Sehne s = 2·r·sin(Δφ/2):

        x = √(h² + (2·r·sin(Δφ/2))²)      (Bohrungsmitte – Bohrungsmitte)

    kreuz = True: zusätzlich die gegenläufigen Stangen (Versatz −k), sie kreuzen die ersten
    in halber Höhe.  Zuschnitt = x + 2·Zugabe (Zugabe negativ = Abzug, z. B. −25 für Gabelköpfe auf der Stange).
    bauart "zigzag" (v7.9.1): n Ankerpunkte je Ring, der obere Ring ist um 180°/n gegen den unteren verdreht, jeder
    Ankerpunkt trägt zwei Stangenenden -> 2n Stangen, Δφ = 180°/n (n = 3: Δφ = 60°, x = √(h² + r²)).
    bauart "triplex": n (gerade) Stangen, Paare benachbarter Punkte kreuzen sich in einer Ebene (k = 1).
    k = 0: senkrechte Stangen (x = h).
    Rückgabe: x, zuschnitt, dphi, sehne, winkel (zur Senkrechten), anzahl, r_mitte (kleinster
    Abstand einer Stange von der Achse, in halber Höhe), kreuzt (bool)."""
    triplex = (bauart == "triplex")   # v7.9.0: n Stangen, benachbarte Paare kreuzen sich in einer Ebene
    zigzag = (bauart == "zigzag")     # v7.9.1: n Ankerpunkte je Ring, oberer Ring um 180°/n verdreht
    n = max(int(n), 3)
    if triplex:
        n += n % 2
        k = 1
    k = int(k) % n
    dphi = (math.pi / n) if zigzag else 2.0 * math.pi * k / n
    sehne = 2.0 * r_mm * math.sin(dphi / 2.0)
    x = math.sqrt(h_mm * h_mm + sehne * sehne)
    gegenlaeufig = bool(kreuz) and k != 0 and 2 * k != n
    if triplex:
        gegenlaeufig = True
    if zigzag:
        gegenlaeufig = False
    return dict(x=x, zuschnitt=x + 2.0 * zugabe_mm, dphi=math.degrees(dphi), sehne=sehne,
                winkel=math.degrees(math.atan2(sehne, h_mm)),
                anzahl=(n if triplex else 2 * n if zigzag else n * (2 if gegenlaeufig else 1)), r_mitte=r_mm * abs(math.cos(dphi / 2.0)),
                kreuzt=gegenlaeufig, n=n, k=k, bauart="triplex" if triplex else ("zigzag" if zigzag else "frei"))


def gitterrohr_freiraum(h_mm: float, r_mm: float, dphi_deg: float, D_mm: float, f_mm: float,
                        feld_halb_deg: float = 0.0, z0_mm: float = 0.0) -> dict:
    """Abstand der Stangen vom Lichtkegel (v7.8.0; z0 = Höhe der unteren Lochebene über dem Spiegel, v7.9.0).

    Die Stange läuft von (r, φ0, z=0) nach (r, φ0+Δφ, z=h) geradlinig; ihr Achsabstand ist
    ρ(t) = r·√((1−t)² + t² + 2t(1−t)·cosΔφ).  Der Lichtkegel eines Feldrandpunkts hat in der Höhe
    z über dem Hauptspiegel (unterer Ring = Spiegelebene) den Radius R(z) = D/2·(1 − z/f) + z·tanθ
    (Bündelversatz durch das Feld).  Rückgabe: Mindestabstand ρ − R [mm] und die Höhe, an der er auftritt."""
    dphi = math.radians(dphi_deg)
    tanth = math.tan(math.radians(feld_halb_deg))
    best, zbest = 1e9, 0.0
    for i in range(0, 201):
        t = i / 200.0
        rho = r_mm * math.sqrt(max((1 - t) ** 2 + t * t + 2 * t * (1 - t) * math.cos(dphi), 0.0))
        z = z0_mm + t * h_mm
        R = D_mm / 2.0 * (1.0 - z / f_mm) + z * tanth
        if rho - R < best:
            best, zbest = rho - R, z
    return dict(abstand=best, z=zbest)


def gitterrohr_auto(D_mm: float, f_mm: float, buet_mm: float = 50.0, fe_mm: float = 30.0,
                    afov_deg: float = 70.0, ring_breite_mm: float = 50.0,
                    achse_mm: float = 30.0, ring_dicke_mm: float = 15.0,
                    loch_ring_mm: float = 25.0, d_innen_mm: float = 0.0,
                    spiegel_tief_mm: float = 0.0) -> dict:
    """Vorgaben aus dem Fangspiegel-Rechner (v7.8.1, erweitert v7.9.0):
    Ring-Innen-Ø = Tubus-Ø aus dem Okular (Gittertubus), Teilkreis der Befestigungspunkte =
    Innen-Ø + Ringbreite (Loch in Ringmitte), Höhe zwischen den Lochebenen
    h = f − A − Achshöhe: Spiegeloberfläche bündig mit der Oberkante des unteren Rings (z = 0),
    Fangspiegelmitte (= optische Achse des Okularauszugs) liegt 'achse_mm' über der Oberkante des
    Hutrings; Ringe gleich dick.
    v7.9.0: Die Lochmitte der Gegenstücke liegt 'loch_ring_mm' von der Ringfläche entfernt (innen, zwischen
    den Ringen) -> Lochabstand h = (f − A − Achshöhe) − Ringdicke − 2·loch_ring; untere Lochebene z0 = loch_ring.
    v8.1.0: 'spiegel_tief_mm' = Spiegeloberfläche liegt so viel UNTER der Oberkante des unteren Rings; die Fangspiegelmitte
    liegt dann entsprechend näher am unteren Ring (h und Stangenlänge kleiner); z0 (Lochebene über dem Spiegel) = loch_ring + tief."""
    tub = tubus_aus_okular(D_mm, f_mm, buet_mm, fe_mm, afov_deg)
    di = d_innen_mm if d_innen_mm > 0 else tub["d_innen"]      # v7.11.0: Innen-Ø aus dem Fangspiegel-Rechner, überschreibbar
    A = di / 2.0 + max(buet_mm, 0.0)
    return dict(d_innen=di, d_innen_auto=tub["d_innen"], d_teil=di + ring_breite_mm,
                d_fang=fangspiegel_d(D_mm, f_mm, buet_mm, di, 10.0, fe_mm, afov_deg), z_sec=f_mm - A, h=f_mm - A - achse_mm - ring_dicke_mm - 2.0 * loch_ring_mm - spiegel_tief_mm,
                A=A, feld_halb=tub["feld_deg"] / 2.0, z0=loch_ring_mm + spiegel_tief_mm, tief=spiegel_tief_mm,
                h_ring=f_mm - A - achse_mm - ring_dicke_mm - spiegel_tief_mm)


def fangspiegel_versatz(A_mm: float, D_mm: float, f_mm: float) -> dict:
    """Versatz des Fangspiegels (v8.1.1): Der Lichtkegel trifft die 45°-Fläche unten breiter als oben; die Spiegelmitte
    muss daher von der Schnittstelle der optischen Achse aus um 'quer' vom Okularauszug WEG und um denselben Betrag
    Richtung Hauptspiegel verschoben werden.  T = tan(Halbwinkel) = D/(2f);  quer = A·T²/(1 − T²)  (A = Fangspiegel–Fokus).
    Rückgabe: quer (mm, in x vom OAZ weg), tiefer (gleicher Wert in z), T."""
    T = (D_mm / 2.0) / f_mm
    q = A_mm * T * T / (1.0 - T * T)
    return dict(quer=q, tiefer=q, T=T)


def gitterrohr_text(D, f, g, fr, h, d_teil) -> str:
    return (
        f"Gitterrohr ({dict(triplex='TripleX', zigzag='Zigzag').get(g.get('bauart'), 'frei')}): Lochabstand h = {h:.0f} mm, Teilkreis-Ø {d_teil:.0f} mm, {g['n']} Punkte je Ring, "
        f"" + (f"oberer Ring um {g['dphi']:.1f}° verdreht" if g.get('bauart') == 'zigzag' else f"Versatz {g['k']} Teilung(en) = {g['dphi']:.1f}°") + f"\n"
        f"Sehne zwischen den Anschlusspunkten: {g['sehne']:.1f} mm   Stange schräg: {g['winkel']:.1f}° zur Senkrechten\n"
        f"Stangenlänge (Bohrungsmitte–Mitte): {g['x']:.1f} mm   →  Zuschnitt {g['zuschnitt']:.1f} mm   "
        f"Anzahl: {g['anzahl']}" + ((" (gekreuzte Paare, Kreuzung in halber Höhe)" if g.get('bauart') == 'triplex' else " (mit gegenläufigen, Kreuzung in halber Höhe)") if g['kreuzt'] else "") + "\n"
        f"Kleinster Achsabstand einer Stange: {g['r_mitte']:.0f} mm; Freiraum zum Lichtkegel (Feldrand): "
        f"{fr['abstand']:.0f} mm bei z = {fr['z']:.0f} mm"
        + ("  ⚠ Stange schneidet den Lichtkegel!" if fr['abstand'] < 0 else ""))


def plt_Rectangle(*a, **k):
    from matplotlib.patches import Rectangle
    return Rectangle(*a, **k)


def gitterrohr_zeichnen(ax1, ax2, D, f, g, h, r, fr, z0=0.0, ring=None, kon=None):
    """Seitenansicht (links) und Draufsicht (rechts) des Gitterrohrs mit Lichtkegel (v7.8.0)."""
    n, k = g["n"], g["k"]
    ph = [2 * math.pi * i / n for i in range(n)]
    _tief = (kon or {}).get("tief", 0.0); _lr = z0 - _tief   # z0 = Lochebene über dem Spiegel, _lr = Lochmitte bis Ringfläche
    ax1.clear(); ax2.clear()
    tx = g.get("bauart") == "triplex"
    zg = g.get("bauart") == "zigzag"
    dphi_r = math.radians(g["dphi"])
    for i in range(n):
        for sgn in (((1 if i % 2 == 0 else -1),) if tx else ((1, -1) if (g["kreuzt"] or zg) else (1,))):
            x0 = r * math.cos(ph[i]); x1 = r * math.cos(ph[i] + sgn * dphi_r)
            ax1.plot([x0, x1], [z0, z0 + h], color="#1565c0", lw=1.2, alpha=0.8)
    ax1.plot([-D / 2, D / 2], [0, 0], color="#444", lw=3)
    zz = np.linspace(0, z0 + h + _lr, 50)
    ax1.fill_betweenx(zz, -(D / 2 * (1 - zz / f)), (D / 2 * (1 - zz / f)), color="#ffe082", alpha=0.6, lw=0)
    ax1.axhline(_tief, color="#888", lw=2); ax1.axhline(z0 + h + _lr, color="#888", lw=2)   # Ring-Innenflächen
    if kon and ring:   # v7.11.0: Ringe, Fangspiegel und Okularauszug andeuten (kon: rt, z_sec, A, d_fang)
        ri, ro = ring; rt = kon["rt"]; zs = kon["z_sec"]; A = kon["A"]; dfs = kon["d_fang"]; zo = z0 + h + _lr   # zo = Unterkante Hutring
        for sg in (-1, 1):
            for z_a in (_tief - rt, zo):
                ax1.add_patch(plt_Rectangle((min(sg * ri, sg * ro), z_a), ro - ri, rt, fc="#cfd4dc", ec="#555", lw=0.8))
        ax1.add_patch(plt_Rectangle((-D / 2, -4), D, 4, fc="#7fb7d6", ec="#444", lw=0.8))                 # Hauptspiegel
        zz2 = np.linspace(0, zs, 50)
        ax1.fill_betweenx(zz2, -(D / 2 * (1 - zz2 / f)), (D / 2 * (1 - zz2 / f)), color="#ffe082", alpha=0.6, lw=0)
        ax1.plot([-dfs * 0.7071, dfs * 0.7071], [zs - dfs * 0.7071, zs + dfs * 0.7071], color="#6a1b9a", lw=3)  # Fangspiegel (45°)
        ax1.plot([0, A], [zs, zs], color="#e6a700", lw=1.2, ls="--")
        ax1.add_patch(plt_Rectangle((ri, zs - 25), A + 40 - ri, 50, fc="#e8d9b0", ec="#555", lw=0.8))      # OAZ-Rohr (2")
        ax1.add_patch(plt_Rectangle((ri, zo + rt), 60, max(zs - 25 - zo - rt, 0), fc="#d8c89a", ec="#555", lw=0.8))   # Sockel
        ax1.plot([A], [zs], marker="x", color="r"); ax1.text(A + 8, zs + 30, "Fokus", color="r", fontsize=7)
        ax1.text(0, zs + dfs * 0.9 + 10, "Fangsp. Ø%.0f" % dfs, ha="center", fontsize=7, color="#6a1b9a")
        ax1.set_xlim(-max(ro, A) - 40, max(ro, A) + 90)
    ax1.set_aspect("equal"); ax1.set_title("Seitenansicht (Lichtkegel gelb)", fontsize=10)
    ax1.set_xlabel("mm"); ax1.set_ylabel("Höhe über Spiegel [mm]")
    th = np.linspace(0, 2 * math.pi, 200)
    if ring:   # v7.9.2: Ring (Innen-/Außenradius) als Kreisring, Ankerpunkte sitzen auf dem Teilkreis in Ringmitte
        ri, ro = ring
        ax2.fill(np.concatenate([ro * np.cos(th), ri * np.cos(th[::-1])]),
                 np.concatenate([ro * np.sin(th), ri * np.sin(th[::-1])]), color="#cfd4dc", ec="#555", lw=1.0,
                 label="Tubusring (Innen-Ø %.0f, Außen-Ø %.0f)" % (2 * ri, 2 * ro))
    ax2.plot(r * np.cos(th), r * np.sin(th), color="#888", lw=1.0, ls=":")
    ax2.plot(D / 2 * np.cos(th), D / 2 * np.sin(th), color="#e6a700", lw=1.5, label="Lichtkegel am Spiegel")
    ax2.plot(g["r_mitte"] * np.cos(th), g["r_mitte"] * np.sin(th), color="#c62828", lw=1, ls="--",
             label="kleinster Stangenabstand")
    for i in range(n):
        for sgn in (((1 if i % 2 == 0 else -1),) if tx else ((1, -1) if (g["kreuzt"] or zg) else (1,))):
            j = ph[i] + sgn * dphi_r
            ax2.plot([r * math.cos(ph[i]), r * math.cos(j)], [r * math.sin(ph[i]), r * math.sin(j)],
                     color="#1565c0", lw=1.0, alpha=0.7)
        ax2.text(1.08 * r * math.cos(ph[i]), 1.08 * r * math.sin(ph[i]), str(i + 1), ha="center", va="center", fontsize=8)
    ax2.set_aspect("equal"); ax2.set_title("Draufsicht (Anker mittig im Ring, Stangen als Sehnen)", fontsize=10)
    ax2.legend(fontsize=7, loc="lower right"); ax2.set_xlabel("mm")


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
        f"Ausleuchtung mit {r['d_pass']:.0f} mm (Okularfeldblende {r['B_ok']:.0f} mm): 100 % bis Ø {r['feld100']:.0f} mm, "
        f"75 % bis Ø {r['feld75']:.0f} mm, am Feldrand {r['rand']*100:.0f} %\n"
        f"   Fangspiegel für 90 % am Feldrand: ≥ {r['d_rand90']:.0f} mm, für 75 %: ≥ {r['d_rand75']:.0f} mm "
        f"(nur Haupt- und Fangspiegel gerechnet)\n"
        f"Blende gegenüber dem OAZ (Hutring-Ø {2*r['R_hut']:.0f} mm, um den Hutring gebogen): "
        f"Fokus–Hutring-Achse {r['x_blende']:.0f} mm, Feld {r['B_ok']:.0f} mm\n"
        f"   Sichtlinie durch das OAZ-Rohr (steilster Winkel {r['w_oaz']:.0f}°); ebene Blende wäre Ø {r['d_geo']:.0f} mm, "
        f"Strahlkegel allein {r['d_kegel']:.0f} mm\n"
        f"   → gebogene Blende: Breite (Sehne) mind. {r['d_blende']:.0f} mm, "
        f"Zuschnitt (Bogenlänge) {r['d_bogen_pass']:.0f} mm"
        + (f"\nTubus aus Übersichtsokular ({r['tub']['B']:.0f} mm Feldblende, wahres Feld {r['tub']['feld_deg']:.2f}°): "
           f"Innen-Ø bei voller Ausleuchtung {r['tub']['d_innen']:.0f} mm (Gittertubus, Wand 0 mm; "
           f"Eingabe Tubus-Ø = 0 verwendet diesen Wert)"
           if r.get("tub") else ""))


def fangspiegel_skizze(ax, D, f, r):
    """Draufsicht (Blick entlang der optischen Achse) auf Hutring, Fangspiegel,
    OAZ-Rohr und die gebogene Blende gegenüber dem OAZ, mit Maßen,
    Lichtkegel (Hauptspiegel → Okular) und der steilsten Sichtlinie durch das
    OAZ-Rohr (v7.0.0, gemeinsam für Desktop und Web).
    Koordinaten: u entlang der OAZ-Achse (Feldblende/Fokus bei u = 0),
    v quer dazu; optische Achse bei u = A."""
    A, R, rf, rt = r["A"], r["R_hut"], r["B_ok"] / 2.0, r["oaz_r"]
    L, dF = r["oaz_l"], r["d_pass"]
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
    from matplotlib.patches import Ellipse
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


def fangspiegel_d(D_mm: float, f_mm: float, buet_mm: float = 50.0,
                  tubus_mm: float = 0.0, bild_mm: float = 10.0,
                  fe_mm: float = 30.0, afov_deg: float = 70.0) -> float:
    """Nur die Geometrie: Fangspiegel-Ø aufgerundet auf ganze mm (v7.0.0).
    tubus_mm = 0: Innen-Ø aus dem Übersichtsokular (v7.6.2, vorher D)."""
    if tubus_mm <= 0:
        tubus_mm = tubus_aus_okular(D_mm, f_mm, buet_mm, fe_mm, afov_deg)["d_innen"]
    d = (tubus_mm / 2.0 + max(buet_mm, 0.0)) * D_mm / f_mm + max(bild_mm, 0.0)
    return float(math.ceil(d - 1e-9))
