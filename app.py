"""
Newton-Teleskop: Kontrast- und Schärfeverlust sphärischer Hauptspiegel
=======================================================================
Streamlit-Version v8.4.0 (2026-11-08)
Entspricht newton_spiegel_rechner.py v8.4.0

Änderungshistorie (Kurzfassung, jüngste zuerst; Einzelheiten in CHANGELOG_vollstaendig.txt und in der Doku):
  v8.5.0 (2026-11-12) NEU Tab Low-Rider (Bauen): Strahl am Fangspiegel um δ nach unten, Okularauszug tiefer; Fangspiegel (kleine/große Achse, Obstruktion) über der Absenkung der Einblickhöhe, Kurvenschar bei gleicher Einblickhöhe, Vergleich größerer Spiegel (Asphärik, Koma) (neues Modul kern_lowrider.py)
  v8.4.0 (2026-11-08) NEU Spiegelzelle: Auflagepunkte 3/6/9/18 (Dreieck, Wippen, Dreiecke, Wippen mit Dreiecken); optimale Lage, Radien, Wippen-/Dreieckslängen aus Dünnplatten-Rechnung (Eigengewicht im Zenit), Durchbiegungskarte, RMS/Strehl, Variantenvergleich (neues Modul kern_zelle.py); Blende-Tab jetzt in der Gruppe Bildqualität; Kennwerte ohne V½; Vergleich übernimmt Okularauszug, Tubus-Ø und Bildfeld aus dem Tab Fangspiegel.
  v8.3.0 (2026-11-08) Wahrnehmung: Näherung Q̃ immer eingezeichnet (Abweichungsfläche nur photopisch), Objektauswahl reduziert auf Mond / Heller Planet / DSO (hell) / Aus / Frei (volle Liste bleibt bei Max. Vergrößerung und Blende); Kennwerte: "Wie gut ist der Spiegel? · Wie weit trägt er?"; Web: Wahrnehmungs-Diagramm größer gezeichnet (12×6,6 Zoll) und alle Diagramme mit 130 dpi, dadurch feiner
  v8.2.1 (2026-11-08) Wahrnehmung: Näherung Q̃ wird bis zur Vergrößerung gezeigt, bis zu der die Netzhautleuchtdichte photopisch bleibt (statt ganz ausgeblendet); bei dunklen Objekten photopische Referenzkurven (exakt + Q̃)
  v8.2.0 (2026-11-08) Gitterrohr: maßstäbliche 3D-Ansicht (neues Modul ansicht3d.py; Seite, gedreht, Draufsicht; OAZ zwischen zwei Spinnenbeinen, Streulichtblende aus Fangspiegel-Rechner), Schalter in Web und Desktop
  v8.1.0 (2026-11-07) Gitterrohr: Eingabe "Spiegel unter Ring-Oberkante" (Spiegelversatz verkürzt h und Stangenlänge); Prüffälle 8" Zigzag und 14" TripleX
  v8.0.0 (2026-11-07) Programm in Module aufgeteilt (kern_optik, kern_foucault, kern_bau, kern_vergleich, diagramme, texte, konstanten); Rechenwerte und Diagramme unverändert (Regressionstest)
  v7.11.0 (2026-11-07) Gitterrohr: Ring-Innen-Ø aus dem Fangspiegel-Rechner, überschreibbar; Achse 40 mm über dem Hutring; Seitenansicht mit Ringen, Fangspiegel und Okularauszug
  v7.10.0 (2026-11-07) Kennwerte: Pfeiltiefe (Mitte–Rand, Kugel R = 2f und Paraboloid) im Steckbrief
  v7.9.2 (2026-11-07) Gitterrohr: Draufsicht zeigt den Tubusring als Kreisring, Ankerpunkte mittig im Ring
  v7.9.1 (2026-11-07) Gitterrohr Zigzag korrigiert: 3 Ankerpunkte je Ring (zwei Stangenenden je Punkt), oberer Ring um 60° verdreht, 6 Stangen
  v7.9.0 (2026-11-07) Gitterrohr: Bauart Zigzag/TripleX; Gabelkopf auf der Stange (Abzug je Ende statt Zugabe), Lochmitte 25 mm von der Ringfläche, Ringdicke
  v7.8.1 (2026-11-07) Gitterrohr: Konstruktionsdaten ergänzt (Ringbreite, Achshöhe über dem Hutring, Gabelloch-Abstand 25 mm als Zugabe je Ende); Teilkreis = Innen-Ø + Ringbreite, h = f − A − Achshöhe
  v7.8.0 (2026-11-07) Neuer Tab Gitterrohr (Bauen): Stangenlänge aus Höhe, Teilkreis, Punktzahl und Versatz; Freiraum zum Lichtkegel; Q_vis-Kern unverändert
  v7.7.1 (2026-11-06) Fangspiegel-Tab: Ausleuchtung des Okularfeldes (100-%-Feld, Helligkeit am Feldrand, Fangspiegel für 90/75 % am Rand); Q_vis-Kern unverändert
  v7.7  (2026-11-05)  Tab Kennwerte als Steckbrief (Fläche · Reichweite · Bildwirkung, Q_vis photopisch bei 30/60/100/150×), übrige Werte unter "Details"
  v7.6  (2026-11-05)  Netzhauthelligkeit (Étendue) geht in Q_vis ein; Fangspiegel-Tab: Tubus-Ø aus dem Übersichtsokular (Gittertubus); Antoniadi-Hinweis beim Seeing
  v7.5  (2026-11-03)  Ein Feld "Objekt" setzt Leuchtdichte und Augenpupille gemeinsam; Objekthelligkeiten korrigiert
  v7.4  (2026-11-03)  Q_vis berücksichtigt die Bildhelligkeit (leuchtdichteabhängige CSF)
  v7.3  (2026-10-31)  Fehler im Kern behoben (externe Prüfung)
  v7.2  (2026-10-31)  Q_vis-Kurven mit Augenpupille; Oberfläche kompakter
  v7.1  (2026-10-30)  Tabs in Gruppen geordnet
  v7.0  (2026-10-29)  Vergleich-Tab (Q_abs), exakte 2D-MTF, Fangspiegel-Blende
  v6.x  (2026-10-25)  Ablese-Cursor, Fangspiegel-Tab, Foucault-Korrekturen

Rechenkern und Diagramm-Methoden liegen in den gemeinsamen Modulen (kern_*.py, diagramme.py, texte.py,
konstanten.py); diese Datei enthält nur die Streamlit-Oberfläche. Start: streamlit run app.py

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


# ══════════════════════════════════════════════════════════════════════════════
# Diagramm-Schicht: 1:1 aus dem Desktop-Programm übernommen
# ══════════════════════════════════════════════════════════════════════════════
# Die Diagramm-Methoden unten sind wörtliche Kopien der _diagramm_*-Methoden
# aus newton_spiegel_rechner.py (v7.0.0-beta). Dadurch sehen Web- und Desktop-
# Version identisch aus; bei Änderungen am Desktop einfach erneut kopieren.
# Nur die Tk-Variablen (StringVar/DoubleVar) werden durch _Var ersetzt.

import types as _types
from diagramme import DiagrammMethoden
from kern_bau import (fangspiegel_rechner, fangspiegel_skizze, fangspiegel_text,
    fangspiegel_zeichnen, gitterrohr_auto, gitterrohr_freiraum, gitterrohr_stangen, gitterrohr_text,
    gitterrohr_zeichnen, pfeiltiefe, tubus_aus_okular)
from ansicht3d import gitter3d_zeichnen
from kern_lowrider import ANSICHTEN as LR_ANSICHTEN, lowrider_text, lowrider_zeichnen
from kern_zelle import MATERIALIEN, OPT, OPT_OHNE, zelle_rechner, zelle_text, zelle_zeichnen
from kern_foucault import (foucault_kennzahlen, foucault_wellenfront, foucault_zonen,
    foucault_zonengrenzen)
from kern_optik import (ANTONIADI_HINWEIS, OBJEKT_HELLIGKEITEN, OBJEKT_PUPILLE, WAHRNEHMUNG_OBJEKTE, _Wb_schieber,
    _qvis_schieber, _strehl_exakt, _wellenfronten, berechne, berechne_vergr, beurteilung,
    d_fang_min, perceived_quality_exakt, v_halb_exakt, v_krit_aus_qvis, v_krit_aus_qvis_von_Wb,
    v_krit_blur_direkt, v_krit_naeh, v_kritisch, v_kritisch_strehl)
from kern_vergleich import vergleich_achsen, vergleich_zeichnen
from konstanten import BG, EYE_RES, VERSION


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
class _Diagramme(DiagrammMethoden):
    """Zustand (Optionen) als _Var-Attribute; die Zeichenmethoden stammen aus diagramme.DiagrammMethoden."""

    def __init__(self):
        self._mtf_objekte = _Var("planeten")
        self._mtf_fokus   = _Var("bestfocus")
        self._vmax_objekt     = _Var(OBJEKT_HELLIGKEITEN[1][0])
        self._vmax_L0         = _Var(600.0)
        self._vmax_seeing     = _Var(1.5)
        self._vmax_D_auge     = _Var(6.0)
        self._vmax_L_schwelle = _Var(1.0)
        self._vmaxD_seeings   = _Var("0.4, 0.8, 1.5, 2.5")
        self._vmaxD_Dmin      = _Var(50.0)
        self._vmaxD_Dmax      = _Var(500.0)
        self._foucault_letzte_kz = None
        self._foucault_letzte_D  = None
        self._wn_V     = None    # Ablese-Cursor (Wahrnehmung)
        self._wn_fang  = 0.0     # Fangspiegel-Ø [mm], 0 = aus


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
             kz_fc=None, D_fc=None, alt1d=False):
    fig, ax = _neue_figur((8, 4.5))
    d = _Diagramme(); d._tab_mtf = (fig, ax, _FakeCanvas())
    d._mtf_objekte = _Var(objekte)
    d._mtf_fokus   = _Var("paraxial" if paraxial else "bestfocus")
    d._mtf_alt1d   = _Var(bool(alt1d))
    d._foucault_letzte_kz = kz_fc       # Vergleichskurve aus dem Foucault-Tab
    d._foucault_letzte_D  = D_fc
    d._diagramm_mtf(D, f, lam, S_slide)
    return fig


def plot_wahrnehmung(D, f, lam, S_slide, S_real, V_cur=None, d_fang=0.0, d_pup=0.0, L_obj=0.0):
    fig, ax = _neue_figur((12, 6.6))
    d = _Diagramme(); d._tab_wahrnehmung = (fig, ax, _FakeCanvas())
    d._wn_V = V_cur
    d._wn_fang = d_fang
    d._wn_pup = d_pup
    d._wn_L = L_obj          # v7.4.0: Leuchtdichte [cd/m²], 0 = photopisch
    d._diagramm_wahrnehmung(D, f, lam, S_slide=S_slide, S_real=S_real)
    return fig


def plot_blende(D, f, lam, D_blend):
    D_bl = min(D_blend, D - 10)
    fig, (ax, ax2) = _neue_figur((13, 5.2), ncols=2)   # v7.3.1: nebeneinander
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

    D   = st.sidebar.slider("Öffnung D [mm]",    min_value=50,  max_value=500,  value=200, step=5)
    f   = st.sidebar.slider("Brennweite f [mm]", min_value=200, max_value=3000, value=1000, step=50)
    lam = 550.0  # fix, λ=550nm

    N = f / D
    st.sidebar.markdown(f"**f/D = {N:.2f}**")

    S_pct = st.sidebar.slider(
        "Strehl [%]", 0, 100, 0,
        help="0 = reale Kugel; alles darüber ist der Ziel-Strehl in Prozent (simuliert "
             "Figurierungsfortschritt), 100 = Paraboloid. Liegt der Wert unter dem Strehl der Kugel, "
             "gilt die Kugel.")

    st.sidebar.markdown("---")
    _NAV = [("Grundlagen", [("grund", "Diagramme"), ("kenn", "Kennwerte"), ("grenzen", "Grenzen (max. Vergr.)")]),
            ("Bildqualität", [("wahr", "Wahrnehmung"), ("mtf", "MTF"), ("blende", "Blende"), ("vergleich", "Vergleich (Beta)")]),
            ("Bauen", [("messen", "Foucault-Test"), ("bauen", "Fangspiegel"), ("gitter", "Gitterrohr"), ("zelle", "Spiegelzelle"), ("lowrider", "Low-Rider")]),
            ("", [("doku", "ℹ️ Dokumentation")])]
    if "nav" not in st.session_state:
        st.session_state["nav"] = "grund"

    def _set_nav(k):
        st.session_state["nav"] = k

    for _gname, _items in _NAV:
        if _gname:
            st.sidebar.caption(_gname)
        else:
            st.sidebar.divider()          # v7.3.1: Dokumentation deutlich von den Gruppen absetzen
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
    _Q_V = (30.0, 60.0, 100.0, 150.0)
    Qp_30, Qp_60, Qp_100, Qp_150 = (perceived_quality_exakt(D, f, lam, v) for v in _Q_V)
    Qs_30, Qs_60, Qs_100, Qs_150 = (_qvis_schieber(D, f, lam, S_slide, v) for v in _Q_V)

    verdict_text, verdict_color = beurteilung(S_slide)
    if S_slide > S_real + 1e-6:
        verdict_text = f"Mit Verbesserung (S={S_slide:.2f}):  {verdict_text}"

    # ── Kennzahlen-Übersicht (kompakt) ──────────────────────────────────────
    def _qc(q):
        return "#2e7d32" if q >= 0.90 else "#e65100" if q >= 0.70 else "#c62828"

    _Vcap = max(2.0 * D, 60.0)               # v7.3.1: Obergrenze 2·D
    def _vk_ex_text(v):
        if v >= _Vcap - 0.5:
            return f"> {_Vcap:.0f}x"
        return f"{v:.0f}x" if v > 30.5 else "< 30x"

    delta_s    = S_real - S_mar
    avp        = r_real["aufl_verlust_pct"]
    alpha_blur = D**3 / (128 * f**2) / f * 206265

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
    rows += _row("V_krit N&#228;herung D/(13&#183;Wb)",
                 f"{Vk_naeh:.0f}x &#8594; {Vk_naeh_s:.0f}x", "#534AB7")
    rows += _row("V_krit Blur (geom.)",
                 f"{Vk_blur:.0f}x  &#945;_blur={alpha_blur:.1f}\"" if Vk_blur > 1 else "< 1x - Blur dominiert immer")
    rows += _row("V_krit MTF&#215;CSF >10%",
                 f"{_vk_ex_text(Vk_exakt)} &#8594; {_vk_ex_text(Vk_exakt_s)}")
    rows += _row("Aufl.verlust",         f"{avp:.1f}%  (Blur {r_real['d_blur_as']:.1f}\")")
    rows += _row("Verlust bei 150x",     f"{rv['verlust']:.1f} mm")
    rows += _row("V_krit Paraboloid / Vmax",
                 f"{'&#8734;' if Vk['Vk_para'] >= 1e8 else format(Vk['Vk_para'], '.0f') + 'x'}  | Vmax: {Vk['Vmax']:.0f}x")

    _kenn_html = f"<table style='border-collapse:collapse;width:100%;margin-bottom:4px'>{rows}</table>"

    # v7.2.4: Kernwerte, Beurteilung und Tabelle erscheinen nur in der Ansicht "Kennwerte".
    if _nav == "kenn":
        st.markdown(_verdict_html, unsafe_allow_html=True)
        _gap = S_slide > S_real + 1e-6
        # Steckbrief: Wie gut ist der Spiegel? · Wie weit trägt er? · Wie wirkt er im Bild?
        st.markdown("**Wie gut ist der Spiegel?**")
        _k1, _k2, _k5 = st.columns(3)
        _k1.metric("Strehl exakt", f"{S_slide:.3f}",
                   delta=f"Sphäre {S_real:.3f}" if _gap else None, delta_color="off")
        _k2.metric("W_PtV best focus", f"{r_real['Wb']:.3f} λ")
        _pf = pfeiltiefe(D, f)
        _k5.metric("Pfeiltiefe (Mitte–Rand)", f"{_pf['s_kugel']:.3f} mm",
                   delta=f"Paraboloid {_pf['s_parabel']:.3f} mm", delta_color="off",
                   help=f"Kugel R = 2f = {_pf['R']:.0f} mm: s = R − √(R² − (D/2)²). Paraboloid: s = D²/(16 f). "
                        f"Unterschied {_pf['diff_um']:.1f} µm.")
        st.markdown("**Wie weit trägt er?**")
        _k3, = st.columns(1)
        _k3.metric("V_krit (MTF×CSF >10 %)", _vk_ex_text(Vk_exakt_s),
                   delta=f"Sphäre {_vk_ex_text(Vk_exakt)}" if _gap else None, delta_color="off")
        st.markdown("**Wie wirkt er im Bild?**  Q_vis, Paraboloid = 1")
        _qk = st.columns(4)
        for _c, _v, _qs, _qp in zip(_qk, _Q_V, (Qs_30, Qs_60, Qs_100, Qs_150), (Qp_30, Qp_60, Qp_100, Qp_150)):
            _c.metric(f"bei {_v:.0f}×", f"{_qs:.3f}",
                      delta=f"Sphäre {_qp:.3f}" if _gap else None, delta_color="off")
        st.caption("Q_vis hier: photopisch, volle Öffnung, ohne Seeing und Fangspiegel, bezogen auf das Paraboloid. Das ist die strengste Bewertung der Spiegelfläche; an dunklen Objekten und mit Augenpupille sieht der Spiegel besser aus (Tab Wahrnehmung).")
        with st.expander("Details"):
            st.markdown(_kenn_html, unsafe_allow_html=True)

    # ── Lazy Tab-Rendering: Figure als PNG-Bytes im session_state ─────────────
    # Figures werden nur neu gezeichnet wenn sich die relevanten Parameter
    # geändert haben. Matplotlib-Figures sind nicht pickle-bar → PNG-Bytes.
    import io as _io

    def _fig_to_png(fig):
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
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
        alt1d = st.checkbox("alte 1D-Näherung zusätzlich zeigen (nur Vergleich)",
                            key="mtf_alt1d",
                            help="Zeigt die frühere eindimensionale MTF-Rechnung gestrichelt. "
                                 "Sie weicht bei starker Aberration (kleiner Strehl) deutlich von der "
                                 "exakten 2D-Rechnung ab.")
        paraxial  = "Paraxial" in fokus_mode
        objekte_k = {"Planeten": "planeten", "Deep-Sky (M42)": "deepsky", "Alle": "alle"}[objekte_mode]
        # Vergleichskurve: zuletzt im Foucault-Tab gemessene Wellenfront (falls
        # für diese Öffnung vorhanden), sonst Strehl-Schieber — wie im Desktop.
        _kz_fc = st.session_state.get("fc_last_kz")
        _D_fc  = st.session_state.get("fc_last_D")
        _fc_key = (round(float(_kz_fc["S_exakt"]), 4), _D_fc) if _kz_fc is not None else None
        _show_cached("fig_mtf", lambda: plot_mtf(D, f, lam, S_slide,
                     paraxial=paraxial, objekte=objekte_k,
                     kz_fc=_kz_fc, D_fc=_D_fc, alt1d=alt1d),
                     _h + (paraxial, objekte_k, _fc_key, alt1d))

    if _nav == "wahr":
        # ── Ablesen: exakte Q_vis-Werte bei gewählter Vergrößerung ───────────
        # Spiegel = Regler in der Seitenleiste (Öffnung, Brennweite, Strehl-Schieber)
        wc1, wc2, wc3 = st.columns([3, 1.3, 1.9])
        _opt_aus  = "Aus (photopisch, ohne Abblenden)"
        _opt_frei = "Frei (L und d_p selbst)"
        _obj_namen = [n for n, _, _ in WAHRNEHMUNG_OBJEKTE]
        _wahl = wc3.selectbox(
            "Objekt  (Helligkeit + Augenpupille)", [_opt_aus] + _obj_namen + [_opt_frei],
            index=2, key="wn_obj",
            help="Ein Feld setzt Bildhelligkeit L [cd/m²] UND Augenpupille d_p zusammen: "
                 "Mond 3500 / 2,5 mm · heller Planet (Jupiter) 600 / 3 · "
                 "DSO (hell) 5 / 6. Eigene Werte unter „Frei“. Bei wenig Licht wandert die Spitze der "
                 "Kontrastempfindlichkeit zu groben Strukturen, ein schlechter Spiegel wirkt dann "
                 "besser. Ab L ≥ 100 cd/m² gilt photopisch. Q̃-Kurven, V_krit und V½ bleiben "
                 "photopisch (Q̃ ist immer eingezeichnet; die Abweichungsfläche nur, wo L ≥ 100 cd/m² gilt). Pupillenkurven sind auf einen "
                 "gleich abgeblendeten Paraboloid normiert; der Öffnungsverlust steckt in Q_abs.")
        if _wahl == _opt_aus:
            L_obj, d_pup = 0.0, 0.0
        elif _wahl == _opt_frei:
            fr1, fr2 = st.columns(2)
            L_obj = fr1.number_input("Leuchtdichte L [cd/m²]", min_value=0.0, max_value=20000.0,
                                     value=5.0, step=0.5, key="wn_L")
            d_pup = fr2.number_input("Augenpupille d_p [mm]  (0 = aus)", min_value=0.0, max_value=9.0,
                                     value=6.0, step=0.5, key="wn_pup")
        else:
            L_obj, d_pup = [(float(l), float(p)) for n, l, p in WAHRNEHMUNG_OBJEKTE if n == _wahl][0]
            wc3.caption(f"L = {L_obj:g} cd/m²  ·  d_p = {d_pup:g} mm")
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

        _show_cached("fig_wahrnehmung", lambda: plot_wahrnehmung(D, f, lam,
                     S_slide, S_real, V_cur=V_cur, d_fang=d_fs, d_pup=d_pup, L_obj=L_obj),
                     _h + (V_cur, d_fs, d_pup, L_obj))

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
        fc1, fcr, fc2, fc3 = st.columns([1.1, 1.5, 2.4, 1.8])      # v7.2.6: alle Auswahlpunkte in einer Zeile
        with fc1:
            n_zonen = st.radio("Zonen:", [4, 6, 8], horizontal=True, index=0, key="fc_n_zonen")
        with fcr:
            rand_g = st.checkbox("Randzone teilen", key="fc_rand",
                                 help="Die äußerste Zone wird in zwei flächengleiche Hälften geteilt "
                                      "(innere und äußere Randhälfte getrennt messen) → n+1 Messzonen. "
                                      "Der Rand ist die schwierigste Zone; so wird er feiner aufgelöst.")
        with fc2:
            use_direct = st.radio(
                "Auswertungs-Modus:",
                ["Polynom-Fit  (glatte Fläche)", "Hybrid  (Zonenfehler)"],
                index=0, horizontal=True, key="fc_modus",
                help="Polynom-Fit: glatte Fläche aus geraden ρ-Potenzen — dämpft "
                     "Messrauschen, mittelt aber lokale Zonenfehler weg.\n"
                     "Hybrid: Polynom + stückweise konstante Zonenresiduen — erfasst "
                     "Ringfehler. Braucht mindestens 6 Zonen; empfohlen, wenn das "
                     "Fit-Residuum > 0.010 λ ist.")
            use_direct = use_direct.startswith("Hybrid")
        with fc3:
            grad_label = st.radio(
                "Polynomgrad:", ["BIC (auto)", "Fest (n//2)"],
                index=0, horizontal=True, key="fc_grad",
                help="BIC wählt den Grad automatisch (kann bei rauschigen Daten sehr "
                     "niedrig ausfallen). Fest koppelt den Grad an die Zonenzahl.")
            grad_modus = "fest" if grad_label.startswith("Fest") else "bic"

        r_zon   = foucault_zonen(D, n_zonen, rand_g)
        r_grenz = foucault_zonengrenzen(D, n_zonen, rand_g)
        nz_tot  = len(r_zon)                      # Zahl der Messzonen (n oder n+1)
        Rc      = 2.0 * f
        df_para_abs = r_zon**2 / (2.0 * Rc)
        df_para     = df_para_abs - df_para_abs[0]

        # Info-Zeile
        info_parts = [f"Z{i+1}: r={rv_:.1f}mm  ΔF_soll={dp:.3f}mm"
                      for i, (rv_, dp) in enumerate(zip(r_zon, df_para))]
        if rand_g:
            st.caption(f"Randzone geteilt: Z{n_zonen} = innere, Z{n_zonen + 1} = äußere Randhälfte (je halbe Fläche).")
        st.caption("Gleichflächige Zonen | Parabolspiegel-Sollwerte rel. Zone 1 "
                   "(+ = Messerschneide weiter weg, FigureXP-Konvention):  " +
                   "  |  ".join(info_parts))
        st.caption("ΔF=0 → Kugel  |  ΔF=ΔF_soll → Parabel  |  Zone 1 = Referenz, fest 0")

        # Eingabetabelle
        st.markdown("**ΔF-Messwerte [mm]  (relativ zu Zone 1):**")

        # Session-State für Messwerte
        key_prefix = f"fc_{D}_{f}_{n_zonen}_{int(rand_g)}"
        df_vals = []
        cols_hdr = st.columns([1, 2, 2, 2, 2, 2])
        cols_hdr[0].markdown("**Zone**")
        cols_hdr[1].markdown("**Radius [mm]**")
        cols_hdr[2].markdown("**Grenze innen–außen [mm]**")
        cols_hdr[3].markdown("**ΔF_soll [mm]**")
        cols_hdr[4].markdown("**ΔF gemessen [mm]**")
        cols_hdr[5].markdown("**Abw. [mm]**")

        for i in range(nz_tot):
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
            for i in range(1, nz_tot):
                k = f"{key_prefix}_z{i}"
                v = round(float(df_para[i]), 3)
                # Widget-Key löschen, dann neu setzen – vermeidet Streamlit-Konflikt
                if k in st.session_state:
                    del st.session_state[k]
                st.session_state[k] = v
            st.rerun()
        if bk2.button("✕  Alle auf Null (Kugel)"):
            for i in range(1, nz_tot):
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
        _fc_res_key = f"fc_result_{D}_{f}_{n_zonen}_{int(rand_g)}"
        if do_calc:
            df_arr = np.array(df_vals)
            try:
                W_z = foucault_wellenfront(r_zon, df_arr, D, f, lam_fc, r_grenz=r_grenz)
                # Hybrid braucht >= 6 Zonen (wie Desktop); sonst Polynom-Modus
                _ud_eff = use_direct and (nz_tot >= 6)
                kz  = foucault_kennzahlen(W_z, r_zon, D, lam_fc,
                                          use_direct=_ud_eff, grad_modus=grad_modus, r_grenz=r_grenz)
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
            W_z = foucault_wellenfront(r_zon, df_arr, D, f, lam_fc, r_grenz=r_grenz)  # für plot_foucault

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
        # v7.3.1: beide Diagramme nebeneinander (das untere lag sonst außerhalb des Bildschirms)
        _namen = [n for n, _ in OBJEKT_HELLIGKEITEN]
        _preset = dict(OBJEKT_HELLIGKEITEN)

        def _objekt_gewaehlt():
            st.session_state["vmax_L0"] = float(_preset[st.session_state["vmax_objekt"]])

        if "vmax_L0" not in st.session_state:
            st.session_state["vmax_L0"] = float(_preset[OBJEKT_HELLIGKEITEN[1][0]])
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.caption("Gegenprobe zur „2×D“-Faustregel: Netzhaut-Flächenhelligkeit L(V) sowie die "
                       "Grenzen durch Helligkeit, Seeing und Spiegelfehler (aktueller Strehl-Schieber).")
            vc1, vc2 = st.columns([2, 1.3])
            objekt_v = vc1.selectbox("Objekt:", _namen, index=1, key="vmax_objekt",
                                     on_change=_objekt_gewaehlt,
                                     help="Füllt L₀ mit dem Referenzwert des Objekts; "
                                          "danach frei überschreibbar.")
            L0_v = vc2.number_input("L₀ [cd/m²]", min_value=0.0, step=10.0,
                                    format="%.1f", key="vmax_L0")
            vc3, vc4, vc5 = st.columns(3)
            seeing_v = vc3.number_input("Seeing [\"]", min_value=0.0, value=1.5,
                                        step=0.1, format="%.1f", key="vmax_seeing",
                                        help=ANTONIADI_HINWEIS)
            D_auge_v = vc4.number_input("Augenpupille [mm]", min_value=0.5, value=6.0,
                                        step=0.5, format="%.1f", key="vmax_D_auge")
            L_schw_v = vc5.number_input("Schwelle [cd/m²]", min_value=0.0, value=1.0,
                                        step=0.1, format="%.2f", key="vmax_L_schwelle")
            _show_cached("fig_vmax",
                         lambda: plot_vmax(D, f, S_slide, objekt_v, L0_v, seeing_v,
                                           D_auge_v, L_schw_v),
                         _h + (objekt_v, L0_v, seeing_v, D_auge_v, L_schw_v))
        with gcol2:
            st.caption("Fundamentale Grenze auch bei perfekter Politur (S=1): Beugung + Seeing, "
                       "im Vergleich zur „2×D“-Regel.")
            seeings_txt = st.text_input("Seeing-Werte [\"], kommagetrennt",
                                        value="0.4, 0.8, 1.5, 2.5", key="vmaxD_seeings",
                                        help=ANTONIADI_HINWEIS)
            wc2, wc3 = st.columns(2)
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
        tubus_v = fc2.number_input("Tubus-Ø [mm] (0 = aus Okular, Gittertubus)", min_value=0,
                                   value=0, step=10, key="fs_tubus")
        bild_v  = fc3.number_input("100%-Bildfeld [mm] (visuell 10, Foto 20–25)", min_value=0,
                                   value=10, step=1, key="fs_bild")
        V_fs    = fc4.number_input("Q_vis bei V [×]", min_value=1, value=150,
                                   step=10, key="fs_V")
        st.session_state["fs_geo"] = (float(buet_v), float(tubus_v), float(bild_v))   # v8.4.0: auch für den Vergleich
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        hut_v  = gc1.number_input("Hutring-Ø für Blende [mm] (0 = Tubus-Ø)", min_value=0,
                                  value=0, step=10, key="fs_hut")
        fe_v   = gc2.number_input("Übersichtsokular f [mm]", min_value=1,
                                  value=30, step=1, key="fs_fe")
        afov_v = gc3.number_input("Okular-AFOV [°]", min_value=1,
                                  value=70, step=1, key="fs_afov")
        st.session_state["fs_oku"] = (float(fe_v), float(afov_v))   # v8.5.0: für den Low-Rider-Tab
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
        _fs_zeilen = [z for z in fangspiegel_text(D, f, S_fs, S_fs_txt, r_fs, V_fs).split("\n")
                      if not z.startswith(("Q_vis bei", "Effektiver Kontrast"))]    # v7.2.6: schlanker
        st.markdown("  \n".join(_fs_zeilen))
        _show_cached("fig_fangspiegel",
                     lambda: plot_fangspiegel(D, f, lam, S_fs, S_fs_txt, r_fs, V_fs),
                     (D, f, lam, S_fs, buet_v, tubus_v, bild_v, V_fs, hut_v, fe_v, afov_v, oazd_v, oazl_v))

    if _nav == "gitter":
        # v7.8.0: Stangenlänge eines Gitterrohrs
        st.subheader("Gitterrohr: Stangenlänge")
        st.caption("Zwei Ringe, je n Anschlusspunkte auf dem Teilkreis; jede Stange verbindet einen Punkt unten mit dem um k "
                   "Teilungen versetzten Punkt oben: x = √(h² + (2·r·sin(Δφ/2))²), Δφ = k·360°/n.")
        bc1, bc2, bc3, bc4 = st.columns(4)
        g_bau = bc1.radio("Bauart", ["Zigzag", "TripleX", "frei"], horizontal=True, key="gt_bau",
                          help="Zigzag: 3 Ankerpunkte je Ring, oberer Ring um 60° verdreht, zwei Stangenenden je Punkt (6 Stangen). frei: n Punkte, Versatz k. TripleX: benachbarte Punkte werden mit gekreuzten Stangen "
                               "verbunden (Paar liegt in einer Ebene, Kreuzung nur verspannt).")
        g_rb  = bc2.number_input("Ringbreite [mm]", min_value=0.0, value=50.0, step=5.0, key="gt_rb")
        g_rt  = bc3.number_input("Ringdicke [mm]", min_value=0.0, value=15.0, step=1.0, key="gt_rt")
        g_ach = bc4.number_input("opt. Achse über Hutring-OK [mm]", min_value=0.0, value=40.0, step=1.0, key="gt_ach")
        hc1, hc2, hc3, hc4 = st.columns(4)
        _di_auto = tubus_aus_okular(D, f, st.session_state.get("fs_buet", 50), st.session_state.get("fs_fe", 30),
                                    st.session_state.get("fs_afov", 70))["d_innen"]
        g_di  = hc3.number_input(f"Ring-Innen-Ø [mm] (0 = aus Fangspiegel-Rechner: {_di_auto:.1f})", min_value=0.0, value=0.0, step=5.0, key="gt_di")
        g_st  = hc4.number_input("Spiegel unter Ring-Oberkante [mm]", min_value=0.0, value=0.0, step=1.0, key="gt_st",
                                 help="0 = Spiegeloberfläche bündig mit der Oberkante des unteren Rings; sitzt der Spiegel tiefer, hier den Versatz nach unten eintragen.")
        g_lr  = hc1.number_input("Lochmitte bis Ringfläche [mm]", min_value=0.0, value=25.0, step=1.0, key="gt_lr")
        g_z   = hc2.number_input("Abzug je Ende [mm] (Gabelkopf: Lochmitte bis Stangenende)", min_value=0.0, value=25.0, step=1.0, key="gt_z")
        _tx = (g_bau == "TripleX"); _zz = (g_bau == "Zigzag"); _bk = {"Zigzag": "zigzag", "TripleX": "triplex", "frei": "frei"}[g_bau]
        _ga = gitterrohr_auto(D, f, st.session_state.get("fs_buet", 50), st.session_state.get("fs_fe", 30),
                              st.session_state.get("fs_afov", 70), g_rb, g_ach, g_rt, g_lr, g_di, g_st)
        gc1, gc2, gc3, gc4 = st.columns(4)
        g_h   = gc1.number_input("Lochabstand h [mm] (0 = automatisch)", min_value=0.0, value=0.0, step=10.0, key="gt_h")
        g_d   = gc2.number_input("Teilkreis-Ø [mm] (0 = Innen-Ø + Ringbreite)", min_value=0.0, value=0.0, step=10.0, key="gt_d")
        g_n   = gc3.number_input("Ankerpunkte je Ring n", min_value=3, max_value=24, value=(3 if _zz else 6), step=1, key="gt_n_" + _bk)
        g_k   = gc4.number_input("Versatz k (Teilungen)", min_value=0, max_value=12, value=1, step=1, key="gt_k", disabled=(_tx or _zz),
                                 help="0 = senkrechte Stangen, 1 = Nachbarpunkt, 2 = übernächster ... (TripleX: fest 1)")
        g_kr  = st.checkbox("gegenläufige Stangen zusätzlich (Kreuz)", value=True, key="gt_kreuz", disabled=(_tx or _zz))
        _h_use = g_h if g_h > 0 else _ga["h"]
        _d_use = g_d if g_d > 0 else _ga["d_teil"]
        _g  = gitterrohr_stangen(_h_use, _d_use / 2.0, int(g_n), int(g_k), -g_z, g_kr, _bk)
        _fr = gitterrohr_freiraum(_h_use, _d_use / 2.0, _g["dphi"], D, f, _ga["feld_halb"], _ga["z0"])
        st.markdown("  \n".join(gitterrohr_text(D, f, _g, _fr, _h_use, _d_use).split("\n")))
        def _plot_gitter():
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2))
            gitterrohr_zeichnen(a1, a2, D, f, _g, _h_use, _d_use / 2.0, _fr, _ga["z0"],
                                (_ga["d_innen"] / 2.0, _ga["d_innen"] / 2.0 + g_rb),
                                dict(rt=g_rt, z_sec=_ga["z_sec"], A=_ga["A"], d_fang=_ga["d_fang"], tief=_ga["tief"]))
            fig.tight_layout()
            return fig
        _show_cached("fig_gitter", _plot_gitter, (D, f, _h_use, _d_use, g_n, g_k, g_z, g_kr, g_bau, g_lr, g_rb, g_rt, g_ach, g_di, g_st))
        if st.checkbox("3D-Ansicht (maßstäblich; Spinne, Blendenhöhe, OAZ-Winkel geschätzt)", value=False, key="gt_3d"):
            def _plot_gitter3d():
                fig = plt.figure(figsize=(14, 7.4))
                gitter3d_zeichnen(fig, D, f, _ga, _g, _h_use, _d_use / 2.0, g_rb, g_rt, g_lr, _bk,
                                  dict(buet=st.session_state.get("fs_buet", 50), fe=st.session_state.get("fs_fe", 30),
                                       afov=st.session_state.get("fs_afov", 70)))
                return fig
            _show_cached("fig_gitter3d", _plot_gitter3d, (D, f, _h_use, _d_use, g_n, g_k, g_z, g_kr, g_bau, g_lr, g_rb, g_rt, g_ach, g_di, g_st,
                                                          st.session_state.get("fs_buet", 50), st.session_state.get("fs_fe", 30),
                                                          st.session_state.get("fs_afov", 70)))

    if _nav == "zelle":
        # v8.4.0: Spiegelzelle – Auflagepunkte 3/6/9/18
        st.subheader("Spiegelzelle: Lage der Auflagepunkte")
        st.caption("Spiegel unter Eigengewicht (Zenit), dünne Platte, Punktlager; Wippen und Dreiecke tragen alle Punkte gleich. "
                   "Die Radien sind das berechnete Optimum für kleinste Oberflächenverformung (nach Abzug von Kippung und Brennweitenänderung).")
        zc1, zc2, zc3, zc4 = st.columns(4)
        z_n = int(zc1.radio("Auflagepunkte", [3, 6, 9, 18], index=1, horizontal=True, key="zl_n"))
        z_t = zc2.number_input("Spiegeldicke [mm]", min_value=1.0, value=25.0, step=1.0, key="zl_t")
        z_mat = "Borosilikat (Pyrex/Duran)"   # nur Vergleich der Varianten, ein Glas genügt
        z_ref = zc3.checkbox("Brennweitenänderung abziehen", value=True, key="zl_ref",
                             help="Aus: klassische Werte (3 Punkte 0,645·R, 6 Punkte 0,68·R). An: Refokussieren ist beim Newton möglich, das Optimum liegt weiter innen.")
        _tz = z_t
        _zp = None
        if z_n in (3, 6):
            _r0 = (OPT if z_ref else OPT_OHNE)[z_n]["r"]
            _rin = st.number_input(f"Radius des Punktkreises r/R (Optimum {_r0:.3f})", min_value=0.1, max_value=0.92,
                                   value=float(round(_r0, 3)), step=0.01, key=f"zl_r_{z_n}_{int(z_ref)}")
            _zp = dict(r=float(_rin))
        def _plot_zelle():
            fig = plt.figure(figsize=(12, 6.6)); fig.patch.set_facecolor(BG)
            zelle_zeichnen(fig, D, _tz, z_n, z_mat, z_ref, lam, _zp)
            return fig
        _show_cached("fig_zelle", _plot_zelle, (D, _tz, z_n, z_mat, z_ref, lam, str(_zp)))
        _zd = zelle_rechner(D, _tz, z_n, z_mat, z_ref, lam, _zp)
        st.code(zelle_text(_zd, _tz, z_mat), language=None)

    if _nav == "lowrider":
        # v8.5.0: Low-Rider-Newton – Strahl nach unten, Okularauszug tiefer
        st.subheader("Low-Rider: Fangspiegel und Einblickhöhe")
        _lb, _lt, _lbi = st.session_state.get("fs_geo", (50.0, 0.0, 10.0))
        _lfe, _laf = st.session_state.get("fs_oku", (30.0, 70.0))
        st.caption("Der Strahl wird am Fangspiegel nicht rechtwinklig, sondern um den Winkel δ nach unten abgelenkt: "
                   "Der Okularauszug sitzt tiefer, der Fangspiegel wird etwas größer. Okularauszug, Tubus und Bildfeld "
                   f"kommen aus dem Tab Fangspiegel (jetzt {_lb:.0f} mm / Tubus-Ø {'aus Okular' if _lt <= 0 else format(_lt, '.0f') + ' mm'} / {_lbi:.0f} mm).")
        lc1, lc2, lc3 = st.columns([2, 2, 1.4])
        _lan = lc1.radio("Ansicht", LR_ANSICHTEN, horizontal=True, key="lr_an")
        _lw = lc2.slider("Strahl nach unten δ [°]", 0, 35, 15, 1, key="lr_w")
        _lDm = lc3.number_input("bis Ø [mm]", min_value=int(D) + 10, max_value=1500,
                                value=int(max(round(D * 1.43 / 10) * 10, D + 10)), step=10, key="lr_dmax",
                                disabled=(_lan != LR_ANSICHTEN[2]))
        _lgeo = dict(buet=_lb, tubus=_lt, bild=_lbi, fe=_lfe, afov=_laf)
        def _plot_lr():
            fig = plt.figure(figsize=(12, 5.6)); fig.patch.set_facecolor(BG)
            lowrider_zeichnen(fig, float(D), float(f), float(_lw), _lan, _lgeo, float(_lDm))
            return fig
        _show_cached("fig_lowrider", _plot_lr, (D, f, _lw, _lan, _lDm, _lb, _lt, _lbi, _lfe, _laf))
        st.code(lowrider_text(float(D), float(f), float(_lw), _lgeo), language=None)

    if _nav == "vergleich":
        # v7.2.1: Spiegel 1 = Hauptspiegel (Seitenleiste, S = Schieberwert), f/D vom Hauptspiegel;
        # nur der Vergleichsspiegel 2 wird hier eingegeben.
        vD1 = float(D)
        if S_slide is not None:
            vS1 = float(S_slide)
        else:
            _, _, _Wb1, _, _ = _wellenfronten(D, f, lam)
            vS1 = float(_strehl_exakt(_Wb1))
        vS1 = min(max(vS1, 0.05), 1.0)
        vBu0, vTu0, vBi0 = st.session_state.get("fs_geo", (50.0, 0.0, 10.0))
        vN = f / D
        st.caption(f"**Beta · Q_abs-Vergleich.** Spiegel 1 = Hauptspiegel: D = {vD1:.0f} mm, f/D = {vN:.2f}, "
                   f"S = {vS1:.3f}{' (Schieber)' if S_slide is not None else ''}; Spiegel 2 hat dasselbe f/D; "
                   f"Fangspiegel automatisch (Okularauszug {vBu0:.0f} mm, Bildfeld {vBi0:.0f} mm, "
                   f"Tubus-Ø {'aus Okular' if vTu0 <= 0 else format(vTu0, '.0f') + ' mm'} aus dem Tab Fangspiegel).")
        r1 = st.columns(4)
        vD2 = r1[0].number_input("Spiegel 2: D [mm]", 50, 1500, 200, 10, key="vg_D2")
        vS2 = r1[1].number_input("Spiegel 2: S", 0.10, 1.0, 1.00, 0.01, format="%.2f", key="vg_S2")
        vV  = r1[2].number_input("V [×]", 10, 1000, 200, 10, key="vg_V")
        vSe = r1[3].number_input("Seeing [″] (0 = aus)", 0.0, 6.0, 1.5, 0.1, key="vg_see",
                                 help=ANTONIADI_HINWEIS)
        # v8.4.0: Okularauszug, Tubus-Ø und Bildfeld kommen aus dem Fangspiegel-Tab (Bauen), keine Doppeleingabe
        vBu, vTu, vBi = st.session_state.get("fs_geo", (50.0, 0.0, 10.0))
        r2 = st.columns(2)
        vRD = r2[0].number_input("Referenz: idealer D [mm]", 50, 1500, 200, 10, key="vg_refD")
        vPu = r2[1].number_input("Augenpupille [mm] (0 = aus)", 0.0, 9.0, 0.0, 0.5, key="vg_pupille",
                                 help="Ist die Austrittspupille D/V größer als die Augenpupille, nutzt das Auge nur "
                                      "die Teilöffnung V·Pupille (blendet das Teleskop ab). Mond 2–3 mm, Planeten "
                                      "3–5 mm, Deep-Sky 6–7 mm.")
        _kz = st.session_state.get("fc_last_kz")
        _Df = st.session_state.get("fc_last_D")
        ck = st.columns(6)
        use_w1 = ck[0].checkbox("Sp. 1 Foucault", key="vg_wf1", help="Spiegel 1 aus Foucault-Test (Wellenfront)")
        use_w2 = ck[1].checkbox("Sp. 2 Foucault", key="vg_wf2", help="Spiegel 2 aus Foucault-Test (Wellenfront)")
        sph1 = ck[2].checkbox("Sp. 1 sphärisch", key="vg_sph1", help="Spiegel 1 sphärisch (aus D, f/D)")
        sph2 = ck[3].checkbox("Sp. 2 sphärisch", key="vg_sph2", help="Spiegel 2 sphärisch (aus D, f/D)")
        v_auge = ck[4].checkbox("Augen-Filter", value=True, key="vg_auge",
                                help="Bild mit Augen-Filter bei V (CSF-Tiefpass)")
        v_mass = ck[5].checkbox("Augenmaßstab", value=True, key="vg_mass",
                                help="Planet wächst mit V")
        _notiz = []
        _wf = {}
        if vS1 < 0.09 and not sph1:      # S -> Wb-Inversion erreicht S < 0,082 nicht (S nicht monoton)
            sph1 = True
            _notiz.append("Spiegel 1: Strehl < 0,09 – es wird die sphärische Wellenfront aus D und f/D verwendet.")
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
                  D2=vD2, S2=vS2, lam=lam, buet=vBu, tubus=vTu, bild=vBi, pupille=vPu,
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
        import pathlib
        _doc_candidates = [
            pathlib.Path(__file__).parent / "documentation_v7_6.html",
            pathlib.Path("documentation_v7_6.html"),
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
                "Dokumentationsdatei **documentation_v7_6.html** nicht gefunden. "
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
| V_krit (Näherung) | D / (13 · W_b) |
| Foucault | Zonen-ΔF → Wellenfront (kumulative Neigungsintegration) |
| Parabel-Soll δl | r²/(4f), rel. Zone 1 |
""")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Newton-Spiegel v{VERSION} · "
        "Formeln: Sacek (telescope-optics.net) · Suiter (Star Testing) · Mannos/Sakrison 1974 (CSF) · "
        "Strehl: exaktes Pupillenintegral"
    )


if __name__ == "__main__" or True:
    main()