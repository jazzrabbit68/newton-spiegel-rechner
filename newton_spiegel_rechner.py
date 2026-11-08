"""
Newton-Teleskop: Kontrast- und Schärfeverlust sphärischer Hauptspiegel  v8.3.0 (2026-11-08) Wahrnehmung: Näherung Q̃ immer eingezeichnet (Abweichungsfläche nur photopisch), Objektauswahl reduziert auf Mond / Heller Planet / DSO (hell) / Aus / Frei (volle Liste bleibt bei Max. Vergrößerung und Blende); Kennwerte: "Wie gut ist der Spiegel? · Wie weit trägt er?"
  v9.0.0 (2026-11-08)
============================================================================================
Änderungshistorie (Kurzfassung, jüngste zuerst; Einzelheiten in CHANGELOG_vollstaendig.txt und in der Doku):
  v9.0.0 (2026-11-08) NEU Spiegelzelle: Auflagepunkte 3/6/9/18 (Dreieck, Wippen, Dreiecke, Wippen mit Dreiecken); optimale Lage, Radien, Wippen-/Dreieckslängen aus Dünnplatten-Rechnung (Eigengewicht im Zenit), Durchbiegungskarte, RMS/Strehl, Variantenvergleich (neues Modul kern_zelle.py)
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

Berücksichtigt:
  - Wellenfrontfehler und Strehl-Quotient (Kontrast)
  - Geometrischen Unschärfefleck am besten Fokus (Schärfe)
  - Vergrößerungsabhängigen Schärfeverlust durch das Auge

Formeln:
  Kontrast (Strehl):
    W_PtV (paraxial)  = D⁴ / (512·f³·λ)          [Sacek §4.1.2, korrigiert 07/2026:
                                                      Reflexion verdoppelt Oberflaechen-
                                                      in Wellenfrontfehler]
    W_PtV (best focus)= W_PtV_paraxial / 4         [Sacek §4.1.2]
    W_RMS             = W_PtV_bestfocus / (1.5·√5) [Sacek]
    Strehl (Näherung) = exp(-(2π·W_RMS)²)          [Maréchal 1947, zit. n. Sacek]
    Strehl (exakt)    = |⟨exp(i·2π·W(ρ))⟩|²        [Standardformel, vgl. Sacek]
    D_eff (Kontrast)  = D · S^0.25                 [eigene Näherung]

  Schärfe:
    r_Airy [arcsec]   = 1.22·λ/D·206265
    d_blur [mm]       = D³ / (128·f²)              [Sacek, kleinster Zerstreuungskreis]
    Dawes [arcsec]    = 116 / D[mm]

  Kritische Vergrößerung:
    V_krit            = D[mm] / (13·Wb)            [empirisch gefittet, s. Chat-
                                                       Historie 08/2026: ersetzt alte,
                                                       nachweislich falsche Formel
                                                       D·0.7·√S — Wb monoton, S bei
                                                       starker Aberration nicht mehr]
    V_wendepunkt_naeh = V_krit / ³√(S²)             [nur für Sigmoid-Näherungskurve
                                                       Q̃(V), nicht V_krit selbst]

  MTF:
    Paraboloid:  analytisch (Kreisapertur)          [Sacek]
    Sphäre rel.: OTF-Faltungsintegral, numerisch    [Sacek]
      Wellenfrontform W(ρ) = Wp·(ρ⁴−ρ²)  [best focus]
                   oder Wp·ρ⁴             [paraxialer Fokus]
      Amplitude: Wp = 4·Wb
        (Wb = PtV am besten Fokus = Wp_paraxial/4
         → W(ρ) = Wp·(ρ⁴−ρ²) = 4·Wb·(ρ⁴−ρ²); konsistent mit Sacek)

  Wahrnehmung:
    CSF nach Mannos & Sakrison (1974):
      csf(f) = 2.6·(0.0192 + 0.114·f)·exp(−(0.114·f)^1.1)   [Mannos/Sakrison 1974]
    Q_vis = ∫(MTF_sph · MTF_para · CSF) dν / ∫(MTF_para · CSF) dν
    Q̃(V) = S + (1−S)·(1−w(V))   [eigene Näherungsformel]
      mit w(V) = 1 / (1 + (V_krit/V)²)

Quellen:
  [Sacek]    V. Sacek, "Notes on Amateur Telescope Optics",
             https://www.telescope-optics.net  (laufend aktualisiert)
  [Suiter]   H.R. Suiter, "Star Testing Astronomical Telescopes",
             1. Aufl. Willmann-Bell (1994); 2. Aufl. (2013)
  [Mannos]   J.L. Mannos, D.J. Sakrison, "The effects of a visual fidelity criterion on the encoding of
             images", IEEE Trans. Inf. Theory 20(4), 525-536 (1974)  — CSF-Formel
  [Barten]   P.G.J. Barten, "Contrast Sensitivity of the Human Eye and Its Effects
             on Image Quality", SPIE Optical Engineering Press (1999), ISBN 0-8194-3496-5

Abhängigkeiten: pip install matplotlib numpy
"""

import math
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import numpy as np
from functools import lru_cache
from diagramme import DiagrammMethoden
from kern_bau import (fangspiegel_rechner, fangspiegel_skizze, fangspiegel_text,
    fangspiegel_zeichnen, gitterrohr_auto, gitterrohr_freiraum, gitterrohr_stangen, gitterrohr_text,
    gitterrohr_zeichnen, pfeiltiefe, qvis_kurve_mit_fangspiegel)
from ansicht3d import gitter3d_zeichnen
from kern_zelle import MATERIALIEN, OPT, OPT_OHNE, zelle_rechner, zelle_text, zelle_zeichnen
from kern_foucault import (foucault_kennzahlen, foucault_wellenfront, foucault_zonen,
    foucault_zonengrenzen)
from kern_optik import (ANTONIADI_HINWEIS, OBJEKT_HELLIGKEITEN, OBJEKT_PUPILLE, WAHRNEHMUNG_OBJEKTE, _L_eff, _L_roh,
    _Wb_schieber, _qvis_schieber, _strehl_exakt, _wellenfronten, berechne, berechne_vergr,
    beurteilung, csf_spitze_cpd, d_fang_min, perceived_quality_exakt, qvis_einzelrechner,
    v_halb_exakt, v_krit_aus_qvis, v_krit_aus_qvis_von_Wb, v_krit_blur_direkt, v_krit_naeh,
    v_kritisch, v_kritisch_strehl)
from kern_vergleich import _wf_kugel, qvis_pupille_kurve, vergleich_achsen, vergleich_zeichnen
from konstanten import ACC, BG, BG2, EYE_RES
import texte
from texte import T


class App(DiagrammMethoden, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(T("win_title"))
        self.resizable(True, True)
        self._maximiere_fenster()
        self.configure(bg=BG)
        self._update_job      = None
        self._compute_cancel  = False
        self._last_params     = None   # zuletzt berechnete Parameter
        self._dirty_tabs      = set()  # Tabs die beim nächsten Wechsel neu gezeichnet werden
        self._build_ui()
        self._baue_navigation()
        # Tab-Wechsel: bei Bedarf das neue Tab nachzeichnen
        self._nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._aktualisieren()

    def _maximiere_fenster(self):
        """Startet das Fenster maximiert (Bildschirmgröße) statt in Default-Breite.

        '-zoomed' funktioniert unter Linux/X11 (Ubuntu) und Windows; falls das
        Attribut nicht unterstützt wird (z.B. manche macOS/Wayland-Setups),
        wird ersatzweise die Geometrie manuell auf die Bildschirmgröße gesetzt.
        """
        try:
            self.attributes("-zoomed", True)
        except tk.TclError:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"{sw}x{sh}+0+0")

    def _sprache_wechseln(self, lang):
        texte.set_sprache(lang)
        self.title(T("win_title"))
        self._texte_aktualisieren()
        self._aktualisieren()

    def _texte_aktualisieren(self):
        """Alle statischen Label-Texte nach Sprachwechsel neu setzen."""
        # Slider-Namen
        for key, widget in self._param_name_widgets.items():
            widget.config(text=T(key))
        # Karten-Überschriften
        for key, widget in self._grp_widgets.items():
            widget.config(text=T(key))
        # Karten-Labels
        for key, widget in self._karten_name_widgets.items():
            widget.config(text=T(key))
        # Tab-Titel
        for key, (nbk, frm) in self._tab_nb.items():
            if nbk is self._nb:
                continue
            nbk.tab(frm, text=T(key))
        for gi, (gkey, keys) in enumerate(self._tab_gruppen):
            self._nb.tab(gi, text=T(gkey))
        for k, (b, tk_) in getattr(self, "_nav_btns", {}).items():
            b.config(text=T(tk_).strip())
        for gk, h in getattr(self, "_nav_heads", {}).items():
            h.config(text=T(gk).strip())
        # MTF-Beschriftungen
        for key, widget in self._mtf_label_widgets.items():
            widget.config(text=T(key))

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=3)

        left  = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Sprachumschalter ─────────────────────────────────────────────────
        lang_frame = tk.Frame(left, bg=BG)
        lang_frame.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 0))
        tk.Label(lang_frame, text=T("lang_lbl"), bg=BG,
                 font=("Helvetica", 9), fg="#888").pack(side="left", padx=(0, 6))
        self._var_lang = tk.StringVar(value=texte._LANG)
        for code, label in [("de", "Deutsch"), ("en", "English")]:
            tk.Radiobutton(lang_frame, text=label, variable=self._var_lang,
                           value=code, bg=BG, fg="#555",
                           activebackground=BG, selectcolor=BG2,
                           font=("Helvetica", 9),
                           command=lambda c=code: self._sprache_wechseln(c)
                           ).pack(side="left", padx=4)

        # ── Eingaben ─────────────────────────────────────────────────────────
        grp_lbl = tk.Label(left, text=T("grp_params"),
                           font=("Helvetica", 11, "bold"), bg=BG)
        grp_lbl.grid(row=1, column=0, columnspan=3, sticky="w", **pad)
        self._grp_widgets = {"grp_params": grp_lbl}
        self._param_name_widgets = {}

        self.var_D      = tk.DoubleVar(value=200)
        self.var_f      = tk.DoubleVar(value=1000)
        self.var_strehl = tk.DoubleVar(value=0)
        self.var_blende = tk.DoubleVar(value=150)
        self.var_lam    = tk.DoubleVar(value=550)   # fix 550 nm, kein Eingabefeld

        # (key, var, min, max, unit, info-label-breite)
        param_defs = [
            ("sl_D",      self.var_D,        50,   500,  "mm",  14),
            ("sl_f",      self.var_f,        200,  4000, "mm",  14),
            ("sl_strehl", self.var_strehl,   0,    100,  "%",   14),
        ]
        self._param_labels = {}   # key → (lbl_val, unit)
        self._param_vars   = {}   # key → (var, mn, mx)
        for i, (key, var, mn, mx, unit, info_w) in enumerate(param_defs, 2):
            lbl_name = tk.Label(left, text=T(key), bg=BG, width=12, anchor="w")
            lbl_name.grid(row=i, column=0, sticky="w", **pad)
            self._param_name_widgets[key] = lbl_name

            # Eingabefeld + Einheit in einem kleinen Frame
            unit_frm = tk.Frame(left, bg=BG)
            unit_frm.grid(row=i, column=1, sticky="w", **pad)

            ent = tk.Entry(unit_frm, textvariable=var, width=9,
                           font=("Helvetica", 10), justify="right",
                           relief="solid", bd=1)
            ent.grid(row=0, column=0)
            ent.bind("<Return>",   self._param_changed)
            ent.bind("<KP_Enter>", self._param_changed)

            tk.Label(unit_frm, text=f" {unit}", bg=BG, fg="#555",
                     font=("Helvetica", 10)).grid(row=0, column=1)

            lbl_val = tk.Label(left, text="", bg=BG, width=info_w, anchor="w",
                               font=("Helvetica", 9), fg="#555")
            lbl_val.grid(row=i, column=2, sticky="w", **pad)
            self._param_labels[key] = (lbl_val, unit)
            self._param_vars[key]   = (var, mn, mx)

        row = len(param_defs) + 2   # 5 Parameter + Titelzeile (row=1) + Startoffset

        # Hinweis + Berechnen-Button
        hint_frm = tk.Frame(left, bg=BG)
        hint_frm.grid(row=row, column=0, columnspan=3, sticky="ew",
                      padx=10, pady=(4, 2))
        tk.Button(hint_frm, text="Berechnen  (Enter)",
                  command=self._param_changed,
                  font=("Helvetica", 9), bg=ACC, fg="white",
                  relief="flat", padx=6, pady=2).pack(side="left")
        row += 1

        # ── Ergebniskarten ────────────────────────────────────────────────────
        self._res = {}
        self._karten_name_widgets = {}

        # v7.1.0: links nur Kernwerte, die übrigen Karten im Tab "Kennwerte"
        # v7.7.0: Steckbrief (drei Fragen) sichtbar, übrige Karten unter "Details"
        self._karten_steck = [
            ("grp_flaeche",    [("rS_exakt", "rS_exakt"), ("rWb", "rWb"), ("rSag", "rSag")]),
            ("grp_reichweite", [("rVkrit_exakt", "rVkrit_exakt"), ("rVhalb", "rVhalb")]),
            ("grp_qvis",       [("rQp30", "rQp30"), ("rQp60", "rQp60"),
                                ("rQp100", "rQp100"), ("rQp150", "rQp150")]),
        ]
        self._karten_rest = [
            ("grp_contrast", [("rN", "rN"), ("rWp", "rWp"), ("rWrms", "rWrms"), ("rS", "rS"),
                              ("rDeff_k", "rDeff_k"),
                              ("rVkrit_naeh", "rVkrit_naeh"),
                              ("rVkrit_blur", "rVkrit_blur")]),
            ("grp_sharpness", [("rAuflVerlust", "rAuflVerlust"),
                               ("rVerlustV", "rVerlustV")]),
        ]
        # v7.2.1: Kernwerte + Urteil stehen jetzt oben im Tab Grundlagen > Kennwerte;
        # links bleiben nur Parameter und die Tab-Knöpfe (s. _baue_navigation).
        self._nav_row = row
        self._nav_left = left

        # ── Diagramm-Tabs ─────────────────────────────────────────────────────
        _st = ttk.Style(self)
        _st.layout("Tabless.TNotebook.Tab", [])          # Reiter ausblenden, Auswahl per Knopf links
        _st.configure("Tabless.TNotebook", borderwidth=0, tabmargins=0)
        self._nb = ttk.Notebook(right, style="Tabless.TNotebook")
        self._nb.pack(fill="both", expand=True)
        self._mtf_label_widgets = {}

        tab_attrs = [
            ("_tab_grund",        "tab_grund"),          # Reihenfolge = Reihenfolge in _tab_gruppen
            ("_tab_kennwerte",    "tab_kennwerte"),
            ("_tab_vmax",         "tab_vmax"),
            ("_tab_vmax_D",       "tab_vmax_D"),
            ("_tab_wahrnehmung",  "tab_wahrnehmung"),
            ("_tab_mtf",          "tab_mtf"),
            ("_tab_vergleich",    "tab_vergleich"),
            ("_tab_foucault",     "tab_foucault"),
            ("_tab_fangspiegel",  "tab_fangspiegel"),
            ("_tab_blende",       "tab_blende"),
            ("_tab_gitter",       "tab_gitter"),
            ("_tab_zelle",        "tab_zelle"),
        ]
        self._tab_keys = [k for _, k in tab_attrs]
        # v7.1.0: Gruppen (oberste Reiter); Gruppen mit mehreren Diagrammen
        # bekommen ein eigenes Unter-Notebook.
        self._tab_gruppen = [
            ("grp_t_grund",     ["tab_grund", "tab_kennwerte", "tab_vmax"]),   # tab_vmax_D teilt sich den Tab (rechte Hälfte)
            ("grp_t_bild",      ["tab_wahrnehmung", "tab_mtf", "tab_vergleich"]),
            ("grp_t_bauen",     ["tab_foucault", "tab_fangspiegel", "tab_blende", "tab_gitter", "tab_zelle"]),
        ]
        self._tab_nb = {}      # tab_key -> (Notebook, Frame)
        self._grp_nb = []      # pro Gruppe: (Gruppen-Frame, Unter-Notebook | None)
        for gkey, keys in self._tab_gruppen:
            if len(keys) > 1:
                gfrm = tk.Frame(self._nb, bg=BG)
                sub = ttk.Notebook(gfrm, style="Tabless.TNotebook")
                sub.pack(fill="both", expand=True)
                sub.bind("<<NotebookTabChanged>>", self._on_tab_changed)
                self._nb.add(gfrm, text=T(gkey))
            else:
                gfrm, sub = tk.Frame(self._nb, bg=BG), None
                self._nb.add(gfrm, text=T(gkey))
            self._grp_nb.append((gfrm, sub))
        for attr, tab_key in tab_attrs:
            gi = next(i for i, (_, ks) in enumerate(self._tab_gruppen)
                      if tab_key in ks or (tab_key == "tab_vmax_D" and "tab_vmax" in ks))
            gfrm, sub = self._grp_nb[gi]
            if tab_key == "tab_vmax_D":            # teilt sich den Tab mit tab_vmax (rechte Hälfte)
                frm = self._tab_nb["tab_vmax"][1]
                self._tab_nb[tab_key] = self._tab_nb["tab_vmax"]
            elif sub is not None:
                frm = tk.Frame(sub, bg=BG)
                sub.add(frm, text=T(tab_key))
                self._tab_nb[tab_key] = (sub, frm)
            else:
                frm = gfrm
                self._tab_nb[tab_key] = (self._nb, frm)
            if attr == "_tab_vmax":
                _l = tk.Frame(frm, bg=BG); _l.pack(side="left", fill="both", expand=True)   # v7.3.1: nebeneinander
                _r = tk.Frame(frm, bg=BG); _r.pack(side="left", fill="both", expand=True)
                self._vmax_cols = (_l, _r)
                frm = _l
            elif attr == "_tab_vmax_D":
                frm = self._vmax_cols[1]
            if attr == "_tab_grund":
                self._baue_grundlagen(frm)
            elif attr == "_tab_kennwerte":
                kr = 0
                self.lbl_verdict = tk.Label(frm, text="", bg=BG,
                                            font=("Helvetica", 10),
                                            wraplength=620, justify="left")
                self.lbl_verdict.grid(row=kr, column=0, columnspan=4,
                                      sticky="w", padx=10, pady=(4, 2))
                kr += 1
                for grp_key, karten in self._karten_steck:
                    grp_w = tk.Label(frm, text=T(grp_key),
                                     font=("Helvetica", 11, "bold"), bg=BG)
                    grp_w.grid(row=kr, column=0, columnspan=4, sticky="w", **pad)
                    self._grp_widgets[grp_key] = grp_w
                    kr += 1
                    kr = self._karten_grid(frm, karten, kr, self._res,
                                           self._karten_name_widgets, pad)
                tk.Label(frm, text="Q_vis hier: photopisch, volle Öffnung, ohne Seeing und Fangspiegel – die strengste Bewertung der Fläche; "
                              "an dunklen Objekten und mit Augenpupille wirkt der Spiegel besser (Tab Wahrnehmung).",
                         bg=BG, fg="#666", font=("Helvetica", 8), wraplength=620, justify="left"
                         ).grid(row=kr, column=0, columnspan=4, sticky="w", padx=10, pady=(2, 0))
                kr += 1
                self.lbl_vkrit = tk.Label(frm, text="", bg=BG,
                                          font=("Helvetica", 11, "bold"),
                                          wraplength=620, justify="left")
                self.lbl_vkrit.grid(row=kr, column=0, columnspan=4,
                                    sticky="w", padx=10, pady=(6, 4))
                kr += 1
                # Details (aufklappbar)
                det = tk.Frame(frm, bg=BG)
                det_row = kr + 1
                def _det_umschalten(det=det, frm=frm, det_row=det_row):
                    if det.winfo_manager():
                        det.grid_remove(); btn.config(text="Details ▸")
                    else:
                        det.grid(row=det_row, column=0, columnspan=4, sticky="ew")
                        btn.config(text="Details ▾")
                btn = tk.Button(frm, text="Details ▸", command=_det_umschalten,
                                font=("Helvetica", 9), relief="flat", bg=BG2, padx=8, pady=2)
                btn.grid(row=kr, column=0, columnspan=4, sticky="w", padx=10, pady=(2, 2))
                dr = 0
                for grp_key, karten in self._karten_rest:
                    grp_w = tk.Label(det, text=T(grp_key),
                                     font=("Helvetica", 11, "bold"), bg=BG)
                    grp_w.grid(row=dr, column=0, columnspan=4, sticky="w", **pad)
                    self._grp_widgets[grp_key] = grp_w
                    dr += 1
                    dr = self._karten_grid(det, karten, dr, self._res,
                                           self._karten_name_widgets, pad)
                for c in range(4):
                    frm.columnconfigure(c, weight=1, uniform="kw")
                    det.columnconfigure(c, weight=1, uniform="kd")
            elif attr == "_tab_mtf":
                btn_frame = tk.Frame(frm, bg=BG)
                btn_frame.pack(side="top", fill="x", padx=8, pady=(6, 0))
                self._mtf_modus = tk.StringVar(value="klassisch")
                row1b = tk.Frame(btn_frame, bg=BG)
                row1b.pack(side="top", anchor="w", pady=(2, 0))
                lbl_fok = tk.Label(row1b, text=T("mtf_fokus_lbl"), bg=BG,
                                   fg="#888", font=("Helvetica", 9))
                lbl_fok.pack(side="left", padx=(0, 6))
                self._mtf_label_widgets["mtf_fokus_lbl"] = lbl_fok
                self._mtf_fokus = tk.StringVar(value="bestfocus")
                for val, key in [("bestfocus", "mtf_bf"), ("paraxial", "mtf_pf")]:
                    rb = tk.Radiobutton(row1b, text=T(key), variable=self._mtf_fokus,
                                        value=val, bg=BG, fg="#666",
                                        activebackground=BG, selectcolor=BG2,
                                        font=("Helvetica", 9),
                                        command=self._aktualisieren)
                    rb.pack(side="left", padx=8)
                    self._mtf_label_widgets[key] = rb
                self._mtf_alt1d = tk.BooleanVar(value=False)
                tk.Checkbutton(row1b, text="alte 1D-Näherung zeigen", variable=self._mtf_alt1d,
                               bg=BG, fg="#666", activebackground=BG, selectcolor=BG2,
                               font=("Helvetica", 9), command=self._aktualisieren
                               ).pack(side="left", padx=14)
                row2 = tk.Frame(btn_frame, bg=BG)
                row2.pack(side="top", anchor="w", pady=(3, 0))
                lbl_obj = tk.Label(row2, text=T("mtf_obj_lbl"), bg=BG, fg=ACC,
                                   font=("Helvetica", 10, "bold"))
                lbl_obj.pack(side="left", padx=(0, 6))
                self._mtf_label_widgets["mtf_obj_lbl"] = lbl_obj
                self._mtf_objekte = tk.StringVar(value="planeten")
                for val, key in [("planeten", "mtf_planets"),
                                  ("deepsky",  "mtf_deepsky"),
                                  ("alle",     "mtf_all")]:
                    rb2 = tk.Radiobutton(row2, text=T(key), variable=self._mtf_objekte,
                                         value=val, bg=BG, fg="#8e44ad",
                                         activebackground=BG, selectcolor=BG2,
                                         font=("Helvetica", 10),
                                         command=self._aktualisieren)
                    rb2.pack(side="left", padx=8)
                    self._mtf_label_widgets[key] = rb2
                fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))
            elif attr == "_tab_wahrnehmung":
                # ── Ablese-Leiste: Cursor-Vergrößerung ──
                wn = tk.Frame(frm, bg=BG)
                wn.pack(side="top", fill="x", padx=8, pady=(6, 0))
                tk.Label(wn, text="Ablesen bei V =", bg=BG, fg="#111",
                         font=("Helvetica", 9, "bold")).pack(side="left")
                self._wn_V_var = tk.StringVar(value="150")
                e_v = tk.Entry(wn, textvariable=self._wn_V_var, width=6,
                               font=("Helvetica", 9))
                e_v.pack(side="left", padx=(3, 0))
                tk.Label(wn, text="×   (oder ins Diagramm klicken;  Spiegel = Regler Öffnung / Brennweite / Strehl-Schieber)",
                         bg=BG, fg="#666", font=("Helvetica", 9)).pack(side="left", padx=(2, 14))

                wn2 = tk.Frame(frm, bg=BG)
                wn2.pack(side="top", fill="x", padx=8, pady=(2, 0))
                tk.Label(wn2, text="Fangspiegel Ø d =", bg=BG, fg="#6a1b9a",
                         font=("Helvetica", 9, "bold")).pack(side="left")
                self._wn_fang_var = tk.StringVar(value="0")
                e_fs = tk.Entry(wn2, textvariable=self._wn_fang_var, width=6,
                                font=("Helvetica", 9))
                e_fs.pack(side="left", padx=(3, 0))
                self._wn_fang_hint = tk.Label(wn2, text="mm   (0 = aus)", bg=BG,
                                              fg="#666", font=("Helvetica", 9))
                self._wn_fang_hint.pack(side="left", padx=(2, 0))
                e_fs.bind("<Return>",   self._wahrnehmung_neu)
                e_fs.bind("<FocusOut>", self._wahrnehmung_neu)
                tk.Label(wn2, text="Augenpupille d_p =", bg=BG, fg="#ef6c00",
                         font=("Helvetica", 9, "bold")).pack(side="left", padx=(16, 0))
                self._wn_pup_var = tk.StringVar(value="3")
                e_pp = tk.Entry(wn2, textvariable=self._wn_pup_var, width=5,
                                font=("Helvetica", 9))
                e_pp.pack(side="left", padx=(3, 0))
                tk.Label(wn2, text="mm  (0 = aus; Mond 2–3, Planet 3–5, Deep-Sky 6–7)",
                         bg=BG, fg="#666", font=("Helvetica", 9)).pack(side="left", padx=(2, 0))
                e_pp.bind("<Return>",   self._wahrnehmung_neu)
                e_pp.bind("<FocusOut>", self._wahrnehmung_neu)
                wn3 = tk.Frame(frm, bg=BG)
                wn3.pack(side="top", fill="x", padx=8, pady=(2, 0))
                tk.Label(wn3, text="Objekt:", bg=BG, fg="#111",
                         font=("Helvetica", 9, "bold")).pack(side="left")
                self._wn_obj_var = tk.StringVar(value="Heller Planet (Jupiter)")
                _cb = ttk.Combobox(wn3, textvariable=self._wn_obj_var, width=15, state="readonly",
                                   values=["Aus (photopisch)"] + [n for n, _, _ in WAHRNEHMUNG_OBJEKTE] + ["Frei"])
                _cb.pack(side="left", padx=(3, 14))
                _cb.bind("<<ComboboxSelected>>", self._wn_objekt_gewaehlt)
                tk.Label(wn3, text="Leuchtdichte L =", bg=BG, fg="#2e7d32",
                         font=("Helvetica", 9, "bold")).pack(side="left")
                self._wn_L_var = tk.StringVar(value="600")
                e_L = tk.Entry(wn3, textvariable=self._wn_L_var, width=7,
                               font=("Helvetica", 9))
                e_L.pack(side="left", padx=(3, 0))
                tk.Label(wn3, text="cd/m²  (Objekt setzt L und Augenpupille zusammen; \"Frei\" = selbst eintragen; 0 = photopisch)",
                         bg=BG, fg="#666", font=("Helvetica", 9)).pack(side="left", padx=(2, 0))
                e_L.bind("<Return>",   self._wahrnehmung_neu)
                e_L.bind("<FocusOut>", self._wahrnehmung_neu)
                for _e in (e_v,):
                    _e.bind("<Return>",   self._wahrnehmung_neu)
                    _e.bind("<FocusOut>", self._wahrnehmung_neu)

                self._wn_readout = tk.Label(frm, text="", bg=BG, fg="#222",
                                            font=("Helvetica", 9), justify="left",
                                            anchor="w")
                self._wn_readout.pack(side="top", fill="x", padx=8, pady=(2, 0))

                fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)

                def _wn_klick(ev, _ax=ax, _self=self):
                    # Klick ins Diagramm setzt die Ablese-Vergrößerung
                    if ev.inaxes is _ax and ev.xdata is not None and ev.button == 1:
                        _self._wn_V_var.set(str(int(round(float(ev.xdata)))))
                        _self._wahrnehmung_neu()
                canvas.mpl_connect("button_press_event", _wn_klick)
                setattr(self, attr, (fig, ax, canvas))
            elif attr == "_tab_blende":
                # v7.2.4: Blendeneingabe (Abblenden des Spiegels) direkt in diesem Tab
                _bf = tk.Frame(frm, bg=BG); _bf.pack(side="top", fill="x", padx=8, pady=(6, 0))
                tk.Label(_bf, text="Blende D =", bg=BG, fg="#1565c0",
                         font=("Helvetica", 9, "bold")).pack(side="left")
                _eb = tk.Entry(_bf, textvariable=self.var_blende, width=7,
                               font=("Helvetica", 10), justify="right", relief="solid", bd=1)
                _eb.pack(side="left", padx=(4, 2))
                _eb.bind("<Return>", self._param_changed); _eb.bind("<KP_Enter>", self._param_changed)
                tk.Label(_bf, text="mm   (Enter; max. Öffnung − 10 mm)", bg=BG, fg="#666",
                         font=("Helvetica", 8)).pack(side="left", padx=(2, 0))
                fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=96)   # v7.3.1: nebeneinander
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG); ax2.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, ax2, canvas))

            elif attr == "_tab_foucault":
                # ── Steuerleiste oben ──────────────────────────────────────
                ctrl = tk.Frame(frm, bg=BG)
                ctrl.pack(side="top", fill="x", padx=8, pady=(6, 2))

                tk.Label(ctrl, text="Zonen:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._foucault_n = tk.IntVar(value=4)
                for n in [4, 6, 8]:
                    tk.Radiobutton(ctrl, text=str(n), variable=self._foucault_n,
                                   value=n, bg=BG, activebackground=BG,
                                   selectcolor=BG2, font=("Helvetica", 9),
                                   command=self._foucault_eingabe_neu
                                   ).pack(side="left", padx=4)

                self._foucault_rand = tk.BooleanVar(value=False)
                tk.Checkbutton(ctrl, text="Randzone teilen (innere + äußere Hälfte messen, n+1 Zonen)",
                               variable=self._foucault_rand, bg=BG, activebackground=BG,
                               selectcolor=BG2, font=("Helvetica", 9),
                               command=self._foucault_eingabe_neu).pack(side="left", padx=(14, 4))

                tk.Label(ctrl, text="    λ = 550 nm (fix)", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left", padx=8)
                self._foucault_lam = tk.DoubleVar(value=550.0)

                # Zeile 2: Auswertungs-Modus + Status
                ctrl2 = tk.Frame(frm, bg=BG)
                ctrl2.pack(side="top", fill="x", padx=8, pady=(0, 2))

                tk.Label(ctrl2, text="Auswertung:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._foucault_use_direct = tk.BooleanVar(value=False)
                tk.Radiobutton(ctrl2,
                               text="Polynom-Fit  (glatte Flaeche)",
                               variable=self._foucault_use_direct,
                               value=False, bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#1565c0",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=6)
                tk.Radiobutton(ctrl2,
                               text="Hybrid  (Zonenfehler)",
                               variable=self._foucault_use_direct,
                               value=True, bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#6a1b9a",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=6)

                self._foucault_werkstatt = tk.BooleanVar(value=True)
                tk.Checkbutton(ctrl2,
                               text="🔧 Werkstatt-Ansicht (wie FigureXP)",
                               variable=self._foucault_werkstatt,
                               bg=BG, activebackground=BG, selectcolor=BG2,
                               fg="#E65100", font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=6)

                tk.Label(ctrl2, text="  Grad:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left", padx=(8, 0))
                self._foucault_grad_modus = tk.StringVar(value="bic")
                tk.Radiobutton(ctrl2,
                               text="BIC (auto)",
                               variable=self._foucault_grad_modus,
                               value="bic", bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#555",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=4)
                tk.Radiobutton(ctrl2,
                               text="Fest (n//2)",
                               variable=self._foucault_grad_modus,
                               value="fest", bg=BG, activebackground=BG,
                               selectcolor=BG2, fg="#555",
                               font=("Helvetica", 9),
                               command=self._foucault_aktualisieren
                               ).pack(side="left", padx=4)

                self._foucault_status = tk.Label(ctrl2, text="", bg=BG,
                                                  font=("Helvetica", 9, "bold"))
                self._foucault_status.pack(side="left", padx=12)

                # ── Zonenradien-Info ──────────────────────────────────────
                self._foucault_info = tk.Label(frm, text="", bg=BG, fg="#555",
                                               font=("Helvetica", 8), anchor="w")
                self._foucault_info.pack(side="top", fill="x", padx=8)

                # ── ΔF-Eingabefelder (dynamisch, max 8 Zonen) ─────────────
                eingabe_frm = tk.Frame(frm, bg=BG)
                eingabe_frm.pack(side="top", fill="x", padx=8, pady=(4, 2))
                self._foucault_eingabe_frm = eingabe_frm
                self._foucault_df_vars = []   # wird in _foucault_eingabe_neu befüllt

                # ── Diagramm ──────────────────────────────────────────────
                fig, (ax_wf, ax_info) = plt.subplots(
                    1, 2, figsize=(9.0, 4.2), dpi=96,
                    gridspec_kw={"width_ratios": [3, 2]})
                fig.patch.set_facecolor(BG)
                ax_wf.set_facecolor(BG); ax_info.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax_wf, ax_info, canvas))

                # Eingabefelder initial aufbauen
                self._foucault_eingabe_neu()

            elif attr == "_tab_vmax":
                # ── Steuerleiste ──────────────────────────────────────────
                ctrlv = tk.Frame(frm, bg=BG)
                ctrlv.pack(side="top", fill="x", padx=8, pady=(6, 2))

                tk.Label(ctrlv, text="Objekt:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_objekt = tk.StringVar(value=OBJEKT_HELLIGKEITEN[1][0])
                objekt_namen = [n for n, _ in OBJEKT_HELLIGKEITEN]
                om = tk.OptionMenu(ctrlv, self._vmax_objekt, *objekt_namen,
                                   command=lambda *_: self._vmax_objekt_gewaehlt())
                om.config(bg=BG, fg="#222", font=("Helvetica", 9),
                          highlightthickness=0)
                om.pack(side="left", padx=(4, 12))

                tk.Label(ctrlv, text="L₀ [cd/m²]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_L0 = tk.DoubleVar(value=600.0)
                tk.Entry(ctrlv, textvariable=self._vmax_L0, width=8,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 12))
                tk.Button(ctrlv, text="↻ Update", font=("Helvetica", 8),
                          command=self._aktualisieren
                          ).pack(side="left", padx=(0, 4))

                ctrlv2 = tk.Frame(frm, bg=BG)
                ctrlv2.pack(side="top", fill="x", padx=8, pady=(0, 4))

                tk.Label(ctrlv2, text="Seeing FWHM [\"]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_seeing = tk.DoubleVar(value=1.5)
                tk.Entry(ctrlv2, textvariable=self._vmax_seeing, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 12))

                tk.Label(ctrlv2, text="Augenpupille [mm]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_D_auge = tk.DoubleVar(value=6.0)
                tk.Entry(ctrlv2, textvariable=self._vmax_D_auge, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 12))

                tk.Label(ctrlv2, text="Helligkeitsschwelle [cd/m²]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmax_L_schwelle = tk.DoubleVar(value=1.0)
                tk.Entry(ctrlv2, textvariable=self._vmax_L_schwelle, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 4))
                tk.Label(frm, text=ANTONIADI_HINWEIS, bg=BG, fg="#666",
                         font=("Helvetica", 8)).pack(side="top", anchor="w", padx=8)

                # ── Diagramm ──────────────────────────────────────────────
                fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

            elif attr == "_tab_vmax_D":
                # ── Steuerleiste ──────────────────────────────────────────
                ctrld = tk.Frame(frm, bg=BG)
                ctrld.pack(side="top", fill="x", padx=8, pady=(6, 4))

                tk.Label(ctrld, text="Seeing-Werte [\"] (kommagetrennt):",
                         bg=BG, fg="#555", font=("Helvetica", 9)
                         ).pack(side="left")
                self._vmaxD_seeings = tk.StringVar(value="0.4, 0.8, 1.5, 2.5")
                tk.Entry(ctrld, textvariable=self._vmaxD_seeings, width=24,
                         font=("Helvetica", 9)).pack(side="left", padx=(4, 12))

                tk.Label(ctrld, text="D-Bereich [mm]:", bg=BG, fg="#555",
                         font=("Helvetica", 9)).pack(side="left")
                self._vmaxD_Dmin = tk.DoubleVar(value=50.0)
                tk.Entry(ctrld, textvariable=self._vmaxD_Dmin, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 2))
                tk.Label(ctrld, text="–", bg=BG).pack(side="left")
                self._vmaxD_Dmax = tk.DoubleVar(value=500.0)
                tk.Entry(ctrld, textvariable=self._vmaxD_Dmax, width=5,
                         font=("Helvetica", 9)).pack(side="left", padx=(2, 4))
                tk.Label(frm, text=ANTONIADI_HINWEIS, bg=BG, fg="#666",
                         font=("Helvetica", 8)).pack(side="top", anchor="w", padx=8)

                # ── Diagramm ──────────────────────────────────────────────
                fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

            elif attr == "_tab_vergleich":
                self._vg_vars = {}
                def _zeile(felder):
                    fr = tk.Frame(frm, bg=BG)
                    fr.pack(side="top", fill="x", padx=8, pady=(4, 0))
                    for lbl, key, val, hint in felder:
                        tk.Label(fr, text=lbl, bg=BG, fg="#1565c0",
                                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 3))
                        v = tk.StringVar(value=val); self._vg_vars[key] = v
                        e = tk.Entry(fr, textvariable=v, width=6, font=("Helvetica", 9))
                        e.pack(side="left")
                        e.bind("<Return>",   self._vergleich_neu)
                        e.bind("<FocusOut>", self._vergleich_neu)
                        if key in ("D1", "S1", "N"):      # v7.2.1: vom Hauptspiegel übernommen
                            e.config(state="readonly")
                        tk.Label(fr, text=hint, bg=BG, fg="#666",
                                 font=("Helvetica", 8)).pack(side="left", padx=(2, 12))
                _zeile([("Spiegel 1 (= Hauptspiegel)  D =", "D1", "200", "mm"), ("S =", "S1", "1.00", ""),
                        ("f/D =", "N", "5", "(Hauptspiegel; gilt auch für Spiegel 2)")])
                _zeile([("Spiegel 2 (Vergleich)  D =", "D2", "200", "mm"), ("S =", "S2", "1.00", "(Fangspiegel automatisch)")])

                _zeile([("V =", "V", "200", "×"), ("Seeing", "seeing", "1.5", "″ (0 = aus; Antoniadi I<0,5 · II 0,5–1 · III 1–2 · IV 2–3 · V>3)"),
                        ("Referenz: idealer", "refD", "200", "mm"),
                        ("Okularauszug", "buet", "50", "mm"),
                        ("Bildfeld", "bild", "10", "mm"),
                        ("Augenpupille", "pupille", "0", "mm (0 = aus; Mond 2–3, Planet 3–5, Deep-Sky 6–7)")])
                fr_wf = tk.Frame(frm, bg=BG); fr_wf.pack(side="top", fill="x", padx=8, pady=(4, 0))
                fr_wf2 = tk.Frame(frm, bg=BG); fr_wf2.pack(side="top", fill="x", padx=8, pady=(2, 0))
                self._vg_wf1 = tk.BooleanVar(value=False)
                self._vg_wf2 = tk.BooleanVar(value=False)
                for _v, _t in [(self._vg_wf1, "Spiegel 1 aus Foucault-Test (Wellenfront)"),
                               (self._vg_wf2, "Spiegel 2 aus Foucault-Test (Wellenfront)")]:
                    tk.Checkbutton(fr_wf, text=_t, variable=_v, bg=BG, fg="#6a1b9a",
                                   activebackground=BG, selectcolor=BG2,
                                   font=("Helvetica", 9, "bold"),
                                   command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_sph1 = tk.BooleanVar(value=False)
                self._vg_sph2 = tk.BooleanVar(value=False)
                for _v, _t in [(self._vg_sph1, "Spiegel 1 sphärisch (aus D, f/D)"),
                               (self._vg_sph2, "Spiegel 2 sphärisch (aus D, f/D)")]:
                    tk.Checkbutton(fr_wf2, text=_t, variable=_v, bg=BG, fg="#6a1b9a",
                                   activebackground=BG, selectcolor=BG2,
                                   font=("Helvetica", 9, "bold"),
                                   command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_auge = tk.BooleanVar(value=True)
                tk.Checkbutton(fr_wf, text="Bild mit Augen-Filter bei V", variable=self._vg_auge,
                               bg=BG, fg="#1565c0", activebackground=BG, selectcolor=BG2,
                               font=("Helvetica", 9, "bold"),
                               command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_mass = tk.BooleanVar(value=True)
                tk.Checkbutton(fr_wf2, text="Augenmaßstab (Planet wächst mit V)", variable=self._vg_mass,
                               bg=BG, fg="#1565c0", activebackground=BG, selectcolor=BG2,
                               font=("Helvetica", 9, "bold"),
                               command=self._vergleich_neu).pack(side="left", padx=(0, 14))
                self._vg_readout = tk.Label(frm, text="", bg=BG, fg="#222",
                                            font=("Helvetica", 10), justify="left",
                                            anchor="w")
                self._vg_readout.pack(side="top", fill="x", padx=8, pady=(4, 0))
                fig = plt.figure(figsize=(11.0, 7.4), dpi=96)
                fig.patch.set_facecolor(BG)
                ax1, ax2, _axb, _axw = vergleich_achsen(fig)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax1, ax2, canvas))

            elif attr == "_tab_fangspiegel":
                fsr = tk.Frame(frm, bg=BG)
                fsr.pack(side="top", fill="x", padx=8, pady=(6, 0))
                self._fs_vars = {}
                for lbl, key, val, hint in [
                        ("Okularauszug (Fokus über Tubus)", "buet", "50", "mm"),
                        ("Tubus-Ø (0 = aus Okular)", "tubus", "0", "mm"),
                        ("100%-Bildfeld", "bild", "10", "mm  (visuell 10, Foto 20–25)"),
                        ("Q_vis bei V =", "V", "150", "×"),
                        ("Hutring-Ø (0 = Tubus-Ø)", "hut", "0", "mm"),
                        ("Übersichtsokular f", "fe", "30", "mm"),
                        ("AFOV", "afov", "70", "°"),
                        ("OAZ-Rohr Innen-Ø", "oazd", "50", "mm"),
                        ("OAZ-Rohr Länge", "oazl", "80", "mm")]:
                    if key == "hut":
                        fsr = tk.Frame(frm, bg=BG)
                        fsr.pack(side="top", fill="x", padx=8, pady=(2, 0))
                    tk.Label(fsr, text=lbl, bg=BG, fg="#6a1b9a",
                             font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 3))
                    v = tk.StringVar(value=val)
                    self._fs_vars[key] = v
                    e = tk.Entry(fsr, textvariable=v, width=5, font=("Helvetica", 9))
                    e.pack(side="left")
                    e.bind("<Return>",   self._fangspiegel_neu)
                    e.bind("<FocusOut>", self._fangspiegel_neu)
                    tk.Label(fsr, text=hint, bg=BG, fg="#666",
                             font=("Helvetica", 8)).pack(side="left", padx=(2, 10))
                self._fs_readout = tk.Label(frm, text="", bg=BG, fg="#222",
                                            font=("Helvetica", 10), justify="left",
                                            anchor="w")
                self._fs_readout.pack(side="top", fill="x", padx=8, pady=(4, 0))
                fig = plt.figure(figsize=(11.0, 5.0), dpi=96)
                fig.patch.set_facecolor(BG)
                ax = fig.add_subplot(1, 2, 1); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

            elif attr == "_tab_gitter":
                gtr = tk.Frame(frm, bg=BG)
                gtr.pack(side="top", fill="x", padx=8, pady=(6, 0))
                self._gt_vars = {}
                for lbl, key, val, hint in [
                        ("Ringbreite", "b", "50", "mm"),
                        ("Ringdicke", "rd", "15", "mm"),
                        ("Loch–Ringfläche", "lr", "25", "mm"),
                        ("Spiegel unter Ring-OK", "st", "0", "mm (0 = bündig)"),
                        ("Achse über Hutring", "a", "40", "mm"),
                        ("Ring-Innen-Ø", "di", "0", "mm (0 = Fangspiegel-Rechner)"),
                        ("Lochabstand h", "h", "0", "mm (0 = auto)"),
                        ("Teilkreis-Ø", "d", "0", "mm (0 = Innen-Ø + Ringbreite)"),
                        ("Punkte je Ring n", "n", "3", ""),
                        ("Versatz k", "k", "1", "(0 senkrecht, 1 Nachbar)"),
                        ("Abzug je Ende", "z", "25", "mm (Gabelkopf)")]:
                    tk.Label(gtr, text=lbl, bg=BG, fg="#6a1b9a",
                             font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 3))
                    v = tk.StringVar(value=val)
                    self._gt_vars[key] = v
                    e = tk.Entry(gtr, textvariable=v, width=5, font=("Helvetica", 9))
                    e.pack(side="left")
                    e.bind("<Return>",   self._gitter_neu)
                    e.bind("<FocusOut>", self._gitter_neu)
                    tk.Label(gtr, text=hint, bg=BG, fg="#666",
                             font=("Helvetica", 8)).pack(side="left", padx=(2, 10))
                self._gt_kreuz = tk.BooleanVar(value=True)
                tk.Checkbutton(frm, text="gegenläufige Stangen zusätzlich (Kreuz)", variable=self._gt_kreuz,
                               bg=BG, fg="#6a1b9a", activebackground=BG, selectcolor=BG2,
                               font=("Helvetica", 9, "bold"), command=self._gitter_neu
                               ).pack(side="top", anchor="w", padx=8)
                self._gt_3d = tk.BooleanVar(value=False)
                tk.Checkbutton(frm, text="3D-Ansicht (maßstäblich; Spinne, Blende, OAZ geschätzt)", variable=self._gt_3d,
                               bg=BG, fg="#6a1b9a", activebackground=BG, selectcolor=BG2,
                               font=("Helvetica", 9, "bold"), command=self._gitter_neu
                               ).pack(side="top", anchor="w", padx=8)
                self._gt_bau = tk.StringVar(value="zigzag")
                _rb = tk.Frame(frm, bg=BG); _rb.pack(side="top", anchor="w", padx=8)
                tk.Label(_rb, text="Bauart:", bg=BG, fg="#6a1b9a", font=("Helvetica", 9, "bold")).pack(side="left")
                for _txt, _val in (("Zigzag (3 Ankerpunkte, 6 Stangen)", "zigzag"), ("TripleX (gekreuzte Paare)", "triplex"),
                                   ("frei (n, k, Kreuz)", "frei")):
                    tk.Radiobutton(_rb, text=_txt, value=_val, variable=self._gt_bau, bg=BG, fg="#6a1b9a",
                                   activebackground=BG, selectcolor=BG2, font=("Helvetica", 9),
                                   command=self._gitter_bau).pack(side="left", padx=(6, 0))
                self._gt_readout = tk.Label(frm, text="", bg=BG, fg="#222",
                                            font=("Helvetica", 10), justify="left", anchor="w")
                self._gt_readout.pack(side="top", fill="x", padx=8, pady=(4, 0))
                fig = plt.figure(figsize=(11.0, 5.0), dpi=96)
                fig.patch.set_facecolor(BG)
                ax = fig.add_subplot(1, 2, 1); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

            elif attr == "_tab_zelle":
                zr = tk.Frame(frm, bg=BG); zr.pack(side="top", fill="x", padx=8, pady=(6, 0))
                self._zl_n = tk.IntVar(value=6)
                tk.Label(zr, text="Auflagepunkte:", bg=BG, fg="#6a1b9a", font=("Helvetica", 9, "bold")).pack(side="left")
                for _n in (3, 6, 9, 18):
                    tk.Radiobutton(zr, text=str(_n), value=_n, variable=self._zl_n, bg=BG, fg="#6a1b9a",
                                   activebackground=BG, selectcolor=BG2, font=("Helvetica", 9),
                                   command=self._zelle_neu).pack(side="left", padx=(4, 0))
                self._zl_t = tk.StringVar(value="25")
                self._zl_r = tk.StringVar(value="0")
                for _lbl, _var, _hint in (("Dicke", self._zl_t, "mm (Standard 25)"), ("Radius r/R", self._zl_r, "(0 = Optimum, nur 3/6)")):
                    tk.Label(zr, text=_lbl, bg=BG, fg="#6a1b9a", font=("Helvetica", 9, "bold")).pack(side="left", padx=(12, 3))
                    _e = tk.Entry(zr, textvariable=_var, width=6, font=("Helvetica", 9)); _e.pack(side="left")
                    _e.bind("<Return>", self._zelle_neu); _e.bind("<FocusOut>", self._zelle_neu)
                    tk.Label(zr, text=_hint, bg=BG, fg="#666", font=("Helvetica", 8)).pack(side="left", padx=(2, 4))
                zr2 = tk.Frame(frm, bg=BG); zr2.pack(side="top", fill="x", padx=8, pady=(2, 0))
                tk.Label(zr2, text="Material:", bg=BG, fg="#6a1b9a", font=("Helvetica", 9, "bold")).pack(side="left")
                self._zl_mat = tk.StringVar(value=list(MATERIALIEN.keys())[0])
                _cb = ttk.Combobox(zr2, textvariable=self._zl_mat, values=list(MATERIALIEN.keys()), state="readonly", width=26)
                _cb.pack(side="left", padx=(4, 12)); _cb.bind("<<ComboboxSelected>>", self._zelle_neu)
                self._zl_ref = tk.BooleanVar(value=True)
                tk.Checkbutton(zr2, text="Brennweitenänderung abziehen (Refokus)", variable=self._zl_ref, bg=BG, fg="#6a1b9a",
                               activebackground=BG, selectcolor=BG2, font=("Helvetica", 9, "bold"),
                               command=self._zelle_neu).pack(side="left")
                fig = plt.figure(figsize=(11.0, 5.2), dpi=96)
                fig.patch.set_facecolor(BG)
                ax = fig.add_subplot(1, 1, 1)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                self._zl_text = tk.Text(frm, height=9, bg=BG, fg="#222", font=("Courier", 9), wrap="none", relief="flat")
                self._zl_text.pack(side="bottom", fill="x", padx=8, pady=(0, 4))
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

            else:
                fs = (7.5, 4.8) if attr == "_tab_beugung" else (5.6, 4.2)
                fig, ax = plt.subplots(figsize=fs, dpi=96)
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                canvas = FigureCanvasTkAgg(fig, master=frm)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                setattr(self, attr, (fig, ax, canvas))

    def _baue_grundlagen(self, frm):
        """Tab "Grundlagen" (v7.1.0): die vier klassischen Diagramme als 2×2-
        Übersicht in einer Figur; Klick auf ein Diagramm vergrößert es, ein
        weiterer Klick (oder Rechtsklick) stellt die Übersicht wieder her."""
        tk.Label(frm, text="Klick auf ein Diagramm = vergrößern / zurück",
                 bg=BG, fg="#888", font=("Helvetica", 8, "italic")
                 ).pack(side="top", anchor="w", padx=8, pady=(4, 0))
        fig = plt.figure(figsize=(9.5, 6.4), dpi=96)
        fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2, 2)
        axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
        for a in axes:
            a.set_facecolor(BG)
        canvas = FigureCanvasTkAgg(fig, master=frm)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._tab_strehl  = (fig, axes[0], canvas)
        self._tab_oeff    = (fig, axes[1], canvas)
        self._tab_deff_D  = (fig, axes[2], canvas)
        self._tab_beugung = (fig, axes[3], canvas)
        self._grund = dict(fig=fig, axes=axes, canvas=canvas, gs=gs, zoom=None)
        canvas.mpl_connect("button_press_event", self._grund_klick)

    def _grund_fonts(self):
        """Übersicht (2×2): Schriften verkleinern; vergrößerte Ansicht: normal."""
        g = self._grund
        fak = 1.0 if g["zoom"] is not None else 0.62
        axs = list(g["axes"])
        tw = getattr(self, "_oeff_twin", None)
        if tw is not None:
            axs.append(tw)
        for a in axs:
            arts = [a.title, a.xaxis.label, a.yaxis.label] + list(a.texts)
            leg = a.get_legend()
            if leg is not None:
                arts += list(leg.get_texts())
            for t in arts:
                if not hasattr(t, "_fs0"):
                    t._fs0 = t.get_fontsize()
                t.set_fontsize(max(t._fs0 * fak, 5))
            if not hasattr(a, "_ls0"):
                a._ls0 = plt.rcParams["xtick.labelsize"]
                a._ls0 = 10.0 if not isinstance(a._ls0, (int, float)) else a._ls0
            a.tick_params(labelsize=a._ls0 * (1.0 if fak == 1.0 else 0.7))

    def _grund_klick(self, ev):
        g = self._grund
        if ev.inaxes is None and g["zoom"] is None:
            return
        fig, axes, gs = g["fig"], g["axes"], g["gs"]
        if g["zoom"] is not None:            # zurück zur Übersicht
            g["zoom"] = None
            for i, a in enumerate(axes):
                a.set_subplotspec(gs[i // 2, i % 2]); a.set_visible(True)
            tw = getattr(self, "_oeff_twin", None)
            if tw is not None:
                tw.set_visible(True)
        else:
            if ev.inaxes not in axes:
                return
            k = axes.index(ev.inaxes)
            g["zoom"] = k
            for i, a in enumerate(axes):
                a.set_visible(i == k)
            tw = getattr(self, "_oeff_twin", None)
            if tw is not None:
                tw.set_visible(k == 1)
            axes[k].set_subplotspec(fig.add_gridspec(1, 1)[0, 0])
        self._grund_fonts()
        fig.tight_layout()
        g["canvas"].draw_idle()

    def _karten_grid(self, parent, karten, start_row, store, name_widgets, pad):
        is_odd_last = len(karten) % 2 == 1
        for j, (str_key, attr) in enumerate(karten):
            r    = start_row + j // 2
            last = is_odd_last and j == len(karten) - 1
            c    = 0 if last else (j % 2) * 2
            cs   = 4 if last else 2
            frm  = tk.Frame(parent, bg=BG2, relief="flat", bd=0, padx=8, pady=5)
            frm.grid(row=r, column=c, columnspan=cs, sticky="ew", padx=6, pady=2)
            lbl_name = tk.Label(frm, text=T(str_key), bg=BG2,
                                font=("Helvetica", 9), fg="#555")
            lbl_name.pack(anchor="w")
            name_widgets[str_key] = lbl_name
            lbl_val = tk.Label(frm, text="–", bg=BG2, font=("Helvetica", 13, "bold"))
            lbl_val.pack(anchor="w")
            store[attr] = lbl_val
        return start_row + math.ceil(len(karten) / 2)

    # ── Aktualisierung ────────────────────────────────────────────────────────

    def _param_changed(self, _=None):
        """Werte klemmen und Neuberechnung starten (nur auf Enter)."""
        try:                                   # Blende (Tab "Blende"): 50 mm ... 600 mm
            _b = self.var_blende.get()
            self.var_blende.set(min(max(_b, 50.0), 600.0))
        except tk.TclError:
            self.var_blende.set(150.0)
        for key, (var, mn, mx) in self._param_vars.items():
            try:
                v = var.get()
                if v < mn: var.set(mn)
                elif v > mx: var.set(mx)
            except tk.TclError:
                var.set(mn)
        # laufenden Job abbrechen und sofort neu starten
        if getattr(self, "_update_job", None):
            self.after_cancel(self._update_job)
            self._update_job = None
        self._start_update()

    def _start_update(self):
        """Startet Berechnung im Hintergrund-Thread."""
        self._compute_cancel = True          # laufenden Thread abbrechen
        params = (self.var_D.get(), self.var_f.get(), self.var_lam.get(),
                  self.var_blende.get(), self.var_strehl.get())
        self._compute_cancel = False
        import threading
        threading.Thread(target=self._compute_thread,
                         args=(params,), daemon=True).start()

    def _compute_thread(self, params):
        """Alle schweren Berechnungen im Hintergrund (kein tkinter)."""
        try:
            D, f, lam, D_bl_raw, S_pct = params
            D_bl = min(D_bl_raw, D - 10)

            r_real       = berechne(D, f, lam)
            S_real       = r_real["strehl"]
            S_slide      = S_pct / 100.0          # v7.2.2: Eingabe = Ziel-Strehl in %, <= Kugel gilt als Kugel
            S_slide      = min(max(S_slide, S_real), 1.0)
            Deff_k_slide = D * S_slide**0.25
            Vk           = v_kritisch(D, f, lam, EYE_RES)
            Vk_slide     = v_kritisch_strehl(D, S_slide)
            rv           = berechne_vergr(D, f, lam, 150.0, EYE_RES)
            Vk_naeh      = Vk["Vk_sph"]
            Vk_blur      = v_krit_blur_direkt(D, f, EYE_RES)
            Vk_exakt     = v_krit_aus_qvis(D, f, lam, rel_schwelle=0.90)
            Vk_halb      = v_halb_exakt(D, f, lam)
            Wb_s         = _Wb_schieber(D, f, lam, S_slide)
            Vk_slide["Vk_sph"] = v_krit_naeh(D, Wb_s)
            Vk_exakt_s   = v_krit_aus_qvis_von_Wb(D, Wb_s, lam, rel_schwelle=0.90)
            _Q_V         = (30.0, 60.0, 100.0, 150.0)
            Qp_30, Qp_60, Qp_100, Qp_150 = (perceived_quality_exakt(D, f, lam, v) for v in _Q_V)
            Qp_s         = tuple(_qvis_schieber(D, f, lam, S_slide, v) for v in _Q_V)

            if self._compute_cancel:
                return

            res = dict(D=D, f=f, lam=lam, D_bl=D_bl, S_pct=S_pct,
                       r_real=r_real, S_real=S_real, S_slide=S_slide,
                       Deff_k_slide=Deff_k_slide, Vk=Vk, Vk_slide=Vk_slide,
                       rv=rv, Vk_naeh=Vk_naeh, Vk_blur=Vk_blur,
                       Vk_exakt=Vk_exakt, Vk_halb=Vk_halb,
                       Vk_exakt_s=Vk_exakt_s,
                       Qp_30=Qp_30, Qp_60=Qp_60, Qp_100=Qp_100, Qp_150=Qp_150,
                       Qp_s=Qp_s)
            self.after(0, lambda r=res: self._aktualisieren(r))
        except Exception:
            import traceback
            msg = traceback.format_exc()
            self.after(0, lambda m=msg: self._zeige_fehler(m))

    def _baue_navigation(self):
        """v7.2.1: Tab-Auswahl als Knöpfe unter den Spiegel-Parametern
        (die Reiter der Notebooks sind ausgeblendet)."""
        left = self._nav_left
        frm = tk.Frame(left, bg=BG)
        frm.grid(row=self._nav_row, column=0, columnspan=3, sticky="ew",
                 padx=10, pady=(10, 0))
        self._nav_btns = {}      # tab_key -> (Button, Text-Key)
        self._nav_heads = {}     # gruppen_key -> Label
        for gkey, keys in self._tab_gruppen:
            if len(keys) > 1:
                h = tk.Label(frm, text=T(gkey).strip(), bg=BG, fg="#888",
                             font=("Helvetica", 8, "bold"), anchor="w")
                h.pack(fill="x", pady=(8, 0))
                self._nav_heads[gkey] = h
                for k in keys:
                    self._nav_knopf(frm, k, k)
            else:
                self._nav_knopf(frm, keys[0], gkey)
        self._nav_aktiv_markieren()

    def _nav_knopf(self, parent, tab_key, text_key):
        b = tk.Button(parent, text=T(text_key).strip(), anchor="w",
                      font=("Helvetica", 10), relief="flat", bd=0,
                      bg=BG2, activebackground="#d0d7e5", padx=10, pady=3,
                      command=lambda k=tab_key: self._tab_waehlen(k))
        b.pack(fill="x", pady=1)
        self._nav_btns[tab_key] = (b, text_key)

    def _nav_aktiv_markieren(self):
        akt = self._aktiver_tab()
        for k, (b, _) in self._nav_btns.items():
            if k == akt:
                b.config(bg=ACC, fg="white", activebackground=ACC, activeforeground="white")
            else:
                b.config(bg=BG2, fg="black", activebackground="#d0d7e5", activeforeground="black")

    def _on_tab_changed(self, _=None):
        """Zeichnet den neu sichtbaren Tab (immer, nicht nur wenn dirty)."""
        if hasattr(self, "_nav_btns"):
            self._nav_aktiv_markieren()
        if not self._last_params:
            return
        active = self._aktiver_tab()
        if active:
            self._dirty_tabs.discard(active)
            self._diagramm_fuer_tab(active, self._last_params)

    def _aktiver_tab(self) -> str:
        """Gibt den Tab-Key des aktuell sichtbaren Tabs zurück."""
        try:
            sel = self._nb.select()
            if not sel:
                return self._tab_keys[0]
            gi = self._nb.index(sel)
            keys = self._tab_gruppen[gi][1]
            sub = self._grp_nb[gi][1]
            if sub is None:
                return keys[0]
            return keys[sub.index(sub.select())]
        except Exception:
            return self._tab_keys[0] if self._tab_keys else ""

    def _tab_waehlen(self, key: str):
        """Wählt einen Tab per Key (Gruppe + Unter-Reiter)."""
        for gi, (_, keys) in enumerate(self._tab_gruppen):
            if key in keys:
                self._nb.select(gi)
                sub = self._grp_nb[gi][1]
                if sub is not None:
                    sub.select(keys.index(key))
                self.update_idletasks()
                return

    def _diagramm_fuer_tab(self, tab_key: str, r: dict):
        """Zeichnet genau das Diagramm für tab_key."""
        D, f, lam   = r["D"], r["f"], r["lam"]
        S_real      = r["S_real"]
        S_slide     = r["S_slide"]
        Deff_k_slide= r["Deff_k_slide"]
        D_bl        = r["D_bl"]
        r_real      = r["r_real"]
        if tab_key == "tab_grund":
            self._diagramm_strehl(D, f, lam, r_real["N"], S_real, S_slide)
            self._diagramm_oeffnung(D, f, lam, r_real["N"], r_real["Deff_k"],
                                    S_sph=S_real, Dk_slide=Deff_k_slide)
            self._diagramm_deff_D(D, r_real["N"], S_slide)
            self._diagramm_beugung(D, f)
            self._grund_fonts()
            self._grund["fig"].tight_layout()
            self._grund["canvas"].draw_idle()
        elif tab_key == "tab_mtf":
            self._diagramm_mtf(D, f, lam, S_slide)
        elif tab_key == "tab_wahrnehmung":
            self._diagramm_wahrnehmung(D, f, lam, S_slide=S_slide, S_real=S_real)
        elif tab_key == "tab_blende":
            self._diagramm_blende(D, f, lam, D_bl)
        elif tab_key == "tab_foucault":
            if hasattr(self, "_foucault_df_vars"):
                D_prev = getattr(self, "_foucault_D_prev", None)
                if D_prev != D:
                    self._foucault_D_prev = D
                    self._foucault_eingabe_neu()
                else:
                    self._foucault_aktualisieren()
        elif tab_key == "tab_vmax":
            self._diagramm_vmax(D, f, S_slide)
            self._diagramm_vmax_D()
        elif tab_key == "tab_vmax_D":
            self._diagramm_vmax_D()
        elif tab_key == "tab_fangspiegel":
            self._diagramm_fangspiegel(D, f, lam, S_slide, S_real)
        elif tab_key == "tab_gitter":
            self._diagramm_gitter(D, f)
        elif tab_key == "tab_zelle":
            self._diagramm_zelle(D, lam)
        elif tab_key == "tab_vergleich":
            self._diagramm_vergleich(lam)

    def _vg_eingaben(self):
        """Eingaben des Vergleich-Tabs (Komma/Punkt, Fallback bei Fehler)."""
        std = dict(D1=200, S1=1.0, D2=200, S2=1.0, N=5, V=200, seeing=1.5,
                   refD=200, buet=50, bild=10, pupille=0.0)
        # v7.2.1: Spiegel 1 und f/D sind der Hauptspiegel (Parameter links, S = Schieberwert)
        try:
            _D = float(self.var_D.get()); _f = float(self.var_f.get())
            _lp = getattr(self, "_last_params", None) or {}
            _S = _lp.get("S_slide")
            if _S is None:
                _S = _strehl_exakt(_wellenfronten(_D, _f, 550.0)[2])
            self._vg_vars["D1"].set(f"{_D:g}")
            self._vg_vars["S1"].set(f"{float(_S):.4f}")
            self._vg_vars["N"].set(f"{_f / _D:.4f}")
        except Exception:
            pass
        out = {}
        for k, v in self._vg_vars.items():
            try:
                x = float(v.get().replace(",", ".").strip())
                if k in ("D1", "D2", "refD", "N", "V") and x <= 0: raise ValueError
                if k in ("S1", "S2"): x = min(max(x, 0.10 if k == 'S2' else 0.05), 1.0)
                if x < 0: raise ValueError
            except ValueError:
                x = std[k]
            if k not in ("S1", "S2", "N", "seeing", "pupille"):
                x = int(round(x))
            out[k] = x
            v.set(f"{x:.2f}" if k in ("S1", "S2") else f"{x:.2f}" if k == "N" else f"{x:g}")
        return out

    def _vergleich_neu(self, *_):
        try:
            self._diagramm_vergleich(getattr(self, "_last_params", {}).get("lam", 550.0))
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _diagramm_vergleich(self, lam):
        """v7.0.0-beta: Q_abs-Vergleich zweier Spiegel + D×S-Karte; optional
        mit der Wellenfront aus dem Foucault-Test (Spiegel 1 und/oder 2)."""
        fig, ax1, ax2, canvas = self._tab_vergleich
        e = self._vg_eingaben()
        notiz = []
        wf = {}
        kz = getattr(self, "_foucault_letzte_kz", None)
        if (self._vg_wf1.get() or self._vg_wf2.get()) and kz is None \
                and hasattr(self, "_foucault_df_vars"):
            try:
                self._foucault_aktualisieren()
                kz = getattr(self, "_foucault_letzte_kz", None)
            except Exception:
                kz = None
        if kz is None:
            if self._vg_wf1.get() or self._vg_wf2.get():
                notiz.append("Keine Foucault-Daten vorhanden — erst im Foucault-Tab Messwerte eingeben.")
            self._vg_wf1.set(False); self._vg_wf2.set(False)
        else:
            Df = float(getattr(self, "_foucault_letzte_D", e["D1"]))
            wfd = dict(rho=kz["rho_f"], W=kz["W_c"], lam=kz.get("lam_nm") or 550.0)
            for k, var in (("1", self._vg_wf1), ("2", self._vg_wf2)):
                if var.get():
                    wf[k] = wfd
                    e["D" + k] = Df
                    e["S" + k] = float(kz["S_exakt"])
                    self._vg_vars["D" + k].set(f"{Df:g}")
                    self._vg_vars["S" + k].set(f"{kz['S_exakt']:.2f}")
            if wf:
                try:                      # f/D des gemessenen Spiegels aus den Hauptreglern
                    e["N"] = float(self.var_f.get()) / float(self.var_D.get())
                    self._vg_vars["N"].set(f"{e['N']:.2f}")
                except Exception:
                    pass
                if not kz.get("kennzahlen_zuverlaessig", True):
                    notiz.append("⚠ Foucault-Fit unzuverlässig (Polynom-Modus, Residuum hoch) — "
                                 "Wellenfront ist geglättet, Ergebnis optimistisch! Hybrid-Modus (n≥6) verwenden.")
        if e["S1"] < 0.09 and not self._vg_sph1.get():   # Inversion S->Wb versagt bei S < 0,082
            self._vg_sph1.set(True)
            notiz.append("Spiegel 1: Strehl < 0,09 – es wird die sphärische Wellenfront aus D und f/D verwendet.")
        fig.clf()
        ax1, ax2, axb, axw = vergleich_achsen(fig, extra=bool(wf) or self._vg_sph1.get() or self._vg_sph2.get())
        for a in (ax1, ax2): a.set_facecolor(BG)
        self._tab_vergleich = (fig, ax1, ax2, canvas)
        p = dict(e, refS=1.0, tubus=0.0, lam=lam,
                 wf1=wf.get("1"), wf2=wf.get("2"), auge=self._vg_auge.get(),
                 sph1=self._vg_sph1.get(), sph2=self._vg_sph2.get(),
                 augenmass=self._vg_mass.get())
        zeilen = vergleich_zeichnen(ax1, ax2, p, axb, axw)
        self._vg_readout.config(text="\n".join(notiz + zeilen))
        fig.tight_layout()
        canvas.draw_idle()

    def _fs_eingaben(self):
        """Ganzzahlige Eingaben des Fangspiegel-Tabs (Komma/Punkt, Fallback)."""
        std = dict(buet=50, tubus=0, bild=10, V=150, hut=0, fe=30, afov=70, oazd=50, oazl=80)
        out = {}
        for k, v in self._fs_vars.items():
            try:
                x = int(round(float(v.get().replace(",", ".").strip())))
                if x < 0 or (k in ("V", "fe", "afov", "oazd", "oazl") and x < 1):
                    raise ValueError
            except ValueError:
                x = std[k]
            out[k] = x
            v.set(str(x))
        return out

    def _gt_eingaben(self):
        """Eingaben des Gitterrohr-Tabs (Komma/Punkt, Fallback bei Fehler)."""
        std = dict(b=50.0, rd=15.0, lr=25.0, st=0.0, a=40.0, di=0.0, h=0.0, d=0.0, n=6, k=1, z=25.0)
        out = {}
        for key, v in self._gt_vars.items():
            try:
                x = float(v.get().replace(",", ".").strip())
                if x < 0 or (key == "n" and x < 3):
                    raise ValueError
                if key in ("n", "k"):
                    x = int(round(x))
            except ValueError:
                x = std[key]
            out[key] = x
            v.set(str(x if key in ("n", "k") else f"{x:g}"))
        return out

    def _gitter_bau(self):
        """Bauart gewechselt: Standard-n setzen (Zigzag 3, sonst 6)."""
        self._gt_vars["n"].set("3" if self._gt_bau.get() == "zigzag" else "6")
        self._gitter_neu()

    def _gitter_neu(self, *_):
        res = getattr(self, "_last_params", None)
        if res is None:
            return
        try:
            self._diagramm_fuer_tab("tab_gitter", res)
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _zelle_neu(self, *_):
        res = getattr(self, "_last_params", None)
        if res is None:
            return
        try:
            self._diagramm_fuer_tab("tab_zelle", res)
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _diagramm_zelle(self, D, lam):
        """Spiegelzelle: Auflagepunkte 3/6/9/18, Durchbiegung, Vergleich (neu in v9.0.0)."""
        fig, ax, canvas = self._tab_zelle
        def _zahl(v, std):
            try:
                x = float(v.get().replace(",", ".").strip())
                return x if x >= 0 else std
            except ValueError:
                return std
        n = int(self._zl_n.get()); t = _zahl(self._zl_t, 25.0)
        t = t if t > 0 else 25.0
        mat = self._zl_mat.get(); ref = bool(self._zl_ref.get())
        p = None
        if n in (3, 6):
            rr = _zahl(self._zl_r, 0.0)
            if 0.1 <= rr <= 0.92:
                p = dict(r=rr)
        d = zelle_zeichnen(fig, D, t, n, mat, ref, lam, p)
        self._zl_text.config(state="normal"); self._zl_text.delete("1.0", "end")
        self._zl_text.insert("1.0", zelle_text(d, t, mat)); self._zl_text.config(state="disabled")
        canvas.draw_idle()

    def _diagramm_gitter(self, D, f):
        """Gitterrohr-Rechner: Stangenlänge, Freiraum zum Lichtkegel, Skizze (neu in v7.8.0)."""
        fig, ax, canvas = self._tab_gitter
        fig.clf()
        _3d = bool(self._gt_3d.get()) if hasattr(self, "_gt_3d") else False
        if not _3d:
            ax1 = fig.add_subplot(1, 2, 1); ax1.set_facecolor(BG)
            ax2 = fig.add_subplot(1, 2, 2); ax2.set_facecolor(BG)
            self._tab_gitter = (fig, ax1, canvas)
        e = self._gt_eingaben()
        fs = getattr(self, "_fs_vars", None)
        buet = float(fs["buet"].get()) if fs else 50.0
        fe = float(fs["fe"].get()) if fs else 30.0
        afov = float(fs["afov"].get()) if fs else 70.0
        ga = gitterrohr_auto(D, f, buet, fe, afov, e["b"], e["a"], e["rd"], e["lr"], e["di"], e["st"])
        h = e["h"] if e["h"] > 0 else ga["h"]
        d = e["d"] if e["d"] > 0 else ga["d_teil"]
        g = gitterrohr_stangen(h, d / 2.0, e["n"], e["k"], -e["z"], bool(self._gt_kreuz.get()),
                               self._gt_bau.get())
        fr = gitterrohr_freiraum(h, d / 2.0, g["dphi"], D, f, ga["feld_halb"], ga["z0"])
        self._gt_readout.config(text=gitterrohr_text(D, f, g, fr, h, d))
        if _3d:
            gitter3d_zeichnen(fig, D, f, ga, g, h, d / 2.0, e["b"], e["rd"], e["lr"], self._gt_bau.get(),
                              dict(buet=buet, fe=fe, afov=afov))
            canvas.draw_idle()
            return
        gitterrohr_zeichnen(ax1, ax2, D, f, g, h, d / 2.0, fr, ga["z0"],
                          (ga["d_innen"] / 2.0, ga["d_innen"] / 2.0 + e["b"]),
                          dict(rt=e["rd"], z_sec=ga["z_sec"], A=ga["A"], d_fang=ga["d_fang"], tief=ga["tief"]))
        fig.tight_layout()
        canvas.draw_idle()

    def _fangspiegel_neu(self, *_):
        res = getattr(self, "_last_params", None)
        if res is None:
            return
        try:
            self._diagramm_fuer_tab("tab_fangspiegel", res)
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _diagramm_fangspiegel(self, D, f, lam, S_slide=None, S_real=None):
        """Fangspiegel-Rechner: Mindestgröße + Q_vis(V) für d_min, +10, +20 mm
        (neu in v6.12.0).  Spiegel: Strehl-Schieber, sonst Sphäre."""
        fig, ax, canvas = self._tab_fangspiegel
        fig.clf()
        ax = fig.add_subplot(1, 2, 1); ax.set_facecolor(BG)
        ax2 = fig.add_subplot(1, 2, 2); ax2.set_facecolor(BG)
        self._tab_fangspiegel = (fig, ax, canvas)
        e = self._fs_eingaben()
        if S_slide is not None:
            S_u, S_txt = S_slide, "Schieber"
        else:
            _, _, Wb_sph, _, _ = _wellenfronten(D, f, lam)
            S_u, S_txt = _strehl_exakt(Wb_sph), "Sphäre"
        r = fangspiegel_rechner(D, f, e["buet"], e["tubus"], e["bild"],
                                S_u, e["V"], lam, e["hut"], e["fe"], e["afov"], e["oazd"], e["oazl"])
        self._fs_readout.config(text=fangspiegel_text(D, f, S_u, S_txt, r, e["V"]))
        fangspiegel_zeichnen(ax, D, f, lam, S_u, S_txt, r, e["V"])
        fangspiegel_skizze(ax2, D, f, r)
        fig.tight_layout()
        canvas.draw_idle()

    def _vmax_objekt_gewaehlt(self):
        """Füllt L₀ mit dem Referenzwert des gewählten Objekt-Presets;
        bleibt danach frei überschreibbar (Preset + manuell, s. Chat)."""
        name = self._vmax_objekt.get()
        for n, L0 in OBJEKT_HELLIGKEITEN:
            if n == name:
                self._vmax_L0.set(L0)
                break
        self._aktualisieren()

    def _zeige_fehler(self, msg: str):
        """Zeigt Berechnungsfehler im Statusbereich an."""
        print("BERECHNUNGSFEHLER:\n", msg)   # immer in Terminal
        try:
            self.lbl_verdict.config(
                text="⚠ Berechnungsfehler (Details in Terminal)", fg="#c62828")
        except Exception:
            pass

    def _aktualisieren(self, res: dict | None = None):
        """UI-Update auf dem Hauptthread. Ohne res: Erststart → Thread starten."""
        if res is None:
            D        = self.var_D.get()
            f        = self.var_f.get()
            lam      = self.var_lam.get()
            D_bl_raw = self.var_blende.get()
            S_pct    = self.var_strehl.get()
            params   = (D, f, lam, D_bl_raw, S_pct)
            import threading
            threading.Thread(target=self._compute_thread,
                             args=(params,), daemon=True).start()
            return
        try:
            self._aktualisieren_ui(res)
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())

    def _aktualisieren_ui(self, res: dict):

        # ── Parameter aus Ergebnis-Dict ────────────────────────────────────
        D, f, lam   = res["D"], res["f"], res["lam"]
        D_bl        = res["D_bl"]
        r_real      = res["r_real"]
        S_real      = res["S_real"]
        S_slide     = res["S_slide"]
        Deff_k_slide= res["Deff_k_slide"]
        Vk          = res["Vk"]
        Vk_slide    = res["Vk_slide"]
        rv          = res["rv"]
        Vk_naeh     = res["Vk_naeh"]
        Vk_blur     = res["Vk_blur"]
        Vk_exakt    = res["Vk_exakt"]
        Vk_halb     = res["Vk_halb"]
        Vk_exakt_s  = res["Vk_exakt_s"]
        Qp_30, Qp_60, Qp_100, Qp_150 = res["Qp_30"], res["Qp_60"], res["Qp_100"], res["Qp_150"]
        Qs_30, Qs_60, Qs_100, Qs_150 = res["Qp_s"]
        self._last_params = res
        # ── Info-Labels neben den Eingabefeldern ──────────────────────────
        r = r_real
        Vk_naeh_s = Vk_slide["Vk_sph"]
        for key, (lbl, unit) in self._param_labels.items():
            if key == "sl_D":
                lbl.config(text=f"f/{f/D:.1f}")
            elif key == "sl_f":
                lbl.config(text=f"f/{f/D:.1f}")
            elif key == "sl_lam":
                lbl.config(text="sichtbares Licht" if 380 <= lam <= 700 else "außerhalb VIS")
            elif key == "sl_strehl":
                lbl.config(text=("= Kugel" if S_slide <= S_real + 1e-6
                                 else "= Paraboloid" if S_slide >= 0.9999
                                 else f"S = {S_slide:.3f}"))
            elif key == "sl_blende":
                lbl.config(text=f"f/{f/D_bl:.1f}")

        # ── Ergebniskarten ─────────────────────────────────────────────────
        rv2 = self._res
        r   = r_real
        _col_ex    = "#2e7d32" if S_real >= 0.95 else "#e65100" if S_real >= 0.80 else "#c62828"
        Vk_naeh_s  = Vk_slide["Vk_sph"]
        delta_ex   = Vk_exakt - Vk_naeh
        col_naeh   = "#534AB7"
        col_blur   = "#2e7d32" if Vk_blur > 50 else "#e65100" if Vk_blur > 20 else "#c62828"
        col_ex     = "#2e7d32" if abs(delta_ex) < 30 else "#e65100"
        avp        = r["aufl_verlust_pct"]

        _Vcap = max(2.0 * D, 60.0)           # v7.3.1: Obergrenze 2·D – darüber keine sinnvolle Vergrößerung
        def _vk_ex_text(Ve):
            if Ve >= _Vcap - 0.5:
                return f"> {_Vcap:.0f}×"
            return f"{Ve:.0f}×" if Ve > 30.5 else "< 30×"

        rv2["rN"].config(    text=f"f/{r['N']:.2f}")
        rv2["rWp"].config(   text=f"{r['Wp']:.4f} λ")
        rv2["rWb"].config(   text=f"{r['Wb']:.4f} λ")
        _pf = pfeiltiefe(D, f)
        rv2["rSag"].config(  text=f"{_pf['s_kugel']:.3f} mm  →  {_pf['s_parabel']:.3f} mm")
        rv2["rWrms"].config( text=f"{r['Wrms']:.5f} λ")
        rv2["rS"].config(    text=f"{S_real:.4f}  →  {S_slide:.4f}")
        rv2["rS_exakt"].config(
            text=f"{S_real:.4f}  →  {S_slide:.4f}", fg=_col_ex)
        rv2["rDeff_k"].config(
            text=f"{r['Deff_k']:.1f} → {Deff_k_slide:.1f} mm")
        rv2["rVkrit_naeh"].config(
            text=f"{Vk_naeh:.0f}×  →  {Vk_naeh_s:.0f}×"
                 f"  |  V½ exakt: {Vk_halb:.0f}×",
            fg=col_naeh)
        rv2["rVhalb"].config(text=f"{Vk_halb:.0f}×")
        rv2["rVkrit_blur"].config(
            text=(f"{Vk_blur:.0f}×  (α_blur={D**3/(128*f**2)/f*206265:.1f}\"  — unabh. von S)"
                  if Vk_blur > 1 else "< 1×  — Blur dominiert immer"),
            fg=col_blur)
        rv2["rVkrit_exakt"].config(
            text=f"{_vk_ex_text(Vk_exakt)}  →  {_vk_ex_text(Vk_exakt_s)}",
            fg=col_ex)
        rv2["rAuflVerlust"].config(
            text=f"{avp:.1f}%  (Blur {r['d_blur_as']:.1f}\")",
            fg="#c62828" if avp > 80 else "#e65100" if avp > 40 else "#2e7d32")
        rv2["rVerlustV"].config(
            text=f"{rv['verlust']:.1f} mm",
            fg="#c62828" if rv["verlust"] > 5 else "#333")

        text, farbe = beurteilung(S_slide)
        if S_slide > S_real + 1e-6:
            text = f"Mit Verbesserung (S={S_slide:.2f}):  {text}"
        self.lbl_verdict.config(text=text, fg=farbe)
        self.lbl_vkrit.config(
            text=f"⚡ Real: V_krit={Vk['Vk_sph']:.0f}×  →  "
                 f"Verbessert (S={S_slide:.2f}): V_krit={Vk_slide['Vk_sph']:.0f}×  |  "
                 f"Paraboloid: V_krit={'unbegrenzt' if Vk['Vk_para'] >= 1e8 else format(Vk['Vk_para'], '.0f') + '×'}  |  V_max={Vk['Vmax']:.0f}×",
            fg="#534AB7")

        def _qcolor(q):
            return "#2e7d32" if q >= 0.90 else "#e65100" if q >= 0.70 else "#c62828"
        rv2["rQp30"].config( text=f"{Qp_30:.3f}  →  {Qs_30:.3f}",  fg=_qcolor(Qs_30))
        rv2["rQp60"].config(text=f"{Qp_60:.3f}  →  {Qs_60:.3f}", fg=_qcolor(Qs_60))
        rv2["rQp100"].config(text=f"{Qp_100:.3f}  →  {Qs_100:.3f}", fg=_qcolor(Qs_100))
        rv2["rQp150"].config(  text=f"{Qp_150:.3f}  →  {Qs_150:.3f}", fg=_qcolor(Qs_150))

        # ── Diagramme ──────────────────────────────────────────────────────
        # Beim ersten Durchlauf (startup) alle Tabs zeichnen, danach nur
        # das sichtbare Tab — Rest wird beim Tab-Wechsel nachgeholt.
        active = self._aktiver_tab()
        alle_tab_keys = [k for _, k in [
            ("_tab_grund",       "tab_grund"),
            ("_tab_kennwerte",   "tab_kennwerte"),
            ("_tab_mtf",         "tab_mtf"),
            ("_tab_wahrnehmung", "tab_wahrnehmung"),
            ("_tab_blende",      "tab_blende"),
            ("_tab_foucault",    "tab_foucault"),
            ("_tab_vmax",        "tab_vmax"),
            ("_tab_vmax_D",      "tab_vmax_D"),
            ("_tab_fangspiegel", "tab_fangspiegel"),
            ("_tab_gitter",      "tab_gitter"),
            ("_tab_zelle",       "tab_zelle"),
            ("_tab_vergleich",   "tab_vergleich"),
        ]]
        startup = not getattr(self, "_first_draw_done", False)
        if startup:
            self._first_draw_done = True
            # Beim ersten Durchlauf nur das sichtbare Tab zeichnen;
            # alle anderen werden beim Tab-Wechsel nachgeholt (lazy).
            self._dirty_tabs = set(alle_tab_keys) - {active}
            self._diagramm_fuer_tab(active, res)
        else:
            self._dirty_tabs = set(alle_tab_keys) - {active}
            self._diagramm_fuer_tab(active, res)

    # ── Foucault-Tab Hilfsmethoden ─────────────────────────────────────────────

    def _foucault_eingabe_neu(self):
        """Eingabefelder neu aufbauen wenn Zonenanzahl geändert."""
        n = self._foucault_n.get()
        rand = bool(getattr(self, "_foucault_rand", None) and self._foucault_rand.get())
        # Alte Widgets entfernen
        for w in self._foucault_eingabe_frm.winfo_children():
            w.destroy()
        self._foucault_df_vars = []

        D = self.var_D.get()
        f = self.var_f.get()
        Rc    = 2.0 * f
        r_zon   = foucault_zonen(D, n, rand)
        r_grenz = foucault_zonengrenzen(D, n, rand)
        df_para_abs = r_zon**2 / (2.0 * Rc)        # absoluter Parabolsollwert (+r²/4f)
        df_para     = df_para_abs - df_para_abs[0]  # relativ zu Zone 1 (Referenz = 0)

        # Info-Text mit Zonenradien und Parabolsollwerten
        info = (("Randzone geteilt: Z%d = innere, Z%d = äußere Randhälfte (je halbe Fläche).   " % (n, n + 1)
                 if rand else "") +
                "Gleichflächige Zonen — Parabolspiegel-Sollwerte, relativ zu Zone 1 "
                "(+ = Messerschneide weiter vom Spiegel weg, FigureXP-Konvention):  " +
                "  |  ".join(
                    f"Z{i+1}: r={rv:.1f}mm  ΔF_soll={dp:.3f}mm"
                    for i, (rv, dp) in enumerate(zip(r_zon, df_para))) +
                "      [ΔF=0 → Kugel  |  ΔF=ΔF_soll → Parabel  |  Zone 1 = Referenz, fest 0]")
        self._foucault_info.config(text=info)

        # Kopfzeile
        headers = ["Zone", "Zonenradius [mm]", "Zonengrenze außen [mm]",
                   "ΔF_soll [mm]  (rel. Zone 1)",
                   "ΔF gemessen [mm]  (rel. Zone 1; = ΔF_soll → Parabel | = 0 → Kugel)", "Abw. [mm]"]
        widths  = [6, 16, 16, 14, 38, 10]
        for col, (hdr, w) in enumerate(zip(headers, widths)):
            tk.Label(self._foucault_eingabe_frm,
                     text=hdr, bg=BG, fg="#555",
                     font=("Helvetica", 9, "bold"), width=w).grid(
                     row=0, column=col, padx=4, pady=2)

        for i, rv in enumerate(r_zon):
            dp = df_para[i]           # Parabolsollwert für diese Zone (rel. Zone 1)
            r_innen  = r_grenz[i-1] if i > 0 else 0.0
            r_aussen = r_grenz[i]
            ist_referenz = (i == 0)
            var = tk.DoubleVar(value=0.0 if ist_referenz else round(dp, 3))
            var.trace_add("write", lambda *_: self._foucault_aktualisieren())
            self._foucault_df_vars.append(var)

            # Zone
            tk.Label(self._foucault_eingabe_frm,
                     text=f"Z{i+1}", bg=BG, fg=ACC,
                     font=("Helvetica", 9, "bold"), width=6).grid(
                     row=i+1, column=0, padx=4, pady=2)
            # Zonenradius
            tk.Label(self._foucault_eingabe_frm,
                     text=f"{rv:.2f}", bg=BG, fg="#333",
                     font=("Helvetica", 9), width=16).grid(
                     row=i+1, column=1, padx=4)
            # Zonengrenze (innen–außen, für Zonenmaske)
            tk.Label(self._foucault_eingabe_frm,
                     text=f"{r_innen:.2f} – {r_aussen:.2f}", bg=BG, fg="#333",
                     font=("Helvetica", 9), width=16).grid(
                     row=i+1, column=2, padx=4)
            # Sollwert (klickbar → Wert übernehmen, außer bei der Referenzzone)
            lbl_soll = tk.Label(self._foucault_eingabe_frm,
                                text=f"{dp:.3f}", bg=BG2, fg="#1565c0",
                                font=("Helvetica", 9, "bold"), width=14,
                                cursor="hand2" if not ist_referenz else "arrow",
                                relief="flat")
            lbl_soll.grid(row=i+1, column=3, padx=4)
            if not ist_referenz:
                lbl_soll.bind("<Button-1>",
                              lambda e, v=var, d=dp: (v.set(round(d, 3)), None))
            # Messwert-Eingabe (Zone 1 = Referenz, fest 0, nicht editierbar)
            if ist_referenz:
                tk.Label(self._foucault_eingabe_frm,
                         text="0.000  (Referenz)", bg=BG, fg="#777",
                         font=("Helvetica", 9, "italic"), width=12).grid(
                         row=i+1, column=4, padx=4, sticky="w")
            else:
                sp = tk.Spinbox(self._foucault_eingabe_frm,
                                from_=-50.0, to=50.0, increment=0.001,
                                textvariable=var, width=12, format="%.3f",
                                font=("Helvetica", 10))
                sp.grid(row=i+1, column=4, padx=4, sticky="w")
            # Abweichung vom Sollwert (wird in _foucault_aktualisieren befüllt)
            lbl_abw = tk.Label(self._foucault_eingabe_frm,
                               text="0.000", bg=BG, fg="#333",
                               font=("Helvetica", 9), width=10)
            lbl_abw.grid(row=i+1, column=5, padx=4)

        # Knopf: alle Zonen auf Parabolsollwerte zurücksetzen
        btn_frame = tk.Frame(self._foucault_eingabe_frm, bg=BG)
        btn_frame.grid(row=len(r_zon)+1, column=0, columnspan=6, pady=6, sticky="w")
        tk.Button(btn_frame, text="↺  Alle auf Parabel-Sollwerte",
                  font=("Helvetica", 9), bg=BG2,
                  command=lambda: [v.set(round(d,3))
                                   for v,d in zip(self._foucault_df_vars, df_para)]
                  ).pack(side="left", padx=4)
        tk.Button(btn_frame, text="✕  Alle auf Null (Kugel)",
                  font=("Helvetica", 9), bg=BG2,
                  command=lambda: [v.set(0.0) for v in self._foucault_df_vars]
                  ).pack(side="left", padx=4)

        self._foucault_aktualisieren()

    def _foucault_aktualisieren(self):
        """Neuberechnung und Diagramm-Update fuer Foucault-Tab."""
        try:
            D          = self.var_D.get()
            f          = self.var_f.get()
            lam        = self._foucault_lam.get()
            n          = self._foucault_n.get()
            rand       = bool(getattr(self, "_foucault_rand", None) and self._foucault_rand.get())
            # _foucault_use_direct wird erst beim Tab-Aufbau erstellt;
            # Default False (Polynom) wenn Tab noch nicht geoeffnet wurde.
            if not hasattr(self, "_foucault_use_direct"):
                self._foucault_use_direct = tk.BooleanVar(value=False)
            use_direct = self._foucault_use_direct.get()
            if not hasattr(self, "_foucault_werkstatt"):
                self._foucault_werkstatt = tk.BooleanVar(value=True)
            werkstatt_ansicht = self._foucault_werkstatt.get()
            if not hasattr(self, "_foucault_grad_modus"):
                self._foucault_grad_modus = tk.StringVar(value="bic")
            grad_modus = self._foucault_grad_modus.get()

            r_zon  = foucault_zonen(D, n, rand)
            r_gr   = foucault_zonengrenzen(D, n, rand)
            nz     = len(r_zon)                      # Zahl der Messzonen (n oder n+1)
            df_arr = np.array([v.get() for v in self._foucault_df_vars])
            df_para_abs = r_zon**2 / (2.0 * 2.0 * f)
            df_para     = df_para_abs - df_para_abs[0]
            abw         = df_arr - df_para

            for i, dv in enumerate(abw):
                for w in self._foucault_eingabe_frm.grid_slaves(row=i+1, column=5):
                    col = "#2e7d32" if abs(dv) < 0.01 else                           "#e65100" if abs(dv) < 0.1  else "#c62828"
                    w.config(text=f"{dv:+.3f}", fg=col)

            W_z = foucault_wellenfront(r_zon, df_arr, D, f, lam, r_grenz=r_gr)

            # Hybrid-Modus braucht mindestens 6 Zonen um zuverlaessig
            # zu sein: bei n=4 hat das BIC-Polynom zu viel Spielraum
            # und schluckt Zonenfehler (n Punkte, nur 2 Koeffizienten
            # -> bei n=4 fast Ueberfit, Residuen zu klein).
            ud_effektiv = use_direct and (nz >= 6)
            kz  = foucault_kennzahlen(W_z, r_zon, D, lam,
                                      use_direct=ud_effektiv,
                                      grad_modus=grad_modus, r_grenz=r_gr)

            rayleigh_ok  = kz["S_exakt"] >= 0.80
            zuverlaessig = kz.get("kennzahlen_zuverlaessig", True)
            s_col = ("#c62828" if not zuverlaessig else
                     "#2e7d32" if kz["S_exakt"] >= 0.95 else
                     "#e65100" if rayleigh_ok else "#c62828")
            modus = "Hybrid" if use_direct else "Polynom"
            warn  = ""
            if use_direct and nz < 6:
                warn = "  INFO: Hybrid-Modus benoetigt n>=6 Zonen -- Polynom-Modus verwendet"
            elif not zuverlaessig:
                warn = (f"  ⚠ UNZUVERLAESSIG: Fit-Res={kz['fit_residuum_rms']:.3f}lambda"
                        f" > 0.010 -- Hybrid-Modus (n>=6) verwenden!")

            # Warnung wenn Zonen noch nahe Kugelfehler (weniger als 20% korrigiert)
            dl_soll_arr = (r_zon**2 - r_zon[0]**2) / (4.0 * f)
            fehlend = np.where(dl_soll_arr > 0.001,
                               (dl_soll_arr - df_arr) / dl_soll_arr, 0.0)
            kugel_zonen = [f"Z{i+1}" for i, feh in enumerate(fehlend)
                           if feh > 0.80 and i > 0]
            if kugel_zonen:
                warn += (f"  !! {", ".join(kugel_zonen)} noch"
                         f" auf Kugelnaehe -- S nicht repraesentativ!")
                s_col = "#c62828"
            status = (
                f"[{modus}]  "
                f"S={kz['S_exakt']:.4f}   "
                f"Wrms={kz['W_rms']:.4f}lambda   "
                f"WPtV={kz['W_ptv']:.4f}lambda   "
                f"Abtrag={kz['Glas_ptv_nm']:.1f}nm   "
                f"V={kz['Glas_volumen_mm3']:.1f}mm3   "
                f"Deff={kz['Deff_k']:.1f}mm   "
                f"{'OK Rayleigh' if rayleigh_ok else 'FAIL Rayleigh'}"
                f"{warn}"
            )
            self._foucault_status.config(text=status, fg=s_col)
            self._foucault_letzte_kz = kz          # 07/2026: fuer MTF-Tab Vergleichskurve
            self._foucault_letzte_D  = D
            self._diagramm_foucault(D, f, lam, r_zon, df_arr, kz,
                                    use_direct=use_direct,
                                    werkstatt_ansicht=werkstatt_ansicht)
        except Exception as e:
            if hasattr(self, "_foucault_status"):
                self._foucault_status.config(
                    text=f"Fehler: {e}", fg="#c62828")

    def _wahrnehmung_lesen(self):
        """Liest Cursor-V und Fangspiegel-Ø als GANZE Zahlen aus den Feldern
        (Nachkommastellen werden gerundet; Komma oder Punkt sind erlaubt).
        Ungültige Eingaben (leer/Text/V≤0/d<0) lassen den letzten Wert stehen;
        das Feld wird danach auf die ganze Zahl zurückgeschrieben."""
        def _ganz(var, fallback, minimum):
            try:
                x = int(round(float(str(var.get()).strip().replace(",", "."))))
                if x >= minimum:
                    return x
            except (tk.TclError, ValueError):
                pass
            return fallback
        V = _ganz(self._wn_V_var, int(getattr(self, "_wn_V", 150)), 1)
        d = _ganz(self._wn_fang_var, int(getattr(self, "_wn_fang", 0)), 0)
        self._wn_V, self._wn_fang = float(V), float(d)   # d: Fangspiegel-Ø [mm], 0 = aus
        self._wn_V_var.set(str(V))
        self._wn_fang_var.set(str(d))
        try:
            p = float(str(self._wn_pup_var.get()).strip().replace(",", "."))
            if p < 0:
                raise ValueError
        except (tk.TclError, ValueError):
            p = float(getattr(self, "_wn_pup", 3.0))
        self._wn_pup = p                                  # Augenpupille [mm], 0 = aus
        self._wn_pup_var.set(f"{p:g}")
        try:                                              # v7.4.0: Leuchtdichte [cd/m²], 0 = photopisch
            Lv = float(str(self._wn_L_var.get()).strip().replace(",", "."))
            if Lv < 0:
                raise ValueError
        except (tk.TclError, ValueError):
            Lv = float(getattr(self, "_wn_L", 0.0))
        self._wn_L = Lv
        self._wn_L_var.set(f"{Lv:g}")
        _name = "Frei"                                    # v7.5.0: Objekt-Anzeige mit den Eingaben abgleichen
        if p == 0 and Lv == 0:
            _name = "Aus (photopisch)"
        else:
            for _n, _L0 in OBJEKT_HELLIGKEITEN:
                if abs(_L0 - Lv) < 1e-9 and abs(OBJEKT_PUPILLE[_n] - p) < 1e-9:
                    _name = _n
        self._wn_obj_var.set(_name)
        return V

    def _wahrnehmung_readout(self, D, f, lam, S_slide, S_real):
        """Exakte Q_vis-Werte bei der Cursor-Vergrößerung für die Spiegel der
        Hauptregler (Öffnung, Brennweite, Strehl-Schieber)."""
        V = self._wn_V

        _Vcap = max(2.0 * D, 60.0)
        def _vtxt(v):
            return f"> {_Vcap:.0f}×" if v >= _Vcap - 0.5 else "≤ 30×" if v <= 30.5 else f"{v:.0f}×"
        def _vk(v):
            return "∞" if v >= 1e8 else f"{v:.0f}×"
        L_o = _L_roh(getattr(self, "_wn_L", 0.0))     # rohe Objekt-Leuchtdichte (v7.6.0: Étendue)
        d_pp = getattr(self, "_wn_pup", 0.0) or 0.0
        L_ey = _L_eff(L_o, D, V, d_pp)                # Netzhautleuchtdichte bei diesem V (None = photopisch)
        q_sph = perceived_quality_exakt(D, f, lam, V, L=L_o, d_p=d_pp)
        _, _, Wb_sph, _, _ = _wellenfronten(D, f, lam)
        S_sph = _strehl_exakt(Wb_sph)
        zeilen = [
            f"V = {V:.0f}×  (λ={lam:.0f} nm)   —   D={D:.0f} mm  f={f:.0f} mm  (f/{f/D:.1f})",
            f"Sphäre   S={S_sph:.3f}:   Q_vis = {q_sph:.4f}   Verlust {(1-q_sph)*100:.1f} %"
            + (f"   [Netzhaut-Leuchtdichte {L_ey:.3g} cd/m², CSF-Spitze ≈ {csf_spitze_cpd(L_ey):.1f} cpd]" if L_ey else ""),
        ]
        if S_slide is not None and S_real is not None and S_slide > S_real + 0.01:
            e = qvis_einzelrechner(D, S_slide, V, lam, L=L_o, d_p=d_pp)
            zeilen.append(
                f"Schieber S={S_slide:.3f}:   Q_vis = {e['Q']:.4f}   Q̃ = {e['Q_naeh']:.4f}   "
                f"Verlust {e['verlust']:.1f} %   Q≥0.90 bis {_vtxt(e['V_q90'])}   V_krit {_vk(e['V_krit'])}")
        d_fs = getattr(self, "_wn_fang", 0.0)
        d_min = d_fang_min(D, f)
        self._wn_fang_hint.config(
            text=f"mm   (0 = aus;  Mindestgröße für D/f: {d_min:.0f} mm)")
        if d_fs > 0:
            S_u = S_slide if S_slide is not None else S_sph
            q_fs = float(qvis_kurve_mit_fangspiegel(D, S_u, d_fs, lam, V)[0])
            zeilen.append(f"Schieber + Fangspiegel  d={d_fs:.0f} mm (ε={min(d_fs/D,0.95):.2f}):   "
                          f"Q_vis = {q_fs:.4f}   (Näherung)")
        if d_pp > 0:
            _, _, Wb_k, _, _ = _wellenfronten(D, f, lam)
            q_pk = float(qvis_pupille_kurve(D, 1.0, [V], d_pp, lam, wf=_wf_kugel(Wb_k, lam), L=L_o)[0])
            txt = f"Mit Augenpupille {d_pp:g} mm (Austrittspupille {D/V:.1f} mm" + \
                  (", Auge blendet ab" if D / V > d_pp else ", volle Öffnung") + \
                  f"):   Sphäre Q_vis = {q_pk:.4f}"
            if S_slide is not None and S_real is not None and S_slide > S_real + 0.01:
                q_ps = float(qvis_pupille_kurve(D, S_slide, [V], d_pp, lam,
                                            wf=_wf_kugel(_Wb_schieber(D, f, lam, S_slide), lam), L=L_o)[0])
                txt += f"   Schieber Q_vis = {q_ps:.4f}"
            zeilen.append(txt)
        zeilen.append("Paraboloid:   Q_vis = 1.0000")
        self._wn_readout.config(text="\n".join(zeilen))

    def _wn_objekt_gewaehlt(self, *_):
        """Objekt gewählt: Leuchtdichte L und Augenpupille d_p gemeinsam setzen (v7.5.0)."""
        n = self._wn_obj_var.get()
        if n.startswith("Aus"):
            self._wn_L_var.set("0"); self._wn_pup_var.set("0")
        else:
            for _n, _l, _p in WAHRNEHMUNG_OBJEKTE:
                if _n == n:
                    self._wn_L_var.set(f"{_l:g}"); self._wn_pup_var.set(f"{_p:g}")
        self._wahrnehmung_neu()

    def _wahrnehmung_neu(self, *_):
        """Nur das Wahrnehmungs-Diagramm neu zeichnen (Cursor)."""
        res = getattr(self, "_last_params", None)
        if res is None:
            return
        try:
            self._diagramm_fuer_tab("tab_wahrnehmung", res)
        except Exception:
            import traceback
            self._zeige_fehler(traceback.format_exc())


if __name__ == "__main__":
    app = App()
    app.mainloop()
