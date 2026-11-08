"""Texte und Sprache (de/en) – gemeinsam für Desktop und Web."""
from konstanten import VERSION, EYE_RES


# ── Mehrsprachigkeit ──────────────────────────────────────────────────────────

_LANG = "de"   # aktive Sprache: "de" oder "en"

STRINGS = {
    # Fenstertitel
    "win_title": {
        "de": f"Newton-Spiegel: Kontrast- und Schärfeverlust  v{VERSION}",
        "en": f"Newton Mirror: Contrast and Sharpness Loss  v{VERSION}",
    },
    # Eingabe-Gruppe
    "grp_params": {"de": "Teleskop-Parameter", "en": "Telescope Parameters"},
    # Slider-Namen
    "sl_D":      {"de": "Öffnung D",       "en": "Aperture D"},
    "sl_f":      {"de": "Brennweite f",    "en": "Focal length f"},
    "sl_lam":    {"de": "Wellenlänge λ",   "en": "Wavelength λ"},
    "sl_strehl": {"de": "Strehl", "en": "Strehl"},
    "sl_blende": {"de": "Blende D [mm]",   "en": "Stop D [mm]"},
    # Karten-Gruppen
    "grp_contrast": {"de": "Kontrast",
                     "en": "Contrast"},
    "grp_sharpness": {"de": "Schärfe bei gewählter Vergrößerung",
                      "en": "Sharpness at selected magnification"},
    "grp_qvis":  {"de": "Wie wirkt er im Bild?  Q_vis (MTF×CSF, Paraboloid = 1; photopisch, volle Öffnung)",
                  "en": "How does it look?  Q_vis (MTF×CSF, paraboloid = 1; photopic, full aperture)"},
    "grp_flaeche": {"de": "Wie gut ist der Spiegel?", "en": "How good is the mirror?"},
    "grp_reichweite": {"de": "Wie weit trägt er?", "en": "How far does it carry?"},
    "rVhalb":       {"de": "V½  (Halbwertspunkt)",    "en": "V½  (half-value point)"},
    # Karten-Labels
    "rN":           {"de": "Öffnungsverhältnis",              "en": "Focal ratio"},
    "rWp":          {"de": "W_PtV paraxial [λ]",              "en": "W_PtV paraxial [λ]"},
    "rWb":          {"de": "W_PtV best focus [λ]",            "en": "W_PtV best focus [λ]"},
    "rSag":         {"de": "Pfeiltiefe Mitte–Rand (Kugel → Parab.)", "en": "Sagitta centre–edge (sphere → parab.)"},
    "rWrms":        {"de": "W_RMS [λ]",                       "en": "W_RMS [λ]"},
    "rS":           {"de": "Strehl Maréchal-Näherung",        "en": "Strehl Maréchal approx."},
    "rS_exakt":     {"de": "Strehl Pupillenintegral",         "en": "Strehl pupil integral"},
    "rDeff_k":      {"de": "Eff. Öffnung Kontrast [mm]",      "en": "Eff. aperture contrast [mm]"},
    "rVkrit_naeh":  {"de": "V_krit  Näherung  = D/(13·Wb)", "en": "V_crit  approx.  = D/(13·Wb)"},
    "rVkrit_blur":  {"de": "V_krit  Blur sichtbar  (geometrisch)",
                     "en": "V_crit  blur visible  (geometric)"},
    "rVkrit_exakt": {"de": "V_krit  Kontrastverlust >10%  (MTF×CSF)",
                     "en": "V_crit  contrast loss >10%  (MTF×CSF)"},
    "rAuflVerlust": {"de": "Aufl.verlust gg. Paraboloid [%]",
                     "en": "Resolut. loss vs. paraboloid [%]"},
    "rVerlustV":    {"de": "Verlust bei dieser Vergr. [mm]",
                     "en": "Loss at this magnif. [mm]"},
    "rQp30":        {"de": "Q_vis  bei 30×",          "en": "Q_vis  at  30×"},
    "rQp60":        {"de": "Q_vis  bei 60×",          "en": "Q_vis  at  60×"},
    "rQp100":       {"de": "Q_vis  bei 100×",         "en": "Q_vis  at 100×"},
    "rQp150":       {"de": "Q_vis  bei 150×",         "en": "Q_vis  at 150×"},
    # Tab-Titel
    "grp_kern":        {"de": "Kernwerte", "en": "Key values"},
    "grp_t_grund":     {"de": "  Grundlagen  ",             "en": "  Basics  "},
    "grp_t_bild":      {"de": "  Bildqualität  ",           "en": "  Image quality  "},
    "grp_t_messen":    {"de": "  Messen  ",                 "en": "  Testing  "},
    "grp_t_grenzen":   {"de": "  Grenzen  ",                "en": "  Limits  "},
    "grp_t_bauen":     {"de": "  Bauen  ",                  "en": "  Building  "},
    "grp_t_vergleich": {"de": "  Vergleich (Beta)  ",       "en": "  Comparison (beta)  "},
    "tab_grund":       {"de": "  Diagramme  ",              "en": "  Charts  "},
    "tab_kennwerte":   {"de": "  Kennwerte  ",              "en": "  Details  "},
    "tab_strehl":      {"de": "  Strehl vs. f/D  ",         "en": "  Strehl vs. f/D  "},
    "tab_oeff":        {"de": "  Eff. Öffnung vs. f/D  ",   "en": "  Eff. aperture vs. f/D  "},
    "tab_deff_D":      {"de": "  D_eff vs. Öffnung  ",      "en": "  D_eff vs. aperture  "},
    "tab_beugung":     {"de": "  Beugungsgrenze  ",         "en": "  Diffraction limit  "},
    "tab_mtf":         {"de": "  MTF  ",                    "en": "  MTF  "},
    "tab_wahrnehmung": {"de": "  Wahrnehmung  ",            "en": "  Perception  "},
    "tab_blende":      {"de": "  Blende  ",                 "en": "  Stop-down  "},
    "tab_foucault":    {"de": "  Foucault-Test  ",          "en": "  Foucault test  "},
    "tab_vmax":        {"de": "  Grenzen (max. Vergr.)  ",   "en": "  Limits (max. magn.)  "},
    "tab_vmax_D":      {"de": "  Vmax vs. Ø / Seeing  ",        "en": "  Vmax vs. D/seeing  "},
    "tab_fangspiegel": {"de": "  Fangspiegel  ",              "en": "  Secondary mirror  "},
    "tab_gitter":      {"de": "  Gitterrohr  ",               "en": "  Truss tube  "},
    "tab_zelle":       {"de": "  Spiegelzelle  ",             "en": "  Mirror cell  "},
    "tab_vergleich":   {"de": "  Vergleich (Beta)  ",         "en": "  Comparison (beta)  "},
    # MTF-Beschriftungen
    "mtf_fokus_lbl":   {"de": "Fokusebene:",     "en": "Focus plane:"},
    "mtf_bf":  {"de": "Best Focus  (ρ⁴−ρ²,  telescope-optics.net)",
                "en": "Best Focus  (ρ⁴−ρ²,  telescope-optics.net)"},
    "mtf_pf":  {"de": "Paraxialer Fokus  (ρ⁴,  Sacek)",
                "en": "Paraxial Focus  (ρ⁴,  Sacek)"},
    "mtf_obj_lbl":  {"de": "Objekte:",   "en": "Objects:"},
    "mtf_planets":  {"de": "Planeten",   "en": "Planets"},
    "mtf_deepsky":  {"de": "Deep-Sky (M42)", "en": "Deep-Sky (M42)"},
    "mtf_all":      {"de": "Alle",       "en": "All"},
    # Bewertungstexte
    "verdict_good": {
        "de": "✓ Sehr gut — nahezu gleichwertig mit Parabolspiegel (Strehl ≥ 0.95)",
        "en": "✓ Very good — nearly equivalent to paraboloid (Strehl ≥ 0.95)",
    },
    "verdict_ok": {
        "de": "⚠  Noch beugungsbegrenzt (Rayleigh), spürbarer Kontrastverlust bei Planeten  (0.80 ≤ Strehl < 0.95)",
        "en": "⚠  Still diffraction-limited (Rayleigh), noticeable contrast loss on planets  (0.80 ≤ Strehl < 0.95)",
    },
    "verdict_bad": {
        "de": "✗  Nicht beugungsbegrenzt — erheblicher Kontrast- und Schärfeverlust, Parabolspiegel dringend empfohlen (Strehl < 0.80)",
        "en": "✗  Not diffraction-limited — significant contrast and sharpness loss, paraboloid strongly recommended (Strehl < 0.80)",
    },
    # Diagramm-Achsen / Titel (kurze Versionen)
    "ax_fD":        {"de": "Öffnungsverhältnis f/D",    "en": "Focal ratio f/D"},
    "ax_strehl":    {"de": "Strehl-Quotient",            "en": "Strehl ratio"},
    "ax_deff_mm":   {"de": "Effektive Öffnung Kontrast [mm]", "en": "Effective aperture contrast [mm]"},
    "ax_aufl_pct":  {"de": "Auflösungsverlust [%]",     "en": "Resolution loss [%]"},
    "ax_D_mm":      {"de": "Öffnung D [mm]",            "en": "Aperture D [mm]"},
    "ax_oeff_loss": {"de": "Öffnungsverlust [%]",       "en": "Aperture loss [%]"},
    "ax_f_mm":      {"de": "Brennweite f [mm]",         "en": "Focal length f [mm]"},
    "ax_nu_norm":   {"de": "Normierte Ortsfrequenz  ν  (0=DC, 1=Beugungsgrenze)",
                     "en": "Normalised spatial frequency  ν  (0=DC, 1=diffraction limit)"},
    "ax_lpmm":      {"de": "Räumliche Frequenz  [Lp/mm]  (fokal)",
                     "en": "Spatial frequency  [lp/mm]  (focal)"},
    "ax_mtf":       {"de": "Kontrastübertragung (MTF)",  "en": "Contrast transfer (MTF)"},
    "ax_vergr":     {"de": "Vergrößerung V  [×]",        "en": "Magnification V  [×]"},
    "ax_qvis":      {"de": "Visueller Qualitätsindex  Q_vis",
                     "en": "Visual quality index  Q_vis"},
    # Blur-Einheit im Ausgabefeld
    "unabh_S":  {"de": "unabh. von S", "en": "indep. of S"},
    "blur_immer": {"de": "< 1×  — Blur dominiert immer",
                   "en": "< 1×  — blur always dominant"},
    "vk_min":   {"de": "< 30×  — Verlust schon bei kleinster Vergr.",
                 "en": "< 30×  — loss already at lowest magnif."},
    # Sprachumschalter
    "lang_lbl": {"de": "Sprache / Language:", "en": "Sprache / Language:"},
}


def T(key: str) -> str:
    """Gibt den String für den aktuellen Schlüssel in der aktiven Sprache zurück."""
    entry = STRINGS.get(key)
    if entry is None:
        return key   # Fallback: Schlüssel selbst
    return entry.get(_LANG, entry.get("de", key))


def set_sprache(lang):
    """Aktive Sprache setzen ("de" oder "en")."""
    global _LANG
    _LANG = lang
