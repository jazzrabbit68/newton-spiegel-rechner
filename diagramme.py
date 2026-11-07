"""Gemeinsame Diagramm-Methoden für Desktop (Tkinter) und Web (Streamlit).
Die Methoden zeichnen in die Achsen, die die Oberfläche in self._tab_xxx = (fig, ax, canvas) bereitstellt;
Optionen kommen aus self._xxx.get() (tk-Variablen oder der _Var-Ersatz im Web).
(v8.0.0: aus newton_spiegel_rechner.py v7.11.0 ausgegliedert, Code unverändert)"""
import math
import numpy as np
import matplotlib.ticker as ticker
from kern_bau import qvis_kurve_mit_fangspiegel
from kern_optik import (Q_vis_blende_batch, _L_eff, _L_roh, _Wb_schieber, _Wb_von_S, _lnS_geglaettet,
    _strehl_exakt, _wellenfronten, berechne, csf, csf_spitze_cpd, d_fang_min, helligkeit_praxis,
    kurven_N, mtf_aus_wellenfront, mtf_kurven, mtf_para, mtf_para_obstr, mtf_sph_rel,
    mtf_sph_rel_1d, perceived_quality_exakt, perceived_quality_exakt_kurven,
    perceived_quality_exakt_kurven_von_Wb, perceived_quality_exakt_von_Wb, v_halb_exakt,
    v_kombiniert_blur, v_krit_naeh, v_max_diffraktion, v_max_helligkeit, v_max_seeing,
    v_min_helligkeit, v_wendepunkt_naeh)
from kern_vergleich import _wf_kugel, qvis_pupille_kurve
from konstanten import ACC, BG, COR, GRN
from texte import T


class _Var:
    """Minimaler Ersatz für tk.StringVar (Rückfall, wenn eine Option nicht von der Oberfläche gesetzt wurde)."""
    def __init__(self, value=None):
        self._v = value
    def get(self):
        return self._v
    def set(self, v):
        self._v = v


tk = type("tk", (), {"StringVar": _Var})   # nur für getattr(..., tk.StringVar(...)) -Rückfälle


class DiagrammMethoden:

    def _diagramm_vmax_D(self):
        """Zeigt V_max(D) für PERFEKTE Optik (S=1): Beugungsgrenze (linear
        in D) kombiniert mit Seeing (aperturunabhängig, konstant) für
        mehrere Seeing-Werte, plus "2×D"-Mythos zum Vergleich.

        Vereinfachung ggü. _diagramm_vmax() (08/2026, auf Wunsch): keine
        Verknüpfung mit Spiegel-Strehl — zeigt nur die FUNDAMENTALE Grenze,
        die auch bei perfekter Politur bleibt: Beugung + Seeing. Die
        Kombination mit tatsächlichem Spiegel-Strehl macht _diagramm_vmax()
        (inzwischen korrekt, seit v_krit_naeh bei Wb→0 sauber gegen
        unendlich geht statt bei der alten D·0,7·√S-Formel zu deckeln).

        Kernaussage: die "2×D"-Regel trifft näherungsweise zu, solange die
        (aperturproportionale) Beugungsgrenze bindend ist (kleine
        Öffnungen) — wird aber zunehmend zu optimistisch, sobald das
        (aperturunabhängige) Seeing übernimmt (große Öffnungen). Der
        Übergang ("Knick") liegt ungefähr dort, wo beide Kurven sich
        schneiden.
        """
        fig, ax, canvas = self._tab_vmax_D
        ax.cla()
        ax.set_facecolor(BG)

        try:
            seeings = [float(s.strip()) for s in self._vmaxD_seeings.get().split(",") if s.strip()]
        except ValueError:
            seeings = [0.4, 0.8, 1.5, 2.5]
        Dmin = max(self._vmaxD_Dmin.get(), 10.0)
        Dmax = max(self._vmaxD_Dmax.get(), Dmin + 10.0)

        D_arr = np.linspace(Dmin, Dmax, 300)
        farben = ["#1565c0", "#2e7d32", "#E65100", "#8e44ad", "#00838f", "#ad1457"]

        for i, seeing in enumerate(seeings):
            V_see = v_max_seeing(seeing)
            V_diff = np.array([v_max_diffraktion(D) for D in D_arr])
            V_komb = np.array([v_kombiniert_blur(v, V_see) for v in V_diff])
            farbe = farben[i % len(farben)]
            ax.plot(D_arr, V_komb * 2.0, color=farbe, lw=2.0,
                   label=f"Seeing {seeing:.1f}\"  (praktisch)")

        ax.plot(D_arr, 2.0 * D_arr, color="#c62828", lw=1.8, ls=":",
               label='"2×D"-Mythos')

        ax.set_xlabel("Öffnung D [mm]", fontsize=9)
        ax.set_ylabel("V_max praktisch  [×]  (perfekte Optik, S=1)", fontsize=9)
        ax.set_title("V_max vs. Öffnung — Beugung + Seeing (S=1)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(D_arr[0], D_arr[-1])
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        canvas.draw()

    def _diagramm_vmax(self, D: float, f: float, S_slide: float):
        """'Sinnvolle Maximalvergrößerung' — Gegenprobe zum 2×D-Mythos.

        Zeigt L(V) (Netzhaut-Flächenhelligkeit) sowie die unabhängigen
        Grenzen V_helligkeit_min, V_seeing (Onset/praktisch, jetzt mit
        Spiegelfehler kombiniert), V_helligkeit_max, dazu die klassische
        2×D-Faustregel zum Vergleich. S. Modul-Docstring bei
        v_min_helligkeit() für die Herleitung und warum keine einzelne
        'V_optimal'-Zahl aus einem Integral verwendet wird.

        WICHTIG (08/2026): V_seeing allein ignoriert die Spiegelqualität.
        Für einen schlechten (v.a. schnellen) Kugelspiegel kann der
        Spiegel selbst mehr Unschärfe beitragen als die Atmosphäre — daher
        wird die Seeing-Grenze jetzt mit dem geometrischen Spiegel-Blur des
        AKTUELLEN Spiegels (S_slide, wie im Rest des Tools) über
        v_kombiniert_blur() kombiniert. Für eine sehr gute Optik (S→1)
        reduziert sich das automatisch auf reines Seeing.
        """
        fig, ax, canvas = self._tab_vmax
        ax.cla()
        ax.set_facecolor(BG)

        L0         = max(self._vmax_L0.get(), 1e-6)
        seeing     = max(self._vmax_seeing.get(), 0.0)
        D_auge     = max(self._vmax_D_auge.get(), 0.5)
        L_schwelle = max(self._vmax_L_schwelle.get(), 1e-6)
        objekt     = self._vmax_objekt.get()

        V_hmin   = v_min_helligkeit(D, D_auge)
        V_seeM   = v_max_seeing(seeing)                 # Onset: Seeing allein
        V_mirror = v_krit_naeh(D, _Wb_schieber(D, f, 550.0, S_slide))    # Onset: Spiegel allein
        V_komb   = v_kombiniert_blur(V_seeM, V_mirror)   # Onset: beide zusammen
        V_komb_prakt = V_komb * 2.0                      # praktisch, s. Konvention V_max=2×V_krit
        V_hmax   = v_max_helligkeit(D, L0, D_auge, L_schwelle)
        V_mythos = 2.0 * D

        V_obergrenze = max(V_mythos, V_komb_prakt, V_hmax if V_hmax > 0 else 0, V_hmin) * 1.15
        V_arr = np.logspace(math.log10(3.0), math.log10(V_obergrenze), 400)
        L_V   = helligkeit_praxis(D, V_arr, L0, D_auge)

        ACC = "#1565c0"; COR = "#E65100"; GRN = "#2e7d32"; VIO = "#8e44ad"; GRA = "#888888"

        ax.plot(V_arr, L_V, color=ACC, lw=2.0, label=f"L(V) — {objekt} (L₀={L0:.1f} cd/m²)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(L_schwelle, color="#999", lw=1.0, ls=":",
                   label=f"Wahrnehmungsschwelle ({L_schwelle:.2g} cd/m²)")

        ax.axvline(V_hmin, color=GRN, lw=1.6, ls="--",
                   label=f"V_helligkeit_min={V_hmin:.0f}×  (AP=Augenpupille)")
        if V_seeM < V_obergrenze:
            ax.axvline(V_seeM, color=GRA, lw=1.0, ls=":",
                       label=f"V_seeing allein={V_seeM:.0f}×  (Seeing {seeing:.1f}\")")
        if V_mirror < V_obergrenze:
            ax.axvline(V_mirror, color=GRA, lw=1.0, ls="-.",
                       label=f"V_spiegel allein={V_mirror:.0f}×  (S={S_slide:.3f})")
        if V_komb < V_obergrenze:
            ax.axvline(V_komb, color=COR, lw=1.2, ls=":",
                       label=f"V_kombiniert Onset={V_komb:.0f}×")
            ax.axvline(V_komb_prakt, color=COR, lw=1.6, ls="--",
                       label=f"V_kombiniert praktisch={V_komb_prakt:.0f}×  (=2×Onset)")
        if 0 < V_hmax < V_obergrenze:
            ax.axvline(V_hmax, color=VIO, lw=1.6, ls="--",
                       label=f"V_helligkeit_max={V_hmax:.0f}×")
        ax.axvline(V_mythos, color="#c62828", lw=1.6, ls=":",
                   label=f"\"2×D\"-Mythos={V_mythos:.0f}×")

        V_ober_sinn = min(x for x in [V_komb_prakt, V_hmax if V_hmax > 0 else 1e9] if x > 0)
        if V_ober_sinn > V_hmin:
            ax.axvspan(V_hmin, V_ober_sinn, color=GRN, alpha=0.08,
                       label=f"sinnvoller Bereich ≈ {V_hmin:.0f}–{V_ober_sinn:.0f}×")

        ax.set_xlim(V_arr[0], V_arr[-1])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Vergrößerung V [×]  (log)", fontsize=9)
        ax.set_ylabel("Netzhaut-Flächenhelligkeit L(V) [cd/m²]", fontsize=9)
        ax.set_title(f"Sinnvolle Maximalvergrößerung  —  D={D:.0f}mm  f/{f/D:.1f}  S={S_slide:.3f}  "
                     f"Seeing={seeing:.1f}\"  Augenpupille={D_auge:.1f}mm",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        canvas.draw()

    def _diagramm_foucault(self, D, f, lam, r_zon, df_arr, kz, use_direct=False,
                           werkstatt_ansicht=False):
        """Foucault-Diagramm: Wellenfrontkurve + Kennzahlentafel."""
        fig, ax_wf, ax_info, canvas = self._tab_foucault
        ax_wf.cla(); ax_info.cla()
        ax_wf.set_facecolor("#f5f5f5")
        ax_info.set_facecolor("#f5f5f5")

        COR = "#E65100"; ACC = "#1565c0"; GRN = "#2e7d32"
        R     = D / 2.0
        rho_z = kz.get("rho_zonen", r_zon / R)      # v7.3.1: Punkte an den Zonengrenzen (dort gilt der Wert)

        # ── Wellenfrontkurve (aus Polynom-Fit, am besten Fokus) ───────────
        rho_fine = kz["rho_f"]
        W_fine   = kz["W_c"]    # pistonbereinigt, bester Fokus

        # Werkstatt-Ansicht (FigureXP-Darstellung): Kurve vertikal gespiegelt,
        # Null im Tal (bei ~0.7R), Huegel in Mitte und Rand -- genau wie
        # FigureXPs "Surface Error Analysis" / "Glass Remaining". Reine
        # Anzeigetransformation y -> W_max - y: Kennzahlen in kz (S_exakt/
        # W_rms/W_ptv) bleiben unveraendert. ACHTUNG Vorzeichen: gegenueber
        # der Standardansicht (Abtrag-Sicht, Maximum im Tal bei 0.7R) ist das
        # exakt die gespiegelte Kurve (Summe beider = PtV).
        W_ref   = float(np.max(W_fine))
        W_fine_disp = (W_ref - W_fine) if werkstatt_ansicht else W_fine

        _grad = kz.get("poly_grad", 1)
        _res  = kz.get("fit_residuum_rms", 0.0)
        _a4   = kz.get("a4", 0.0)
        if use_direct:
            lbl = f"Wellenfront (Hybrid Grad {_grad}, res={_res:.3f}lambda)"
        elif _res > 0.010:
            lbl = f"Wellenfront (Polynom Grad {_grad}, a4={_a4:.3f}) WARN res={_res:.3f}"
        else:
            lbl = f"Wellenfront (Polynom Grad {_grad}, a4={_a4:.3f})"
        ax_wf.plot(rho_fine, W_fine_disp, color=ACC, lw=2.0, label=lbl)

        # Zonenpunkte: y-Position auf rechter Achse (nm Glas)
        # Da matplotlib nur eine Daten-y-Achse kennt, rechnen wir die nm-Werte
        # über die Inverse der secondary_yaxis-Transformation in λ-Einheiten zurück.
        # zu_lam(g) = g * 2/lam + W_min  →  Punkt sitzt visuell auf der nm-Skala.
        W_z    = kz["W_zonen"]
        Glas_z = kz.get("Glas_zonen")
        W_min  = kz.get("W_min")
        if Glas_z is not None and W_min is not None:
            # Punkte liegen auf der Kurve (gleiche Referenz W_max wie Abtrag).
            y_punkte = (W_ref - np.asarray(kz["W_c_zonen"])) if werkstatt_ansicht \
                       else np.asarray(kz["W_c_zonen"])
            Glas_lbl = Glas_z
            Wm_disc  = float(np.mean(W_z))
            ax_wf.scatter(rho_z, y_punkte, color=COR, s=50, zorder=5,
                          label=f"Messzonen (n={len(r_zon)})")
            for i, (rho, yp) in enumerate(zip(rho_z, y_punkte)):
                wc  = W_z[i] - Wm_disc   # unverschoben, fuer λ-Angabe im Label
                txt = f"Z{i+1}: {Glas_lbl[i]:.0f} nm\n({wc:+.3f}λ)"
                ax_wf.annotate(txt,
                               xy=(rho, yp), xytext=(0, 10),
                               textcoords="offset points",
                               fontsize=7, ha="center", color=COR)
        else:
            # Fallback ohne lam_nm: nur λ-Skala
            Wm_disc  = float(np.mean(W_z))
            W_c_disc = (W_ref - (W_z - Wm_disc)) if werkstatt_ansicht else (W_z - Wm_disc)
            ax_wf.scatter(rho_z, W_c_disc, color=COR, s=50, zorder=5,
                          label=f"Messzonen (n={len(r_zon)})")
            for i, (rho, wc) in enumerate(zip(rho_z, W_c_disc)):
                ax_wf.annotate(f"Z{i+1}: {wc:+.3f}λ",
                               xy=(rho, wc), xytext=(0, 10),
                               textcoords="offset points",
                               fontsize=7, ha="center", color=COR)

        ax_wf.axhline(0,      color="#999",    lw=0.8)
        if not werkstatt_ansicht:
            ax_wf.axhline(+0.125, color="#1565c0", lw=0.7, ls=":", alpha=0.7,
                          label="±λ/8 (Rayleigh)")
            ax_wf.axhline(-0.125, color="#1565c0", lw=0.7, ls=":", alpha=0.7)
        else:
            ax_wf.axhline(+0.25, color="#1565c0", lw=0.7, ls=":", alpha=0.7,
                          label="λ/4 (Rayleigh, PtV)")
        ax_wf.set_xlim(-0.02, 1.05)
        ax_wf.set_xlabel("Normierter Zonenradius ρ = r/R", fontsize=8)
        ax_wf.set_ylabel("Oberflächenfehler, Wellenfront-Skala  [Einheiten von λ]"
                         if werkstatt_ansicht else
                         "Wellenfrontfehler W(ρ)  [Einheiten von λ]", fontsize=8)
        ax_wf.set_title("Werkstatt-Ansicht (wie FigureXP: Surface Error / Glass Remaining)"
                        if werkstatt_ansicht else
                        "Gemessene Wellenfront (bester Fokus, pistonbereinigt)",
                        fontsize=8, fontweight="bold")
        ax_wf.legend(fontsize=7,
                     loc="upper center" if werkstatt_ansicht else "lower center")
        ax_wf.grid(True, alpha=0.25)

        # ── Zweite y-Achse: Materialabtrag [nm Glas], referenziert auf den
        #    am wenigsten bearbeiteten Punkt (Minimum der Wellenfrontkurve) ──
        W_min = kz.get("W_min")
        if W_min is not None:
            if werkstatt_ansicht:
                zu_glas = lambda w: np.asarray(w) * lam / 2.0
                zu_lam  = lambda g: np.asarray(g) * 2.0 / lam
                _glas_lbl = "Glass Remaining [nm]  (0 = Tal: dort nichts abtragen, wie FigureXP)"
            else:
                zu_glas = lambda w: (W_ref - np.asarray(w)) * lam / 2.0
                zu_lam  = lambda g: W_ref - np.asarray(g) * 2.0 / lam
                _glas_lbl = "Materialabtrag [nm Glas]  (0 = Tal: dort nichts abtragen)"
            ax_glas = ax_wf.secondary_yaxis("right", functions=(zu_glas, zu_lam))
            ax_glas.set_ylabel(_glas_lbl, fontsize=8, color="#6a1b9a")
            ax_glas.tick_params(colors="#6a1b9a", labelsize=7)

        # ── Kennzahlentafel ───────────────────────────────────────────────
        ax_info.axis("off")
        zuverlaessig = kz.get("kennzahlen_zuverlaessig", True)
        rayleigh_ok  = kz["S_exakt"] >= 0.80
        if not zuverlaessig:
            s_col = "#c62828"   # unzuverlaessig -> immer rot, unabh. vom Zahlenwert
        else:
            s_col = GRN if kz["S_exakt"] >= 0.95 else \
                    "#e65100" if rayleigh_ok else "#c62828"

        _s_exakt_str = f"{kz['S_exakt']:.4f}"
        _wptv_str    = f"{kz['W_ptv']:.4f} λ"
        _wrms_str    = f"{kz['W_rms']:.4f} λ"
        if not zuverlaessig:
            _s_exakt_str += "  ⚠ unzuverlässig"
            _wptv_str    += "  ⚠"
            _wrms_str    += "  ⚠"

        zeilen = [
            ("Öffnung / f-Zahl",  f"D = {D:.0f} mm  f/{f/D:.1f}"),
            ("Wellenlänge",        f"λ = {lam:.0f} nm"),
            ("", ""),
            ("W_rms",              _wrms_str),
            ("W_PtV",              _wptv_str),
            ("Wb (aus Fit)",       f"{kz['Wb_fit']:.4f} λ"),
            ("Fit-Residuum RMS",   f"{kz['fit_residuum_rms']:.4f} λ"
                                   + ("  ⚠ > 0.010 (Grenze)" if not zuverlaessig else "")),
            ("", ""),
            ("Abtrag RMS (Glas)",  f"{kz['W_rms']*lam/2.0:.1f} nm"
                                   if kz.get("Glas_ptv_nm") is not None else "–"),
            ("Abtrag PtV (Glas)",  f"{kz['Glas_ptv_nm']:.1f} nm"
                                   if kz.get("Glas_ptv_nm") is not None else "–"),
            ("Abtrag V (Glas)",  f"{kz['Glas_volumen_mm3']:.1f} mm³"
                                   if kz.get("Glas_volumen_mm3") is not None else "–"),
            ("", ""),
            ("Strehl  Maréchal",   f"{kz['S_mar']:.4f}"),
            ("Strehl  exakt",      _s_exakt_str),
            ("D_eff  (Kontrast)",  f"{kz['Deff_k']:.1f} mm"),
            ("", ""),
            ("Beurteilung",
             "⚠ Zonenfehler nicht erfasst — Hybrid-Modus nutzen!" if not zuverlaessig
             else ("✓ Rayleigh erfüllt (S ≥ 0.80)" if rayleigh_ok
                   else "✗ Rayleigh nicht erfüllt")),
        ]

        y = 0.97
        for lbl, val in zeilen:
            if not lbl and not val:
                y -= 0.04; continue
            is_strehl  = lbl.startswith("Strehl") or lbl in ("W_rms", "W_PtV")
            is_verdict = lbl == "Beurteilung"
            ax_info.text(0.03, y, lbl + ":", transform=ax_info.transAxes,
                         fontsize=8, color="#555", va="top")
            ax_info.text(0.52, y, val, transform=ax_info.transAxes,
                         fontsize=9,
                         fontweight="bold" if (is_strehl or is_verdict) else "normal",
                         color=s_col if (is_strehl or is_verdict) else "#222",
                         va="top",
                         fontfamily="monospace" if not is_verdict else "sans-serif")
            y -= 0.075

        ax_info.set_title("Auswertung", fontsize=8, fontweight="bold")
        _titel_zuv = "  ⚠ UNZUVERLÄSSIG" if not zuverlaessig else ""
        fig.suptitle(
            f"Foucault-Auswertung  —  D={D:.0f}mm  f/{f/D:.1f}  "
            f"S={kz['S_exakt']:.4f}  W_rms={kz['W_rms']:.4f}λ{_titel_zuv}",
            fontsize=8, fontweight="bold", color=(s_col if not zuverlaessig else "black"))
        fig.tight_layout()
        canvas.draw()

    # ── Diagramme ─────────────────────────────────────────────────────────────

    def _ax_fmt(self, ax):
        ax.grid(True, color="#ddd", lw=0.5)

    def _diagramm_strehl(self, D, f, lam, N_akt, S_akt, S_sph=None):
        fig, ax, canvas = self._tab_strehl
        ax.cla()
        ns, sts, _, _, sts_exakt = kurven_N(D, lam)
        _, _, Wb_akt, _, S_akt_mar = _wellenfronten(D, f, lam)  # S_akt_mar = Maréchal
        ax.fill_between(ns, S_sph if S_sph else S_akt, 1.0,
                        color=ACC, alpha=0.10, label="Bereich Paraboloid→Sphäre")
        ax.axhline(1.00, color=GRN,       lw=1.5, ls="-",  label="Paraboloid (S=1.0)")
        # Maréchal: gestrichelt, deutlich dicker, andere Farbe
        ax.plot(ns, sts,       color="#c07000", lw=2.2, ls="--",
                label="Strehl Maréchal (Näherung, gueltig S>0.5)")
        # Exakt: durchgezogen, Hauptkurve
        ax.plot(ns, sts_exakt, color="#534AB7", lw=2.0, ls="-",
                label="Strehl exakt (Pupillenintegral)")
        ax.axhline(0.80, color="#BA7517", lw=1.2, ls="--", label="Rayleigh S=0.80")
        ax.axhline(0.95, color="#888",    lw=1.0, ls=":",  label="S=0.95")
        ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
        # Hinweistext: Marechal ungueltig unter f/8
        ax.axvline(8.0, color="#c07000", lw=0.7, ls=":", alpha=0.5)
        ax.text(8.1, 0.05, "Marechal\n< f/8 ungenau", fontsize=7,
                color="#c07000", alpha=0.8, va="bottom")
        ax.scatter([N_akt], [S_akt_mar], color="#c07000", s=60, zorder=5,
                   marker="s")
        ax.scatter([N_akt], [S_akt],    color="#534AB7", s=50, zorder=5,
                   marker="D")
        ax.annotate(f"f/{N_akt:.1f}  S={S_akt:.3f} (exakt) / "
                    f"S={S_akt_mar:.3f} (Maréchal)",
                    xy=(N_akt, S_akt), xytext=(8, 10),
                    textcoords="offset points", fontsize=9, color="#333",
                    arrowprops=dict(arrowstyle="-", color="#aaa"))
        if S_sph and S_sph > S_akt + 0.002:
            ax.axhline(S_sph, color=COR, lw=1.5, ls="-.",
                       label=f"Verbessert S={S_sph:.3f}")
            ax.annotate(f"S={S_sph:.3f}", xy=(14.5, S_sph),
                        fontsize=9, color=COR, va="bottom")
        ax.set_xlim(3, 15); ax.set_ylim(0, 1.08)
        ax.set_xlabel(T("ax_fD"), fontsize=10)
        ax.set_ylabel(T("ax_strehl"), fontsize=10)
        ax.set_title(f"Strehl vs. f/D  (D={D:.0f}mm, λ={lam:.0f}nm)", fontsize=10)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(fontsize=9, loc="lower right")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_oeffnung(self, D, f, lam, N_akt, Dk_akt,
                            Ds_akt=None, S_sph=None, Dk_slide=None, Ds_slide=None):
        fig, ax, canvas = self._tab_oeff
        ax.cla()
        _tw = getattr(self, "_oeff_twin", None)
        if _tw is not None and _tw in fig.axes:
            fig.delaxes(_tw)

        ns, _, deff_ks, aufl_verluste, _ = kurven_N(D, lam)
        r_akt   = berechne(D, f, lam)
        avp_akt = r_akt["aufl_verlust_pct"]

        ax.fill_between(ns, deff_ks, D, color=ACC, alpha=0.10,
                        label="Kontrast-Verlust gg. Paraboloid")
        ax.axhline(D, color="#555", lw=1.5, ls="-", label=f"Paraboloid = {D:.0f}mm")
        ax.plot(ns, deff_ks, color=ACC, lw=2, label="Eff. Öffnung (Kontrast)")
        ax.axvline(N_akt, color="#ccc", lw=0.8, ls=":")
        ax.scatter([N_akt], [Dk_akt], color=ACC, s=60, zorder=5)
        ax.annotate(f"D_eff={Dk_akt:.0f}mm",
                    xy=(N_akt, Dk_akt), xytext=(N_akt+0.3, Dk_akt-15),
                    fontsize=8, color=ACC)
        if Dk_slide is not None and Dk_slide > Dk_akt + 1:
            ax.scatter([N_akt], [Dk_slide], color=GRN, s=120, zorder=6,
                       marker="*", label=f"Schieber D_eff={Dk_slide:.0f}mm")
        ax.set_xlim(3, 15)
        ax.set_xlabel(T("ax_fD"), fontsize=10)
        ax.set_ylabel(T("ax_deff_mm"), fontsize=10, color=ACC)
        ax.tick_params(axis="y", labelcolor=ACC)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"f/{x:.0f}" if x == int(x) else ""))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        ax2 = ax.twinx()
        self._oeff_twin = ax2
        ax2.set_facecolor("none")
        ax2.fill_between(ns, aufl_verluste, 0, color=COR, alpha=0.08)
        ax2.plot(ns, aufl_verluste, color=COR, lw=2,
                 label="Aufl.verlust gg. Paraboloid [%]")
        ax2.scatter([N_akt], [avp_akt], color=COR, s=60, zorder=5)
        ax2.annotate(f"{avp_akt:.0f}%",
                     xy=(N_akt, avp_akt), xytext=(N_akt+0.3, avp_akt+2),
                     fontsize=8, color=COR, fontweight="bold")
        ax2.axhline(50, color=COR, lw=0.8, ls=":", alpha=0.5)
        ax2.set_ylabel(T("ax_aufl_pct"), fontsize=10, color=COR)
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis="y", labelcolor=COR)
        ax2.spines["right"].set_color(COR)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="center right")
        ax.set_title(f"Eff. Öffnung & Aufl.verlust vs. f/D  (D={D:.0f}mm)", fontsize=10)
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_deff_D(self, D_akt, N_akt, S_slide=None):
        fig, ax, canvas = self._tab_deff_D
        ax.cla()

        D_arr    = np.linspace(50, 400, 300)
        N_list   = [5, 6, 7, 8, 10]
        colors_N = {5:"#c62828", 6:"#D85A30", 7:"#c8a020", 8:"#2e7d32", 10:"#1565c0"}

        for N in N_list:
            Wb_arr = np.array([_wellenfronten(D, D * N, 550.0)[2] for D in D_arr])
            loss_pct = (1.0 - np.exp(0.25 * _lnS_geglaettet(Wb_arr))) * 100
            col = colors_N[N]
            loss_roh = (1.0 - np.exp(0.25 * _lnS_geglaettet(Wb_arr, roh=True))) * 100
            ax.plot(D_arr, loss_roh, color=col, lw=1.0, ls="--", alpha=0.35, zorder=1)
            ax.plot(D_arr, loss_pct, color=col, lw=2, label=f"f/{N}", zorder=3)
            ax.text(D_arr[-1] + 3, loss_pct[-1], f"f/{N}",
                    color=col, fontsize=9, va="center", fontweight="bold")

        Wb_akt   = _wellenfronten(D_akt, D_akt * N_akt, 550.0)[2]
        S_akt    = _strehl_exakt(Wb_akt)          # exakt, für Vergleich mit Schieber
        loss_akt = (1.0 - np.exp(0.25 * float(_lnS_geglaettet(Wb_akt)))) * 100  # auf der Kurve
        ax.scatter([D_akt], [loss_akt], color=ACC, s=80, zorder=6)
        ax.annotate(f"D={D_akt:.0f}mm f/{N_akt:.1f}  -{loss_akt:.0f}% Öffnung",
                    xy=(D_akt, loss_akt), xytext=(D_akt + 15, loss_akt + 5),
                    fontsize=9, color=ACC, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))

        if S_slide is not None and S_slide > S_akt + 0.01:
            loss_slide = (1 - S_slide**0.25) * 100
            ax.scatter([D_akt], [loss_slide], color=GRN, s=80, zorder=7, marker="*")
            ax.annotate(f"Verbessert S={S_slide:.2f}  -{loss_slide:.0f}%",
                        xy=(D_akt, loss_slide), xytext=(D_akt + 15, loss_slide - 6),
                        fontsize=9, color=GRN, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=GRN, lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRN, alpha=0.85))
            ax.annotate("", xy=(D_akt, loss_slide), xytext=(D_akt, loss_akt),
                        arrowprops=dict(arrowstyle="->", color=GRN, lw=1.5))

        ax.axhline(20, color="#BA7517", lw=1.5, ls="--", label="20% Verlust (spürbar)")
        ax.axhline(50, color="#c62828", lw=1.0, ls=":",  label="50% Verlust (erheblich)")
        ax.text(52, 21, "20% Verlust", fontsize=8, color="#BA7517", fontweight="bold")
        ax.text(52, 51, "50% Verlust", fontsize=8, color="#c62828", fontweight="bold")
        ax.axhline((1 - 0.8**0.25)*100, color="#888", lw=1.0, ls=":",
                   label=f"Rayleigh S=0.80 ({(1-0.8**0.25)*100:.0f}% Verlust)")
        ax.text(52, (1-0.8**0.25)*100 + 1, "Rayleigh S=0.80", fontsize=8, color="#888")

        ax.set_xlim(50, 420)
        ax.set_ylim(0, min(100, max(90, loss_akt + 15)))
        ax.set_xlabel(T("ax_D_mm"), fontsize=10)
        ax.set_ylabel(T("ax_oeff_loss"), fontsize=10)
        from matplotlib.lines import Line2D
        proxy = [
            Line2D([0],[0], color="#555", lw=2,   label="D × S^0.25  (geglättet)"),
            Line2D([0],[0], color="#555", lw=1.0, ls="--", alpha=0.5,
                   label="D × S^0.25  (exakter Strehl, ungeglättet)"),
        ]
        ax.legend(handles=proxy, fontsize=9, loc="upper left")
        ax.set_title("Öffnungsverlust vs. Öffnung — D × S^0.25  (Kontrast)", fontsize=9)
        ax.text(0.5, -0.13,
                "Gilt für V→∞ (reine Kontrast-/MTF-Form, ohne Augenauflösung/CSF) — "
                "bei kleiner Vergrößerung besser: Tab „Q_vis vs. Vergrößerung“\n"
                "Durchgezogen: Oszillation des exakten Strehl ab S < ~0.5 geglättet; "
                "gestrichelt: exakte Kurve. Dort ist S^0.25 nur noch qualitativ.",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#777", style="italic")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_beugung(self, D_akt, f_akt):
        fig, ax, canvas = self._tab_beugung
        ax.cla()

        lam_mm = 550e-6
        f_arr  = np.linspace(200, 2000, 500)

        def D_grenz(S):
            Wb = _Wb_von_S(S)
            return (Wb * 4.0 * 512 * lam_mm) ** 0.25 * f_arr ** 0.75  # 07/2026: 1024->512

        D_95 = D_grenz(0.95)
        D_80 = D_grenz(0.80)
        D_50 = D_grenz(0.50)

        ax.fill_between(f_arr, 0,    D_95, color="#2e7d32", alpha=0.10)
        ax.fill_between(f_arr, D_95, D_80, color="#f9a825", alpha=0.12)
        ax.fill_between(f_arr, D_80, 300,  color="#c62828", alpha=0.07)

        ax.text(300,  40,  "S ≥ 0.95  (sehr gut)",              color="#2e7d32", fontsize=8, alpha=0.9)
        ax.text(300, 155,  "0.80 ≤ S < 0.95  (gut)",            color="#c8860a", fontsize=8, alpha=0.9)
        ax.text(300, 255,  "S < 0.80  (nicht beugungsbegrenzt)", color="#c62828", fontsize=8, alpha=0.9)

        ax.plot(f_arr, D_95, color="#1565c0", lw=2,   label="S=0.95")
        ax.plot(f_arr, D_80, color="#2e7d32", lw=2.5, label="S=0.80  (Rayleigh)")
        ax.plot(f_arr, D_50, color="#BA7517", lw=1.5, ls="--", label="S=0.50")

        for N in [5, 6, 7, 8, 10]:
            D_fN = np.minimum(f_arr / N, 300)
            ax.plot(f_arr, D_fN, color="#ccc", lw=0.8, ls="--")
            y_label = 1900.0 / N
            if y_label < 285:
                ax.text(1920, y_label, f"f/{N}", color="#999", fontsize=8, va="center")

        ax.scatter([f_akt], [D_akt], color=ACC, s=100, zorder=7)
        x_off = -200 if f_akt > 1500 else 80
        ax.annotate(f"D={D_akt:.0f}mm  f/{f_akt/D_akt:.1f}",
                    xy=(f_akt, D_akt), xytext=(f_akt + x_off, D_akt + 25),
                    fontsize=9, color=ACC, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.85))

        ax.set_xlim(200, 2000); ax.set_ylim(0, 300)
        ax.set_xlabel(T("ax_f_mm"), fontsize=10)
        ax.set_ylabel(T("ax_D_mm"), fontsize=10)
        ax.set_title("Beugungsgrenze sphärischer Spiegel  (λ=550nm)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_mtf(self, D, f, lam, S_slide=None, V=150.0):
        fig, ax, canvas = self._tab_mtf
        ax.cla()
        for ax2_old in fig.get_axes()[1:]:
            fig.delaxes(ax2_old)

        freqs_as, fc, mp_arr, msa, msr_arr, strehl = mtf_kurven(D, f, lam)

        all_details = [
            ("Jupiter\nGürtel",      5.0,      "#4a90d9", "planeten"),
            ("Jupiter\nFestons",     1.5,      "#4a90d9", "planeten"),
            ("Mars\nPol",            1.0,      "#d94a4a", "planeten"),
            ("Saturn\nCassini",      0.5,      "#c8a020", "planeten"),
            ("Trapezium\nAB (~9\")", 9.0,      "#8e44ad", "deepsky"),
            ("M42\nFilamente",       3.0,      "#8e44ad", "deepsky"),
            ("Trapezium\nCD (~2\")", 2.0,      "#8e44ad", "deepsky"),
            ("M13\nHalo-Sterne",     2.5,      "#c0392b", "deepsky"),
            ("M13\nAußensterne",     1.5,      "#c0392b", "deepsky"),
            ("M13\nKern",            0.7,      "#c0392b", "deepsky"),
            ("Dawes\nGrenze",        116.0/D,  "#888",    "beide"),
        ]

        paraxial    = getattr(self, '_mtf_fokus',
                              tk.StringVar(value="bestfocus")).get() == "paraxial"
        fokus_label = "Paraxialer Fokus  (ρ⁴)" if paraxial else "Best Focus  (ρ⁴−ρ²)"
        N_KURVE     = 300
        lam_mm      = lam * 1e-6
        Wb_k        = D**4 / (512.0 * f**3 * lam_mm) / 4.0  # 07/2026: 1024->512
        fc_fokal    = 1.0 / (lam_mm * (f / D))

        nu_k2  = np.linspace(0, 1, N_KURVE)
        mp_k2  = np.array([mtf_para(nu)                              for nu in nu_k2])
        msr_k2 = np.array([mtf_sph_rel(nu, Wb_k, paraxial=paraxial) for nu in nu_k2])
        ms_k2  = mp_k2 * msr_k2

        ax.plot(nu_k2, mp_k2,  color=GRN, lw=2.2, ls="-",
                label="Paraboloid  (Beugungsgrenze)")
        ax.plot(nu_k2, ms_k2,  color=COR, lw=2.2, ls="-",
                label=f"Sphäre absolut  (S={strehl:.3f})")

        # v7.3.1: optional die ALTE 1D-Näherung (Linienintegral, bis v6.x) zum Vergleich
        if getattr(self, '_mtf_alt1d', None) is not None and self._mtf_alt1d.get():
            msr_1d = np.array([mtf_sph_rel_1d(nu, Wb_k, paraxial=paraxial) for nu in nu_k2])
            ax.plot(nu_k2, mp_k2 * msr_1d, color=COR, lw=1.6, ls=(0, (4, 2)), alpha=0.75,
                    label="Sphäre absolut – alte 1D-Näherung (nur Vergleich)")

        # v7.2.3: Schieber-Kurve (verbesserter Spiegel) UND – falls für dieses D
        # vorhanden – die Foucault-Messung als getrennte Kurven (relative Kurve entfällt).
        kz_fc = getattr(self, '_foucault_letzte_kz', None)
        D_fc  = getattr(self, '_foucault_letzte_D', None)
        if S_slide is not None and S_slide > strehl + 0.002:
            Wb_sl = _Wb_schieber(D, f, lam, S_slide)
            msr_sl = np.array([mtf_sph_rel(nu, Wb_sl, paraxial=paraxial) for nu in nu_k2])
            ax.plot(nu_k2, mp_k2 * msr_sl, color=ACC, lw=2.0, ls="--",
                    label=f"Verbessert (Schieber)  S={S_slide:.3f}")
        if kz_fc is not None and D_fc == D:      # v7.3.1: immer zeigen, wenn Messdaten für diese Öffnung vorliegen (früher nur bei S_Foucault > S_Spiegel)
            # v7.3.1: MTF direkt aus der GEMESSENEN Wellenfront (best focus, Polynom bzw. Hybrid) –
            # nicht aus Wb_fit mit reinem ρ⁴−ρ²-Modell (das passte bei höheren Polynomgraden nicht
            # zum angezeigten Strehl).
            ms_fc = mtf_aus_wellenfront(kz_fc['rho_f'], kz_fc['W_c'], 0.0, nu=nu_k2)
            ax.plot(nu_k2, ms_fc, color="#1a7abf", lw=1.8, ls="-.",
                    label=f"Foucault gemessen  (S={kz_fc['S_exakt']:.3f})")

        _obj = getattr(self, '_mtf_objekte', None)
        _obj = _obj.get() if _obj is not None else "alle"
        for label, d_as, col, kat in all_details:
            if _obj != "alle" and kat not in (_obj, "beide"):
                continue                      # v7.3.1: Objektauswahl (Planeten / Deep-Sky / Alle) wirkt jetzt
            nu_det = (1.0 / (2.0 * d_as)) / fc
            if nu_det >= 1.0:
                continue
            ax.axvline(nu_det, color=col, lw=0.8, ls=":", alpha=0.7)
            mp_v = float(np.interp(nu_det, nu_k2, mp_k2))
            ax.text(nu_det + 0.005, mp_v - 0.07,
                    label.replace("\n", " "), fontsize=7, color=col, rotation=90, va="top")

        ax2x = ax.twiny()
        ax2x.set_facecolor("none")
        ax2x.set_xlim(0, fc_fokal)
        ax2x.set_xlabel(T("ax_lpmm"), fontsize=9)
        ax2x.tick_params(axis="x", labelsize=8)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel(T("ax_nu_norm"), fontsize=9)
        ax.set_ylabel(T("ax_mtf"), fontsize=10)
        ax.set_title(
            f"MTF-Kurven [{fokus_label}] — D={D:.0f}mm  f/{f/D:.1f}  "
            f"Strehl={strehl:.3f}  fc={fc_fokal:.1f} Lp/mm  λ={lam:.0f}nm",
            fontsize=9)
        fokus_color = "#2e7d32" if paraxial else ACC
        ax.text(0.98, 0.97,
                f"{fokus_label}\n"
                + ("Vergleichbar mit Sacek (telescope-optics.net)" if paraxial
                   else "Vergleichbar mit Zemax / telescope-optics.net"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=fokus_color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=fokus_color, alpha=0.85))
        ax.legend(fontsize=9, loc="lower left")
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()

    def _diagramm_wahrnehmung(self, D, f, lam,
                               S_slide=None, S_real=None):
        """Visueller Qualitätsindex Q_vis vs. Vergrößerung.

        Zwei Kurven pro Spiegel:
          · Q_vis exakt  = MTF×CSF-Integral ohne w(V)         (durchgezogen)
          · Q̃ Näherung  = S + (1−S)·(1−w(V))  [w(V) mit V_wendepunkt_naeh,
                           NICHT V_krit — s. Chat-Historie 08/2026:
                           V_krit selbst passt die Kurvenform schlecht,
                           separater gröberer Korrekturfaktor für S<1]
                           (gestrichelt)
        """
        fig, ax, canvas = self._tab_wahrnehmung
        ax.cla()

        # Ablese-Cursor (10/2026, v6.10.0): V_cur=None -> kein Cursor.
        # Desktop: Werte aus den Eingabefeldern lesen + Textzeile aktualisieren;
        # Web: Attribute werden von plot_wahrnehmung() gesetzt.
        if hasattr(self, "_wn_V_var"):
            self._wahrnehmung_lesen()
            self._wahrnehmung_readout(D, f, lam, S_slide, S_real)
        V_cur = getattr(self, "_wn_V", None)
        L_obj = _L_roh(getattr(self, "_wn_L", 0.0))   # rohe Objekt-Leuchtdichte [cd/m²]; None = aus (photopisch)
        d_pup = getattr(self, "_wn_pup", 0.0) or 0.0   # Augenpupille [mm], 0 = aus

        _, _, Wb, _, S_mar = _wellenfronten(D, f, lam)
        S_exakt = _strehl_exakt(Wb)

        V_krit     = v_krit_naeh(D, Wb)
        V_wende    = v_wendepunkt_naeh(D, Wb)
        V_halb     = v_halb_exakt(D, f, lam)
        V_max_plot = 400.0
        V_min_plot = 30.0
        if V_cur is not None:                 # Achse so erweitern, dass V sichtbar bleibt
            V_max_plot = max(V_max_plot, V_cur * 1.15)
            V_min_plot = min(V_min_plot, V_cur * 0.85)
        V_arr      = np.linspace(V_min_plot, V_max_plot, 150)

        Qe_sph  = perceived_quality_exakt_kurven(D, f, lam, V_arr, L=L_obj, d_p=d_pup)
        f_para  = D * 50.0
        Qe_para = perceived_quality_exakt_kurven(D, f_para, lam, V_arr, L=L_obj, d_p=d_pup)

        def Q_naeh(V_a, S, Wb_val):
            Vk = v_wendepunkt_naeh(D, Wb_val)
            w  = 1.0 / (1.0 + (Vk / np.maximum(V_a, 1e-9))**2)
            return S + (1 - S) * (1 - w)

        Qn_sph  = Q_naeh(V_arr, S_exakt, Wb)
        Qn_para = Q_naeh(V_arr, 1.0, 1e-9)

        has_slide = (S_slide is not None and S_real is not None
                     and S_slide > S_exakt + 0.002)
        if has_slide:
            Wb_s      = _Wb_schieber(D, f, lam, S_slide)
            S_ex_s    = S_slide
            S_ex_s_ph = _strehl_exakt(Wb_s)
            Qe_slide  = perceived_quality_exakt_kurven_von_Wb(D, Wb_s, lam, V_arr, L=L_obj, d_p=d_pup)
            Qn_slide  = Q_naeh(V_arr, S_ex_s_ph, Wb_s)

        # v7.6.0: kleinste Netzhautleuchtdichte im Plotbereich (Étendue, bei V_max); Q̃/V_krit/V½ gelten nur photopisch
        L_min = None if L_obj is None else _L_eff(L_obj, D, V_max_plot, d_pup)
        L_phot = L_min is None
        L_min_val = L_obj if L_min is None else L_min
        strehl_label = f"Sphäre f/{f/D:.1f}  S={S_exakt:.3f} (exakt)"
        if not L_phot:
            strehl_label += f"  L={L_obj:g} cd/m²"

        # v7.2.0: Kurven MIT Abblenden durch die Augenpupille (zusätzlich)
        if d_pup > 0:
            Qp_sph = qvis_pupille_kurve(D, 1.0, V_arr, d_pup, lam, wf=_wf_kugel(Wb, lam), L=L_obj)
            if has_slide:
                Qp_slide = qvis_pupille_kurve(D, S_slide, V_arr, d_pup, lam,
                                              wf=_wf_kugel(Wb_s, lam), L=L_obj)   # v7.5.1: wf statt S->Wb-Inversion

        if L_phot:      # Q̃-Näherung gilt nur photopisch
            ax.fill_between(V_arr, Qe_sph, Qn_sph,
                            color="#534AB7", alpha=0.12, label="Abweichung exakt ↔ Näherung")
        ax.fill_between(V_arr, Qe_sph, Qe_para,
                        color=COR, alpha=0.10, label="Wahrnehmungsverlust")
        ax.plot(V_arr, Qe_para, color=GRN, lw=2.0, ls="-",  label="Paraboloid  exakt")
        if L_phot:
            ax.plot(V_arr, Qn_para, color=GRN, lw=1.0, ls="--", alpha=0.4)
        ax.plot(V_arr, Qe_sph,  color=COR, lw=2.2, ls="-",
                label=f"{strehl_label}  exakt")
        if L_phot:
            ax.plot(V_arr, Qn_sph,  color=COR, lw=1.5, ls="--", alpha=0.75,
                    label=f"Sphäre  Näherung Q̃  (S={S_exakt:.3f})")

        if has_slide:
            ax.plot(V_arr, Qe_slide, color=ACC, lw=2.0, ls="-",
                    label=f"Schieber exakt  S={S_ex_s:.3f}")
            if L_phot:
                ax.plot(V_arr, Qn_slide, color=ACC, lw=1.2, ls="--", alpha=0.7,
                        label=f"Schieber Näherung  S={S_ex_s:.3f}")

        if d_pup > 0:
            ax.plot(V_arr, Qp_sph, color=COR, lw=2.4, ls="-.",
                    label=f"Sphäre MIT Augenpupille {d_pup:g} mm  (exakt)")
            if has_slide:
                ax.plot(V_arr, Qp_slide, color=ACC, lw=2.2, ls="-.",
                        label=f"Schieber MIT Augenpupille {d_pup:g} mm  (exakt)")
            V_stop = D / d_pup
            if V_min_plot <= V_stop <= V_max_plot:
                ax.axvline(V_stop, color="#ef6c00", lw=1.2, ls="-.", alpha=0.8)
                ax.text(V_stop + 2, 0.03, f"Abblendgrenze D/d_p = {V_stop:.0f}×",
                        fontsize=7, color="#ef6c00", rotation=90, va="bottom", alpha=0.9)

        ax.axhline(S_exakt, color=COR, lw=1.0, ls=":", alpha=0.75)
        ax.text(V_max_plot * 0.98, S_exakt + 0.012,
                f"Asymptote Q̃→S={S_exakt:.3f}",
                fontsize=8, color=COR, ha="right", alpha=0.85)

        for V_m, col, ls, lbl in [
                (D / 7.0, "#888888", ":",  f"V_min={D/7:.0f}×"),
                (V_krit,  "#534AB7", "--", f"V_krit={V_krit:.0f}×  (10%-Schwelle)"),
                (V_wende, "#c62828", "-.", f"V_wendepunkt≈{V_wende:.0f}×  (Näherung, =V_krit/³√S²)"),
                (V_halb,  "#00838f", ":",  f"V½_exakt={V_halb:.0f}×  (echter Halbwertspunkt)")]:
            if V_min_plot <= V_m <= V_max_plot:
                ax.axvline(V_m, color=col, lw=1.0, ls=ls, alpha=0.65)
                ax.text(V_m + 2, 0.03, lbl,
                        fontsize=7, color=col, rotation=90, va="bottom", alpha=0.8)

        for q_thresh, col, lbl in [(0.9, GRN, "Q=0.90 (sehr gut)"),
                                    (0.7, "#BA7517", "Q=0.70 (spürbar)")]:
            ax.axhline(q_thresh, color=col, lw=1.0, ls=":", alpha=0.7)
            ax.text(V_min_plot * 1.03, q_thresh + 0.008, lbl, fontsize=8, color=col, alpha=0.85)

        # ── Fangspiegel-Effekt auf den Schieber-Spiegel (zusätzliche Kurve) ──
        d_fs = getattr(self, "_wn_fang", 0.0) or 0.0
        S_fs = S_slide if S_slide is not None else S_exakt
        if d_fs > 0:
            Qe_fs = qvis_kurve_mit_fangspiegel(D, S_fs, d_fs, lam, V_arr)
            ax.plot(V_arr, Qe_fs, color="#6a1b9a", lw=2.4, ls="-",
                    label=f"Schieber + Fangspiegel  d={d_fs:.0f}mm "
                          f"(ε={min(d_fs/D, 0.95):.2f}, S={S_fs:.3f})  Näherung")

        # ── Ablese-Cursor: exakter Wert bei der gewählten Vergrößerung ───────
        if V_cur is not None:
            ax.axvline(V_cur, color="#111", lw=1.3, ls="-", alpha=0.8, zorder=4)
            pts = [(perceived_quality_exakt(D, f, lam, V_cur, L=L_obj, d_p=d_pup), COR,
                    "Sphäre {q:.3f}"),
                   (1.0, GRN, "Parab. 1.000")]
            if has_slide:
                pts.append((perceived_quality_exakt_von_Wb(D, Wb_s, lam, V_cur, L=L_obj, d_p=d_pup),
                            ACC, "Schieber {q:.3f}"))
            if d_fs > 0:
                pts.append((float(qvis_kurve_mit_fangspiegel(D, S_fs, d_fs, lam, V_cur)[0]),
                            "#6a1b9a", "Schieber+FS {q:.3f}"))
            if d_pup > 0:
                pts.append((float(qvis_pupille_kurve(D, 1.0, [V_cur], d_pup, lam, L=L_obj,
                                                     wf=_wf_kugel(Wb, lam))[0]),
                            COR, "Sphäre+Pupille {q:.3f}"))
            # Beschriftungen von oben nach unten, mit Mindestabstand (Punkte
            # liegen oft dicht beieinander, z.B. Parabol 1.000 / Schieber 0.983)
            axh_pt = fig.get_size_inches()[1] * 72.0 * 0.72      # grobe Achsenhöhe [pt]
            y_top  = 1.28
            vorher_q, vorher_off = None, None
            for q, col, txt in sorted(pts, key=lambda t: -t[0]):
                ax.scatter([V_cur], [q], color=col, s=46, zorder=7,
                           edgecolor="white", linewidth=0.8)
                off = 2.0
                if vorher_q is not None:
                    gap = (vorher_q - q) / y_top * axh_pt        # Abstand der Punkte [pt]
                    off = vorher_off - 11.0 + gap if gap < 11.0 else 2.0
                ax.annotate(txt.format(q=q), xy=(V_cur, q), xytext=(7, off - 5),
                            textcoords="offset points", fontsize=8,
                            fontweight="bold", color=col, zorder=8,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.8))
                vorher_q, vorher_off = q, off

        max_abw = float(np.max(np.abs(Qe_sph - Qn_sph)))
        if L_phot:
            L_info = ""
        elif d_pup > 0:
            L_info = (f"\nObjekt L = {L_obj:g} cd/m², am Auge L·min(1,(D/(V·d_p))²)"
                      f"\n  = {L_min_val:.3g} cd/m² bei {V_max_plot:.0f}× (CSF-Spitze ≈ {csf_spitze_cpd(L_min_val):.1f} cpd)"
                      f"\n(V_krit, V½: photopisch; Q̃-Kurven ausgeblendet)")
        else:
            L_info = (f"\nLeuchtdichte L = {L_obj:g} cd/m²: CSF-Spitze ≈ {csf_spitze_cpd(L_obj):.1f} cpd"
                      f"\n(V_krit, V½: photopisch; Q̃-Kurven ausgeblendet)")
        ax.text(0.015, 0.985,
                f"V_krit (10%-Schwelle, exakt)   = {V_krit:.0f}×   [D/(13·Wb)]\n"
                f"V_wendepunkt (Näherungsformel) = {V_wende:.0f}×   [V_krit/³√(S²)]\n"
                f"V½ (echter Halbwertspunkt)     = {V_halb:.0f}×   [nur via Integral]"
                + L_info
                + ("\nPupillenkurven: rel. zum gleich abgeblendeten Paraboloid (Öffnungsverlust → Q_abs)" if d_pup > 0 else ""),
                transform=ax.transAxes, fontsize=7, family="monospace",
                va="top", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.85))
        ax.set_xlim(V_min_plot, V_max_plot)
        ax.set_ylim(0, 1.28 if V_cur is not None else 1.12)   # Kopfraum fuer Info-Kasten + Cursor-Werte
        ax.set_xlabel(T("ax_vergr"), fontsize=10)
        ax.set_ylabel(T("ax_qvis"), fontsize=10)
        ax.set_title(
            f"Visuelle Wahrnehmung  —  D={D:.0f}mm  f/{f/D:.1f}"
            f"  |  S_exakt={S_exakt:.3f}" + (f"  max.Abw.={max_abw:.3f}" if L_phot else "")
            + (f"  |  Cursor V={V_cur:.0f}×" if V_cur is not None else ""),
            fontsize=9)
        ax.legend(fontsize=8, loc="lower right",
                  title="— exakt (MTF×CSF)   - - Näherung Q̃(V)   -· mit Augenpupille",
                  title_fontsize=7)
        self._ax_fmt(ax); fig.tight_layout(); canvas.draw_idle()


    def _diagramm_blende(self, D, f, lam, D_blend):
        """Blenden-Diagramm: Q̃(V) für Original und Blendenstufen + Leuchtdichte-Effekt."""
        fig, ax, ax2, canvas = self._tab_blende
        ax.cla(); ax2.cla()

        V_arr      = np.linspace(30, 400, 80)   # 80 Punkte reichen für Plot (war 150)
        V_max_plot = 400.0

        d_fang   = d_fang_min(D, f)
        eps_orig = d_fang / D

        def S_von_D(Db):
            _, _, Wb, _, _ = _wellenfronten(Db, f, lam)
            return _strehl_exakt(Wb)

        # Sinnvoller Vergrößerungsbereich — abhängig von Original-Öffnung
        V_min_orig  = D / 7.0
        V_max_AP    = D / 0.5
        _, _, Wb_orig, _, _ = _wellenfronten(D, f, lam)
        V_krit_orig = v_krit_naeh(D, Wb_orig)
        V_max_orig  = min(V_max_AP, V_krit_orig * 2.0)
        V_min_blend = D_blend / 7.0
        V_max_AP_bl = D_blend / 0.5
        _, _, Wb_bl, _, _ = _wellenfronten(D_blend, f, lam)
        V_krit_bl   = v_krit_naeh(D_blend, Wb_bl)
        V_max_blend = min(V_max_AP_bl, V_krit_bl * 2.0)

        _trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
        _N_MTF  = 40                                    # MTF-Stützpunkte (war 80)
        _nu     = np.linspace(0.0, 1.0, _N_MTF)        # einmal berechnen, nicht per V

        # MTF-Arrays für Original-Paraboloid (hängen nicht von V ab) — einmal berechnen
        _mp_orig = np.array([mtf_para(ni) for ni in _nu])
        _fc_orig = D / (lam * 1e-6 * 206265)

        _q_cache: dict = {}   # lokaler Cache: Db → (Q_arr, S, Vk)

        def Q_opt_kurve(Db):
            """Optische Qualität normiert auf Original-Paraboloid — MTF×CSF.

            Obstruktion: ε = d_fang/Db — wächst beim Abblenden.
            Ergebnis wird gecacht: gleiche Blende wird nicht zweimal gerechnet.
            """
            key = round(Db, 2)
            if key in _q_cache:
                return _q_cache[key]

            _, _, Wb, _, _ = _wellenfronten(Db, f, lam)
            S       = _strehl_exakt(Wb)          # gecacht durch @lru_cache
            Vk      = v_krit_naeh(Db, Wb)
            eps     = min(d_fang / Db, 0.99)
            fc_bl   = Db / (lam * 1e-6 * 206265)
            f_ratio = fc_bl / _fc_orig

            # Blenden-MTF-Arrays einmal für alle V berechnen
            nu_bl = _nu / max(f_ratio, 1e-6)
            mp_bl = np.array([mtf_para_obstr(min(ni, 1.0), eps) for ni in nu_bl])
            ms_bl = np.array([mtf_sph_rel(min(ni, 1.0), Wb)     for ni in nu_bl])
            cutoff_mask = _nu > f_ratio
            mp_bl[cutoff_mask] = 0.0
            ms_bl[cutoff_mask] = 0.0
            mtf_bl = ms_bl * mp_bl   # kombinierte Blenden-MTF (V-unabhängig)

            Q_arr = np.zeros(len(V_arr))
            for j, V in enumerate(V_arr):
                w     = np.array([csf(ni * _fc_orig * 3600.0 / V) for ni in _nu])
                num   = float(_trapz(mtf_bl  * w, _nu))
                denom = float(_trapz(_mp_orig * w, _nu))
                Q_arr[j] = num / max(denom, 1e-12)

            result = (Q_arr, S, Vk)
            _q_cache[key] = result
            return result

        # ── Original-Spiegel ─────────────────────────────────────────────────
        Q_orig, S_orig, Vk_orig = Q_opt_kurve(D)
        ax.plot(V_arr, Q_orig, color=COR, lw=2.5, ls="-",
                label=f"Original  D={D:.0f}mm  S={S_orig:.3f}  V½={Vk_orig:.0f}×")
        ax.axhline(float(Q_orig[-1]), color=COR, lw=0.8, ls=":", alpha=0.5)
        ax.text(V_max_plot * 0.98, float(Q_orig[-1]) + 0.012,
                f"Asymptote orig={float(Q_orig[-1]):.3f}",
                fontsize=7, color=COR, ha="right", alpha=0.85)

        # ── Blendenstufen: 4 Zwischenwerte zwischen D_blend und D ─────────────
        D_steps = []
        if D_blend < D * 0.98:
            # gleichmäßig verteilt von D_blend bis knapp unter D
            n_steps = 4
            for i in range(1, n_steps):
                D_steps.append(D_blend + i * (D - D_blend) / n_steps)

        colors_blend = ["#1565c0", "#2e7d32", "#7b1fa2", "#c8860a"]
        for i, Db in enumerate(D_steps):
            Q_b, S_b, Vk_b = Q_opt_kurve(Db)
            col = colors_blend[i % len(colors_blend)]
            pct = Db / D * 100
            ax.plot(V_arr, Q_b, color=col, lw=1.2, ls="--", alpha=0.6,
                    label=f"D={Db:.0f}mm ({pct:.0f}%)  S={S_b:.3f}  V½={Vk_b:.0f}×")

        # ── Gewählte Blende ───────────────────────────────────────────────────
        if D_blend < D * 0.98:
            Q_bl, S_bl, Vk_bl = Q_opt_kurve(D_blend)
            pct_bl = D_blend / D * 100
            ax.plot(V_arr, Q_bl, color=ACC, lw=2.5, ls="-",
                    label=f"Blende  D={D_blend:.0f}mm ({pct_bl:.0f}%)  "
                          f"S={S_bl:.3f}  V½={Vk_bl:.0f}×")
            ax.axhline(float(Q_bl[-1]), color=ACC, lw=0.8, ls=":", alpha=0.5)
            # Asymptote blend: unterhalb der Linie, damit kein Overlap mit orig
            ax.text(V_max_plot * 0.98, float(Q_bl[-1]) - 0.022,
                    f"Asymptote blend={float(Q_bl[-1]):.3f}",
                    fontsize=7, color=ACC, ha="right", va="top", alpha=0.85)

            # Gewinn/Verlust durch Blende — links von V½ (weg von opt-Blende-Annotation)
            Q_bl_at_Vk = float(np.interp(Vk_orig, V_arr, Q_bl))
            Q_or_at_Vk = float(np.interp(Vk_orig, V_arr, Q_orig))
            delta = Q_bl_at_Vk - Q_or_at_Vk
            x_ann = max(Vk_orig - 80, 50)
            ax.annotate(
                f"bei V={Vk_orig:.0f}×:\n"
                f"Original Q={Q_or_at_Vk:.2f}\n"
                f"Blende   Q={Q_bl_at_Vk:.2f}  "
                f"({'+'if delta>=0 else ''}{delta:.2f})",
                xy=(Vk_orig, Q_or_at_Vk),
                xytext=(x_ann, Q_or_at_Vk + 0.12),
                fontsize=8, color=ACC if delta >= 0 else "#c62828",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACC, alpha=0.88))

        # ── "optimale Blende" — max Q je Vergrößerung (aus Stichproben) ────────
        D_scan  = np.linspace(D * 0.40, D, 8)    # 8 Stichproben reichen (war 12)
        Q_stack = np.array([Q_opt_kurve(Db)[0] for Db in D_scan])
        Q_opt_line = Q_stack.max(axis=0)

        # Optimale Blende pro Vergrößerung — welches D gibt max Q?
        D_opt_arr = np.array([D_scan[Q_stack[:, j].argmax()] for j in range(len(V_arr))])

        # Repräsentativer Wert: bei V_krit_orig
        idx_vk  = int(np.argmin(np.abs(V_arr - V_krit_orig)))
        D_opt_vk = D_opt_arr[idx_vk]
        Q_opt_vk = Q_opt_line[idx_vk]
        # Und im sinnvollen Bereich: Mittelwert der optimalen Blende
        mask_sinn = (V_arr >= V_min_orig) & (V_arr <= V_max_orig)
        D_opt_mean = float(D_opt_arr[mask_sinn].mean()) if mask_sinn.any() else D_opt_vk

        ax.plot(V_arr, Q_opt_line, color=GRN, lw=1.8, ls="-.",
                label=f"Optimale Blende (max Q je V)  ⌀≈{D_opt_mean:.0f}mm im sinnv. Bereich")

        # Annotation opt. Blende: rechts und unterhalb, weg von Blende-Box
        x_opt = min(V_krit_orig + 50, 340)
        y_opt = max(Q_opt_vk - 0.12, 0.05)
        ax.annotate(
            f"bei V={V_krit_orig:.0f}×:\nopt. Blende ≈{D_opt_vk:.0f}mm\nQ={Q_opt_vk:.3f}",
            xy=(V_krit_orig, Q_opt_vk),
            xytext=(x_opt, y_opt),
            fontsize=8, color=GRN, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GRN, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRN, alpha=0.88)
        )

        # ── Referenzlinien Q=0.90 / Q=0.70 ──────────────────────────────────
        for q_thresh, col, lbl in [(0.9, GRN, "Q=0.90"), (0.7, "#BA7517", "Q=0.70")]:
            ax.axhline(q_thresh, color=col, lw=0.8, ls=":", alpha=0.6)
            ax.text(32, q_thresh + 0.008, lbl, fontsize=7, color=col, alpha=0.8)

        if 30 <= Vk_orig <= V_max_plot:
            ax.axvline(Vk_orig, color=COR, lw=1.0, ls="--", alpha=0.5)
            ax.text(Vk_orig + 2, 0.03, f"V½={Vk_orig:.0f}×",
                    fontsize=7, color=COR, rotation=90, va="bottom", alpha=0.7)

        # ── Sinnvoller Vergrößerungsbereich ──────────────────────────────────
        ax.axvspan(30, min(V_min_orig, V_max_plot),
                   color="#888", alpha=0.08, zorder=0)
        if V_min_orig > 32:
            ax.text((30 + min(V_min_orig, V_max_plot)) / 2, 0.96,
                    "nicht sinnvoll\n(zu niedrig)",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color="#999", style="italic", zorder=1)
        if V_max_orig < V_max_plot:
            ax.axvspan(min(V_max_orig, V_max_plot), V_max_plot,
                       color="#888", alpha=0.08, zorder=0)
            ax.text((V_max_orig + V_max_plot) / 2, 0.96,
                    "nicht sinnvoll\n(zu hoch)",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color="#999", style="italic", zorder=1)
        for V_m, lbl in [
            (V_min_orig, f"V_min={V_min_orig:.0f}×\n(AP=7mm)"),
            (V_max_orig, f"V_max={V_max_orig:.0f}×\n({'AP=0.5mm' if V_max_orig >= V_max_AP*0.99 else '2×V_krit'})"),
        ]:
            if 30 < V_m < V_max_plot:
                ax.axvline(V_m, color="#999", lw=1.0, ls=":", alpha=0.8)
                ax.text(V_m + 2, 0.72, lbl, fontsize=7, color="#777",
                        rotation=90, va="center", alpha=0.9)
        if D_blend < D * 0.98:
            for V_m, lbl in [
                (V_min_blend, f"V_min={V_min_blend:.0f}×\n(Blende)"),
                (V_max_blend, f"V_max={V_max_blend:.0f}×\n({'AP' if V_max_blend >= V_max_AP_bl*0.99 else '2×V_krit'} Blende)"),
            ]:
                if 30 < V_m < V_max_plot:
                    ax.axvline(V_m, color=ACC, lw=1.0, ls=":", alpha=0.6)
                    ax.text(V_m + 2, 0.50, lbl, fontsize=7, color=ACC,
                            rotation=90, va="center", alpha=0.8)

        ax.set_xlim(30, V_max_plot)
        ax.set_ylim(0, min(1.15, max(float(Q_orig.max()), float(Q_bl.max()) if D_blend < D*0.98 else 0) * 1.15))
        ax.set_xlabel("")
        ax.set_ylabel("Q_vis normiert auf Original-Paraboloid  (MTF×CSF)", fontsize=9)

        # Rayleigh-Hinweis: ins Diagramm oben links (nicht über dem Titel)
        rayleigh_ok = S_orig >= 0.80
        titel_farbe = "#2e7d32" if rayleigh_ok else "#c62828"
        rayleigh_text = (
            "✓ Rayleigh S ≥ 0.80 — Abblenden nicht sinnvoll"
            if rayleigh_ok else
            f"✗ Rayleigh nicht erfüllt (S={S_orig:.3f}) — Abblenden kann helfen"
        )
        ax.text(0.02, 0.97, rayleigh_text,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontweight="bold", color=titel_farbe,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=titel_farbe, alpha=0.9))

        ax.set_title(
            f"Blendeneffekt  —  D={D:.0f}mm  f/{f/D:.1f}  S={S_orig:.3f}"
            f"  |  Blende={D_blend:.0f}mm ({D_blend/D*100:.0f}%)"
            f"  |  d_fang={d_fang:.0f}mm  ε={eps_orig:.3f}→{d_fang/D_blend:.3f}",
            fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        self._ax_fmt(ax)

        # ── Unterer Panel: Gain-Faktor mit Leuchtdichte-Effekt ────────────────
        objekte_lum = [
            ("Jupiter",            600, "#e65100"),
            ("Saturn",             200, "#c8860a"),
            ("Schwacher Planet",    50, "#1565c0"),
            ("Heller Nebel",         5, "#6a1b9a"),
            ("Schwaches Deep-Sky",   0.5, "#1b5e20"),
        ]
        L_arr_lum = [L for _, L, _ in objekte_lum]
        # MTF einmal pro Öffnung berechnen, dann über V×L vektorisieren
        Q_bl_mat  = Q_vis_blende_batch(D_blend, f, lam, V_arr, L_arr_lum, D_ref=D)
        Q_ori_mat = Q_vis_blende_batch(D,       f, lam, V_arr, L_arr_lum, D_ref=D)
        gain_mat  = Q_bl_mat / np.maximum(Q_ori_mat, 1e-12)   # (nV, nL)

        for li, (name, L_obj, col) in enumerate(objekte_lum):
            gain = gain_mat[:, li]
            ls = "-" if L_obj >= 500 else ("--" if L_obj >= 5 else ":")
            ax2.plot(V_arr, gain, color=col, lw=1.8, ls=ls,
                     label=f"{name}  (L≈{L_obj:.0f} cd/m²)")

        ax2.axhline(1.0, color="#888", lw=1.2, ls="-")
        ax2.text(V_max_plot * 0.98, 1.02, "Abblenden lohnt sich ↑",
                 fontsize=8, color="#2e7d32", ha="right")
        ax2.text(V_max_plot * 0.98, 0.96, "Abblenden schadet ↓",
                 fontsize=8, color="#c62828", ha="right")

        # Y-Limits dynamisch — gain_mat schon berechnet
        y_max = min(float(gain_mat.max()) * 1.1, 4.0)
        y_min = max(float(gain_mat.min()) * 0.95, 0.3)
        ax2.set_ylim(y_min, y_max)
        ax2.fill_between(V_arr, 1.0, y_max, color="#2e7d32", alpha=0.04)
        ax2.fill_between(V_arr, y_min, 1.0, color="#c62828", alpha=0.06)

        # Rayleigh-Linie: ab welcher Blende ist S=0.80 erreicht?
        D_rayleigh = None
        for Db in np.linspace(D * 0.4, D, 40):   # 40 reichen; _strehl_exakt gecacht
            if S_von_D(Db) >= 0.80:
                D_rayleigh = Db
                break
        if D_rayleigh is not None and D_rayleigh > D * 0.42:
            ax2.axvline(V_max_plot * 0.5, color="#888", lw=0, ls=":")  # dummy
            ax.axhline(0.80, color="#BA7517", lw=1.0, ls="--", alpha=0.7)
            ax.text(32, 0.81, f"Rayleigh S=0.80  (D_blend≥{D_rayleigh:.0f}mm)",
                    fontsize=7, color="#BA7517", alpha=0.9)

        if 30 <= Vk_orig <= V_max_plot:
            ax2.axvline(Vk_orig, color=COR, lw=1.0, ls="--", alpha=0.5)

        # ── Sinnvoller Bereich im unteren Panel ──────────────────────────────
        ax2.axvspan(30, min(V_min_orig, V_max_plot),
                    color="#888", alpha=0.08, zorder=0)
        if V_min_orig > 32:
            ax2.text((30 + min(V_min_orig, V_max_plot)) / 2, 0.96,
                     "nicht sinnvoll\n(zu niedrig)",
                     transform=ax2.get_xaxis_transform(), ha="center", va="top",
                     fontsize=7, color="#999", style="italic", zorder=1)
        if V_max_orig < V_max_plot:
            ax2.axvspan(min(V_max_orig, V_max_plot), V_max_plot,
                        color="#888", alpha=0.08, zorder=0)
            ax2.text((V_max_orig + V_max_plot) / 2, 0.96,
                     "nicht sinnvoll\n(zu hoch)",
                     transform=ax2.get_xaxis_transform(), ha="center", va="top",
                     fontsize=7, color="#999", style="italic", zorder=1)
        for V_m, lbl in [(V_min_orig, f"V_min={V_min_orig:.0f}×"),
                          (V_max_orig, f"V_max={V_max_orig:.0f}×")]:
            if 30 < V_m < V_max_plot:
                ax2.axvline(V_m, color="#999", lw=1.0, ls=":", alpha=0.8)
                ax2.text(V_m + 2, y_min + (y_max-y_min)*0.05, lbl,
                         fontsize=7, color="#777", rotation=90, va="bottom")
        if D_blend < D * 0.98:
            for V_m, lbl in [(V_min_blend, f"V_min={V_min_blend:.0f}×\n(Blende)"),
                              (V_max_blend, f"V_max={V_max_blend:.0f}×\n(Blende)")]:
                if 30 < V_m < V_max_plot:
                    ax2.axvline(V_m, color=ACC, lw=1.0, ls=":", alpha=0.6)
                    ax2.text(V_m + 2, y_min + (y_max-y_min)*0.25, lbl,
                             fontsize=7, color=ACC, rotation=90, va="bottom")

        ax2.set_xlim(30, V_max_plot)
        ax2.set_xlabel(T("ax_vergr"), fontsize=10)
        ax2.set_ylabel("Gain-Faktor  Q_blend / Q_orig", fontsize=9)
        ax2.legend(fontsize=7, loc="upper right", ncol=2)
        self._ax_fmt(ax2)

        # v7.3.1: Texte außerhalb des sichtbaren Bereichs (z.B. Q-Linien-Beschriftung, "Abblenden lohnt sich")
        # nicht zeichnen – sie standen sonst frei neben/unter dem Diagramm.
        for _a in (ax, ax2):
            _y0, _y1 = sorted(_a.get_ylim()); _x0, _x1 = sorted(_a.get_xlim())
            for _t in list(_a.texts):
                if type(_t).__name__ == "Text" and _t.get_transform() == _a.transData:
                    _x, _y = _t.get_position()
                    if not (_x0 <= _x <= _x1 and _y0 <= _y <= _y1):
                        _t.remove()
        fig.tight_layout(); canvas.draw_idle()
