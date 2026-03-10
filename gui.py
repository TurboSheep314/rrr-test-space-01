import os
import json
import threading
import urllib.request
import zipfile
from pathlib import Path
import webbrowser
import folium
import subprocess
import sys

import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
# pip install pywebview

from src.geo_utils import load_zip_shapes
from src.variance_analysis import compute_relative_variance_cv
from src.composite_score import compute_composite_score
from src.heatmap import create_zip_heatmap


# ----------------------------
# Helpers (ported from app.py)
# ----------------------------
def to_numeric_loose(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA, "NA": pd.NA})
    x = x.str.replace(r"[\$,]", "", regex=True)
    x = x.str.replace("%", "", regex=False)
    x = x.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def canon(col: str) -> str:
    s = str(col).strip().lower()
    for ch in ["(", ")", "%", "$", ",", "/", "-", "_"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def load_scores(sheet_id: str, gid: int = 0) -> pd.DataFrame:
    """
    Same logic as Streamlit version: load public sheet CSV, detect Zip & Overall Score,
    normalize Zip, keep only real 5-digit zips.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    compare_url = f"https://docs.google.com/spreadsheets/d/1MSXgoA67XpADAKwzwTYMwuj2qQS8RsAkBq5oH8b0PA4/edit?gid=0#gid=0"
    
    df = pd.read_csv(url, dtype=str)

    df_compare = pd.read_csv(compare_url, dtype=str)

    df.columns = df.columns.map(lambda c: str(c).strip())
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

    zip_col = None
    overall_col = None

    for c in df.columns:
        cc = canon(c)

        if zip_col is None and (
            cc == "zip"
            or "zip code" in cc
            or "zipcode" in cc
            or "postal" in cc
            or "zcta" in cc
        ):
            zip_col = c

        if overall_col is None and (
            cc == "overall score"
            or ("overall" in cc and "score" in cc)
            or cc in ["overall", "score", "total score"]
        ):
            overall_col = c

    if zip_col is None:
        raise ValueError(f"Missing ZIP column. Found: {list(df.columns)}")
    if overall_col is None:
        raise ValueError(f"Missing Overall Score column. Found: {list(df.columns)}")

    df = df.rename(columns={zip_col: "Zip", overall_col: "Overall Score"})

    z = (
        df["Zip"].astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
    )
    z = z.replace("", pd.NA)
    z = z.str.zfill(5).str[-5:]

    df["Zip"] = z
    df = df[df["Zip"].notna()]
    df = df[df["Zip"].str.match(r"^\d{5}$", na=False)]
    df = df[df["Zip"] != "00000"]

    return df


def ensure_zcta_shapes() -> Path:
    """
    Download TIGER/Line ZCTA520 shapefile if needed.
    Returns path to .shp file.
    """
    url = "https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip"
    extract_dir = Path("data/zcta_cache")
    shp_path = extract_dir / "tl_2024_us_zcta520.shp"

    if not shp_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_path = extract_dir / "tl_2024_us_zcta520.zip"
        urllib.request.urlretrieve(url, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    return shp_path


# ----------------------------
# GUI
# ----------------------------


class ZipHeatmapGUI(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)

        self.app_dir = Path(__file__).resolve().parent
        self.config_path = self.app_dir / "sheets.json"
        self.out_dir = self.app_dir / "out"
        self.out_dir.mkdir(exist_ok=True)

        # Runtime state (replaces st.session_state)
        self.state = {
            "zip_gdf": None,
            "df": None,
            "merged": None,
            "top2": [],
            "map_center": None,   # (lat, lon)
            "map_zoom": None,     # int
            "map_html": None,     # Path
            "last_default_center": None,
            "last_default_zoom": None,
        }

        self.sheets_cfg = self._load_sheets_cfg()

        self._build_ui()

        self._slider_job = None
        self._recompute_running = False

        # Build dataset label mapping (city label -> key)
        self.options = self._build_dataset_options()
        self.dataset_combo["values"] = [city for city, _ in self.options]
        if self.options:
            self.dataset_combo.current(0)

        # Initialize view + auto build
        self.log_line("Ready. Building map...")
        self.build_map()

    def _load_gamma(self) -> float | None:
        gamma_path = self.app_dir / "out" / "intake_profile.json"
        if not gamma_path.exists():
            return None
        try:
            data = json.loads(gamma_path.read_text())
            gamma = data.get("personalization", {}).get("gamma")
            if gamma is None:
                return None
            return float(gamma)
        except Exception:
            return None

    def _load_sheets_cfg(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError("Missing sheets.json (put it in the same folder as gui.py)")
        try:
            return json.loads(self.config_path.read_text())
        except Exception as e:
            raise ValueError(f"Could not parse sheets.json: {e}") from e

    def _build_dataset_options(self):
        options = []
        for dataset_key, cfg in self.sheets_cfg.items():
            city = cfg.get("comparison_city", dataset_key)
            options.append((city, dataset_key))
        options.sort(key=lambda x: x[0].lower())
        return options

    def _build_ui(self):
        self.master.title("ZIP Heatmap (Local GUI)")
        self.master.geometry("1200x800")

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        right = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsw")
        right.grid(row=0, column=1, sticky="nsew")

        # LEFT: controls in tabs
        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)

        tab_data = ttk.Frame(nb, padding=10)
        tab_weights = ttk.Frame(nb, padding=10)
        tab_actions = ttk.Frame(nb, padding=10)

        nb.add(tab_data, text="Dataset")
        nb.add(tab_weights, text="Weights")
        nb.add(tab_actions, text="Run / View")

        # Dataset tab
        ttk.Label(tab_data, text="Comparison city dataset").grid(row=0, column=0, sticky="w")
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(tab_data, textvariable=self.dataset_var, state="readonly", width=35)
        self.dataset_combo.grid(row=1, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(tab_data, text="Map scope").grid(row=2, column=0, sticky="w")
        self.scope_var = tk.StringVar(value="Massachusetts (fast)")
        self.scope_combo = ttk.Combobox(
            tab_data,
            textvariable=self.scope_var,
            values=["All US (slow)", "Massachusetts (fast)"],
            state="readonly",
            width=35,
        )
        self.scope_combo.grid(row=3, column=0, sticky="ew", pady=(4, 12))

        # Optional override for sheet_id/gid if you want
        self.override_sheet = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab_data, text="Override sheet_id/gid (advanced)", variable=self.override_sheet,
                        command=self._toggle_override).grid(row=4, column=0, sticky="w")

        self.sheet_id_var = tk.StringVar(value="")
        self.gid_var = tk.StringVar(value="0")

        self.sheet_id_entry = ttk.Entry(tab_data, textvariable=self.sheet_id_var, width=40, state="disabled")
        self.gid_entry = ttk.Entry(tab_data, textvariable=self.gid_var, width=20, state="disabled")

        ttk.Label(tab_data, text="Sheet ID").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.sheet_id_entry.grid(row=6, column=0, sticky="ew", pady=(4, 6))
        ttk.Label(tab_data, text="GID").grid(row=7, column=0, sticky="w")
        self.gid_entry.grid(row=8, column=0, sticky="w", pady=(4, 0))

        tab_data.columnconfigure(0, weight=1)

        # Weights tab
        ttk.Label(tab_weights, text="Weights (top-2 variance only)", font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.slider_frame = ttk.Frame(tab_weights)
        self.slider_frame.grid(row=1, column=0, sticky="ew")

        # Placeholders; populated after compute_top2()
        self.w1_var = tk.DoubleVar(value=0.50)
        self.w2_var = tk.DoubleVar(value=0.50)
        self.col1_label = tk.StringVar(value="(top1 not computed yet)")
        self.col2_label = tk.StringVar(value="(top2 not computed yet)")

        
        self.slider1 = self._make_slider(
            self.slider_frame,
            "Top-1",
            self.col1_label,
            self.w1_var,
            0,
        )

        self.slider2 = self._make_slider(
            self.slider_frame,
            "Top-2",
            self.col2_label,
            self.w2_var,
            1,
        )

        # Disable sliders until data is loaded
        self.slider1.state(["disabled"])
        self.slider2.state(["disabled"])

        tab_weights.columnconfigure(0, weight=1)

        # Actions tab
        ttk.Button(tab_actions, text="Build Map", command=self.build_map).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(tab_actions, text="Reset Map View", command=self.reset_view).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(tab_actions, text="Open Map Window (embedded)", command=self.open_map_window).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(tab_actions, text="Open Map HTML in Browser", command=self.open_map_in_browser).grid(row=3, column=0, sticky="ew")
        ttk.Button(tab_actions,text="Refresh Map in Browser",command=self.open_map_in_browser).grid(row=4, column=0, sticky="ew", pady=(8, 0))

        tab_actions.columnconfigure(0, weight=1)

        # RIGHT: logs + quick info
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Status / Logs", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w")

        self.log = tk.Text(right, height=30)
        self.log.grid(row=1, column=0, sticky="nsew", pady=(8, 8))

        self.status_var = tk.StringVar(value="Idle")

        ttk.Label(
            right,
            textvariable=self.status_var,
            font=("Arial", 10, "italic")
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))

        self.quick = tk.Text(right, height=8)
        self.quick.grid(row=2, column=0, sticky="ew")

        right.columnconfigure(0, weight=1)

    def load_data(self):
        threading.Thread(target=self._load_data_worker, daemon=True).start()

    def recompute_map(self):
        if self.state["df"] is None or self.state["zip_gdf"] is None:
            messagebox.showwarning("No Data", "Load data first.")
            return
        threading.Thread(target=self._recompute_worker, daemon=True).start()

    def _load_data_worker(self):
        try:
            _, cfg, city = self._get_selected_cfg()
            sheet_id, gid = self._resolve_sheet_id_gid(cfg)

            self.log_line(f"Loading data for {city}")

            df = load_scores(sheet_id, gid)
            zip_gdf = load_zip_shapes(str(ensure_zcta_shapes()))

            if self.scope_var.get() == "Massachusetts (fast)":
                zip_gdf = zip_gdf[zip_gdf["Zip"].str.startswith("0")]

            self.state["df"] = df
            self.state["zip_gdf"] = zip_gdf

            self.log_line(f"Loaded scores={len(df):,}, shapes={len(zip_gdf):,}")
        except Exception as e:
            self.log_line(f"ERROR: {e}")
    def _recompute_worker(self):
        try:
            df = self.state["df"].copy()
            zip_gdf = self.state["zip_gdf"]

            exclude_cols = {"Town", "Zip", "Overall Score"}
            for c in df.columns:
                if c not in exclude_cols:
                    df[c] = to_numeric_loose(df[c])

            cv = compute_relative_variance_cv(df)
            top2 = cv.index[:2].tolist()
            self.state["top2"] = top2

            self.master.after(0, lambda: self._update_slider_labels(top2))

            columns = []
            weights = {}
            if len(top2) >= 1:
                columns.append(top2[0])
                weights[top2[0]] = float(self.w1_var.get())
            if len(top2) >= 2:
                columns.append(top2[1])
                weights[top2[1]] = float(self.w2_var.get())

            df["Composite Score"] = compute_composite_score(df, columns, weights)

            df["Zip"] = df["Zip"].astype(str).str.zfill(5)
            zip_gdf["Zip"] = zip_gdf["Zip"].astype(str).str.zfill(5)

            merged = zip_gdf.merge(df, on="Zip", how="inner")
            self.state["merged"] = merged

            gamma = self._load_gamma()
            score_col = "Composite Score"
            if gamma is not None:
                merged["Composite Score (Calibrated)"] = merged["Composite Score"] * gamma
                score_col = "Composite Score (Calibrated)"

            if self.state["map_center"] is None:
                minx, miny, maxx, maxy = merged.total_bounds
                self.state["map_center"] = ((miny + maxy) / 2, (minx + maxx) / 2)
                self.state["map_zoom"] = 9

            m = create_zip_heatmap(
                merged,
                score_col,
                center=self.state["map_center"],
                zoom=self.state["map_zoom"],
            )

            html_path = self.out_dir / "zip_heatmap.html"
            m.save(str(html_path))
            self.state["map_html"] = html_path

            self.log_line("Recomputed map (fast).")

        except Exception as e:
            self.log_line(f"ERROR: {e}")
        finally:
            self._recompute_running = False
            self.master.after(0, lambda: self.status_var.set("Ready"))
            self.master.after(
                0,
                lambda: webbrowser.open(Path(html_path).resolve().as_uri())
            )

    def _toggle_override(self):
        if self.override_sheet.get():
            self.sheet_id_entry.config(state="normal")
            self.gid_entry.config(state="normal")
        else:
            self.sheet_id_entry.config(state="disabled")
            self.gid_entry.config(state="disabled")

    def _on_slider_change(self):
    # Cancel previous scheduled recompute
        if self._slider_job is not None:
            self.after_cancel(self._slider_job)

    # Schedule a recompute 300ms after last change
        self._slider_job = self.after(300, self._safe_recompute)

    def _safe_recompute(self):
        self._slider_job = None

    # Only recompute if data is already loaded
        if self.state["df"] is None or self.state["zip_gdf"] is None:
            return

        # Avoid overlapping recomputes
        if getattr(self, "_recompute_running", False):
            return

        self._recompute_running = True
        self.status_var.set("Updating map…")
        threading.Thread(target=self._recompute_worker, daemon=True).start()

    def _make_slider(self, parent, label_prefix, col_label_var, value_var, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=6)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=f"{label_prefix} column:").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=col_label_var).grid(row=0, column=1, sticky="w")

        scale = ttk.Scale(
            frame,
            from_=0.0,
            to=1.0,
            variable=value_var,
            orient="horizontal",
            command=lambda _: self._on_slider_change()
        )
        scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        val_lbl = ttk.Label(frame, textvariable=value_var, width=6)
        val_lbl.grid(row=1, column=2, sticky="e", padx=(8, 0))

        parent.columnconfigure(0, weight=1)
        return scale

    def log_line(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def set_quick_info(self, text: str):
        self.quick.delete("1.0", "end")
        self.quick.insert("end", text)

    def _get_selected_cfg(self):
        # Map selected city label -> dataset key
        city = self.dataset_var.get()
        if not city:
            return None, None, None

        mapping = {c: k for c, k in self.options}
        dataset_key = mapping.get(city)
        cfg = self.sheets_cfg.get(dataset_key, {})

        comparison_city = cfg.get("comparison_city", city)
        return dataset_key, cfg, comparison_city

    def _resolve_sheet_id_gid(self, cfg: dict):
        """
        Your Streamlit app currently hardcodes SHEET_ID/GID.
        Here, we prefer sheets.json if present, otherwise fall back to the same defaults.
        """
        if self.override_sheet.get():
            sid = self.sheet_id_var.get().strip()
            gid = int(self.gid_var.get().strip())
            if not sid:
                raise ValueError("Override enabled but Sheet ID is empty.")
            return sid, gid

        sid = cfg.get("spreadsheet_id")
        gid = int(cfg.get("gid", 0))

        # Fallback to your hardcoded values if config lacks them
        if not sid:
            sid = "1aZmL78kZZgcKa4anOHkKYC-TRQyaHrJm"
        if "gid" not in cfg:
            gid = 1097485755

        return sid, gid

    def reset_view(self):
        if self.state["last_default_center"] is None:
            messagebox.showinfo("Reset", "No map built yet.")
            return
        self.state["map_center"] = self.state["last_default_center"]
        self.state["map_zoom"] = self.state["last_default_zoom"]
        self.log_line(f"Reset view to default center={self.state['map_center']} zoom={self.state['map_zoom']}")
        messagebox.showinfo("Reset", "Map view reset. Rebuild or reopen map window to see it.")

    def build_map(self):
        # Run heavy work in a thread
        threading.Thread(target=self._build_map_worker, daemon=True).start()

    def _build_map_worker(self):
        try:
            _, cfg, comparison_city = self._get_selected_cfg()
            if cfg is None:
                raise ValueError("No dataset selected.")

            self.log_line(f"Building map for: {comparison_city}")
            sheet_id, gid = self._resolve_sheet_id_gid(cfg)
            self.log_line(f"Using sheet_id={sheet_id}, gid={gid}")

            # Load scores
            df = load_scores(sheet_id, gid=gid)
            self.state["df"] = df
            self.log_line(f"Loaded scores rows={len(df):,} cols={len(df.columns)}")

            # Load shapes
            shp_path = ensure_zcta_shapes()
            zip_gdf = load_zip_shapes(str(shp_path))
            self.log_line(f"Loaded ZCTA shapes rows={len(zip_gdf):,}")

            # Scope filter
            scope = self.scope_var.get()
            if scope == "Massachusetts (fast)":
                zip_gdf = zip_gdf[zip_gdf["Zip"].astype(str).str.startswith("0")]
                self.log_line(f"Scope=MA fast → shapes rows={len(zip_gdf):,}")

            self.state["zip_gdf"] = zip_gdf

            # Numeric conversion
            exclude_cols = {"Town", "Zip", "Overall Score"}
            for c in df.columns:
                if c in exclude_cols:
                    continue
                df[c] = to_numeric_loose(df[c])

            candidates = [c for c in df.columns if c not in exclude_cols]
            nonnull = df[candidates].notna().sum().sort_values(ascending=False)
            keep = nonnull[nonnull >= 5].index.tolist()

            if len(keep) == 0:
                top2 = []
            else:
                cv = compute_relative_variance_cv(df)
                top2 = cv.index[:2].tolist()

            self.state["top2"] = top2
            self.log_line(f"Top-2 variance columns: {top2 if top2 else '(none)'}")

            # Update slider labels on UI thread
            self.master.after(0, lambda: self._update_slider_labels(top2))

            # Weights from sliders (if fewer than 2 columns, only use what's available)
            columns = []
            weights = {}
            if len(top2) >= 1:
                columns.append(top2[0])
                weights[top2[0]] = float(self.w1_var.get())
            if len(top2) >= 2:
                columns.append(top2[1])
                weights[top2[1]] = float(self.w2_var.get())

            if len(top2) < 2:
                self.log_line("WARNING: Not enough numeric columns for two variance sliders; using available columns only.")

            # Composite score
            df["Composite Score"] = compute_composite_score(df, columns, weights)

            # Normalize zips and filter to valid geometry overlap
            df["Zip"] = df["Zip"].astype(str).str.zfill(5)
            zip_gdf["Zip"] = zip_gdf["Zip"].astype(str).str.zfill(5)

            valid_zips = set(df["Zip"]).intersection(set(zip_gdf["Zip"]))
            before = len(df)
            df = df[df["Zip"].isin(valid_zips)].copy()
            after = len(df)

            dropped = before - after
            self.log_line(f"Filtered to valid geometries: kept={after:,} dropped={dropped:,}")

            # Merge
            merged = zip_gdf.merge(df, on="Zip", how="inner")
            self.state["merged"] = merged

            matched_zips = set(merged["Zip"].astype(str))
            all_data_zips = set(df["Zip"].astype(str))
            missing_zips = sorted(list(all_data_zips - matched_zips))

            # Center + default zoom
            minx, miny, maxx, maxy = merged.total_bounds
            center_lat = (miny + maxy) / 2
            center_lon = (minx + maxx) / 2

            default_zoom = 9 if scope == "Massachusetts (fast)" else 4
            self.state["last_default_center"] = (center_lat, center_lon)
            self.state["last_default_zoom"] = default_zoom

            # Persistent view (like session_state)
            if self.state["map_center"] is None:
                self.state["map_center"] = (center_lat, center_lon)
            if self.state["map_zoom"] is None:
                self.state["map_zoom"] = default_zoom

            # Apply adaptive gain (gamma) if available
            gamma = self._load_gamma()
            score_col = "Composite Score"
            if gamma is not None:
                merged["Composite Score (Calibrated)"] = merged["Composite Score"] * gamma
                score_col = "Composite Score (Calibrated)"

            # Build folium map
            m = create_zip_heatmap(
                merged,
                score_col,
                center=self.state["map_center"],
                zoom=self.state["map_zoom"],
            )
            

        # Inject JS hook to report pan/zoom
            m.get_root().html.add_child(
                folium.Element(f"""
                <script>
                function reportView(map) {{
                    map.on('moveend zoomend', function() {{
                        const c = map.getCenter();
                        window.pywebview.api.update_view(c.lat, c.lng, map.getZoom());
                    }});
                }}
                </script>
                """))
            
            for k in m._children:
                if m._children[k].__class__.__name__ == "Map":
                    m._children[k].add_child(
                        folium.Element("<script>reportView(this);</script>")
                    )


            # Save HTML
            html_path = self.out_dir / "zip_heatmap.html"
            m.save(str(html_path))
            self.state["map_html"] = html_path

            # Quick info panel
            info = (
                f"Dataset: {comparison_city}\n"
                f"Scope: {scope}\n"
                f"Rows merged: {len(merged):,}\n"
                f"Top2: {top2}\n"
                f"Missing zips: {len(missing_zips)}\n"
                f"HTML: {html_path}\n"
            )
            self.master.after(0, lambda: self.set_quick_info(info))

            self.log_line(f"Map saved: {html_path}")
            self.log_line(f"Unmatched ZIP codes count: {len(missing_zips)}")
            if missing_zips:
                self.log_line(f"First 25 unmatched: {missing_zips[:25]}")

        except Exception as e:
            self.log_line(f"ERROR: {e}")
            self.master.after(0, lambda: messagebox.showerror("Build Map Failed", str(e)))

        self.master.after(0, lambda: self.slider1.state(["!disabled"]))
        self.master.after(0, lambda: self.slider2.state(["!disabled"]))

    def _update_slider_labels(self, top2):
        if len(top2) >= 1:
            self.col1_label.set(top2[0])
        else:
            self.col1_label.set("(none)")
        if len(top2) >= 2:
            self.col2_label.set(top2[1])
        else:
            self.col2_label.set("(none)")

    def open_map_window(self):
        html_path = self.state.get("map_html")
        if not html_path or not Path(html_path).exists():
            messagebox.showwarning("No Map Yet", "Build the map first.")
            return
        webbrowser.open(Path(html_path).resolve().as_uri())

      
    def open_map_in_browser(self):
        html_path = self.state.get("map_html")
        if not html_path or not Path(html_path).exists():
            messagebox.showwarning("No Map Yet", "Build the map first.")
            return
        webbrowser.open(Path(html_path).resolve().as_uri())


def main():
    # Run the LLM intake chat first (blocking). This writes out/intake_profile.json.
    chat_script = Path(__file__).resolve().parent / "scripts" / "minimal_llama_chat.py"
    if chat_script.exists():
        subprocess.run([sys.executable, str(chat_script)], check=False)
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    ZipHeatmapGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
