"""Earth System Trend Detective — Tactical Telemetry Satellite Monitor."""

from __future__ import annotations

import os
import time
import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

try:
    from scipy.stats import t as student_t
except ImportError:
    student_t = None

st.set_page_config(
    page_title="Earth System Trend Detective | Tactical Telemetry",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUST_API_URL = os.environ.get("NASA_QUERY_URL", "http://localhost:8080")

COLORS = {
    "bg": "#050505",
    "panel": "#0e1117",
    "cyan": "#00f3ff",
    "crimson": "#ff2a6d",
    "text": "#e8f0f2",
    "muted": "#75838d",
    "green": "#9dffb0",
}

METRICS = {
    "ndwi": {
        "name": "Water Index",
        "short": "NDWI",
        "unit": "NDWI",
        "color": COLORS["cyan"],
        "trend": "declining surface water",
        "clip_bounds": (-0.35, 0.88),
        "slope_impact": -0.0187,
    },
    "lst_c": {
        "name": "Land Surface Temperature",
        "short": "LST",
        "unit": "°C",
        "color": COLORS["crimson"],
        "trend": "warming",
        "clip_bounds": (-40.0, 70.0),
        "slope_impact": 0.04,
    },
    "ndbi": {
        "name": "Urban Index",
        "short": "NDBI",
        "unit": "NDBI",
        "color": "#ff38c8",
        "trend": "urban expansion",
        "clip_bounds": (-0.55, 0.88),
        "slope_impact": 0.0040,
    },
    "ndsi": {
        "name": "Snow Index",
        "short": "NDSI",
        "unit": "NDSI",
        "color": "#38bdf8",
        "trend": "declining snow and ice",
        "clip_bounds": (-0.35, 0.90),
        "slope_impact": -0.0040,
    },
    "salinity_index": {
        "name": "Salinity Index",
        "short": "SALINITY",
        "unit": "SI",
        "color": "#ffbf47",
        "trend": "rising salinity",
        "clip_bounds": (0.0, 1.0),
        "slope_impact": 0.0035,
    },
}

REGIONS_CONFIG = {
    "sea_aquisition": {
        "title": "Aral Sea (Desiccation & Salinization)",
        "selector_label": "Aral Sea",
        "subtitle": "Aral Sea · desiccation and salinization",
        "files": ["sea_aquisition.parquet", "data/sea_aquisition.parquet", "raw_landsat_grid.parquet"],
        "lat": 45.20, "lon": 59.80, "zoom": 6.2,
        "scale_km": 50, "mode": "water",
        "primary_idx": "ndwi", "primary_label": "Mean Water (NDWI)",
        "target_metrics": ("ndwi", "lst_c", "salinity_index"),
        "grid_source_label": "LOCAL SEA PARQUET GRID",
    },
    "urban_aquisition": {
        "title": "Las Vegas & Lake Mead (Urban Growth vs. Water Depletion)",
        "selector_label": "Las Vegas & Lake Mead",
        "subtitle": "Las Vegas & Lake Mead · urban growth and water depletion",
        "files": ["urban_aquisition.parquet", "data/urban_aquisition.parquet"],
        "lat": 36.285, "lon": -114.765, "zoom": 8.3,
        "scale_km": 20, "mode": "urban_water",
        "primary_idx": "ndwi", "primary_label": "Mean Water (NDWI)",
        "target_metrics": ("ndwi", "ndbi", "lst_c", "salinity_index"),
        "grid_source_label": "LOCAL URBAN PARQUET GRID",
    },
    "snow_ice_aquisition": {
        "title": "Harvard & Columbia Glaciers, Alaska (Missing Ice)",
        "selector_label": "Harvard & Columbia Glaciers",
        "subtitle": "Harvard & Columbia Glaciers, Alaska · ice retreat",
        "files": ["snow_ice_aquisition.parquet", "data/snow_ice_aquisition.parquet"],
        "lat": 61.12, "lon": -147.38, "zoom": 8.1,
        "scale_km": 15, "mode": "ice",
        "primary_idx": "ndsi", "primary_label": "Mean Ice (NDSI)",
        "target_metrics": ("ndsi", "ndwi", "lst_c", "salinity_index"),
        "grid_source_label": "LOCAL GLACIER PARQUET GRID",
    },
}

LV_VALLEY_POLYGON = np.array([
    [-115.335, 36.320], [-115.220, 36.310], [-115.085, 36.285],
    [-115.025, 36.235], [-115.015, 36.125], [-114.915, 36.115],
    [-114.935, 36.010], [-115.055, 35.965], [-115.175, 35.960],
    [-115.275, 36.005], [-115.330, 36.085], [-115.355, 36.165],
    [-115.340, 36.245],
])

LV_1984_ANCHOR_POLYGON = np.array([
    [-115.235, 36.220], [-115.115, 36.230], [-115.040, 36.185],
    [-115.045, 36.090], [-115.130, 36.065], [-115.205, 36.085],
    [-115.240, 36.150],
])

_PICKABLE_COLS = [
    "longitude", "latitude", "r", "g", "b",
    "reason", "severity", "ndwi", "z_ndwi", "ndbi", "z_ndbi",
    "ndsi", "z_ndsi", "lst_c", "z_lst",
]
_SIM_COLS = ["longitude", "latitude", "r", "g", "b"]


def telemetry_layout() -> None:
    """Inject Tactical Telemetry CSS styling matching the dark dashboard specification."""
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp, [data-testid="stAppViewContainer"] { background: #050505; color: #e8f0f2; }
        [data-testid="stSidebar"] { background: #080a0d; border-right: 1px solid #1b2930; }
        @media (min-width: 768px) {
            [data-testid="stSidebar"] { width: 20rem !important; min-width: 20rem !important; }
        }
        .st-key-monitoring_region [data-baseweb="select"] { font-size: .8rem; }
        .st-key-monitoring_region [data-baseweb="select"] > div {
            min-height: 2.7rem; padding-left: .75rem; padding-right: .75rem;
        }
        .st-key-monitoring_region [data-baseweb="select"] input,
        .st-key-monitoring_region [data-baseweb="select"] span {
            font-size: .8rem !important; line-height: 1.3; letter-spacing: 0;
        }
        [data-testid="stHeader"] { background: rgba(5,5,5,.92); }
        html, body, [class*="css"] { font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace; }
        h1, h2, h3 { letter-spacing: .025em; }
        h1 { font-size: clamp(1.7rem, 2.6vw, 2.35rem) !important; margin-top: .15rem !important; }
        .eyebrow { color: #00f3ff; font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; }
        .terminal-title { border-bottom: 1px solid #20343a; padding: 0 0 1rem; margin-bottom: 1.1rem; }
        .panel-head {
            display:flex; justify-content:space-between; align-items:center; gap:1rem;
            color:#dce7e9; font-size:.82rem; letter-spacing:.08em; text-transform:uppercase;
            border-bottom:1px solid #1a2b30; padding:0 0 .55rem; margin:.15rem 0 .65rem;
        }
        .panel-head span:last-child { color:#71848a; font-size:.68rem; text-align:right; }
        .telemetry-card {
            box-sizing:border-box; min-width:0; overflow:hidden;
            background:#0e1117; border:1px solid #1c3037; padding:.7rem; min-height:76px;
        }
        .telemetry-label {
            color:#71848a; font-size:.60rem; line-height:1.35; letter-spacing:.09em;
            text-transform:uppercase; margin-bottom:.35rem; overflow-wrap:anywhere;
        }
        .telemetry-value {
            color:#e8f0f2; font-size:clamp(.78rem, 1vw, .98rem); line-height:1.35;
            font-weight:700; white-space:normal; overflow-wrap:anywhere;
        }
        .telemetry-value.cyan { color:#00f3ff; text-shadow:0 0 12px #00f3ff66; }
        .telemetry-value.hot { color:#ff2a6d; text-shadow:0 0 12px #ff2a6d77; }
        .small-note { color:#71848a; font-size:.72rem; line-height:1.55; }
        .map-legend {
            display:flex; flex-wrap:wrap; gap:.9rem; align-items:center;
            margin:.4rem 0 .75rem; color:#9aabb1; font-size:.66rem; letter-spacing:.06em;
        }
        .map-legend-item { display:inline-flex; align-items:center; gap:.38rem; }
        .legend-swatch { width:.62rem; height:.62rem; display:inline-block; border:1px solid #ffffff30; }
        .scale-strip {
            display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center;
            gap:1rem; padding:.45rem 0; margin:.35rem 0 .15rem;
            color:#9aabb1; font-size:.66rem; letter-spacing:.06em; text-transform:uppercase;
        }
        div[data-testid="stPlotlyChart"], div[data-testid="stVegaLiteChart"] {
            border:1px solid #192a30; background:#080b0e; padding:.25rem;
        }
        div.stButton > button { border-radius:0; }
        hr { border-color:#1b2b30; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def points_in_polygon(lons: np.ndarray, lats: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon mask."""
    inside = np.zeros(len(lons), dtype=bool)
    p1x, p1y = poly[0]
    for i in range(1, len(poly) + 1):
        p2x, p2y = poly[i % len(poly)]
        cond = ((p1y > lats) != (p2y > lats)) & (lons < (p2x - p1x) * (lats - p1y) / (p2y - p1y + 1e-12) + p1x)
        inside ^= cond
        p1x, p1y = p2x, p2y
    return inside


def resolve_parquet_path(region_key: str) -> str:
    """Return first existing parquet path (checks current dir, project root, and backend/)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    for candidate in REGIONS_CONFIG[region_key]["files"]:
        for path in [
            candidate,
            os.path.join(project_root, candidate),
            os.path.join(project_root, "backend", candidate),
        ]:
            if os.path.exists(path):
                return path
    return REGIONS_CONFIG[region_key]["files"][0]


def _rolling_median_3(mat: np.ndarray) -> np.ndarray:
    """Exact 2D row-wise 3-step trailing rolling median (min_periods=1) matching pandas."""
    out = np.empty_like(mat, dtype=np.float64)
    out[:, 0] = mat[:, 0]
    if mat.shape[1] > 1:
        out[:, 1] = 0.5 * (mat[:, 0] + mat[:, 1])
    if mat.shape[1] > 2:
        out[:, 2:] = np.median(np.stack([mat[:, :-2], mat[:, 1:-1], mat[:, 2:]], axis=0), axis=0)
    return out


@st.cache_resource
def load_and_compute_region(region_key: str) -> pd.DataFrame:
    """Load parquet, lock the 1984 anchor mask, and compute 1985-2026 changes + default RGB colors."""
    cfg = REGIONS_CONFIG[region_key]
    path = resolve_parquet_path(region_key)
    if not os.path.exists(path):
        st.error(f"Dataset file `{path}` not found. Make sure `.parquet` files are in the project root.")
        st.stop()

    raw = pd.read_parquet(path)
    fill_cols = ["ndwi", "ndbi", "ndsi", "salinity_index", "lst_c"]
    for col in fill_cols:
        if col not in raw.columns:
            raw[col] = 0.0

    lons_raw = raw["longitude"].to_numpy()
    lats_raw = raw["latitude"].to_numpy()
    u_lons = np.unique(lons_raw)
    u_lats = np.unique(lats_raw)
    if len(u_lons) > 2 and len(u_lats) > 2:
        dx = np.median(np.diff(u_lons)) * np.cos(np.radians(cfg["lat"]))
        dy = np.median(np.diff(u_lats))
        if dy / (dx + 1e-9) > 1.65:
            keep_mask = np.isin(lons_raw, u_lons[::2])
            raw = raw.loc[keep_mask]
            lons_raw = raw["longitude"].to_numpy()
            lats_raw = raw["latitude"].to_numpy()
            u_lons = np.unique(lons_raw)
            u_lats = np.unique(lats_raw)

    years_raw = raw["year"].to_numpy()
    u_years = np.unique(years_raw)

    lat_i = np.searchsorted(u_lats, lats_raw)
    lon_i = np.searchsorted(u_lons, lons_raw)
    pair_key = lat_i.astype(np.int64) * len(u_lons) + lon_i
    u_pairs, c_idx = np.unique(pair_key, return_inverse=True)
    y_idx = np.searchsorted(u_years, years_raw)

    n_coords = len(u_pairs)
    n_years = len(u_years)
    c_lat = u_lats[u_pairs // len(u_lons)]
    c_lon = u_lons[u_pairs % len(u_lons)]

    rev = slice(None, None, -1)
    c_idx_r = c_idx[rev]
    y_idx_r = y_idx[rev]

    mats: dict[str, np.ndarray] = {}
    for col in fill_cols:
        mat = np.full((n_coords, n_years), np.nan, dtype=np.float64)
        mat[c_idx_r, y_idx_r] = raw[col].to_numpy(dtype=np.float64)[rev]
        for _ in range(3):
            na = np.isnan(mat)
            if not na.any():
                break
            mat[:, 1:] = np.where(na[:, 1:], mat[:, :-1], mat[:, 1:])
            na = np.isnan(mat)
            mat[:, :-1] = np.where(na[:, :-1], mat[:, 1:], mat[:, :-1])
        if np.isnan(mat).any():
            med = np.nanmedian(mat, axis=1, keepdims=True)
            mat = np.where(np.isnan(mat), med, mat)
        mats[col] = mat

    mask_86 = u_years <= 1986
    mask_92 = u_years <= 1992

    has_nan_86 = np.isnan(mats["ndwi"][:, mask_86]).any()
    has_nan_92 = any(np.isnan(mats[c][:, mask_92]).any() for c in fill_cols)

    anc_ndwi = (np.nanmax if has_nan_86 else np.max)(mats["ndwi"][:, mask_86], axis=1, keepdims=True)
    _mean = np.nanmean if has_nan_92 else np.mean
    _std = np.nanstd if has_nan_92 else np.std
    mean_ndwi = _mean(mats["ndwi"][:, mask_92], axis=1, keepdims=True)
    std_ndwi = _std(mats["ndwi"][:, mask_92], axis=1, ddof=1, keepdims=True)
    mean_ndbi = _mean(mats["ndbi"][:, mask_92], axis=1, keepdims=True)
    std_ndbi = _std(mats["ndbi"][:, mask_92], axis=1, ddof=1, keepdims=True)
    mean_ndsi = _mean(mats["ndsi"][:, mask_92], axis=1, keepdims=True)
    std_ndsi = _std(mats["ndsi"][:, mask_92], axis=1, ddof=1, keepdims=True)
    mean_si = _mean(mats["salinity_index"][:, mask_92], axis=1, keepdims=True)
    mean_lst = _mean(mats["lst_c"][:, mask_92], axis=1, keepdims=True)
    std_lst = _std(mats["lst_c"][:, mask_92], axis=1, ddof=1, keepdims=True)

    std_ndwi_f = np.where(np.isnan(std_ndwi), 0.04, std_ndwi) + 0.03
    std_ndbi_f = np.where(np.isnan(std_ndbi), 0.04, std_ndbi) + 0.03
    std_ndsi_f = np.where(np.isnan(std_ndsi), 0.05, std_ndsi) + 0.03
    std_lst_f = np.where(np.isnan(std_lst), 2.0, std_lst) + 0.5

    z_ndwi = np.round((mats["ndwi"] - mean_ndwi) / std_ndwi_f, 2)
    z_ndbi = np.round((mats["ndbi"] - mean_ndbi) / std_ndbi_f, 2)
    z_ndsi = np.round((mats["ndsi"] - mean_ndsi) / std_ndsi_f, 2)
    z_lst = np.round((mats["lst_c"] - mean_lst) / std_lst_f, 2)

    ndwi_r = np.round(mats["ndwi"], 3)
    ndbi_r = np.round(mats["ndbi"], 3)
    ndsi_r = np.round(mats["ndsi"], 3)
    si_r = np.round(mats["salinity_index"], 3)
    lst_r = np.round(mats["lst_c"], 1)

    years_2d = np.broadcast_to(u_years[None, :], (n_coords, n_years))
    base_urban = np.zeros((n_coords, n_years), dtype=bool)
    is_water_now = np.zeros((n_coords, n_years), dtype=bool)

    if cfg["mode"] == "ice":
        was_glacier = (mean_ndsi > 0.42) & (mean_si > 0.22)
        is_still_glacier = (ndsi_r > 0.38) & (si_r > 0.21)

        anom_ice = (years_2d > 1984) & was_glacier & (~is_still_glacier)
        anom_water = np.zeros((n_coords, n_years), dtype=bool)
        anom_urban = np.zeros((n_coords, n_years), dtype=bool)
        anom_temp = np.zeros((n_coords, n_years), dtype=bool)
        anom_combined = np.zeros((n_coords, n_years), dtype=bool)
        is_water_now = (~is_still_glacier) & (ndwi_r > 0.15) & (si_r <= 0.22)

        albedo_drop_z = np.clip((mean_si - si_r) / 0.08, 0.0, None)
        severity = np.round(np.where(anom_ice, np.maximum(np.abs(z_ndsi), albedo_drop_z), 0.0), 2)

    elif cfg["mode"] == "urban_water":
        anc_ndbi = np.nanmedian(mats["ndbi"][:, mask_86], axis=1, keepdims=True)
        in_mead_basin = (
            (c_lon[:, None] > -114.91)
            & (c_lat[:, None] <= 36.465)
            & ~((c_lat[:, None] > 36.22) & (c_lon[:, None] < -114.48))
        )
        water_anchor_1984 = in_mead_basin & ((anc_ndwi > -0.02) | (mean_ndwi > -0.01))

        still_water = ndwi_r > -0.035
        is_water_now = water_anchor_1984 & np.where(years_2d == 1984, True, still_water)
        anom_water = (years_2d > 1984) & water_anchor_1984 & (~is_water_now)

        in_valley_poly = points_in_polygon(c_lon, c_lat, LV_VALLEY_POLYGON)[:, None]
        in_1984_core_poly = points_in_polygon(c_lon, c_lat, LV_1984_ANCHOR_POLYGON)[:, None]
        old_henderson = (((c_lon + 114.985) ** 2 + (c_lat - 36.035) ** 2) < 0.032 ** 2)[:, None]
        boulder_city = (((c_lon + 114.835) ** 2 + (c_lat - 35.978) ** 2) < 0.022 ** 2)[:, None]

        base_urban_2d = (~water_anchor_1984) & (in_1984_core_poly | old_henderson | boulder_city)
        base_urban = np.broadcast_to(base_urban_2d, (n_coords, n_years)).copy()

        ndbi_roll = _rolling_median_3(ndbi_r)
        dist_from_1984_core = np.sqrt(((c_lon[:, None] + 115.14) * 0.81) ** 2 + (c_lat[:, None] - 36.16) ** 2)
        max_reach_for_year = 0.065 + 0.135 * np.clip((years_2d - 1984) / 42.0, 0.0, 1.0)

        ndbi_urban_shift = (np.abs(ndbi_roll - anc_ndbi) >= 0.012) | (ndbi_roll <= -0.04) | (np.abs(z_ndbi) >= 0.8)
        raw_expansion = (
            in_valley_poly
            & (~water_anchor_1984)
            & (~base_urban)
            & (years_2d > 1984)
            & (dist_from_1984_core <= max_reach_for_year)
            & ndbi_urban_shift
        )
        anom_urban = np.maximum.accumulate(raw_expansion, axis=1).astype(bool)

        anom_temp = (years_2d > 1984) & (z_lst >= 2.6)
        anom_combined = (anom_water | anom_urban) & anom_temp
        anom_ice = np.zeros((n_coords, n_years), dtype=bool)

        urban_sev = np.where(anom_urban, np.maximum(np.abs(z_ndbi), 2.2), 0.0)
        water_sev = np.where(
            anom_water,
            np.maximum(np.abs(z_ndwi), np.clip((anc_ndwi - ndwi_r) / 0.05, 2.0, None)),
            0.0,
        )
        severity = np.round(water_sev + urban_sev + np.where(anom_temp, np.clip(z_lst, 0.0, None), 0.0), 2)

    else:
        was_water = anc_ndwi > 0.0
        is_water_now = ndwi_r > 0.0
        anom_water = (years_2d > 1984) & was_water & ((z_ndwi <= -2.0) | (ndwi_r < -0.05))
        anom_urban = np.zeros((n_coords, n_years), dtype=bool)
        anom_temp = (years_2d > 1984) & (z_lst >= 1.8)
        anom_combined = anom_water & anom_temp
        anom_ice = np.zeros((n_coords, n_years), dtype=bool)
        severity = np.round(
            np.where(anom_water, np.abs(z_ndwi), 0.0)
            + np.where(anom_temp, np.clip(z_lst, 0.0, None), 0.0),
            2,
        )

    sev_norm = np.clip((severity - 1.8) / 4.5, 0.0, 1.0)
    if cfg["mode"] == "ice":
        disp_glacier = (ndsi_r > 0.38) & (si_r > 0.21)
        conds = [anom_ice, disp_glacier, is_water_now]
        r_def = np.select(conds, [(255 - sev_norm * 55).astype(int), 245, 15], default=145)
        g_def = np.select(conds, [(115 - sev_norm * 100).astype(int), 250, 110], default=135)
        b_def = np.select(conds, [(40 - sev_norm * 20).astype(int), 255, 245], default=120)
    else:
        conds = [is_water_now, anom_water, anom_urban, base_urban, anom_combined, anom_temp]
        r_def = np.select(
            conds,
            [25, (255 - sev_norm * 30).astype(int), (255 - sev_norm * 45).astype(int), 124, (255 - sev_norm * 75).astype(int), 255],
            default=185,
        )
        g_def = np.select(
            conds,
            [135, (155 - sev_norm * 80).astype(int), (56 - sev_norm * 56).astype(int), 92, (35 - sev_norm * 35).astype(int), (225 - sev_norm * 65).astype(int)],
            default=170,
        )
        b_def = np.select(
            conds,
            [255, 0, (200 - sev_norm * 40).astype(int), 230, (60 - sev_norm * 35).astype(int), 0],
            default=145,
        )

    df = pd.DataFrame({
        "latitude": np.repeat(c_lat, n_years),
        "longitude": np.repeat(c_lon, n_years),
        "year": np.tile(u_years, n_coords),
        "ndwi": ndwi_r.ravel(),
        "ndbi": ndbi_r.ravel(),
        "ndsi": ndsi_r.ravel(),
        "salinity_index": si_r.ravel(),
        "lst_c": lst_r.ravel(),
        "z_ndwi": z_ndwi.ravel(),
        "z_ndbi": z_ndbi.ravel(),
        "z_ndsi": z_ndsi.ravel(),
        "z_lst": z_lst.ravel(),
        "base_urban": base_urban.ravel(),
        "is_water_now": is_water_now.ravel(),
        "anom_ice": anom_ice.ravel(),
        "anom_water": anom_water.ravel(),
        "anom_urban": anom_urban.ravel(),
        "anom_temp": anom_temp.ravel(),
        "anom_combined": anom_combined.ravel(),
        "severity": severity.ravel(),
        "r_def": r_def.ravel(),
        "g_def": g_def.ravel(),
        "b_def": b_def.ravel(),
    })
    return df


def fetch_region_data(region_key: str) -> pd.DataFrame:
    """Fetch processed dataset from Rust backend (if updated) or fallback to cached Python."""
    import socket
    from urllib.parse import urljoin

    bound_url = os.environ.get("NASA_QUERY_URL")
    base_url = bound_url or RUST_API_URL
    try:
        if not bound_url:
            with socket.create_connection(("127.0.0.1", 8080), timeout=0.05):
                pass
        target_url = urljoin(base_url.rstrip("/") + "/", "anomalies_all")
        resp = requests.get(target_url, params={"region": region_key}, timeout=5.0 if bound_url else 0.25)
        resp.raise_for_status()
        df_rust = pd.DataFrame(resp.json()).rename(columns={"lat": "latitude", "lon": "longitude"})
        if not {"base_urban", "is_water_now", "anom_ice", "r_def"}.issubset(df_rust.columns):
            return load_and_compute_region(region_key)
        return df_rust.sort_values(["latitude", "longitude", "year"]).reset_index(drop=True)
    except Exception:
        return load_and_compute_region(region_key)


def compute_tangent_radius(df_year: pd.DataFrame, center_lat: float) -> tuple[int, int]:
    """Calculate circle radius = 0.50 * grid step so adjacent tiles touch tangentially."""
    lons = np.sort(df_year["longitude"].unique())
    lats = np.sort(df_year["latitude"].unique())
    dlon = float(np.median(np.diff(lons))) if len(lons) > 1 else 0.01
    dlat = float(np.median(np.diff(lats))) if len(lats) > 1 else 0.01

    dx_m = dlon * 111320.0 * np.cos(np.radians(center_lat))
    dy_m = dlat * 111132.0
    min_step_m = min(dx_m, dy_m)
    return max(int(round(min_step_m * 0.50)), 15), int(round(dy_m, -1))


@st.cache_resource
def get_region_bundle(region_key: str) -> dict:
    """Build in-memory per-year and per-cell O(1) index bundle for fast reruns."""
    cfg = REGIONS_CONFIG[region_key]
    df_all = fetch_region_data(region_key)
    years = np.sort(df_all["year"].unique()).astype(int)
    n_years = len(years)

    if len(df_all) % max(n_years, 1) == 0 and (df_all["year"].iloc[:n_years].to_numpy() == years).all():
        by_year = {int(yr): df_all.iloc[yi::n_years].reset_index(drop=True) for yi, yr in enumerate(years)}
        coords_lat = df_all["latitude"].iloc[::n_years].to_numpy(dtype=float)
        coords_lon = df_all["longitude"].iloc[::n_years].to_numpy(dtype=float)
        contiguous = True
    else:
        by_year = {int(yr): grp.reset_index(drop=True) for yr, grp in df_all.groupby("year")}
        coords = df_all[["latitude", "longitude"]].drop_duplicates()
        coords_lat = coords["latitude"].to_numpy(dtype=float)
        coords_lon = coords["longitude"].to_numpy(dtype=float)
        contiguous = False

    sample_year_df = by_year[int(years[-1])]
    cell_radius, cell_res_m = compute_tangent_radius(sample_year_df, cfg["lat"])

    return {
        "df_all": df_all,
        "years": years,
        "n_years": n_years,
        "by_year": by_year,
        "coords_lat": coords_lat,
        "coords_lon": coords_lon,
        "contiguous": contiguous,
        "cell_radius": cell_radius,
        "cell_res_m": cell_res_m,
    }


def prepare_year_frame(df_source: pd.DataFrame, year: int, reg_cfg: dict, anomaly_mode: str, metric_choice: str, only_anomalies: bool):
    """Filter single year and compute RGB colors and KPIs for interactive view."""
    if len(df_source) <= 150_000 and (df_source["year"] == year).all():
        df_year = df_source.copy()
    else:
        df_year = df_source[df_source["year"] == year].copy()

    if anomaly_mode == "ice":
        df_year["active_anom"] = df_year["anom_ice"]
    elif anomaly_mode == "combined":
        df_year["active_anom"] = df_year["anom_combined"]
    elif anomaly_mode == "water":
        df_year["active_anom"] = df_year["anom_water"]
    elif anomaly_mode == "urban":
        df_year["active_anom"] = df_year["anom_urban"]
    elif anomaly_mode == "temp":
        df_year["active_anom"] = df_year["anom_temp"]
    else:
        df_year["active_anom"] = df_year["anom_water"] | df_year["anom_urban"] | df_year["anom_temp"]

    is_glacier_surface = (reg_cfg["mode"] == "ice") & (df_year["ndsi"] > 0.38) & (df_year["salinity_index"] > 0.21)
    is_liquid_water = df_year["is_water_now"]

    df_year["reason"] = np.select(
        [
            df_year["anom_ice"],
            df_year["anom_water"],
            df_year["anom_urban"],
            df_year["base_urban"],
            df_year["anom_combined"],
            df_year["anom_temp"],
            is_glacier_surface,
            is_liquid_water,
        ],
        [
            "GLACIER ANOMALY (Missing Ice): Retreated vs. 1984 Anchor",
            "WATER ANOMALY: Receded shoreline (NDWI drop vs. 1984)",
            "URBAN EXPANSION: Post-1984 built-up infrastructure (NDBI)",
            "1984 URBAN ANCHOR: Original 1984 Las Vegas / Henderson core",
            "CRITICAL (Compound): Water/Urban shift + Thermal shock",
            "THERMAL ANOMALY: Land surface overheating",
            "Normal (Permanent Glacier / Snow cover)",
            "Normal (Liquid Water)",
        ],
        default="Normal (Arid Desert / Bedrock)",
    )

    df_display = df_year[df_year["active_anom"]].copy() if only_anomalies else df_year.copy()

    if metric_choice == "status" and anomaly_mode in ("all_types", "ice"):
        df_display["r"] = df_display["r_def"]
        df_display["g"] = df_display["g_def"]
        df_display["b"] = df_display["b_def"]
        return df_year, df_display, is_liquid_water

    if metric_choice == "status":
        sev_norm = ((df_display["severity"] - 1.8) / 4.5).clip(0.0, 1.0)
        disp_water = df_display["is_water_now"]

        if anomaly_mode == "urban":
            conds = [df_display["anom_urban"], df_display["base_urban"], disp_water]
            df_display["r"] = np.select(conds, [(255 - sev_norm * 45).astype(int), 124, 25], default=185)
            df_display["g"] = np.select(conds, [(56 - sev_norm * 56).astype(int), 92, 135], default=170)
            df_display["b"] = np.select(conds, [(200 - sev_norm * 40).astype(int), 230, 255], default=145)
        elif anomaly_mode == "water":
            conds = [df_display["anom_water"], disp_water]
            df_display["r"] = np.select(conds, [(255 - sev_norm * 35).astype(int), 25], default=185)
            df_display["g"] = np.select(conds, [(165 - sev_norm * 95).astype(int), 135], default=170)
            df_display["b"] = np.select(conds, [0, 255], default=145)
        elif anomaly_mode == "temp":
            conds = [df_display["anom_temp"], disp_water]
            df_display["r"] = np.select(conds, [255, 25], default=185)
            df_display["g"] = np.select(conds, [(230 - sev_norm * 90).astype(int), 135], default=170)
            df_display["b"] = np.select(conds, [0, 255], default=145)
        else:
            conds = [df_display["anom_combined"], disp_water]
            df_display["r"] = np.select(conds, [(255 - sev_norm * 85).astype(int), 25], default=185)
            df_display["g"] = np.select(conds, [(40 - sev_norm * 40).astype(int), 135], default=170)
            df_display["b"] = np.select(conds, [(60 - sev_norm * 40).astype(int), 255], default=145)

    elif metric_choice == "severity":
        has_sev = df_display["severity"] > 0
        s_norm = (df_display["severity"] / 6.0).clip(0.0, 1.0)
        df_display["r"] = np.where(has_sev, 255, 45)
        df_display["g"] = np.where(has_sev, ((1.0 - s_norm) * 210).astype(int), 55)
        df_display["b"] = np.where(has_sev, 20, 72)

    else:
        vals = df_display["salinity_index"]
        vmin, vmax = vals.quantile(0.05), vals.quantile(0.95)
        norm = ((vals - vmin) / (vmax - vmin + 1e-6)).clip(0, 1)
        df_display["r"] = (norm * 255).astype(int)
        df_display["g"] = ((1 - norm) * 180).astype(int)
        df_display["b"] = 40

    return df_year, df_display, is_liquid_water


def get_physical_regime_bands(mode: str, x_min: int, x_max: int) -> tuple[pd.DataFrame, list[str], list[str], tuple[float, float]]:
    """Return physical threshold bands, legend labels, colors, and Y-axis limits."""
    if mode == "ice":
        bands = pd.DataFrame([
            {"xmin": x_min, "xmax": x_max, "ymin": 0.42, "ymax": 1.00, "regime": "Permanent Glacier / Snow (> 0.42)"},
            {"xmin": x_min, "xmax": x_max, "ymin": 0.25, "ymax": 0.42, "regime": "Active Melting / Thinning (0.25 – 0.42)"},
            {"xmin": x_min, "xmax": x_max, "ymin": -0.30, "ymax": 0.25, "regime": "Ice Lost / Bedrock & Fjord (< 0.25)"},
        ])
        domain = [
            "Permanent Glacier / Snow (> 0.42)",
            "Active Melting / Thinning (0.25 – 0.42)",
            "Ice Lost / Bedrock & Fjord (< 0.25)",
        ]
        return bands, domain, ["#38bdf8", "#f59e0b", "#ef4444"], (-0.1, 0.9)

    bands = pd.DataFrame([
        {"xmin": x_min, "xmax": x_max, "ymin": 0.05, "ymax": 0.80, "regime": "Healthy Open Water (> 0.05)"},
        {"xmin": x_min, "xmax": x_max, "ymin": -0.05, "ymax": 0.05, "regime": "Shoreline Retreat / Shallow (-0.05 – 0.05)"},
        {"xmin": x_min, "xmax": x_max, "ymin": -0.60, "ymax": -0.05, "regime": "Desiccated Bed / Arid Land (< -0.05)"},
    ])
    domain = [
        "Healthy Open Water (> 0.05)",
        "Shoreline Retreat / Shallow (-0.05 – 0.05)",
        "Desiccated Bed / Arid Land (< -0.05)",
    ]
    return bands, domain, ["#38bdf8", "#f59e0b", "#ef4444"], (-0.45, 0.65)


def robust_z_score(series: np.ndarray | pd.Series | float,
                   reference: np.ndarray | pd.Series | None = None):
    """Score values with the unbounded median absolute deviation Z-score."""
    values_array = np.asarray(series, dtype=float)
    reference_array = values_array if reference is None else np.asarray(reference, dtype=float)
    reference_array = reference_array[np.isfinite(reference_array)]
    if reference_array.size == 0:
        result = np.full_like(values_array, np.nan, dtype=float)
    else:
        median = np.median(reference_array)
        mad = np.median(np.abs(reference_array - median))
        if mad == 0:
            result = np.zeros_like(values_array, dtype=float)
        else:
            result = 0.6745 * (values_array - median) / mad
    return float(result) if result.ndim == 0 else result


def t_critical_975(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value."""
    if student_t is not None:
        return float(student_t.ppf(0.975, degrees_of_freedom))

    z = 1.959963984540054
    v = float(max(degrees_of_freedom, 1))
    return float(
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )


def get_active_cell_timeseries(
    bundle_or_df: dict | pd.DataFrame,
    active_lat: float,
    active_lon: float,
) -> tuple[pd.DataFrame, float, float]:
    """Snap coordinates to nearest grid cell and return its sorted 1984–2026 record in O(1)."""
    if isinstance(bundle_or_df, dict):
        lats = bundle_or_df["coords_lat"]
        lons = bundle_or_df["coords_lon"]
        dist2 = np.square(lats - active_lat) + np.square((lons - active_lon) * np.cos(np.radians(active_lat)))
        idx = int(np.argmin(dist2))
        snap_lat = float(lats[idx])
        snap_lon = float(lons[idx])
        df_all = bundle_or_df["df_all"]
        if bundle_or_df["contiguous"]:
            n_years = bundle_or_df["n_years"]
            cell_df = df_all.iloc[idx * n_years : (idx + 1) * n_years].copy()
        else:
            cell_df = df_all[(df_all["latitude"] == snap_lat) & (df_all["longitude"] == snap_lon)].sort_values("year").copy()
        return cell_df, snap_lat, snap_lon

    coords = bundle_or_df[["latitude", "longitude"]].drop_duplicates()
    lats = coords["latitude"].to_numpy(dtype=float)
    lons = coords["longitude"].to_numpy(dtype=float)
    dist2 = np.square(lats - active_lat) + np.square((lons - active_lon) * np.cos(np.radians(active_lat)))
    nearest = coords.iloc[int(np.argmin(dist2))]
    snap_lat = float(nearest["latitude"])
    snap_lon = float(nearest["longitude"])
    cell_df = bundle_or_df[(bundle_or_df["latitude"] == snap_lat) & (bundle_or_df["longitude"] == snap_lon)].sort_values("year").copy()
    return cell_df, snap_lat, snap_lon


def build_map_legend_html(mode: str, metric_choice: str) -> str:
    """Return Tactical Telemetry legend swatches matching the preserved RGB palette."""
    if metric_choice == "severity":
        swatches = [
            ("#2d3748", "NORMAL (0)"),
            ("#ffd214", "MODERATE (Z≈2)"),
            ("#ff6b14", "HIGH (Z≈4)"),
            ("#ff1414", "EXTREME (Z>6)"),
        ]
    elif metric_choice == "salinity_index":
        swatches = [
            ("#00b428", "LOW SALINITY / WATER"),
            ("#b46e28", "MODERATE SALINITY"),
            ("#ff0028", "DENSE SALT CRUST"),
        ]
    elif mode == "ice":
        swatches = [
            ("#f5faff", "GLACIER / SNOW"),
            ("#0f6ef5", "FJORD WATER"),
            ("#e63919", "MISSING ICE"),
            ("#918778", "BEDROCK"),
        ]
    elif mode == "urban_water":
        swatches = [
            ("#1987ff", "WATER"),
            ("#ff8c00", "WATER LOSS"),
            ("#7c5ce6", "1984 URBAN CORE"),
            ("#ff38c8", "URBAN GROWTH"),
            ("#ffd700", "THERMAL"),
            ("#b9aa91", "DESERT"),
        ]
    else:
        swatches = [
            ("#1987ff", "WATER"),
            ("#b9aa91", "DRY LAKEBED / SALT FLATS"),
            ("#ff8c00", "WATER ANOMALY"),
            ("#ffd700", "THERMAL"),
            ("#dc143c", "COMBINED"),
        ]
    items = "".join(
        f"<span class='map-legend-item'><i class='legend-swatch' style='background:{color}'></i>{label}</span>"
        for color, label in swatches
    )
    return f"<div class='map-legend'>{items}<span class='map-legend-item'>□ ACTIVE CELL</span></div>"


def compute_map_status_label(df_year: pd.DataFrame, mode: str, anomaly_mode: str, year_val: int) -> str:
    """Summarize flagged/non-water cell counts for the 01 / Interactive Grid header."""
    if df_year.empty:
        return f"NO MAP DATA · {year_val}"
    if mode == "ice":
        active, label = df_year["anom_ice"], "ICE-RETREAT CELLS"
    elif mode == "urban_water":
        if anomaly_mode == "urban":
            active, label = df_year["anom_urban"], "URBAN-GROWTH CELLS"
        elif anomaly_mode == "water":
            active, label = df_year["anom_water"], "WATER-LOSS CELLS"
        elif anomaly_mode == "temp":
            active, label = df_year["anom_temp"], "THERMAL CELLS"
        elif anomaly_mode == "combined":
            active, label = df_year["anom_combined"], "COMPOUND CELLS"
        else:
            active, label = df_year["active_anom"], "ANOMALOUS CELLS"
    else:
        if anomaly_mode == "temp":
            active, label = df_year["anom_temp"], "THERMAL CELLS"
        elif anomaly_mode == "combined":
            active, label = df_year["anom_combined"], "COMPOUND CELLS"
        elif anomaly_mode == "water":
            active, label = df_year["anom_water"] | (~df_year["is_water_now"]), "DRY / WATER-LOSS CELLS"
        else:
            active, label = df_year["active_anom"] | (~df_year["is_water_now"]), "DRY / ANOMALOUS CELLS"

    count = int(active.sum())
    total = len(df_year)
    return f"{label} · {count:,}/{total:,} ({100.0 * count / max(total, 1):.1f}%) · {year_val}"


@st.cache_data(max_entries=128)
def get_cached_frame_payload(
    region_key: str,
    year_val: int,
    anomaly_mode: str,
    metric_choice: str,
    pickable: bool,
) -> tuple[str, str]:
    """Compute and C-serialize minimal frame columns once per (region, year, channel, layer, mode)."""
    bundle = get_region_bundle(region_key)
    reg_cfg = REGIONS_CONFIG[region_key]
    df_year_src = bundle["by_year"][int(year_val)]
    df_year, df_display, _ = prepare_year_frame(
        df_year_src, int(year_val), reg_cfg, anomaly_mode, metric_choice, only_anomalies=False
    )
    status_label = compute_map_status_label(df_year, reg_cfg["mode"], anomaly_mode, int(year_val))
    cols = _PICKABLE_COLS if pickable else _SIM_COLS
    data_json = df_display[cols].to_json(orient="records")
    return data_json, status_label


def build_grid_deck(
    df_or_json: pd.DataFrame | str,
    reg_cfg: dict,
    cell_radius: int,
    cell_res_m: int,
    active_lat: float,
    active_lon: float,
    pickable: bool = True,
) -> pdk.Deck:
    """Build PyDeck map preserving exact ScatterplotLayer radius, RGB colors, and ViewState."""
    if isinstance(df_or_json, str):
        data_json = df_or_json
    else:
        cols = _PICKABLE_COLS if pickable else _SIM_COLS
        data_json = df_or_json[cols].to_json(orient="records")

    grid_layer = pdk.Layer(
        "ScatterplotLayer",
        id="earth-grid",
        data="__FAST_GRID_DATA__",
        get_position="[longitude, latitude]",
        get_fill_color="[r, g, b, 255]",
        get_radius=cell_radius,
        pickable=pickable,
        auto_highlight=pickable,
    )
    layers = [grid_layer]

    if pickable:
        half_lat = max(cell_res_m, 1500) / (2 * 111132.0)
        half_lon = max(cell_res_m, 1500) / (2 * 111320.0 * np.cos(np.radians(active_lat)))
        active_poly = [[
            [active_lon - half_lon, active_lat - half_lat],
            [active_lon + half_lon, active_lat - half_lat],
            [active_lon + half_lon, active_lat + half_lat],
            [active_lon - half_lon, active_lat + half_lat],
        ]]
        active_layer = pdk.Layer(
            "PolygonLayer",
            id="active-cell",
            data=[{"polygon": active_poly[0]}],
            get_polygon="polygon",
            get_fill_color=[255, 255, 255, 35],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=False,
        )
        layers.append(active_layer)

    tooltip = {
        "text": (
            "{reason}\n"
            "Coords: {latitude}, {longitude}\n"
            "Severity: {severity}\n"
            "NDWI: {ndwi} (Z: {z_ndwi}) | NDBI: {ndbi} (Z: {z_ndbi})\n"
            "NDSI: {ndsi} (Z: {z_ndsi}) | Temp: {lst_c} °C (Z: {z_lst})"
        )
    } if pickable else None

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=reg_cfg["lat"],
            longitude=reg_cfg["lon"],
            zoom=reg_cfg["zoom"],
        ),
        tooltip=tooltip,
    )
    orig_to_json = deck.to_json
    deck.to_json = lambda: orig_to_json().replace('"__FAST_GRID_DATA__"', data_json, 1)
    return deck


def build_historical_altair_chart(cell_ts: pd.DataFrame, reg_cfg: dict, hist_metric: str) -> tuple[alt.Chart, float]:
    """Render the dark dual-axis Altair historical trend chart and compute 2010-onward OLS slope."""
    meta = METRICS[hist_metric]
    cols = ["year", hist_metric] if hist_metric == "lst_c" else ["year", hist_metric, "lst_c"]
    clean_ts = cell_ts[cols].dropna().sort_values("year").copy()
    min_year, max_year = int(clean_ts["year"].min()), int(clean_ts["year"].max())

    recent = clean_ts[clean_ts["year"] >= 2010]
    if len(recent) < 2:
        recent = clean_ts
    slope, _ = np.polyfit(recent["year"], recent[hist_metric], 1)

    if hist_metric == reg_cfg["primary_idx"]:
        bands_hist, domain, colors, y_lim = get_physical_regime_bands(reg_cfg["mode"], min_year, max_year)
        bg_hist = alt.Chart(bands_hist).mark_rect(opacity=0.14).encode(
            x=alt.X("xmin:Q", scale=alt.Scale(domain=[min_year, max_year], nice=False), axis=alt.Axis(format="d", title="Year", tickCount=9)),
            x2="xmax:Q",
            y=alt.Y("ymin:Q", scale=alt.Scale(domain=list(y_lim)), axis=alt.Axis(title=meta["short"], titleColor="#38bdf8")),
            y2="ymax:Q",
            color=alt.Color("regime:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient="bottom", columns=1, title="Physical Index Zones")),
        )
        base_left = bg_hist
    else:
        vmin, vmax = float(clean_ts[hist_metric].min()), float(clean_ts[hist_metric].max())
        pad = max((vmax - vmin) * 0.15, 0.05 if hist_metric != "lst_c" else 2.0)
        y_lim = (vmin - pad, vmax + pad)
        base_left = None

    idx_line = alt.Chart(clean_ts).mark_line(
        color="#38bdf8" if hist_metric != "lst_c" else "#fb923c",
        strokeWidth=2.4,
        point=alt.OverlayMarkDef(size=28, color="#38bdf8" if hist_metric != "lst_c" else "#fb923c"),
    ).encode(
        x=alt.X("year:Q", scale=alt.Scale(domain=[min_year, max_year], nice=False), axis=alt.Axis(format="d", title="Year", tickCount=9)),
        y=alt.Y(
            f"{hist_metric}:Q",
            scale=alt.Scale(domain=list(y_lim)),
            axis=alt.Axis(title=meta["short"] if hist_metric != "lst_c" else "Land Surface Temp (°C)", titleColor="#38bdf8" if hist_metric != "lst_c" else "#fb923c"),
        ),
        tooltip=[
            alt.Tooltip("year:Q", format="d", title="Year"),
            alt.Tooltip(f"{hist_metric}:Q", format=".3f" if hist_metric != "lst_c" else ".1f", title=meta["name"]),
        ] + ([alt.Tooltip("lst_c:Q", format=".1f", title="Surface Temp (°C)")] if hist_metric != "lst_c" else []),
    )

    left_layer = (base_left + idx_line) if base_left is not None else idx_line

    if hist_metric == "lst_c":
        chart = left_layer
    else:
        temp_min = max(0.0, float(clean_ts["lst_c"].min()) - 4.0)
        temp_max = float(clean_ts["lst_c"].max()) + 4.0
        temp_line = alt.Chart(clean_ts).mark_line(color="#fb923c", strokeWidth=1.8, strokeDash=[4, 3]).encode(
            x=alt.X("year:Q", scale=alt.Scale(domain=[min_year, max_year], nice=False)),
            y=alt.Y("lst_c:Q", scale=alt.Scale(domain=[temp_min, temp_max]), axis=alt.Axis(title="Land Surface Temp (°C)", titleColor="#fb923c", orient="right")),
        )
        chart = alt.layer(left_layer, temp_line).resolve_scale(y="independent")

    styled = (
        chart.properties(height=340)
        .configure(background="#080b0e")
        .configure_view(strokeWidth=0, fill="#080b0e")
        .configure_axis(gridColor="#142126", domainColor="#263940", tickColor="#263940", labelColor="#9aabb1", titleColor="#9aabb1", labelFont="IBM Plex Mono, Consolas, monospace", titleFont="IBM Plex Mono, Consolas, monospace")
        .configure_legend(labelColor="#9aabb1", titleColor="#dce7e9", labelFont="IBM Plex Mono, Consolas, monospace", titleFont="IBM Plex Mono, Consolas, monospace")
    )
    return styled, float(slope)


def chart_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply dark Tactical Telemetry theme to Plotly forecast figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#080b0e",
        height=height,
        margin=dict(l=12, r=16, t=18, b=12),
        font=dict(family="IBM Plex Mono, Consolas, monospace", color="#9aabb1", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#0e1117", bordercolor="#00f3ff", font_color="#e8f0f2"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#142126", zeroline=False, linecolor="#263940")
    fig.update_yaxes(showgrid=True, gridcolor="#142126", zeroline=False, linecolor="#263940")
    return fig


def build_forecast(
    df: pd.DataFrame,
    key: str,
    extraction_pct: int = 0,
    drought_intensity: int = 0,
) -> tuple[go.Figure, dict[str, float]]:
    """Build 2026–2036 scenario trajectories using recent (>=2010) OLS slope and 95% prediction interval."""
    meta = METRICS[key]
    clean = df[["year", key]].dropna().sort_values("year").copy()
    if len(clean) < 3:
        raise ValueError("At least three valid annual observations are required for an OLS forecast")

    recent = clean[clean["year"] >= 2010]
    if len(recent) < 3:
        recent = clean

    x_all = clean["year"].to_numpy(dtype=float)
    y_all = clean[key].to_numpy(dtype=float)
    x_fit = recent["year"].to_numpy(dtype=float)
    y_fit = recent[key].to_numpy(dtype=float)

    ols_slope, ols_intercept = np.polyfit(x_fit, y_fit, 1)
    residual = y_fit - (ols_slope * x_fit + ols_intercept)
    dof = max(len(x_fit) - 2, 1)
    residual_sd = float(np.sqrt(np.sum(residual**2) / dof))

    anchor_year = float(x_all[-1])
    future = np.arange(anchor_year, anchor_year + 11, dtype=float)
    mean_x = x_fit.mean()
    sxx = max(float(np.sum((x_fit - mean_x) ** 2)), 1e-9)
    prediction_se = residual_sd * np.sqrt(1.0 + 1.0 / len(x_fit) + (future - mean_x) ** 2 / sxx)
    margin = t_critical_975(dof) * prediction_se

    extraction_pressure = extraction_pct / 35.0
    drought_pressure = drought_intensity / 100.0
    stress_level = min(1.0, 0.65 * extraction_pressure + 0.65 * drought_pressure)
    slope_impact = meta["slope_impact"]

    favorable_direction = -1.0 if key in {"lst_c", "salinity_index", "ndbi"} else 1.0
    recovery_coefficient = max(abs(float(ols_slope)) * 0.2, abs(slope_impact) * 0.2)
    best_slope = float(ols_slope + favorable_direction * recovery_coefficient)
    average_slope = float(ols_slope + slope_impact * stress_level)
    worst_slope = float(ols_slope + slope_impact * 1.25)

    clip_lo, clip_hi = meta["clip_bounds"]
    years_from_anchor = future - anchor_year
    best_projected = np.clip(y_all[-1] + best_slope * years_from_anchor, clip_lo, clip_hi)
    average_projected = np.clip(y_all[-1] + average_slope * years_from_anchor, clip_lo, clip_hi)
    worst_projected = np.clip(y_all[-1] + worst_slope * years_from_anchor, clip_lo, clip_hi)

    best_2036 = float(best_projected[-1])
    average_2036 = float(average_projected[-1])
    worst_2036 = float(worst_projected[-1])

    fig = go.Figure()
    scenario_low = np.minimum(best_projected, worst_projected)
    scenario_high = np.maximum(best_projected, worst_projected)
    fig.add_trace(go.Scatter(x=future, y=scenario_low, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=future, y=scenario_high, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="#ff2a6d12", name="Best–worst scenario range", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=future, y=average_projected + margin, mode="lines", line_width=0, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=future, y=average_projected - margin, mode="lines", line_width=0,
        fill="tonexty", fillcolor="#00f3ff20", name="95% prediction interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=x_all, y=y_all, mode="lines", name="Historical annual composite", line=dict(color=meta["color"], width=2)))
    fig.add_trace(go.Scatter(x=future, y=best_projected, mode="lines", name="Best scenario · no added stress", line=dict(color=COLORS["green"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=future, y=average_projected, mode="lines", name="Average scenario · selected inputs", line=dict(color=COLORS["cyan"], width=2.8, dash="dash")))
    fig.add_trace(go.Scatter(x=future, y=worst_projected, mode="lines", name="Worst scenario · maximum stress", line=dict(color=COLORS["crimson"], width=2, dash="longdash")))
    fig.add_vline(x=anchor_year, line_color="#ffffff77", line_dash="dot", annotation_text="NOW")
    chart_theme(fig, 350)
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    fig.update_xaxes(title="YEAR", range=[float(x_all[0]), float(future[-1])])
    fig.update_yaxes(title=meta["unit"])

    stats = {
        "slope": average_slope,
        "ols_slope": float(ols_slope),
        "best_slope": best_slope,
        "worst_slope": worst_slope,
        "2036": average_2036,
        "best_2036": best_2036,
        "worst_2036": worst_2036,
        "delta_2036": average_2036 - best_2036,
        "margin": float(margin[-1]),
        "stress_level": stress_level,
    }
    return fig, stats


def build_anomaly_channel_options(mode: str) -> tuple[list[str], dict[str, str]]:
    """Return anomaly mode keys and display labels for the selected region."""
    if mode == "ice":
        return ["ice"], {"ice": "Glacier Anomaly (Missing Ice)"}
    if mode == "urban_water":
        return (
            ["all_types", "combined", "water", "urban", "temp"],
            {
                "all_types": "All Anomalies",
                "combined": "Compound Anomaly (Water/Urban + Heat)",
                "water": "Lake Mead Desiccation (NDWI vs. 1984)",
                "urban": "Urban Infrastructure (NDBI 1984 & Expansion)",
                "temp": "Thermal Anomaly (LST °C Spike)",
            },
        )
    return (
        ["all_types", "combined", "water", "temp"],
        {
            "all_types": "All Anomalies",
            "combined": "Compound Anomaly (Water Loss + Heat)",
            "water": "Water Anomaly (NDWI Desiccation)",
            "temp": "Thermal Anomaly (LST °C Spike)",
        },
    )


def render_sidebar() -> tuple[str, dict]:
    """Render MISSION CONTROL / EO-01 sidebar controls and return selected region key and config."""
    with st.sidebar:
        st.markdown('<div class="eyebrow">MISSION CONTROL / EO-01</div>', unsafe_allow_html=True)
        st.markdown("## DASHBOARD CONTROLS")
        selected_region = st.selectbox(
            "MONITORING REGION",
            options=list(REGIONS_CONFIG.keys()),
            format_func=lambda k: REGIONS_CONFIG[k]["selector_label"],
            key="monitoring_region",
        )
        reg_cfg = REGIONS_CONFIG[selected_region]
        st.caption(reg_cfg["subtitle"].upper())
        st.markdown("---")
        st.markdown(
            '<div class="small-note">ARCHIVE · LANDSAT 4–9<br>'
            'GRID RESOLUTION · REGION DEPENDENT<br>'
            'REFERENCE · 1984–1992</div>',
            unsafe_allow_html=True,
        )
    return selected_region, reg_cfg


def main() -> None:
    telemetry_layout()
    selected_region, reg_cfg = render_sidebar()
    bundle = get_region_bundle(selected_region)

    if st.session_state.get("active_region") != selected_region:
        st.session_state.active_lat = reg_cfg["lat"]
        st.session_state.active_lon = reg_cfg["lon"]
        st.session_state.active_region = selected_region
    st.session_state.setdefault("active_lat", reg_cfg["lat"])
    st.session_state.setdefault("active_lon", reg_cfg["lon"])

    min_yr, max_yr = int(bundle["years"][0]), int(bundle["years"][-1])
    cell_radius = bundle["cell_radius"]
    cell_res_m = bundle["cell_res_m"]

    st.markdown(
        "<div class='terminal-title'>"
        "<div class='eyebrow'>EARTH SYSTEM TREND DETECTIVE / TACTICAL TELEMETRY</div>"
        "<h1>REGIONAL CHANGE MONITOR</h1></div>",
        unsafe_allow_html=True,
    )

    explorer_tab, predictive_tab, evolution_tab = st.tabs(
        ["Tactical Explorer", "Predictive Scenario Engine", "Anomaly Evolution"]
    )

    with explorer_tab:
        map_col, map_controls = st.columns([2.05, 1.0], gap="large")
        with map_controls:
            st.markdown("<div class='panel-head'><span>MAP CONTROLS</span><span>MAP ONLY</span></div>", unsafe_allow_html=True)
            selected_year = st.slider("OBSERVATION YEAR", min_yr, max_yr, max_yr, 1, key="observation_year")
            chan_opts, chan_labels = build_anomaly_channel_options(reg_cfg["mode"])
            if st.session_state.get("map_anomaly_channel") not in chan_opts:
                st.session_state["map_anomaly_channel"] = chan_opts[0]
            anomaly_mode = st.selectbox(
                "ANOMALY CHANNEL",
                options=chan_opts,
                format_func=lambda x: chan_labels[x],
                key="map_anomaly_channel",
            )
            color_options = ["status", "severity", "salinity_index"] if selected_region == "sea_aquisition" else ["status", "severity"]
            if st.session_state.get("map_color_layer") not in color_options:
                st.session_state["map_color_layer"] = color_options[0]
            metric_choice = st.selectbox(
                "MAP COLOR LAYER",
                options=color_options,
                format_func=lambda x: {
                    "status": "Hydrological & Anomaly Status",
                    "severity": "Severity Heatmap",
                    "salinity_index": "Salinity Heatmap (Aral Sea)",
                }[x],
                key="map_color_layer",
            )
            st.caption("These controls update the heatmap only.")

        map_widget_key = f"basin_map_{selected_region}_{selected_year}"
        saved_state = st.session_state.get(map_widget_key)
        saved_selection = saved_state.get("selection", {}) if isinstance(saved_state, dict) else getattr(saved_state, "selection", {})
        saved_objects = saved_selection.get("objects", {}) if hasattr(saved_selection, "get") else {}
        selected_objects = saved_objects.get("earth-grid", [])
        if selected_objects:
            sel_coords = (float(selected_objects[0]["latitude"]), float(selected_objects[0]["longitude"]))
            last_sel_key = f"_last_sel_{map_widget_key}"
            if st.session_state.get(last_sel_key) != sel_coords:
                st.session_state.active_lat, st.session_state.active_lon = sel_coords
                st.session_state[last_sel_key] = sel_coords

        data_json, map_status = get_cached_frame_payload(
            selected_region, selected_year, anomaly_mode, metric_choice, pickable=True
        )

        with map_col:
            st.markdown(
                f"<div class='panel-head'><span>01 / INTERACTIVE GRID · {reg_cfg['subtitle'].upper()}</span>"
                f"<span>{map_status}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(build_map_legend_html(reg_cfg["mode"], metric_choice), unsafe_allow_html=True)
            deck = build_grid_deck(
                data_json,
                reg_cfg,
                cell_radius,
                cell_res_m,
                st.session_state.active_lat,
                st.session_state.active_lon,
                pickable=True,
            )
            st.pydeck_chart(
                deck,
                width="stretch",
                height=520,
                on_select="rerun",
                selection_mode="single-object",
                key=map_widget_key,
            )
            st.markdown(
                f"<div class='scale-strip'>"
                f"<div>📏 MAP SCALE · <span style='display:inline-block;width:64px;border-bottom:2px solid #e6edf3;border-left:2px solid #e6edf3;border-right:2px solid #e6edf3;height:5px;margin:0 6px;'></span> ~{reg_cfg['scale_km']} KM · CELL STEP ~{cell_res_m} M</div>"
                f"<div>ANOMALY SEVERITY · MODERATE (Z≈2) <span style='display:inline-block;width:96px;height:8px;background:linear-gradient(90deg,#ffae34,#e62020,#8a0014);border-radius:3px;margin:0 6px;vertical-align:middle;'></span> EXTREME (Z&gt;5)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='small-note'>{reg_cfg['grid_source_label']} · {cell_res_m / 1000:.1f} KM GRID · "
                f"ACTIVE CELL {st.session_state.active_lat:.4f}, {st.session_state.active_lon:.4f}</div>",
                unsafe_allow_html=True,
            )

        chart_col, chart_controls = st.columns([4.2, 1.15], gap="large")
        with chart_controls:
            st.markdown("<div class='panel-head'><span>CHART CONTROLS</span><span>HISTORY ONLY</span></div>", unsafe_allow_html=True)
            target_opts = reg_cfg["target_metrics"]
            if st.session_state.get("history_metric") not in target_opts:
                st.session_state["history_metric"] = target_opts[0]
            hist_metric = st.selectbox(
                "TARGET METRIC",
                target_opts,
                format_func=lambda k: f"{METRICS[k]['name']}  /  {METRICS[k]['short']}",
                key="history_metric",
            )
            st.caption("The chart always shows 1984–2026.")

        hist_meta = METRICS[hist_metric]
        cell_ts, snap_lat, snap_lon = get_active_cell_timeseries(
            bundle, st.session_state.active_lat, st.session_state.active_lon
        )

        with chart_col:
            st.markdown(
                f"<div class='panel-head'><span>02 / HISTORICAL TREND · {hist_meta['name'].upper()}</span>"
                f"<span>{snap_lat:.3f}° · {snap_lon:.3f}°</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"### Historical Trend ({min_yr}–{max_yr})")
            if cell_ts[hist_metric].notna().any():
                alt_chart, slope_2010 = build_historical_altair_chart(cell_ts, reg_cfg, hist_metric)
                st.altair_chart(alt_chart, width="stretch")

                baseline_series = cell_ts.loc[cell_ts["year"] <= 1992, hist_metric].to_numpy(dtype=float)
                baseline_series = baseline_series[np.isfinite(baseline_series)]
                baseline_median = float(np.median(baseline_series)) if baseline_series.size else np.nan
                baseline_mad = float(np.median(np.abs(baseline_series - baseline_median))) if baseline_series.size else np.nan
                z_vals = robust_z_score(cell_ts[hist_metric].to_numpy(dtype=float), baseline_series)
                spike_count = int((np.abs(z_vals) > 2).sum()) if np.isfinite(z_vals).any() else 0
                valid_vals = cell_ts[hist_metric].dropna()
                latest_val = float(valid_vals.iloc[-1]) if not valid_vals.empty else np.nan
                current_z = robust_z_score(latest_val, baseline_series)

                baseline_note = (
                    f"BASELINE MEDIAN {baseline_median:.3f}, MAD {baseline_mad:.4f}"
                    if np.isfinite(baseline_median) and np.isfinite(baseline_mad)
                    else "BASELINE STATISTICS UNAVAILABLE"
                )
                latest_note = f"LATEST OBSERVED Z {current_z:+.2f}σ" if np.isfinite(current_z) else "LATEST OBSERVED Z —"
                st.markdown(
                    f"<div class='small-note' style='color:#c4d0d4; margin-top:.35rem;'>"
                    f"2010-onward linear trend: {slope_2010:+.4f} {hist_meta['unit']}/yr</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='small-note'>ACTIVE CELL HISTORY · {baseline_note} · {latest_note} · {spike_count} ANOMALY YEARS (MAD Z > 2)</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No historical observations are available for this cell and metric.")

    with predictive_tab:
        st.markdown(
            "<div class='eyebrow'>03 / 2026–2036 · SCENARIO MODELING</div>"
            "<h2>Predictive Scenario Engine</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='small-note'>Adjust basin stress assumptions and compare best, average, and worst 2036 trajectories.</div>",
            unsafe_allow_html=True,
        )
        forecast_opts = reg_cfg["target_metrics"]
        if st.session_state.get("forecast_metric") not in forecast_opts:
            st.session_state["forecast_metric"] = forecast_opts[0]
        forecast_metric = st.selectbox(
            "FORECAST METRIC",
            forecast_opts,
            format_func=lambda k: f"{METRICS[k]['name']}  /  {METRICS[k]['short']}",
            key="forecast_metric",
        )
        forecast_meta = METRICS[forecast_metric]

        with st.container(border=True):
            st.markdown("<div class='panel-head'><span>BASIN STRESS SIMULATOR</span><span>SCENARIO INPUTS</span></div>", unsafe_allow_html=True)
            control_cols = st.columns([1, 1, 1])
            with control_cols[0]:
                extraction_pct = st.slider(
                    "Upstream agricultural extraction", min_value=0, max_value=35,
                    value=35, step=1, format="%d%%", key="basin_extraction",
                )
            with control_cols[1]:
                drought_intensity = st.slider(
                    "Climate drought intensity", min_value=0, max_value=100,
                    value=40, step=5, format="%d%%", key="basin_drought",
                )
            with control_cols[2]:
                is_pro = st.toggle("Show model diagnostics", value=True, key="forecast_diagnostics")
                st.markdown("<div class='small-note'>Scenario stress is applied to the recent OLS slope from 2026 onward.</div>", unsafe_allow_html=True)

        forecast_ts, _, _ = get_active_cell_timeseries(
            bundle, st.session_state.active_lat, st.session_state.active_lon
        )
        forecast_clean = forecast_ts[["year", forecast_metric]].dropna()
        forecast_baseline = forecast_ts.loc[forecast_ts["year"] <= 1992, forecast_metric].to_numpy(dtype=float)
        forecast_baseline = forecast_baseline[np.isfinite(forecast_baseline)]
        forecast_latest = float(forecast_clean.iloc[-1][forecast_metric]) if not forecast_clean.empty else np.nan
        forecast_current_z = robust_z_score(forecast_latest, forecast_baseline)

        if len(forecast_clean) < 3:
            st.info("A forecast requires at least three observed annual values for the selected cell and metric.")
        else:
            forecast_fig, forecast_stats = build_forecast(forecast_clean, forecast_metric, extraction_pct, drought_intensity)
            st.markdown("<div class='panel-head'><span>03 / FORECAST RANGE</span><span>BEST · AVERAGE · WORST</span></div>", unsafe_allow_html=True)
            st.plotly_chart(forecast_fig, width="stretch", config={"displayModeBar": False})

            if is_pro:
                st.markdown("<div class='panel-head'><span>MODEL DIAGNOSTICS</span><span>RECENT OLS / 95% PREDICTION INTERVAL</span></div>", unsafe_allow_html=True)
                risk = "CRITICAL" if forecast_stats["stress_level"] >= 0.65 or (np.isfinite(forecast_current_z) and abs(forecast_current_z) > 2) else "ELEVATED"
                stat_cols = st.columns(4)
                stat_values = [
                    ("BASE OLS β", f"{forecast_stats['ols_slope']:+.5f} {forecast_meta['unit']}/yr"),
                    ("AVERAGE β", f"{forecast_stats['slope']:+.5f} {forecast_meta['unit']}/yr"),
                    ("2036 PROJECTION", f"{forecast_stats['2036']:.3f} {forecast_meta['unit']}"),
                    ("EARLY WARNING", risk),
                ]
                for col, (label, value) in zip(stat_cols, stat_values):
                    with col:
                        tone = "hot" if label == "EARLY WARNING" and risk == "CRITICAL" else "cyan"
                        st.markdown(
                            f"<div class='telemetry-card'><div class='telemetry-label'>{label}</div>"
                            f"<div class='telemetry-value {tone}'>{value}</div></div>",
                            unsafe_allow_html=True,
                        )
                st.markdown(
                    f"<div class='small-note' style='margin-top:.35rem;'>95% PREDICTION ENVELOPE · "
                    f"{forecast_stats['2036'] - forecast_stats['margin']:.3f} TO {forecast_stats['2036'] + forecast_stats['margin']:.3f} {forecast_meta['unit']} · "
                    f"TREND: {forecast_meta['trend'].upper()}</div>",
                    unsafe_allow_html=True,
                )

            stress_level = forecast_stats["stress_level"]
            if stress_level == 0:
                impact_text = "No added extraction or drought stress is applied; the average line follows the observed trend."
            else:
                sources = []
                if extraction_pct:
                    sources.append(f"{extraction_pct}% upstream extraction reduces downstream inflow")
                if drought_intensity:
                    sources.append(f"{drought_intensity}% drought compounds seasonal water loss")
                impacts = {
                    "ndwi": f"The water-index scenario declines by {abs(forecast_stats['delta_2036']):.3f}, increasing shoreline retreat and exposed lakebed.",
                    "ndbi": f"The urban-index scenario shifts by {abs(forecast_stats['delta_2036']):.3f}, expanding impervious built-up heat islands.",
                    "ndsi": f"The snow-index scenario declines by {abs(forecast_stats['delta_2036']):.3f}, accelerating glacier terminus retreat.",
                    "salinity_index": f"The salinity scenario rises by {abs(forecast_stats['delta_2036']):.3f}, increasing salt-concentration and tipping-point risk.",
                    "lst_c": f"The temperature scenario adds {abs(forecast_stats['delta_2036']):.2f} °C as dry soils lose evaporative cooling.",
                }
                impact_text = "; ".join(sources) + ". " + impacts[forecast_metric]
            stress_label = "HIGH" if stress_level >= 0.65 else "MODERATE" if stress_level >= 0.3 else "LOW"
            st.markdown(
                f"<div class='telemetry-card' style='margin-top:.8rem;border-color:#00f3ff66'>"
                f"<div class='telemetry-label'>BASIN STRESS IMPACT · {stress_label} · 2036 SCENARIO</div>"
                f"<div class='small-note' style='color:#c4d0d4'>{impact_text}</div>"
                f"<div class='small-note' style='margin-top:.45rem'>2036 RANGE · BEST {forecast_stats['best_2036']:.3f} · "
                f"AVERAGE {forecast_stats['2036']:.3f} · WORST {forecast_stats['worst_2036']:.3f} {forecast_meta['unit']}</div></div>",
                unsafe_allow_html=True,
            )

    with evolution_tab:
        st.markdown(
            "<div class='eyebrow'>04 / HISTORICAL CHANGE · YEAR BY YEAR</div>"
            "<h2>Anomaly Evolution</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='small-note' style='color:#c4d0d4; font-size:.82rem; margin-bottom:.8rem;'>"
            "Choose a year range, then click Run Simulation to animate mapped water and anomaly classes from start to finish.</div>",
            unsafe_allow_html=True,
        )
        anim_range = st.slider(
            "ANIMATION YEAR RANGE",
            min_yr,
            max_yr,
            (min_yr, max_yr),
            1,
            key="anim_year_range",
        )
        if reg_cfg["mode"] == "ice":
            evo_swatches = [
                ("#f5faff", "GLACIER / SNOW"),
                ("#0f6ef5", "FJORD WATER"),
                ("#e63919", "MISSING ICE"),
                ("#918778", "BEDROCK"),
            ]
        elif reg_cfg["mode"] == "urban_water":
            evo_swatches = [
                ("#b9aa91", "DESERT"),
                ("#1987ff", "WATER"),
                ("#ff8c00", "WATER LOSS"),
                ("#7c5ce6", "1984 URBAN"),
                ("#ff38c8", "URBAN GROWTH"),
                ("#ffd700", "THERMAL"),
            ]
        else:
            evo_swatches = [
                ("#b9aa91", "DRY BASIN"),
                ("#1987ff", "WATER"),
                ("#ff8c00", "WATER LOSS"),
                ("#ffd700", "THERMAL"),
                ("#dc143c", "COMBINED"),
            ]
        evo_legend = "".join(
            f"<span class='map-legend-item'><i class='legend-swatch' style='background:{c}'></i>{lbl}</span>"
            for c, lbl in evo_swatches
        )
        st.markdown(f"<div class='map-legend'>{evo_legend}</div>", unsafe_allow_html=True)

        run_sim = st.button("▶ Run Year-by-Year Simulation", type="primary", key="run_year_by_year_sim")
        evo_progress = st.empty()
        evo_title = st.empty()
        evo_map = st.empty()

        if run_sim:
            sim_years = [int(y) for y in bundle["years"] if anim_range[0] <= int(y) <= anim_range[1]]
            total_steps = max(len(sim_years), 1)
            default_chan = "ice" if reg_cfg["mode"] == "ice" else "all_types"
            for idx, yr in enumerate(sim_years):
                evo_progress.progress(
                    (idx + 1) / total_steps,
                    text=f"Simulating {reg_cfg['selector_label']}: {yr} ({idx + 1}/{total_steps})",
                )
                sim_json, sim_status = get_cached_frame_payload(
                    selected_region, yr, default_chan, "status", pickable=False
                )
                evo_title.markdown(
                    f"<div class='panel-head'><span>04 / SIMULATION · {reg_cfg['subtitle'].upper()}</span>"
                    f"<span>{sim_status}</span></div>",
                    unsafe_allow_html=True,
                )
                evo_map.pydeck_chart(
                    build_grid_deck(
                        sim_json, reg_cfg, cell_radius, cell_res_m, reg_cfg["lat"], reg_cfg["lon"], pickable=False
                    ),
                    width="stretch",
                    height=520,
                )
                time.sleep(0.08)
            evo_progress.empty()

    st.markdown("---")
    st.markdown(
        "<div class='small-note' style='text-align:center'>EARTH SYSTEM TREND DETECTIVE · NASA SPACE APPS · LANDSAT MAY–OCTOBER COMPOSITE TELEMETRY</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
