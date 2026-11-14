"""Earth System Trend Detective — a self-contained satellite telemetry dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import requests
import streamlit as st
from pathlib import Path

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
    "lst_c": {
        "name": "Land Surface Temperature",
        "short": "LST",
        "unit": "°C",
        "color": COLORS["crimson"],
        "trend": "warming",
    },
    "ndvi": {
        "name": "Vegetation Index",
        "short": "NDVI",
        "unit": "NDVI",
        "color": COLORS["green"],
        "trend": "declining vegetation vigor",
    },
    "ndbi": {
        "name": "Urban Index",
        "short": "NDBI",
        "unit": "NDBI",
        "color": "#ff38c8",
        "trend": "urban expansion",
    },
    "ndsi": {
        "name": "Snow Index",
        "short": "NDSI",
        "unit": "NDSI",
        "color": "#38bdf8",
        "trend": "declining snow and ice",
    },
    "ndwi": {
        "name": "Water Index",
        "short": "NDWI",
        "unit": "NDWI",
        "color": COLORS["cyan"],
        "trend": "declining surface water",
    },
    "salinity_index": {
        "name": "Salinity Index",
        "short": "SALINITY",
        "unit": "SI",
        "color": "#ffbf47",
        "trend": "rising salinity",
    },
}

REGION = {"lat": 45.20, "lon": 59.80, "elevation": 53}
MAP_REGIONS = {
    "aral": {
        "title": "Aral Sea · desiccation and salinization",
        "selector_label": "Aral Sea",
        "center": (45.20, 59.80), "bounds": (58.00, 43.30, 61.80, 46.80),
        "radius": 2900, "cell_size_m": 4000, "parquet": None, "mode": "water",
        "target_metrics": ("ndwi", "lst_c", "ndvi", "salinity_index"),
        "palette": {
            "land": [195, 175, 145], "water": [30, 144, 255], "dry": [194, 164, 96],
            "urban_core": [124, 92, 230],
            "urban": [255, 56, 200], "thermal": [255, 215, 0], "water_loss": [255, 140, 0],
            "ice": [56, 189, 248], "ice_loss": [255, 120, 40], "combined": [255, 20, 60],
        },
        "quantiles": {"water_low": 10, "water_presence": 70, "heat_high": 90, "urban_high": 90, "ice_low": 10, "ice_high": 75, "core": 75},
    },
    "lake_mead": {
        "title": "Las Vegas & Lake Mead · urban growth and water depletion",
        "selector_label": "Las Vegas & Lake Mead",
        "center": (36.285, -114.765), "bounds": (-115.48, 35.82, -114.05, 36.75),
        "radius": 750, "cell_size_m": 1500, "parquet": "urban_aquisition.parquet", "mode": "urban_water",
        "target_metrics": ("ndwi", "lst_c", "ndbi", "salinity_index"),
        "palette": {
            "land": [156, 157, 149], "water": [34, 145, 242], "urban_core": [133, 105, 222],
            "urban": [255, 75, 198], "thermal": [255, 205, 76], "water_loss": [255, 143, 54],
            "ice": [56, 189, 248], "ice_loss": [255, 120, 40], "combined": [239, 54, 92],
        },
        "quantiles": {"water_low": 12, "water_presence": 88, "heat_high": 95, "urban_high": 93, "ice_low": 10, "ice_high": 75, "core": 78},
    },
    "alaska_glaciers": {
        "title": "Harvard & Columbia Glaciers, Alaska · ice retreat",
        "selector_label": "Harvard & Columbia Glaciers",
        "center": (61.12, -147.38), "bounds": (-148.05, 60.90, -146.65, 61.50),
        "radius": 600, "cell_size_m": 1200, "parquet": "snow_ice_aquisition.parquet", "mode": "ice",
        "target_metrics": ("ndsi", "ndwi", "lst_c", "salinity_index"),
        "palette": {
            "land": [103, 119, 132], "water": [42, 111, 166], "urban_core": [124, 92, 230],
            "urban": [255, 56, 200], "thermal": [255, 215, 0], "water_loss": [255, 140, 0],
            "ice": [80, 200, 238], "ice_loss": [255, 120, 58], "combined": [255, 65, 90],
        },
        "quantiles": {"water_low": 10, "water_presence": 82, "heat_high": 92, "urban_high": 92, "ice_low": 12, "ice_high": 75, "core": 75},
    },
}


def regional_quantile_flags(
    frame: pd.DataFrame,
    column: str,
    percentile: float,
    tail: str,
    group_column: str = "year",
    region_key: str | None = None,
) -> pd.Series:
    """Flag only the requested regional percentile tail, independently by year."""
    flags = pd.Series(False, index=frame.index, dtype=bool)
    if region_key is not None:
        domains = {
            "aral": {"ndwi", "lst", "lst_c"},
            "lake_mead": {"ndwi", "lst", "lst_c", "ndbi"},
            "alaska_glaciers": {"ndsi"},
        }
        metric = column.removeprefix("z_")
        if metric not in domains[region_key]:
            return flags
    if column not in frame:
        return flags

    groups = frame.groupby(group_column).groups.values() if group_column in frame else [frame.index]
    for indices in groups:
        values = pd.to_numeric(frame.loc[indices, column], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if finite.size < 5 or finite.nunique() < 2:
            continue
        threshold = float(np.percentile(finite.to_numpy(dtype=float), percentile))
        if tail == "upper":
            flags.loc[finite.index] = finite > threshold
        else:
            flags.loc[finite.index] = finite < threshold
    return flags


def classify_regional_cells(cells: pd.DataFrame, region_key: str) -> pd.DataFrame:
    """Apply distribution-relative anomaly bins using the active region's data."""
    quantiles = MAP_REGIONS[region_key]["quantiles"]
    cells = cells.copy()
    for column in ("z_ndwi", "z_lst_c", "z_lst", "z_ndbi", "z_ndsi"):
        if column not in cells:
            source = {"z_lst": "z_lst_c"}.get(column)
            if source and source in cells:
                cells[column] = cells[source]
            else:
                cells[column] = np.nan
    if "z_lst" not in cells or cells["z_lst"].eq(0).all():
        cells["z_lst"] = cells["z_lst_c"]

    # Lake Mead uses stable physical index boundaries, not annual quantiles,
    # which would mark a fixed share of the surrounding desert every year.
    if region_key == "lake_mead":
        ndwi = pd.to_numeric(cells.get("ndwi", np.nan), errors="coerce")
        base_ndwi = pd.to_numeric(cells.get("base_ndwi", pd.Series(np.nan, index=cells.index)), errors="coerce")
        ndbi = pd.to_numeric(cells.get("ndbi", np.nan), errors="coerce")
        z_ndbi = pd.to_numeric(cells.get("z_ndbi", np.nan), errors="coerce")
        z_lst = pd.to_numeric(cells.get("z_lst", np.nan), errors="coerce")
        cells["base_urban"] = False
        cells["is_water_now"] = ndwi > 0.0
        water_loss = (base_ndwi > 0.0) & (ndwi <= 0.0)
        cells["anom_water"] = water_loss
        cells["anom_urban"] = (
            (ndbi > 0.05) & (ndwi < -0.1) & (z_ndbi > 1.5)
            & ~cells["is_water_now"] & ~water_loss
        )
        cells["anom_temp"] = (z_lst > 3.5) & ~cells["is_water_now"] & ~cells["anom_urban"] & ~cells["base_urban"]
        cells["anom_ice"] = False
        cells["is_glacier"] = False
        cells["anom_combined"] = False
        cells["anom_desiccation"] = False
        return cells

    cells["anom_water"] = regional_quantile_flags(cells, "z_ndwi", quantiles["water_low"], "lower", region_key=region_key)
    cells["anom_temp"] = regional_quantile_flags(cells, "z_lst", quantiles["heat_high"], "upper", region_key=region_key)
    if region_key == "lake_mead":
        cells["anom_urban"] = regional_quantile_flags(cells, "z_ndbi", quantiles["urban_high"], "upper", region_key=region_key) & (cells["z_ndbi"] > 0)
    else:
        cells["anom_urban"] = False

    if region_key == "alaska_glaciers":
        baseline_snow = cells.get("base_ndsi", cells.get("ndsi", pd.Series(0.0, index=cells.index)))
        finite_snow = pd.to_numeric(baseline_snow, errors="coerce").dropna()
        snow_floor = float(np.percentile(finite_snow, quantiles["core"])) if not finite_snow.empty else np.inf
        cells["anom_ice"] = (baseline_snow >= snow_floor) & regional_quantile_flags(cells, "z_ndsi", quantiles["ice_low"], "lower", region_key=region_key)
        cells["is_glacier"] = regional_quantile_flags(cells, "ndsi", quantiles["ice_high"], "upper", region_key=region_key)
    else:
        cells["anom_ice"] = False
        cells["is_glacier"] = False

    if region_key == "lake_mead":
        urban_reference = cells.get("base_ndbi", cells.get("ndbi", pd.Series(0.0, index=cells.index)))
        finite_urban = pd.to_numeric(urban_reference, errors="coerce").dropna()
        urban_floor = float(np.percentile(finite_urban, quantiles["core"])) if not finite_urban.empty else np.inf
        cells["base_urban"] = pd.to_numeric(urban_reference, errors="coerce") > urban_floor
    else:
        cells["base_urban"] = False

    cells["anom_combined"] = ((cells["anom_water"] & cells["anom_temp"]) if region_key == "aral"
                              else False)
    if region_key == "aral":
        # A per-year percentile always marks roughly the same share as water,
        # hiding the shoreline retreat in the time slider. NDWI has an
        # absolute water/land boundary around zero, so classify each year
        # against that fixed threshold.
        cells["is_water_now"] = pd.to_numeric(cells["ndwi"], errors="coerce") > 0
        ndwi = pd.to_numeric(cells["ndwi"], errors="coerce")
        base_ndwi = pd.to_numeric(cells.get("base_ndwi", pd.Series(np.nan, index=cells.index)), errors="coerce")
        base_lst = pd.to_numeric(cells.get("base_lst_c", pd.Series(np.nan, index=cells.index)), errors="coerce")
        lst = pd.to_numeric(cells.get("lst_c", pd.Series(np.nan, index=cells.index)), errors="coerce")
        z_lst = pd.to_numeric(cells.get("z_lst", pd.Series(np.nan, index=cells.index)), errors="coerce")
        z_ndwi = pd.to_numeric(cells.get("z_ndwi", pd.Series(np.nan, index=cells.index)), errors="coerce")
        historical_wet = cells.get("ever_water", pd.Series(False, index=cells.index)).fillna(False).astype(bool)
        historically_water = historical_wet | (base_ndwi > 0.0) | (z_ndwi < -1.5)
        lost_water = historically_water & (ndwi <= 0.0)
        cells["ever_water"] = historically_water
        cells["anom_water"] = cells["anom_water"].fillna(False).astype(bool) | lost_water
        backend_dry = cells.get("anom_desiccation", pd.Series(False, index=cells.index)).fillna(False).astype(bool)
        cells["anom_desiccation"] = backend_dry | (ndwi < -0.05)
        backend_combined = cells.get("anom_combined", pd.Series(False, index=cells.index)).fillna(False).astype(bool)
        elevated_heat = (z_lst > 1.5) | (lst > base_lst)
        cells["anom_combined"] = backend_combined | (lost_water & elevated_heat)
    elif region_key == "lake_mead":
        cells["is_water_now"] = regional_quantile_flags(cells, "ndwi", quantiles["water_presence"], "upper", region_key=region_key)
    else:
        cells["is_water_now"] = False
        cells["anom_desiccation"] = False
    return cells


def map_anomaly_summary(cells: pd.DataFrame, region_key: str, anomaly_filter: str) -> str:
    """Summarize actual flagged cells; basin-mean z-scores hide widespread retreat."""
    if cells.empty:
        return "NO MAP DATA"
    classified = classify_regional_cells(cells, region_key)
    flags = lambda name: classified.get(name, pd.Series(False, index=classified.index)).fillna(False).astype(bool)

    if region_key == "aral":
        ndwi = pd.to_numeric(classified.get("ndwi", np.nan), errors="coerce")
        dry = flags("anom_desiccation") | (ndwi < -0.05) | ~np.isfinite(ndwi)
        if anomaly_filter == "Extreme Heat (LST)":
            active, label = flags("anom_temp") | flags("anom_combined"), "THERMAL / COMBINED CELLS"
        elif anomaly_filter == "All Anomalies":
            active, label = dry | flags("anom_water") | flags("anom_temp") | flags("anom_combined"), "DRY / ANOMALOUS CELLS"
        else:
            active, label = dry | flags("anom_water") | flags("anom_combined"), "DRY / WATER-LOSS CELLS"
    elif region_key == "lake_mead":
        if anomaly_filter == "Urban Growth (NDBI)":
            active, label = flags("anom_urban"), "URBAN-GROWTH CELLS"
        elif anomaly_filter == "Extreme Heat (LST)":
            active, label = flags("anom_temp"), "HEAT-ANOMALY CELLS"
        elif anomaly_filter == "Water Shift (NDWI)":
            active, label = flags("anom_water"), "WATER-LOSS CELLS"
        else:
            active = flags("anom_water") | flags("anom_temp") | flags("anom_urban")
            label = "ANOMALOUS CELLS"
    else:
        active, label = flags("anom_ice"), "ICE-RETREAT CELLS"

    count = int(active.sum())
    total = len(classified)
    return f"{label} · {count:,}/{total:,} ({100 * count / total:.1f}%)"


def annual_metric_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Apply the friend's annual groupby mean and preserve missing years as NaN."""
    if frame.empty or metric not in frame or "year" not in frame:
        return pd.DataFrame(columns=["year", metric])
    observations = frame[["year", metric]].copy()
    observations[metric] = pd.to_numeric(observations[metric], errors="coerce")
    annual = observations.groupby("year", as_index=False)[metric].mean()
    annual = annual.dropna(subset=[metric])
    if annual.empty:
        return annual
    full_years = np.arange(1984, 2027)
    return annual.set_index("year").reindex(full_years).rename_axis("year").reset_index()


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


def telemetry_layout() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp, [data-testid="stAppViewContainer"] { background: #050505; color: #e8f0f2; }
        [data-testid="stSidebar"] { background: #080a0d; border-right: 1px solid #1b2930; }
        @media (min-width: 768px) {
            [data-testid="stSidebar"] { width: 22rem !important; min-width: 22rem !important; }
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
        h1 { font-size: clamp(1.7rem, 3vw, 2.5rem) !important; }
        .eyebrow { color: #00f3ff; font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; }
        .terminal-title { border-bottom: 1px solid #20343a; padding: 0 0 1rem; margin-bottom: 1.1rem; }
        .status-strip { display:flex; flex-wrap:wrap; gap:.6rem; margin:.2rem 0 1.2rem; }
        .status-chip { border:1px solid #183139; background:#091014; color:#aab8be; padding:.35rem .6rem; font-size:.7rem; letter-spacing:.08em; }
        .status-chip.live { color:#00f3ff; border-color:#00f3ff66; box-shadow:0 0 12px #00f3ff18; }
        .panel-head { display:flex; justify-content:space-between; align-items:center; gap:1rem; color:#dce7e9; font-size:.85rem; letter-spacing:.08em; text-transform:uppercase; border-bottom:1px solid #1a2b30; padding:0 0 .65rem; margin:.15rem 0 .65rem; }
        .panel-head span:last-child { color:#71848a; font-size:.68rem; }
        .telemetry-card { box-sizing:border-box; min-width:0; overflow:hidden; background:#0e1117; border:1px solid #1c3037; padding:.7rem; min-height:76px; }
        .telemetry-label { color:#71848a; font-size:.60rem; line-height:1.35; letter-spacing:.09em; text-transform:uppercase; margin-bottom:.35rem; overflow-wrap:anywhere; }
        .telemetry-value { color:#e8f0f2; font-size:clamp(.78rem, 1vw, .98rem); line-height:1.35; font-weight:700; white-space:normal; overflow-wrap:anywhere; }
        .telemetry-value.cyan { color:#00f3ff; text-shadow:0 0 12px #00f3ff66; }
        .telemetry-value.hot { color:#ff2a6d; text-shadow:0 0 12px #ff2a6d77; }
        .small-note { color:#71848a; font-size:.72rem; line-height:1.55; }
.map-legend { display:flex; flex-wrap:wrap; gap:.9rem; align-items:center; margin:.4rem 0 .75rem; color:#9aabb1; font-size:.66rem; letter-spacing:.06em; }
.map-legend-item { display:inline-flex; align-items:center; gap:.38rem; }
.legend-swatch { width:.62rem; height:.62rem; display:inline-block; border:1px solid #ffffff30; }
        div[data-testid="stPlotlyChart"] { border:1px solid #192a30; background:#080b0e; }
        div.stButton > button { border-radius:0; }
        hr { border-color:#1b2b30; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_theme(fig: go.Figure, height: int = 360) -> go.Figure:
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


@st.cache_data
def load_local_region_grid(region_key: str) -> pd.DataFrame:
    """Load and classify a friend's ingested Las Vegas or Alaska grid."""
    cfg = MAP_REGIONS[region_key]
    filename = cfg["parquet"]
    app_dir = Path(__file__).resolve().parent
    candidates = [app_dir / filename, app_dir / "data" / filename, Path.cwd() / filename,
                  Path.cwd() / "data" / filename]
    parquet_path = next((path for path in candidates if path.exists()), None)
    if parquet_path is None:
        return pd.DataFrame()

    cells = pd.read_parquet(parquet_path)
    coordinate_columns = ["latitude", "longitude"]
    if not all(column in cells for column in coordinate_columns):
        return pd.DataFrame()
    cells["year"] = pd.to_numeric(cells["year"], errors="coerce")
    cells = cells.dropna(subset=["year", *coordinate_columns]).copy()
    cells["year"] = cells["year"].astype(int)
    metrics = ["ndwi", "ndbi", "ndsi", "salinity_index", "lst_c"]
    for metric in metrics:
        if metric not in cells:
            cells[metric] = np.nan
        cells[metric] = pd.to_numeric(cells[metric], errors="coerce")

    coords = ["latitude", "longitude"]
    baseline = cells[cells["year"] <= 1992]
    cell_index = pd.MultiIndex.from_frame(cells[coords])
    robust_baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in metrics:
        medians = baseline.groupby(coords)[metric].median()
        baseline_medians = baseline.groupby(coords)[metric].transform("median")
        deviations = baseline[coords].copy()
        deviations["absolute_deviation"] = np.abs(baseline[metric].to_numpy() - baseline_medians.to_numpy())
        mads = deviations.groupby(coords)["absolute_deviation"].median()
        robust_baselines[metric] = (
            medians.reindex(cell_index).to_numpy(dtype=float),
            mads.reindex(cell_index).to_numpy(dtype=float),
        )
        median_values, mad_values = robust_baselines[metric]
        z_values = np.full(len(cells), np.nan, dtype=float)
        metric_values = cells[metric].to_numpy(dtype=float)
        valid_mad = np.isfinite(mad_values) & (mad_values > 0) & np.isfinite(metric_values)
        z_values[valid_mad] = 0.6745 * (metric_values[valid_mad] - median_values[valid_mad]) / mad_values[valid_mad]
        cells[f"z_{metric}"] = z_values

    base_ndsi = cells[cells["year"] <= 1986].groupby(coords)["ndsi"].median().reindex(cell_index).to_numpy(dtype=float)
    base_ndbi = baseline.groupby(coords)["ndbi"].median().reindex(cell_index).to_numpy(dtype=float)
    base_ndwi = baseline.groupby(coords)["ndwi"].median().reindex(cell_index).to_numpy(dtype=float)
    cells["base_ndsi"] = base_ndsi
    cells["base_ndbi"] = base_ndbi
    cells["base_ndwi"] = base_ndwi
    cells = classify_regional_cells(cells, region_key)
    cells["severity"] = cells[["z_ndwi", "z_lst_c", "z_ndbi", "z_ndsi"]].abs().sum(axis=1, min_count=1)
    return cells


@st.cache_data(ttl=3600)
def fetch_aral_grid_history() -> pd.DataFrame:
    """Load annual Aral slices once so every year shares the same coordinate index."""
    annual_frames = []
    for year in range(1984, 2027):
        try:
            response = requests.get("http://127.0.0.1:8080/anomalies", params={"year": year}, timeout=1.5)
            response.raise_for_status()
            records = response.json()
            if records:
                annual_frames.append(pd.DataFrame(records))
        except requests.HTTPError as error:
            # Missing years are expected in the Landsat archive; other HTTP
            # failures indicate the service is unavailable for this load.
            if error.response is not None and error.response.status_code == 404:
                continue
            break
        except requests.RequestException:
            break
        except ValueError:
            continue

    if not annual_frames:
        return pd.DataFrame()
    history = pd.concat(annual_frames, ignore_index=True)
    history = history.rename(columns={"lat": "latitude", "lon": "longitude"})
    history["year"] = pd.to_numeric(history["year"], errors="coerce")
    history = history.dropna(subset=["year", "latitude", "longitude"]).copy()
    history["year"] = history["year"].astype(int)
    return history


@st.cache_data(ttl=3600)
def fetch_grid_observations(year: int, region_key: str) -> pd.DataFrame:
    """Return the selected annual slice on a stable basin-wide coordinate index."""
    if region_key != "aral":
        regional_data = load_local_region_grid(region_key)
        if regional_data.empty:
            return regional_data
        return regional_data[regional_data["year"] == year].copy()

    history = fetch_aral_grid_history()
    if history.empty:
        return pd.DataFrame()
    coordinates = history[["latitude", "longitude"]].drop_duplicates().sort_values(
        ["latitude", "longitude"]
    )
    history["ndwi"] = pd.to_numeric(history["ndwi"], errors="coerce")
    baseline = history[history["year"].between(1984, 1992)]
    baseline_by_cell = baseline.groupby(["latitude", "longitude"])[["ndwi", "lst_c"]].median()
    baseline_by_cell = baseline_by_cell.rename(columns={"ndwi": "base_ndwi", "lst_c": "base_lst_c"})
    baseline_wet = baseline.groupby(["latitude", "longitude"])["ndwi"].max() > 0.0
    prior_wet = history[history["year"] < year].groupby(["latitude", "longitude"])["ndwi"].max() > 0.0
    annual = history[history["year"] == year].drop_duplicates(
        ["latitude", "longitude"], keep="last"
    )
    cells = coordinates.merge(annual, on=["latitude", "longitude"], how="left", validate="one_to_one")
    cells = cells.merge(baseline_by_cell, left_on=["latitude", "longitude"], right_index=True, how="left", validate="many_to_one")
    cell_index = pd.MultiIndex.from_frame(coordinates[["latitude", "longitude"]])
    cells["ever_water"] = (
        baseline_wet.reindex(cell_index).fillna(False).to_numpy(dtype=bool)
        | prior_wet.reindex(cell_index).fillna(False).to_numpy(dtype=bool)
    )
    cells["year"] = int(year)
    return cells


@st.cache_data(ttl=60)
def fetch_cell_history(region_key: str, latitude: float, longitude: float, metric: str) -> pd.DataFrame:
    """Get the annual signal for the active map cell when regional data exists."""
    if region_key == "aral":
        try:
            response = requests.get(
                "http://127.0.0.1:8080/time-series",
                params={"lat": latitude, "lon": longitude, "metric": metric},
                timeout=1.5,
            )
            response.raise_for_status()
            points = response.json().get("timeseries", [])
            rows = [
                {"year": point["year"], metric: point.get("values", {}).get(metric)}
                for point in points
            ]
            history = pd.DataFrame(rows).dropna(subset=[metric])
            return annual_metric_means(history, metric)
        except (requests.RequestException, ValueError, KeyError):
            return pd.DataFrame()

    regional_data = load_local_region_grid(region_key)
    if regional_data.empty or metric not in regional_data:
        return pd.DataFrame()
    latitudes = regional_data["latitude"].to_numpy(dtype=float)
    longitudes = regional_data["longitude"].to_numpy(dtype=float)
    distance = np.square(latitudes - latitude) + np.square((longitudes - longitude) * np.cos(np.radians(latitude)))
    nearest = regional_data.iloc[int(np.argmin(distance))]
    cell = regional_data[
        (regional_data["latitude"] == nearest["latitude"])
        & (regional_data["longitude"] == nearest["longitude"])
    ]
    return annual_metric_means(cell, metric)


def make_grid_map(selected_year: int, selected_lat: float, selected_lon: float,
                  grid_data: pd.DataFrame, anomaly_filter: str,
                  region_key: str) -> pdk.Deck:
    """Render available annual source observations as selectable map cells."""
    region = MAP_REGIONS[region_key]
    if grid_data.empty:
        raise ValueError("Cannot render a grid without source observations")
    cells = grid_data.copy()

    cells = cells.rename(columns={"lat": "latitude", "lon": "longitude"})
    cells["latitude"] = pd.to_numeric(cells["latitude"], errors="coerce")
    cells["longitude"] = pd.to_numeric(cells["longitude"], errors="coerce")
    valid_coordinates = np.isfinite(cells["latitude"]) & np.isfinite(cells["longitude"])
    cells = cells.loc[valid_coordinates].copy()
    if cells.empty:
        raise ValueError("Grid observations have no valid latitude/longitude coordinates")
    if "z_lst" not in cells and "z_lst_c" in cells:
        cells["z_lst"] = cells["z_lst_c"]
    for column, default in {
        "ndwi": np.nan, "ndsi": np.nan, "lst_c": np.nan, "z_ndwi": np.nan, "z_lst": np.nan,
        "z_lst_c": np.nan, "z_ndbi": np.nan, "z_ndsi": np.nan, "ndbi": np.nan,
        "anom_water": False, "anom_temp": False, "anom_combined": False, "anom_desiccation": False,
        "anom_urban": False, "base_urban": False, "anom_ice": False,
        "is_glacier": False, "is_water_now": False,
    }.items():
        if column not in cells:
            cells[column] = default
    cells = classify_regional_cells(cells, region_key)
    cells["epoch"] = selected_year
    if region["mode"] == "ice":
        active = cells["anom_ice"].astype(bool)
    elif anomaly_filter == "Urban Growth (NDBI)":
        active = cells["anom_urban"].astype(bool)
    elif anomaly_filter == "Glacier Retreat (NDSI)":
        active = cells["anom_ice"].astype(bool)
    elif anomaly_filter == "Extreme Heat (LST)":
        active = cells["anom_temp"].astype(bool)
        if region_key == "aral":
            active |= cells["anom_combined"].astype(bool)
    elif anomaly_filter == "Drought (NDVI)":
        active = cells["anom_water"].astype(bool)
        if region_key == "aral":
            active |= cells["anom_combined"].astype(bool)
    elif anomaly_filter == "Water Shift (NDWI)":
        active = cells["anom_water"].astype(bool)
        if region_key == "aral":
            active |= cells["anom_combined"].astype(bool)
    else:
        active = (cells["anom_water"].astype(bool) | cells["anom_temp"].astype(bool)
                  | cells["anom_urban"].astype(bool) | cells["anom_ice"].astype(bool))
        if region_key == "aral":
            active |= cells["anom_combined"].astype(bool)

    combined = cells["anom_combined"].astype(bool) if region_key == "aral" else pd.Series(False, index=cells.index)
    if region_key == "aral":
        # Water-loss cells remain visible across channel selections so dried
        # former lake cells cannot fall back to the neutral basin fill.
        water = cells["anom_water"].astype(bool)
    elif region_key == "lake_mead":
        water = cells["anom_water"].astype(bool) & active
    else:
        water = pd.Series(False, index=cells.index)
    thermal = cells["anom_temp"].astype(bool) & active if region_key in {"aral", "lake_mead"} else pd.Series(False, index=cells.index)
    urban = cells["anom_urban"].astype(bool) & active if region_key == "lake_mead" else pd.Series(False, index=cells.index)
    urban_core = pd.Series(False, index=cells.index)
    ice_loss = cells["anom_ice"].astype(bool) & active if region_key == "alaska_glaciers" else pd.Series(False, index=cells.index)
    open_water = cells["is_water_now"].astype(bool) if region_key in {"aral", "lake_mead"} else pd.Series(False, index=cells.index)
    aral_dry = cells["anom_desiccation"].astype(bool) if region_key == "aral" else pd.Series(False, index=cells.index)
    if region_key == "aral":
        ndwi_values = pd.to_numeric(cells["ndwi"], errors="coerce")
        aral_dry |= ~np.isfinite(ndwi_values) | (ndwi_values < -0.05)
    glacier = cells["is_glacier"].astype(bool) if region_key == "alaska_glaciers" else pd.Series(False, index=cells.index)
    palette = region["palette"]
    colors = np.tile(palette["land"], (len(cells), 1))
    colors[open_water] = palette["water"]
    if region_key == "aral":
        # Lay the opaque sandy basin beneath anomaly colors so hot spots remain legible.
        colors[aral_dry] = palette["dry"]
    colors[urban_core] = palette["urban_core"]
    colors[urban] = palette["urban"]
    colors[glacier] = palette["ice"]
    colors[thermal] = palette["thermal"]
    colors[water] = palette["water_loss"]
    colors[ice_loss] = palette["ice_loss"]
    colors[combined] = palette["combined"]
    cells[["r", "g", "b"]] = colors.tolist()
    cells["alpha"] = 210
    if region_key == "aral":
        cells.loc[aral_dry, "alpha"] = 205
        cells.loc[open_water, "alpha"] = 240
        cells.loc[water | thermal | combined, "alpha"] = 245
    metric_columns = [column for column in ("ndwi", "ndsi", "ndbi", "lst_c", "salinity_index") if column in cells]
    no_metric_data = ~cells[metric_columns].notna().any(axis=1) if metric_columns else pd.Series(True, index=cells.index)
    # Keep the coordinate lattice visible when an annual pixel is absent;
    # for Aral these cells are rendered as dry/salt-flat instead of gaps.
    if region_key != "aral":
        cells.loc[no_metric_data, "alpha"] = 0
    if region_key == "aral":
        cells["diagnostic"] = np.select(
            [combined, water, thermal, aral_dry, open_water],
            ["CRITICAL: COMPOUND ANOMALY", "WATER ANOMALY", "THERMAL ANOMALY",
             "DRY LAKEBED / DESICCATION / SALT FLATS", "NORMAL / WATER"],
            default="NORMAL / DRY HISTORICAL CELL",
        )
    else:
        cells["diagnostic"] = np.select(
            [aral_dry, combined, ice_loss, water, urban, thermal, glacier, urban_core, open_water],
            ["DRY LAKEBED / DESICCATION / SALT FLATS", "CRITICAL: COMPOUND ANOMALY", "GLACIER RETREAT", "WATER ANOMALY", "URBAN EXPANSION",
             "THERMAL ANOMALY", "GLACIER / SNOW", "1984 URBAN CORE", "NORMAL / WATER"],
            default="NORMAL / DRY HISTORICAL CELL",
        )
    if region_key != "aral":
        cells.loc[no_metric_data, "diagnostic"] = "NO SOURCE METRIC DATA"
    cells["z_score"] = cells.get("z_lst", pd.Series(np.full(len(cells), np.nan), index=cells.index))
    cell_half_lat = region["cell_size_m"] / (2 * 111132.0)
    cell_half_lon = region["cell_size_m"] / (2 * 111320.0 * np.cos(np.deg2rad(selected_lat)))
    active_bbox = [[
        [selected_lon-cell_half_lon, selected_lat-cell_half_lat],
        [selected_lon+cell_half_lon, selected_lat-cell_half_lat],
        [selected_lon+cell_half_lon, selected_lat+cell_half_lat],
        [selected_lon-cell_half_lon, selected_lat+cell_half_lat],
    ]]
    min_lon = float(cells["longitude"].min()) - cell_half_lon
    max_lon = float(cells["longitude"].max()) + cell_half_lon
    min_lat = float(cells["latitude"].min()) - cell_half_lat
    max_lat = float(cells["latitude"].max()) + cell_half_lat
    map_bounds = (min_lon, min_lat, max_lon, max_lat)
    grid_layer = pdk.Layer(
        "ScatterplotLayer", id="aral-grid", data=cells,
        get_position="[longitude, latitude]", get_fill_color="[r, g, b, alpha]",
        get_radius=region["radius"],
        radius_min_pixels=4 if region_key == "aral" else 2,
        radius_max_pixels=16 if region_key == "aral" else 8,
        pickable=True, auto_highlight=True,
    )
    selected_layer = pdk.Layer(
        "PolygonLayer", id="active-cell", data=[{"polygon": active_bbox[0]}],
        get_polygon="polygon", get_fill_color=[255, 255, 255, 12],
        get_line_color=[255, 255, 255, 255], line_width_min_pixels=2,
        stroked=True, filled=True, pickable=False,
    )
    min_lon, min_lat, max_lon, max_lat = map_bounds
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    mercator_y = lambda latitude: (1 - np.arcsinh(np.tan(np.radians(np.clip(latitude, -85, 85)))) / np.pi) / 2
    x_span = (max_lon - min_lon) / 360
    y_span = abs(mercator_y(max_lat) - mercator_y(min_lat))
    fit_width = 520 if region_key == "aral" else 440
    fit_height = 480 if region_key == "aral" else 340
    fit_zoom = min(
        np.log2(fit_width / (512 * x_span)),
        np.log2(fit_height / (512 * y_span)),
    ) - 0.45
    constrained_view = pdk.View(
        type="MapView",
        controller={
            "maxBounds": [[min_lon, min_lat], [max_lon, max_lat]],
            "maxBoundsPadding": 12,
            "rubberBand": False,
        },
    )
    deck = pdk.Deck(
        layers=[grid_layer, selected_layer],
        views=[constrained_view],
        initial_view_state=pdk.ViewState(
            latitude=center_lat, longitude=center_lon, zoom=fit_zoom,
            min_zoom=fit_zoom, max_zoom=16,
        ),
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={"text": "{diagnostic}\nCoordinates: {latitude}, {longitude}\nNDWI: {ndwi} (Z: {z_ndwi})\nTemperature: {lst_c} °C (Z: {z_lst})\nEpoch: {epoch}"},
    )
    return deck


def make_history_chart(df: pd.DataFrame, key: str, baseline: np.ndarray) -> go.Figure:
    meta = METRICS[key]
    hist = annual_metric_means(df, key)
    baseline = np.asarray(baseline, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    median = float(np.median(baseline)) if baseline.size else np.nan
    mad = float(np.median(np.abs(baseline - median))) if baseline.size else np.nan
    hist["z"] = robust_z_score(hist[key], baseline)
    hist["rolling_mean"] = hist[key].rolling(window=9, min_periods=3, center=True).mean()
    hist["rolling_std"] = hist[key].rolling(window=9, min_periods=3, center=True).std()
    missing_value = hist[key].isna()
    hist.loc[missing_value, ["rolling_mean", "rolling_std"]] = np.nan
    hist["band_low"] = hist["rolling_mean"] - 2 * hist["rolling_std"]
    hist["band_high"] = hist["rolling_mean"] + 2 * hist["rolling_std"]
    spikes = hist[hist["z"].abs() > 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist["band_high"], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip", connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist["band_low"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="#00f3ff16", name="Rolling ±2σ volatility band",
        hoverinfo="skip", connectgaps=False,
    ))
    if np.isfinite(median):
        fig.add_hline(y=median, line=dict(color="#9aabb155", dash="dash", width=1), annotation_text="BASELINE MEDIAN",
                      annotation_font=dict(color="#71848a", size=9), annotation_position="bottom right")
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist[key], mode="lines+markers", name=f"Annual composite {meta['short']}",
        line=dict(color=meta["color"], width=2.2), marker=dict(size=4), connectgaps=False,
        hovertemplate="YEAR %{x:.0f}<br>ANNUAL COMPOSITE MEAN %{y:.3f} " + meta["unit"] + "<extra></extra>",
    ))
    if not spikes.empty:
        fig.add_trace(go.Scatter(
            x=spikes["year"], y=spikes[key], mode="markers", name="ANNUAL ANOMALY ≥2σ",
            marker=dict(size=8, color=COLORS["crimson"], symbol="diamond", line=dict(color="#fff", width=.8)),
            text=[f"{v:+.2f}σ" for v in spikes["z"]],
            hovertemplate="ANNUAL ALERT %{x:.0f}<br>MEAN %{y:.3f}<br>Z %{text}<extra></extra>",
        ))
    chart_theme(fig, 390)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="YEAR · ANNUAL COMPOSITE", range=[1984, 2027], dtick=5)
    fig.update_yaxes(title=meta["unit"])
    return fig


def t_critical_975(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value."""
    if student_t is not None:
        return float(student_t.ppf(0.975, degrees_of_freedom))

    # Cornish-Fisher expansion fallback, accurate for the long annual record.
    z = 1.959963984540054
    v = float(degrees_of_freedom)
    return float(
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )


def build_forecast(
    df: pd.DataFrame,
    key: str,
    extraction_pct: int = 0,
    drought_intensity: int = 0,
) -> tuple[go.Figure, dict[str, float]]:
    meta = METRICS[key]
    annual = annual_metric_means(df, key)
    annual = annual[np.isfinite(annual[key]) & np.isfinite(annual["year"])]
    if len(annual) < 3:
        raise ValueError("At least three valid annual observations are required for an OLS forecast")
    x = annual["year"].to_numpy(dtype=float)
    y = annual[key].to_numpy(dtype=float)
    ols_slope, ols_intercept = np.polyfit(x, y, 1)
    residual = y - (ols_slope * x + ols_intercept)
    dof = len(x) - 2
    residual_sd = float(np.sqrt(np.sum(residual**2) / dof))
    anchor_year = float(x[-1])
    future = np.arange(anchor_year, 2036, dtype=float)
    mean_x = x.mean()
    sxx = np.sum((x - mean_x) ** 2)
    prediction_se = residual_sd * np.sqrt(1 + 1 / len(x) + (future - mean_x) ** 2 / sxx)
    margin = t_critical_975(dof) * prediction_se
    extraction_pressure = extraction_pct / 30.0
    drought_pressure = drought_intensity / 100.0
    stress_level = min(1.0, 0.65 * extraction_pressure + 0.65 * drought_pressure)
    slope_impact = {
        "ndwi": -0.0050,
        "ndvi": -0.0025,
        "ndbi": 0.0040,
        "ndsi": -0.0025,
        "salinity_index": 0.0035,
        "lst_c": 0.04,
    }[key]
    # The selected extraction and drought controls adjust the average trend.
    # Best remains a recovery case; worst is a fixed high-stress reference so
    # moving the controls changes the average line rather than the worst bound.
    favorable_direction = -1.0 if key in {"lst_c", "salinity_index", "ndbi"} else 1.0
    recovery_coefficient = max(abs(float(ols_slope)) * 0.2, abs(slope_impact) * 0.2)
    best_slope = float(ols_slope + favorable_direction * recovery_coefficient)
    average_slope = float(ols_slope + slope_impact * stress_level)
    worst_slope = float(ols_slope + slope_impact * 1.25)
    # All scenarios share the final 2026 observed annual mean as their anchor.
    years_from_anchor = future - anchor_year
    best_projected = y[-1] + best_slope * years_from_anchor
    average_projected = y[-1] + average_slope * years_from_anchor
    worst_projected = y[-1] + worst_slope * years_from_anchor
    best_2035 = float(best_projected[-1])
    average_2035 = float(average_projected[-1])
    worst_2035 = float(worst_projected[-1])

    fig = go.Figure()
    scenario_low = np.minimum(best_projected, worst_projected)
    scenario_high = np.maximum(best_projected, worst_projected)
    fig.add_trace(go.Scatter(x=future, y=scenario_low, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future, y=scenario_high, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="#ff2a6d12", name="Best–worst scenario range",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future, y=average_projected + margin, mode="lines", line_width=0,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future, y=average_projected - margin, mode="lines", line_width=0,
                             fill="tonexty", fillcolor="#00f3ff20", name="95% prediction interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Historical annual composite",
                             line=dict(color=meta["color"], width=2)))
    fig.add_trace(go.Scatter(x=future, y=best_projected, mode="lines", name="Best scenario · no added stress",
                             line=dict(color=COLORS["green"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=future, y=average_projected, mode="lines", name="Average scenario · selected inputs",
                             line=dict(color=COLORS["cyan"], width=2.8, dash="dash")))
    fig.add_trace(go.Scatter(x=future, y=worst_projected, mode="lines", name="Worst scenario · maximum stress",
                             line=dict(color=COLORS["crimson"], width=2, dash="longdash")))
    fig.add_vline(x=anchor_year, line_color="#ffffff77", line_dash="dot", annotation_text="NOW")
    chart_theme(fig, 350)
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    fig.update_xaxes(title="YEAR", range=[1984, 2035])
    fig.update_yaxes(title=meta["unit"])
    stats = {"slope": average_slope, "ols_slope": float(ols_slope),
             "best_slope": best_slope, "worst_slope": worst_slope,
             "ols_intercept": float(ols_intercept), "sd": residual_sd,
             "2035": average_2035, "best_2035": best_2035,
             "worst_2035": worst_2035, "delta_2035": average_2035 - best_2035,
             "margin": float(margin[-1]), "2026": float(y[-1]),
             "stress_level": stress_level, "dof": float(dof),
             "t_critical": t_critical_975(dof)}
    return fig, stats


telemetry_layout()

with st.sidebar:
    st.markdown('<div class="eyebrow">MISSION CONTROL / EO-01</div>', unsafe_allow_html=True)
    st.markdown("## DASHBOARD CONTROLS")
    selected_region = st.selectbox(
        "MONITORING REGION", list(MAP_REGIONS),
        format_func=lambda key: MAP_REGIONS[key]["selector_label"], key="monitoring_region",
    )
    selected_region_config = MAP_REGIONS[selected_region]
    st.caption(selected_region_config["title"].upper())
    st.markdown("---")
    st.markdown('<div class="small-note">ARCHIVE · LANDSAT 4–9<br>GRID RESOLUTION · REGION DEPENDENT<br>REFERENCE · 1984–1992</div>', unsafe_allow_html=True)


if st.session_state.get("active_region") != selected_region:
    center_lat, center_lon = selected_region_config["center"]
    st.session_state.active_lat = center_lat
    st.session_state.active_lon = center_lon
    st.session_state.active_region = selected_region
    st.session_state.pop("basin_map", None)
st.session_state.setdefault("active_lat", selected_region_config["center"][0])
st.session_state.setdefault("active_lon", selected_region_config["center"][1])

st.markdown(
    "<div class='terminal-title'><div class='eyebrow'>EARTH SYSTEM TREND DETECTIVE / TACTICAL TELEMETRY</div>"
    "<h1>REGIONAL CHANGE MONITOR</h1></div>",
    unsafe_allow_html=True,
)

explorer_tab, predictive_tab = st.tabs(["Tactical Explorer", "Predictive Scenario Engine"])

with explorer_tab:
    map_column_widths = [1.65, 1.15] if selected_region == "aral" else [2.0, 1.15]
    map_col, map_controls = st.columns(map_column_widths, gap="large")
    with map_controls:
        st.markdown("<div class='panel-head'><span>Map Controls</span><span>MAP ONLY</span></div>", unsafe_allow_html=True)
        selected_year = st.slider("OBSERVATION YEAR", 1984, 2026, 2026, 1, key="observation_year")
    if selected_region == "alaska_glaciers":
        anomaly_channels = ["All Anomalies", "Glacier Retreat (NDSI)"]
    elif selected_region == "lake_mead":
        anomaly_channels = ["All Anomalies", "Extreme Heat (LST)", "Water Shift (NDWI)", "Urban Growth (NDBI)"]
    else:
        anomaly_channels = ["All Anomalies", "Extreme Heat (LST)", "Drought (NDVI)", "Water Shift (NDWI)"]
    with map_controls:
        anomaly_filter = st.selectbox("ANOMALY CHANNEL", anomaly_channels, key="map_anomaly_channel")
        st.caption("These controls update the heatmap only.")

    map_grid_data = fetch_grid_observations(selected_year, selected_region)
    map_has_coordinates = False
    if not map_grid_data.empty and {"latitude", "longitude"}.issubset(map_grid_data.columns):
        map_coordinate_values = map_grid_data[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
        map_has_coordinates = bool(np.isfinite(map_coordinate_values.to_numpy(dtype=float)).all(axis=1).any())
    map_widget_key = f"basin_map_{selected_region}_{selected_year}"
    saved_selection = st.session_state.get(map_widget_key)
    saved_selection = saved_selection.get("selection", {}) if isinstance(saved_selection, dict) else getattr(saved_selection, "selection", {})
    saved_objects = saved_selection.get("objects", {}) if hasattr(saved_selection, "get") else {}
    selected_objects = saved_objects.get("aral-grid", [])
    if selected_objects:
        selected_cell = selected_objects[0]
        st.session_state.active_lat = float(selected_cell["latitude"])
        st.session_state.active_lon = float(selected_cell["longitude"])
    map_status = map_anomaly_summary(map_grid_data, selected_region, anomaly_filter)

    with map_col:
        st.markdown(f"<div class='panel-head'><span>01 / Interactive Grid · {selected_region_config['title']}</span><span>{map_status} · {selected_year}</span></div>", unsafe_allow_html=True)
        if selected_region == "alaska_glaciers":
            legend = "<span class='map-legend-item'><i class='legend-swatch' style='background:#38bdf8'></i>GLACIER / SNOW</span><span class='map-legend-item'><i class='legend-swatch' style='background:#ff7828'></i>ICE RETREAT</span>"
        elif selected_region == "lake_mead":
            legend = "<span class='map-legend-item'><i class='legend-swatch' style='background:#1e90ff'></i>WATER</span><span class='map-legend-item'><i class='legend-swatch' style='background:#ff8c00'></i>WATER LOSS</span><span class='map-legend-item'><i class='legend-swatch' style='background:#ff38c8'></i>URBAN GROWTH</span><span class='map-legend-item'><i class='legend-swatch' style='background:#ffd700'></i>HEAT</span>"
        else:
            legend = "<span class='map-legend-item'><i class='legend-swatch' style='background:#1e90ff'></i>WATER</span><span class='map-legend-item'><i class='legend-swatch' style='background:#c2a460'></i>DRY LAKEBED / SALT FLATS</span><span class='map-legend-item'><i class='legend-swatch' style='background:#ff8c00'></i>WATER ANOMALY</span><span class='map-legend-item'><i class='legend-swatch' style='background:#ffd700'></i>THERMAL</span><span class='map-legend-item'><i class='legend-swatch' style='background:#ff143c'></i>COMBINED</span>"
        st.markdown(f"<div class='map-legend'>{legend}<span class='map-legend-item'>□ ACTIVE CELL</span></div>", unsafe_allow_html=True)
        if map_grid_data.empty or not map_has_coordinates:
            st.info(f"No {selected_year} grid observations are available for {selected_region_config['selector_label']}. Check the Rust API or the region's Parquet file.")
        else:
            basin_deck = make_grid_map(
                selected_year, st.session_state.active_lat, st.session_state.active_lon,
                map_grid_data, anomaly_filter, selected_region,
            )
            st.pydeck_chart(
                basin_deck, width="stretch",
                height=750 if selected_region == "aral" else 780,
                on_select="rerun", selection_mode="single-object", key=map_widget_key,
            )
        if not map_grid_data.empty:
            grid_source = "LIVE RUST GRID" if selected_region == "aral" else "LOCAL INGESTED LANDSAT GRID"
        else:
            grid_source = "SOURCE GRID UNAVAILABLE"
        st.markdown(f"<div class='small-note'>{grid_source} · {selected_region_config['cell_size_m'] / 1000:.1f} KM GRID · ACTIVE CELL {st.session_state.active_lat:.4f}, {st.session_state.active_lon:.4f}</div>", unsafe_allow_html=True)

    chart_col, chart_controls = st.columns([4.2, 1.15], gap="large")
    with chart_controls:
        st.markdown("<div class='panel-head'><span>Chart Controls</span><span>HISTORY ONLY</span></div>", unsafe_allow_html=True)
        target_metric_options = selected_region_config["target_metrics"]
        if st.session_state.get("history_metric") not in target_metric_options:
            st.session_state["history_metric"] = target_metric_options[0]
        metric_choice = st.selectbox("TARGET METRIC", target_metric_options, format_func=lambda k: f"{METRICS[k]['name']}  /  {METRICS[k]['unit']}", key="history_metric")
        st.caption("The chart always shows 1984–2026.")
    meta = METRICS[metric_choice]
    cell_history = fetch_cell_history(selected_region, st.session_state.active_lat, st.session_state.active_lon, metric_choice)
    history_source = "ACTIVE CELL HISTORY" if not cell_history.empty else "CELL HISTORY UNAVAILABLE"
    if cell_history.empty:
        cell_history = pd.DataFrame(columns=["year", metric_choice])
    baseline_series = cell_history.loc[cell_history["year"] <= 1992, metric_choice].to_numpy(dtype=float)
    baseline_series = baseline_series[np.isfinite(baseline_series)]
    baseline_median = float(np.median(baseline_series)) if baseline_series.size else np.nan
    baseline_mad = float(np.median(np.abs(baseline_series - baseline_median))) if baseline_series.size else np.nan
    annual_history = annual_metric_means(cell_history, metric_choice)
    annual_history_z = robust_z_score(annual_history[metric_choice], baseline_series)
    spike_count = int((np.abs(annual_history_z) > 2).sum()) if np.isfinite(annual_history_z).any() else 0
    valid_history = annual_history.dropna(subset=[metric_choice])
    latest_value = float(valid_history.iloc[-1][metric_choice]) if not valid_history.empty else np.nan
    current_z = robust_z_score(latest_value, baseline_series)

    with chart_col:
        st.markdown(f"<div class='panel-head'><span>02 / Cell Composite History · {meta['name']}</span><span>{st.session_state.active_lat:.3f}° · {st.session_state.active_lon:.3f}°</span></div>", unsafe_allow_html=True)
        if annual_history[metric_choice].notna().any():
            st.plotly_chart(make_history_chart(cell_history, metric_choice, baseline_series), width="stretch", config={"displayModeBar": False})
        else:
            st.info("No historical observations are available for this cell and metric.")
        baseline_note = f"BASELINE MEDIAN {baseline_median:.3f}, MAD {baseline_mad:.4f}" if np.isfinite(baseline_median) and np.isfinite(baseline_mad) else "BASELINE STATISTICS UNAVAILABLE"
        latest_note = f"LATEST OBSERVED Z {current_z:+.2f}σ" if np.isfinite(current_z) else "LATEST OBSERVED Z —"
        st.markdown(f"<div class='small-note'>{history_source} · {baseline_note} · {latest_note} · {spike_count} ANOMALY YEARS (MAD Z > 2)</div>", unsafe_allow_html=True)

with predictive_tab:
    st.markdown("<div class='eyebrow'>03 / 2026–2035 · SCENARIO MODELING</div><h2>Predictive Scenario Engine</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Adjust basin stress assumptions and compare best, average, and worst 2035 trajectories.</div>", unsafe_allow_html=True)
    forecast_metric_options = selected_region_config["target_metrics"]
    if st.session_state.get("forecast_metric") not in forecast_metric_options:
        st.session_state["forecast_metric"] = forecast_metric_options[0]
    forecast_metric = st.selectbox(
        "FORECAST METRIC", forecast_metric_options,
        format_func=lambda k: f"{METRICS[k]['name']}  /  {METRICS[k]['unit']}",
        key="forecast_metric",
    )
    forecast_meta = METRICS[forecast_metric]
    with st.container(border=True):
        st.markdown("<div class='panel-head'><span>Basin Stress Simulator</span><span>SCENARIO INPUTS</span></div>", unsafe_allow_html=True)
        control_cols = st.columns([1, 1, 1])
        with control_cols[0]:
            extraction_pct = st.slider(
                "Upstream agricultural extraction", min_value=0, max_value=30,
                value=0, step=1, format="%d%%", key="basin_extraction",
            )
        with control_cols[1]:
            drought_intensity = st.slider(
                "Climate drought intensity", min_value=0, max_value=100,
                value=0, step=5, format="%d%%", key="basin_drought",
            )
        with control_cols[2]:
            is_pro = st.toggle("Show model diagnostics", value=True, key="forecast_diagnostics")
            st.markdown("<div class='small-note'>Scenario stress is applied to the OLS slope from 2026 onward.</div>", unsafe_allow_html=True)

    forecast_cell_history = fetch_cell_history(
        selected_region, st.session_state.active_lat, st.session_state.active_lon, forecast_metric,
    )
    forecast_history = annual_metric_means(forecast_cell_history, forecast_metric)
    forecast_history = forecast_history[np.isfinite(forecast_history[forecast_metric])]
    forecast_baseline = forecast_cell_history.loc[
        forecast_cell_history["year"] <= 1992, forecast_metric
    ].to_numpy(dtype=float) if not forecast_cell_history.empty else np.array([], dtype=float)
    forecast_baseline = forecast_baseline[np.isfinite(forecast_baseline)]
    forecast_latest = float(forecast_history.iloc[-1][forecast_metric]) if not forecast_history.empty else np.nan
    forecast_current_z = robust_z_score(forecast_latest, forecast_baseline)
    if len(forecast_history) < 3:
        st.info("A forecast requires at least three observed annual values for the selected cell and metric.")
    else:
        forecast_fig, forecast_stats = build_forecast(forecast_history, forecast_metric, extraction_pct, drought_intensity)
        st.markdown("<div class='panel-head'><span>03 / Forecast Range</span><span>BEST · AVERAGE · WORST</span></div>", unsafe_allow_html=True)
        st.plotly_chart(forecast_fig, width="stretch", config={"displayModeBar": False})
        if is_pro:
            st.markdown("<div class='panel-head'><span>Model Diagnostics</span><span>ANNUAL OLS / 95% PREDICTION INTERVAL</span></div>", unsafe_allow_html=True)
            risk = "CRITICAL" if forecast_stats["stress_level"] >= 0.65 or (np.isfinite(forecast_current_z) and abs(forecast_current_z) > 2) else "ELEVATED"
            stat_cols = st.columns(4)
            stat_values = [
                ("BASE OLS β", f"{forecast_stats['ols_slope']:+.5f} {forecast_meta['unit']}/yr"),
                ("AVERAGE β", f"{forecast_stats['slope']:+.5f} {forecast_meta['unit']}/yr"),
                ("2035 PROJECTION", f"{forecast_stats['2035']:.3f} {forecast_meta['unit']}"),
                ("EARLY WARNING", risk),
            ]
            for col, (label, value) in zip(stat_cols, stat_values):
                with col:
                    st.markdown(f"<div class='telemetry-card'><div class='telemetry-label'>{label}</div><div class='telemetry-value {'hot' if label == 'EARLY WARNING' and risk == 'CRITICAL' else 'cyan'}'>{value}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-note'>95% PREDICTION ENVELOPE · {forecast_stats['2035'] - forecast_stats['margin']:.3f} TO {forecast_stats['2035'] + forecast_stats['margin']:.3f} {forecast_meta['unit']} · TREND: {forecast_meta['trend'].upper()}</div>", unsafe_allow_html=True)

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
                "ndwi": f"The water-index scenario declines by {abs(forecast_stats['delta_2035']):.3f}, increasing shoreline retreat and exposed lakebed.",
                "ndvi": f"The vegetation scenario loses {abs(forecast_stats['delta_2035']):.3f} index units, leaving less cover on dry soils.",
                "salinity_index": f"The salinity scenario rises by {abs(forecast_stats['delta_2035']):.3f}, increasing salt-concentration and tipping-point risk.",
                "lst_c": f"The temperature scenario adds {abs(forecast_stats['delta_2035']):.2f} °C as dry soils lose evaporative cooling.",
            }
            impact_text = "; ".join(sources) + ". " + impacts[forecast_metric]
        stress_label = "HIGH" if stress_level >= 0.65 else "MODERATE" if stress_level >= 0.3 else "LOW"
        st.markdown(
            f"<div class='telemetry-card' style='margin-top:.8rem;border-color:#00f3ff66'>"
            f"<div class='telemetry-label'>BASIN STRESS IMPACT · {stress_label} · 2035 SCENARIO</div>"
            f"<div class='small-note' style='color:#c4d0d4'>{impact_text}</div>"
            f"<div class='small-note' style='margin-top:.45rem'>2035 RANGE · BEST {forecast_stats['best_2035']:.3f} · "
            f"AVERAGE {forecast_stats['2035']:.3f} · WORST {forecast_stats['worst_2035']:.3f} {forecast_meta['unit']}</div></div>",
            unsafe_allow_html=True,
        )


st.markdown("---")
st.markdown("<div class='small-note' style='text-align:center'>EARTH SYSTEM TREND DETECTIVE · NASA SPACE APPS · LANDSAT MAY–OCTOBER COMPOSITE TELEMETRY</div>", unsafe_allow_html=True)
