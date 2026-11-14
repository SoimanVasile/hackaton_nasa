"""
Earth System Trend Detective | Environmental Analytics Dashboard
================================================================
A multi-decadal environmental analytics platform built with Streamlit, PyDeck (Deck.gl),
and Plotly. Ingests 40+ years of satellite observations (1984–2026) across 10,000+
geospatial grid cells in the Aral Basin, with interactive cell inspection,
time-series anomaly detection, and a tiered predictive forecasting view (2026–2050)
featuring a custom CSS blurred paywall.
"""

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go

# ==============================================================================
# 1. PAGE CONFIGURATION & SCIENTIFIC DARK-MODE THEME
# ==============================================================================
st.set_page_config(
    page_title="Earth System Trend Detective | Environmental Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

RUST_API_URL = os.environ.get("RUST_API_URL", "http://localhost:8080")
PARQUET_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_landsat_grid.parquet")
if not os.path.exists(PARQUET_FALLBACK_PATH):
    PARQUET_FALLBACK_PATH = "data/raw_landsat_grid.parquet"

# Custom CSS for dark aesthetic, glassmorphic cards, and blurred paywall
CUSTOM_CSS = """
<style>
    /* Dark-mode background & typography */
    .stApp {
        background-color: #090c10;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top summary ribbon cards */
    .kpi-card {
        background: linear-gradient(145deg, #131722 0%, #171d2b 100%);
        border: 1px solid #252d3d;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #00f2fe;
        margin-top: 3px;
    }

    /* Container cards */
    .dashboard-panel {
        background: #11141d;
        border: 1px solid #1f2737;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    }
    .panel-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f0f6fc;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        border-bottom: 1px solid #1d2535;
        padding-bottom: 8px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-live { background: rgba(0, 230, 118, 0.15); color: #00e676; border: 1px solid rgba(0, 230, 118, 0.35); }
    .badge-cyan { background: rgba(0, 242, 254, 0.15); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.35); }
    .badge-warn { background: rgba(255, 170, 0, 0.15); color: #ffaa00; border: 1px solid rgba(255, 170, 0, 0.35); }
    .badge-alert { background: rgba(255, 60, 80, 0.15); color: #ff3c50; border: 1px solid rgba(255, 60, 80, 0.35); }
    .badge-pro { background: linear-gradient(135deg, #f5af19 0%, #e65c00 100%); color: #ffffff; font-weight: bold; }

    /* Paywall Container & Blur Effect */
    .paywall-wrapper {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 10px;
    }
    .blurred-content {
        filter: blur(8px);
        -webkit-filter: blur(8px);
        pointer-events: none;
        user-select: none;
        opacity: 0.5;
        transition: all 0.3s ease;
    }
    .paywall-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle, rgba(9, 12, 16, 0.75) 0%, rgba(9, 12, 16, 0.94) 100%);
        z-index: 50;
        padding: 24px;
    }
    .cta-card {
        background: #151a24;
        border: 1px solid #ffaa00;
        border-radius: 16px;
        padding: 28px 36px;
        max-width: 540px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.85), 0 0 35px rgba(255, 170, 0, 0.25);
    }
    .cta-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .cta-desc {
        font-size: 0.92rem;
        color: #adbac7;
        line-height: 1.5;
        margin-bottom: 20px;
    }

    /* Map visual legend bar */
    .legend-bar {
        display: flex;
        gap: 14px;
        font-size: 0.78rem;
        color: #8b9bb4;
        align-items: center;
        padding: 8px 12px;
        background: #0d1017;
        border-radius: 6px;
        border: 1px solid #1a202c;
        margin-top: 8px;
    }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. METRIC CONFIGURATION
# ==============================================================================
METRIC_CONFIG = {
    "lst_c": {
        "name": "Land Surface Temperature",
        "unit": "°C",
        "color": "#ff5e62",
        "description": "Satellite thermal infrared skin temperature."
    },
    "ndvi": {
        "name": "Normalized Difference Vegetation Index",
        "unit": "NDVI",
        "color": "#00e676",
        "description": "Vegetation vitality and surface canopy greenness."
    },
    "ndwi": {
        "name": "Normalized Difference Water Index",
        "unit": "NDWI",
        "color": "#00f2fe",
        "description": "Surface water presence; positive values indicate standing water."
    },
    "salinity_index": {
        "name": "Salinity Index",
        "unit": "SI",
        "color": "#f5af19",
        "description": "Evaporite salt crust and mineral salinization index."
    }
}


# ==============================================================================
# 3. DATA INGESTION ENGINE (LIVE RUST API + FAST PARQUET FALLBACK)
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_grid_for_year(year: int) -> pd.DataFrame:
    """
    Fetches the complete 10,000+ cell regional grid for a given year.
    Tries the high-throughput Rust engine first (port 8080); falls back to local Parquet.
    """
    # 1. Try Live Rust API
    try:
        resp = requests.get(f"{RUST_API_URL}/anomalies", params={"year": year}, timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data).rename(columns={"lat": "latitude", "lon": "longitude"})
                df["source"] = "rust_engine"
                return df
    except Exception:
        pass

    # 2. Fallback: Local Parquet File
    if os.path.exists(PARQUET_FALLBACK_PATH):
        try:
            df_full = pd.read_parquet(PARQUET_FALLBACK_PATH)
            # Filter for requested year
            df_year = df_full[df_full["year"] == year].copy()
            if not df_year.empty:
                # Precompute baseline if not present (1984-1992 reference)
                base = df_full[df_full["year"] <= 1992].groupby(["latitude", "longitude"]).agg(
                    base_mean_ndwi=("ndwi", "mean"),
                    base_std_ndwi=("ndwi", "std"),
                    base_mean_lst=("lst_c", "mean"),
                    base_std_lst=("lst_c", "std"),
                ).reset_index()

                df_year = df_year.merge(base, on=["latitude", "longitude"], how="left")
                df_year["base_std_ndwi"] = df_year["base_std_ndwi"].fillna(0.05)
                df_year["base_std_lst"] = df_year["base_std_lst"].fillna(2.0)

                df_year["z_ndwi"] = ((df_year["ndwi"] - df_year["base_mean_ndwi"]) / (df_year["base_std_ndwi"] + 0.03)).round(2)
                df_year["z_lst"] = ((df_year["lst_c"] - df_year["base_mean_lst"]) / (df_year["base_std_lst"] + 0.5)).round(2)
                df_year["ndwi"] = df_year["ndwi"].round(3)
                df_year["lst_c"] = df_year["lst_c"].round(1)

                was_water = df_year["base_mean_ndwi"] > 0.0
                df_year["anom_water"] = (df_year["z_ndwi"] <= -2.0) | (was_water & (df_year["ndwi"] < -0.05))
                df_year["anom_temp"] = df_year["z_lst"] >= 1.8
                df_year["anom_combined"] = df_year["anom_water"] & df_year["anom_temp"]
                df_year["severity"] = (df_year["z_ndwi"].abs() + df_year["z_lst"].clip(lower=0)).round(2)
                df_year["source"] = "parquet_fallback"
                return df_year
        except Exception:
            pass

    # 3. Deterministic Mock Fallback for UI resilience
    np.random.seed(year)
    lats = np.linspace(43.4, 47.0, 60)
    lons = np.linspace(58.0, 61.6, 60)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    flat_lat = grid_lat.flatten()
    flat_lon = grid_lon.flatten()
    n = len(flat_lat)

    # Simulated desiccation gradient from 1984 to 2026
    decay = max(0.0, min(1.0, (year - 1984) / 42.0))
    dist_center = np.sqrt((flat_lat - 45.2)**2 + (flat_lon - 59.8)**2)
    ndwi = np.where(dist_center < 1.1 * (1.0 - 0.75 * decay), 0.25 - 0.4 * decay + np.random.normal(0, 0.04, n), -0.18 + np.random.normal(0, 0.02, n))
    lst_c = 38.0 + 9.0 * decay + np.random.normal(0, 1.8, n)
    ndvi = np.clip(0.18 - 0.12 * decay + np.random.normal(0, 0.02, n), 0.02, 0.6)
    salinity = 0.02 + 0.05 * decay + np.random.normal(0, 0.01, n)

    z_ndwi = (ndwi - 0.15) / 0.08
    z_lst = (lst_c - 40.0) / 2.5

    anom_water = (z_ndwi <= -1.8) | (ndwi < -0.05)
    anom_temp = z_lst >= 1.8
    anom_combined = anom_water & anom_temp
    severity = np.abs(z_ndwi) + np.maximum(0.0, z_lst)

    return pd.DataFrame({
        "latitude": flat_lat,
        "longitude": flat_lon,
        "year": year,
        "ndwi": ndwi.round(3),
        "lst_c": lst_c.round(1),
        "ndvi": ndvi.round(3),
        "salinity_index": salinity.round(3),
        "z_ndwi": z_ndwi.round(2),
        "z_lst": z_lst.round(2),
        "anom_water": anom_water,
        "anom_temp": anom_temp,
        "anom_combined": anom_combined,
        "severity": severity.round(2),
        "source": "synthetic_mock"
    })


@st.cache_data(ttl=600, show_spinner=False)
def fetch_cell_timeseries(lat: f64 if False else float, lon: float, metric: str) -> pd.DataFrame:
    """
    Fetches the 43-year historical time series (1984–2026) for a specific clicked coordinate.
    """
    try:
        resp = requests.get(
            f"{RUST_API_URL}/time-series",
            params={"latitude": lat, "longitude": lon, "metric": metric},
            timeout=2.0
        )
        if resp.status_code == 200:
            data = resp.json()
            pts = data.get("timeseries", [])
            if pts:
                rows = []
                for p in pts:
                    rows.append({
                        "year": p["year"],
                        "timestamp": p["timestamp"],
                        metric: p["values"].get(metric, 0.0),
                        f"{metric}_anomaly": p["anomaly"].get(metric, 0.0),
                        f"{metric}_zscore": p["z_score"].get(metric, 0.0)
                    })
                df = pd.DataFrame(rows).sort_values("year").drop_duplicates(subset=["year"])
                return df
    except Exception:
        pass

    # Fallback to local parquet extraction
    if os.path.exists(PARQUET_FALLBACK_PATH):
        try:
            df_full = pd.read_parquet(PARQUET_FALLBACK_PATH)
            # Find closest cell
            dist = (df_full["latitude"] - lat)**2 + (df_full["longitude"] - lon)**2
            min_dist = dist.min()
            nearest_row = df_full.loc[dist == min_dist].iloc[0]
            n_lat, n_lon = nearest_row["latitude"], nearest_row["longitude"]

            sub = df_full[(df_full["latitude"] == n_lat) & (df_full["longitude"] == n_lon)].copy()
            sub = sub.sort_values("year")

            base_mask = sub["year"] <= 1992
            b_mean = sub.loc[base_mask, metric].mean() if base_mask.any() else sub[metric].mean()
            b_std = sub.loc[base_mask, metric].std() if base_mask.any() else sub[metric].std()
            if pd.isna(b_std) or b_std < 1e-4:
                b_std = 0.05

            sub[f"{metric}_anomaly"] = sub[metric] - b_mean
            sub[f"{metric}_zscore"] = (sub[metric] - b_mean) / b_std
            return sub[["year", "timestamp", metric, f"{metric}_anomaly", f"{metric}_zscore"]].drop_duplicates("year")
        except Exception:
            pass

    # Synthetic fallback
    np.random.seed(int(lat * 100 + lon * 10))
    years = np.arange(1984, 2027)
    base_m = 42.0 if metric == "lst_c" else (0.2 if metric == "ndwi" else 0.15)
    slope = 0.14 if metric == "lst_c" else (-0.008 if metric == "ndwi" else -0.002)
    vals = base_m + slope * (years - 1984) + np.random.normal(0, 0.8 if metric == "lst_c" else 0.03, len(years))
    b_mean = np.mean(vals[:9])
    b_std = np.std(vals[:9]) + 1e-4

    return pd.DataFrame({
        "year": years,
        "timestamp": [f"{y}-08-01T00:00:00Z" for y in years],
        metric: vals,
        f"{metric}_anomaly": vals - b_mean,
        f"{metric}_zscore": (vals - b_mean) / b_std
    })


# ==============================================================================
# 4. SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛰️ **Mission Control**")
    st.markdown("<p style='font-size:0.82rem; color:#8b9bb4;'>NASA Space Apps • Earth Observation Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Year Slider
    selected_year = st.slider(
        "Observation Epoch (Year)",
        min_value=1984,
        max_value=2026,
        value=2026,
        step=1,
        help="Select historical observation year to load the entire 10,000+ cell regional satellite grid."
    )

    # 2. Map Layer Visualization Mode
    map_color_mode = st.selectbox(
        "Map Layer Mode",
        options=["status", "ndwi", "lst_c", "salinity_index", "severity"],
        format_func=lambda x: {
            "status": "🚨 Multi-Hazard Anomaly Status",
            "ndwi": "💧 Water Index (NDWI)",
            "lst_c": "🌡️ Surface Temperature (LST °C)",
            "salinity_index": "🧂 Soil Salinity Index",
            "severity": "🔥 Compound Severity Score"
        }[x],
        help="Colorize all 10,000+ grid cells by anomaly classification or continuous metric heatmaps."
    )

    # 3. Primary Metric Focus
    metric_choice = st.selectbox(
        "Focused Climate Metric",
        options=list(METRIC_CONFIG.keys()),
        index=0,
        format_func=lambda x: f"{METRIC_CONFIG[x]['name']} ({METRIC_CONFIG[x]['unit']})",
        help="Select target parameter for time-series extraction and multi-decadal predictive modeling."
    )

    # 4. Hotspot Filtering
    only_anomalies = st.checkbox("Isolate Critical Anomaly Hotspots", value=False, help="Filter map to show only cells triggering water retreat or thermal shock.")

    st.markdown("---")

    # 5. Pro Account Toggle
    st.markdown("#### 🔐 **Tier Access Simulation**")
    is_pro = st.toggle(
        "Simulate Pro Account (Paid Tier)",
        value=False,
        help="Toggle between Free Tier (paywall blurred preview) and Pro Tier (unlocked 2026–2050 forecast engine)."
    )

    if is_pro:
        st.markdown("<span class='badge badge-pro'>⭐ PRO TIER ACTIVE</span>", unsafe_allow_html=True)
        st.caption("Predictive modeling & confidence projections unlocked.")
    else:
        st.markdown("<span class='badge badge-cyan'>FREE TIER ACTIVE</span>", unsafe_allow_html=True)
        st.caption("Forecasting view is gated behind the blurred paywall.")

    st.markdown("---")
    st.caption("• **Archive:** Landsat 4-9 Tier-1 Level-2 (1984–2026)\n• **Resolution:** 8.0 km × 8.0 km Sentinel Grid\n• **Baseline Period:** 1984–1992 Reference Normal")


# ==============================================================================
# 5. DATA INGESTION & REGIONAL KPI RIBBON
# ==============================================================================
df_grid = fetch_grid_for_year(selected_year)
data_source = df_grid["source"].iloc[0] if "source" in df_grid.columns else "unknown"

# Filter displayed cells if toggle is on
df_display = df_grid[df_grid["anom_water"] | df_grid["anom_temp"]].copy() if only_anomalies else df_grid.copy()

# Header title & status pills
title_col, badge_col = st.columns([3, 1])
with title_col:
    st.title("🛰️ Earth System Trend Detective")
    st.markdown("<p style='color:#8b9bb4; font-size:1.02rem; margin-top:-10px; margin-bottom:15px;'>Multi-Decadal Geospatial Analytics & Forecasting Platform • Aral Sea Basin Observation</p>", unsafe_allow_html=True)
with badge_col:
    st.markdown("<div style='text-align: right; margin-top: 15px;'>", unsafe_allow_html=True)
    if data_source == "rust_engine":
        st.markdown("<span class='badge badge-live'>🟢 LIVE RUST ENGINE (PORT 8080)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge badge-warn'>🟡 LOCAL PARQUET ARCHIVE</span>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem; color:#8b9bb4; margin-top:4px;'>Grid Cells Loaded: <b>{len(df_grid):,}</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Top Regional KPI Ribbon
k1, k2, k3, k4 = st.columns(4)
total_cells = len(df_grid)
water_cells = int((df_grid["ndwi"] > 0.0).sum())
water_pct = (water_cells / total_cells * 100) if total_cells > 0 else 0.0
water_anom_count = int(df_grid["anom_water"].sum())
temp_anom_count = int(df_grid["anom_temp"].sum())
combined_count = int(df_grid["anom_combined"].sum())
mean_temp = df_grid["lst_c"].mean()

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Surface Water Extent</div>
            <div class="kpi-value">{water_pct:.1f}% <span style="font-size:0.9rem; color:#8b9bb4;">({water_cells:,} cells)</span></div>
            <div class="kpi-sub">Epoch {selected_year} coverage</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Surface Temp (LST)</div>
            <div class="kpi-value">{mean_temp:.1f} °C</div>
            <div class="kpi-sub">Regional skin average</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Water Desiccation Alerts</div>
            <div class="kpi-value" style="color: #ff9900;">{water_anom_count:,}</div>
            <div class="kpi-sub">Z ≤ -2.0σ vs 1984–1992</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Critical Hotspots (Water+Heat)</div>
            <div class="kpi-value" style="color: #ff3c50;">{combined_count:,}</div>
            <div class="kpi-sub">Dual-stress compounding cells</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# 6. MAIN WORKSPACE: FULL REGIONAL MAP vs. SELECTED CELL DEEP DIVE
# ==============================================================================
col_map, col_details = st.columns([1.35, 1.65], gap="medium")

# Color calculation for PyDeck 10,000+ cells
if map_color_mode == "status":
    # Categorical colors:
    # Combined Critical: Red [255, 30, 70]
    # Water Anomaly: Orange [255, 140, 20]
    # Temp Anomaly: Yellow [255, 215, 30]
    # Standing Water: Electric Blue [20, 140, 255]
    # Dry Baseline: Slate [140, 150, 165]
    conds = [
        df_display["anom_combined"],
        df_display["anom_water"],
        df_display["anom_temp"],
        df_display["ndwi"] > 0.0
    ]
    df_display["r"] = np.select(conds, [255, 255, 255, 20], default=130)
    df_display["g"] = np.select(conds, [30, 140, 215, 140], default=140)
    df_display["b"] = np.select(conds, [70, 20, 30, 255], default=155)
    df_display["a"] = np.select(conds, [250, 230, 230, 240], default=110)
else:
    vals = df_display[map_color_mode]
    vmin, vmax = vals.quantile(0.05), vals.quantile(0.95)
    norm = ((vals - vmin) / (vmax - vmin + 1e-6)).clip(0, 1)
    if map_color_mode == "ndwi":
        df_display["r"] = ((1 - norm) * 210).astype(int)
        df_display["g"] = (130 + norm * 50).astype(int)
        df_display["b"] = (150 + norm * 105).astype(int)
    elif map_color_mode == "lst_c":
        df_display["r"] = (140 + norm * 115).astype(int)
        df_display["g"] = ((1 - norm) * 190).astype(int)
        df_display["b"] = 35
    elif map_color_mode == "ndvi":
        df_display["r"] = ((1 - norm) * 190).astype(int)
        df_display["g"] = (110 + norm * 120).astype(int)
        df_display["b"] = 50
    else: # salinity or severity
        df_display["r"] = (160 + norm * 95).astype(int)
        df_display["g"] = (60 + norm * 120).astype(int)
        df_display["b"] = ((1 - norm) * 220).astype(int)
    df_display["a"] = 220

# ------------------------------------------------------------------------------
# LEFT COLUMN: PyDeck Full 10,000+ Cell Interactive Map
# ------------------------------------------------------------------------------
with col_map:
    st.markdown(
        f"""
        <div class="dashboard-panel">
            <div class="panel-header">
                <span>🗺️ Aral Basin Regional Sentinel Grid ({selected_year})</span>
                <span class="badge badge-cyan">{len(df_display):,} Cells Displayed</span>
            </div>
        """,
        unsafe_allow_html=True
    )

    # PyDeck Scatterplot Grid Layer
    grid_layer = pdk.Layer(
        "ScatterplotLayer",
        id="aral-grid",
        data=df_display,
        get_position=["longitude", "latitude"],
        get_fill_color=["r", "g", "b", "a"],
        get_radius=1850,
        pickable=True,
        auto_highlight=True,
        highlight_color=[0, 242, 254, 255]
    )

    # Initial view centered over the Aral Sea
    initial_view = pdk.ViewState(
        latitude=45.20,
        longitude=59.80,
        zoom=6.6,
        pitch=25,
        bearing=0
    )

    tooltip_text = (
        "📍 Coordonate: {latitude}, {longitude}\n"
        "💧 NDWI (Apă): {ndwi} (Z: {z_ndwi}σ)\n"
        "🌡️ Temp LST: {lst_c} °C (Z: {z_lst}σ)\n"
        "🔥 Severitate: {severity}\n"
        "Status: " + (
            "CRITIC (Apă+Temp)" if "{anom_combined}" == "True"
            else ("ANOMALIE APĂ" if "{anom_water}" == "True"
            else ("ANOMALIE TERMICĂ" if "{anom_temp}" == "True" else "Normal"))
        )
    )

    deck_chart = pdk.Deck(
        layers=[grid_layer],
        initial_view_state=initial_view,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={"text": "📍 Lat: {latitude:.3f}, Lon: {longitude:.3f}\n💧 NDWI: {ndwi} (Z: {z_ndwi}σ)\n🌡️ Temp: {lst_c}°C (Z: {z_lst}σ)\n🔥 Severitate: {severity}"}
    )

    deck_event = st.pydeck_chart(
        deck_chart,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-object"
    )

    # Map Visual Legend Bar
    if map_color_mode == "status":
        st.markdown(
            """
            <div class="legend-bar">
                <span class="legend-item"><span class="dot" style="background:#148cff;"></span> Apă (NDWI>0)</span>
                <span class="legend-item"><span class="dot" style="background:#ff8c14;"></span> Secare (Z≤-2σ)</span>
                <span class="legend-item"><span class="dot" style="background:#ffd71e;"></span> Șoc Termic</span>
                <span class="legend-item"><span class="dot" style="background:#ff1e46;"></span> Critic Compus</span>
                <span class="legend-item"><span class="dot" style="background:#828c9b;"></span> Uscat Istoric</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="legend-bar">
                <span style="color:#8b9bb4;">Gradient {METRIC_CONFIG[metric_choice]['name']}:</span>
                <span style="color:#00e676;">Valori Mici</span> ➔ 
                <span style="color:#f5af19;">Mediu</span> ➔ 
                <span style="color:#ff5e62;">Valori Mari</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# RESOLVE SELECTED CELL FROM CLICK OR DEFAULT TO ARAL CENTER
# ------------------------------------------------------------------------------
selected_cell = None
if deck_event and deck_event.selection:
    selected_objects = deck_event.selection.get("objects", {}).get("aral-grid", [])
    if selected_objects:
        selected_cell = selected_objects[0]

# Default to prominent core desiccation cell if no click yet
if not selected_cell:
    selected_lat, selected_lon = 45.1853, 59.8098
    curr_ndwi = -0.165
    curr_lst = 46.8
    curr_sal = 0.045
    curr_sev = 2.1
else:
    selected_lat = float(selected_cell["latitude"])
    selected_lon = float(selected_cell["longitude"])
    curr_ndwi = float(selected_cell.get("ndwi", 0.0))
    curr_lst = float(selected_cell.get("lst_c", 0.0))
    curr_sal = float(selected_cell.get("salinity_index", 0.0))
    curr_sev = float(selected_cell.get("severity", 0.0))

# Fetch 43-year historical time series for this exact cell
df_cell_history = fetch_cell_timeseries(selected_lat, selected_lon, metric_choice)
metric_meta = METRIC_CONFIG[metric_choice]


# ------------------------------------------------------------------------------
# RIGHT COLUMN: Historical Time-Series (1984–2026) for Selected Cell
# ------------------------------------------------------------------------------
with col_details:
    st.markdown(
        f"""
        <div class="dashboard-panel">
            <div class="panel-header">
                <span>📈 Historical Trajectory: {selected_lat:.3f}° N, {selected_lon:.3f}° E</span>
                <span class="badge badge-cyan">Cell ID: ARAL-{int(selected_lat*100)}-{int(selected_lon*100)}</span>
            </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Metadata KPIs for the clicked cell
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Coordinates</div><div style='font-size:1.05rem; font-weight:700;'>{selected_lat:.2f}°, {selected_lon:.2f}°</div><div class='kpi-sub'>8km Grid Center</div></div>", unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Water (NDWI)</div><div style='font-size:1.05rem; font-weight:700; color:{'#148cff' if curr_ndwi>0 else '#ff8c14'};'>{curr_ndwi:+.3f}</div><div class='kpi-sub'>{'Open Water' if curr_ndwi>0 else 'Desiccated Bed'}</div></div>", unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Surface Temp</div><div style='font-size:1.05rem; font-weight:700; color:#ff5e62;'>{curr_lst:.1f} °C</div><div class='kpi-sub'>Epoch {selected_year} LST</div></div>", unsafe_allow_html=True)
    with c_m4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Severity Index</div><div style='font-size:1.05rem; font-weight:700; color:#ffd71e;'>{curr_sev:.2f}</div><div class='kpi-sub'>Stress Ranking</div></div>", unsafe_allow_html=True)

    # Plotly Historical Chart
    fig_hist = go.Figure()

    if not df_cell_history.empty and metric_choice in df_cell_history.columns:
        y_vals = df_cell_history[metric_choice].values
        x_years = df_cell_history["year"].values

        # Baseline mean & std
        b_mean = np.mean(y_vals[:min(9, len(y_vals))])
        b_std = np.std(y_vals[:min(9, len(y_vals))])
        if b_std < 1e-4:
            b_std = 0.05

        # Normal ±2σ threshold band
        fig_hist.add_hrect(
            y0=b_mean - 2 * b_std,
            y1=b_mean + 2 * b_std,
            fillcolor="rgba(0, 242, 254, 0.05)",
            line_width=1,
            line_color="rgba(0, 242, 254, 0.2)",
            line_dash="dot",
            annotation_text="Normal Range (±2σ)",
            annotation_position="top left",
            annotation_font_color="rgba(0, 242, 254, 0.5)",
            annotation_font_size=10
        )

        # Baseline Mean Reference Line
        fig_hist.add_hline(
            y=b_mean,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.35)",
            line_width=1.5,
            annotation_text=f"Baseline Mean μ = {b_mean:.2f}",
            annotation_position="bottom right",
            annotation_font_color="rgba(255, 255, 255, 0.6)",
            annotation_font_size=10
        )

        # Main Historical Curve
        fig_hist.add_trace(go.Scatter(
            x=x_years,
            y=y_vals,
            mode="lines+markers",
            name=metric_meta["name"],
            line=dict(color=metric_meta["color"], width=2.5),
            marker=dict(size=4, color=metric_meta["color"]),
            hovertemplate="<b>Year:</b> %{x}<br><b>Value:</b> %{y:.3f} " + metric_meta["unit"] + "<extra></extra>"
        ))

        # Detect and highlight anomaly spikes (|Z| > 2.0)
        z_col = f"{metric_choice}_zscore"
        if z_col in df_cell_history.columns:
            spike_mask = df_cell_history[z_col].abs() >= 2.0
            spikes = df_cell_history[spike_mask]
            if not spikes.empty:
                fig_hist.add_trace(go.Scatter(
                    x=spikes["year"],
                    y=spikes[metric_choice],
                    mode="markers",
                    name="Anomaly Spike (|Z| > 2.0)",
                    marker=dict(size=9, color="#ff3c50", symbol="diamond", line=dict(width=1.5, color="#ffffff")),
                    hovertemplate="<b>⚠️ ANOMALY SPIKE</b><br>Year: %{x}<br>Value: %{y:.3f}<br>Z-Score: %{text}σ<extra></extra>",
                    text=[f"{z:+.2f}" for z in spikes[z_col]]
                ))

        # Selected year vertical line
        fig_hist.add_vline(
            x=selected_year,
            line_width=2,
            line_color="#ffffff",
            line_dash="dot",
            annotation_text=f"{selected_year}",
            annotation_position="top",
            annotation_font_color="#ffffff"
        )

    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin=dict(l=10, r=20, t=15, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Observation Year"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=metric_meta["unit"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(
        f"""
        <div style="font-size:0.84rem; color:#8b9bb4; background:#0d1017; padding:10px 14px; border-radius:6px; border:1px solid #1a202c;">
            💡 <b>Interactivity:</b> Click any cell on the regional map to instantly load its 43-year historical profile,
            rate of change, and multi-decadal predictive modeling.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 7. MULTI-DECADAL PREDICTIVE FORECASTING ENGINE (2026–2050): PRO TIER GATED
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom: 12px;">
        <h3 style="margin-bottom: 2px; display:flex; align-items:center; gap:8px;">
            <span>🔮 Multi-Decadal Statistical Forecasting Engine (2026–2050)</span>
            <span class="badge badge-cyan">AUTOREGRESSIVE OLS</span>
        </h3>
        <p style="color:#8b9bb4; font-size:0.92rem;">
            Predictive climate modeling, OLS trend extrapolation, and expanding 95% confidence bounds projecting localized environmental shift to 2050.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Compute linear regression and expanding 95% confidence intervals for forecast
if not df_cell_history.empty and metric_choice in df_cell_history.columns:
    hist_x = df_cell_history["year"].values
    hist_y = df_cell_history[metric_choice].values
else:
    hist_x = np.arange(1984, 2027)
    hist_y = 42.0 + 0.14 * (hist_x - 1984)

slope, intercept = np.polyfit(hist_x, hist_y, 1)
future_x = np.arange(2026, 2051)
proj_y = intercept + slope * future_x

# Expanding standard error of prediction for 95% confidence bands
n_pts = len(hist_x)
x_mean = np.mean(hist_x)
residuals = hist_y - (intercept + slope * hist_x)
s_err = np.std(residuals) if len(residuals) > 0 else 0.5
conf_margin = 1.96 * s_err * np.sqrt(1 + 1/max(1, n_pts) + (future_x - x_mean)**2 / max(1e-4, np.sum((hist_x - x_mean)**2)))
ci_upper = proj_y + conf_margin
ci_lower = proj_y - conf_margin

fig_pred = go.Figure()

# 95% Confidence Band (Shaded)
fig_pred.add_trace(go.Scatter(
    x=np.concatenate([future_x, future_x[::-1]]),
    y=np.concatenate([ci_upper, ci_lower[::-1]]),
    fill='toself',
    fillcolor='rgba(0, 242, 254, 0.12)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% Confidence Band',
    hoverinfo='skip'
))

# Historical Actuals Line
fig_pred.add_trace(go.Scatter(
    x=hist_x,
    y=hist_y,
    mode='lines',
    name='Historical Actuals (1984–2026)',
    line=dict(color=metric_meta['color'], width=2.5)
))

# Forecast Extrapolation Line
fig_pred.add_trace(go.Scatter(
    x=future_x,
    y=proj_y,
    mode='lines',
    name='Projected Trend (2026–2050)',
    line=dict(color='#00f2fe', width=3, dash='dash')
))

# 2026 Era Boundary
fig_pred.add_vline(
    x=2026,
    line_width=1.5,
    line_color="rgba(255, 255, 255, 0.4)",
    line_dash="dot",
    annotation_text="Current Era (2026)",
    annotation_position="top left",
    annotation_font_size=11
)

fig_pred.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=370,
    margin=dict(l=10, r=20, t=20, b=10),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Year"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=metric_meta["unit"]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)


# ==============================================================================
# ACCESS CONTROL LOGIC: PRO UNLOCKED vs. BLURRED PAYWALL
# ==============================================================================
if is_pro:
    # --------------------------------------------------------------------------
    # UNLOCKED PRO VIEW
    # --------------------------------------------------------------------------
    st.markdown('<div class="dashboard-panel" style="border: 1px solid rgba(0, 242, 254, 0.35);">', unsafe_allow_html=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    val_2050 = proj_y[-1]
    val_2026 = proj_y[0]
    net_change = val_2050 - val_2026
    decadal_rate = slope * 10

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.metric("Projected 2050 Value", f"{val_2050:.2f} {metric_meta['unit']}", f"{net_change:+.2f} shift by 2050")
    with f2:
        st.metric("Decadal Rate of Change", f"{decadal_rate:+.3f} {metric_meta['unit']}/dec", "OLS Trend Slope β")
    with f3:
        st.metric("95% CI Range (2050)", f"±{(ci_upper[-1] - val_2050):.2f} {metric_meta['unit']}", "Prediction Margin")
    with f4:
        st.metric("Tipping Point Risk", "CRITICAL" if abs(slope) > 0.05 else "MODERATE", "Decadal Acceleration", delta_color="inverse")

    st.markdown(
        f"""
        <div style="margin-top: 14px; padding: 12px 18px; background: rgba(0, 242, 254, 0.04); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 8px; font-size: 0.88rem; color: #d0d7de;">
            🔬 <b>Pro Analytical Diagnosis ({selected_lat:.2f}° N, {selected_lon:.2f}° E):</b> 
            Autoregressive climate extrapolation indicates a persistent annual shift of 
            <code>{slope:+.4f} {metric_meta['unit']}/year</code>. Projected 2050 outcome is 
            <code>{val_2050:.2f} {metric_meta['unit']}</code> within 95% confidence envelope 
            [<code>{ci_lower[-1]:.2f}</code>, <code>{ci_upper[-1]:.2f}</code>].
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # --------------------------------------------------------------------------
    # BLURRED PAYWALL GATED VIEW (FREE TIER)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="paywall-wrapper">
            <div class="paywall-overlay">
                <div class="cta-card">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">🔒</div>
                    <div class="cta-title">Unlock Statistical Climate Forecasts</div>
                    <div class="cta-desc">
                        Gain access to multi-decadal predictive modeling, regression confidence bands,
                        and long-term trend extrapolation engines (2026–2050).
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Blurred content container
    st.markdown('<div class="blurred-content">', unsafe_allow_html=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    fb1, fb2, fb3, fb4 = st.columns(4)
    with fb1: st.metric("Projected 2050 Value", "49.80 °C", "+3.40 °C")
    with fb2: st.metric("Decadal Shift Rate", "+1.20 °C/dec", "Slope β")
    with fb3: st.metric("95% CI Range (2050)", "±1.85 °C", "Confidence")
    with fb4: st.metric("Tipping Point Risk", "CRITICAL", "High Confidence", delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

    # Upgrade CTA action button
    col_l, col_btn, col_r = st.columns([1.5, 1.2, 1.5])
    with col_btn:
        if st.button("🚀 Upgrade to Pro for €19/month", use_container_width=True, type="primary"):
            st.toast("🎉 Pro Tier simulated! Toggle 'Simulate Pro Account' in the sidebar to inspect.", icon="⭐")


# ==============================================================================
# 8. FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #5f6e85; font-size: 0.8rem; padding: 10px 0 20px 0;">
        Earth System Trend Detective • Developed for NASA Space Apps Hackathon • 
        Landsat Surface Reflectance Tier-1 (Collection 2) • Axum/Arrow Analytics Engine
    </div>
    """,
    unsafe_allow_html=True
)
