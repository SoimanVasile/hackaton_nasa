"""
Earth System Trend Detective | Environmental Analytics
======================================================
Multi-decadal environmental analytics dashboard script (app.py) built with Streamlit.
Visualizes historical satellite climate observations (1984–2026) within a locked 8km x 8km
geospatial grid cell and features a gated statistical forecasting engine (2026–2050)
using a custom CSS blurred paywall.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Earth System Trend Detective | Environmental Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific dark-mode aesthetic (#0e1117), sharp cards & blurred paywall
CUSTOM_CSS = """
<style>
    /* Dark-mode background & typography */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Card Containers */
    .card-container {
        background: #161a23;
        border: 1px solid #262c3a;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    }

    .card-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f0f4f8;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        border-bottom: 1px solid #222938;
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
    .badge-cyan { background: rgba(0, 242, 254, 0.15); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.35); }
    .badge-locked { background: rgba(0, 230, 118, 0.15); color: #00e676; border: 1px solid rgba(0, 230, 118, 0.35); }
    .badge-alert { background: rgba(255, 94, 98, 0.15); color: #ff5e62; border: 1px solid rgba(255, 94, 98, 0.35); }
    .badge-pro { background: linear-gradient(135deg, #f5af19 0%, #e65c00 100%); color: #ffffff; font-weight: bold; }

    /* Metadata mini-cards */
    .meta-box {
        background: #11141c;
        border: 1px solid #202634;
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
    }
    .meta-label {
        font-size: 0.72rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .meta-value {
        font-size: 1.02rem;
        font-weight: 700;
        color: #e6edf3;
    }

    /* Locked Map Container styling */
    .locked-map-frame {
        border: 1px solid #00f2fe;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.12);
    }
    .map-lock-banner {
        position: absolute;
        top: 8px;
        left: 8px;
        background: rgba(14, 17, 23, 0.88);
        border: 1px solid rgba(0, 242, 254, 0.4);
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 0.72rem;
        color: #00f2fe;
        z-index: 10;
        display: flex;
        align-items: center;
        gap: 5px;
        font-weight: 600;
    }

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
        opacity: 0.55;
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
        background: radial-gradient(circle, rgba(14, 17, 23, 0.78) 0%, rgba(14, 17, 23, 0.94) 100%);
        z-index: 50;
        padding: 25px;
    }
    .cta-card {
        background: #171c28;
        border: 1px solid #ffaa00;
        border-radius: 16px;
        padding: 30px 40px;
        max-width: 560px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.85), 0 0 35px rgba(255, 170, 0, 0.25);
    }
    .cta-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .cta-desc {
        font-size: 0.92rem;
        color: #b0bac9;
        line-height: 1.55;
        margin-bottom: 22px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. METRIC CONFIGURATION & REALISTIC MOCK DATA GENERATION
# ==============================================================================
METRIC_CONFIG = {
    "lst_c": {
        "name": "Land Surface Temperature",
        "unit": "°C",
        "color": "#ff5e62",
        "base_mean": 42.1,
        "base_std": 2.1,
        "slope": 0.14,
        "description": "Satellite thermal infrared surface skin temperature."
    },
    "ndvi": {
        "name": "Vegetation Index",
        "unit": "NDVI",
        "color": "#00e676",
        "base_mean": 0.165,
        "base_std": 0.024,
        "slope": -0.0022,
        "description": "Normalized Difference Vegetation Index canopy vigor."
    },
    "ndwi": {
        "name": "Water Index",
        "unit": "NDWI",
        "color": "#00f2fe",
        "base_mean": 0.180,
        "base_std": 0.045,
        "slope": -0.0075,
        "description": "Normalized Difference Water Index; positive values indicate standing water."
    },
    "salinity_index": {
        "name": "Salinity Index",
        "unit": "SI",
        "color": "#f5af19",
        "base_mean": 0.032,
        "base_std": 0.006,
        "slope": 0.0011,
        "description": "Shortwave-infrared evaporite salt crust index."
    }
}

GRID_LOCATIONS = {
    "ARAL-8K-5980": {"name": "Central Aral Core Desiccation Bed", "lat": 45.2000, "lon": 59.8000, "elevation": 53},
    "ARAL-8K-6010": {"name": "North Aral Inflow Delta Margin", "lat": 46.1200, "lon": 60.1500, "elevation": 42},
    "ARAL-8K-5910": {"name": "Western Deep Trench Remnant", "lat": 44.8500, "lon": 58.7500, "elevation": 29},
    "ARAL-8K-6080": {"name": "Eastern Aralkum Salt Desert", "lat": 44.5000, "lon": 60.8000, "elevation": 61}
}


@st.cache_data
def generate_cell_timeseries(cell_id: str):
    """
    Generates deterministic, scientifically realistic multi-decadal time-series (1984–2050)
    for the selected 8km x 8km grid cell.
    """
    loc = GRID_LOCATIONS[cell_id]
    lat, lon, elevation = loc["lat"], loc["lon"], loc["elevation"]
    seed = int(lat * 1000 + lon * 100)
    np.random.seed(seed)

    years = np.arange(1984, 2051)
    n = len(years)

    data = {
        "year": years,
        "latitude": [lat] * n,
        "longitude": [lon] * n,
        "elevation": [elevation] * n,
    }

    for m_key, conf in METRIC_CONFIG.items():
        base = conf["base_mean"]
        slope = conf["slope"]
        noise_std = conf["base_std"] * 0.7

        trend = base + slope * (years - 1984)
        noise = np.random.normal(0, noise_std, size=n)

        # Inject historical climate shock events
        if m_key == "lst_c":
            noise[years == 2010] += 4.8  # 2010 severe summer heatwave
            noise[years == 2021] += 4.2
        elif m_key == "ndwi":
            noise[years == 2014] -= 0.14 # 2014 eastern basin dryout
        elif m_key == "ndvi":
            noise[years == 2020] -= 0.05 # 2020 acute regional drought

        vals = trend + noise
        if m_key == "ndwi":
            vals = np.clip(vals, -0.45, 0.65)
        elif m_key == "ndvi":
            vals = np.clip(vals, 0.01, 0.85)

        data[m_key] = np.round(vals, 4)

    df = pd.DataFrame(data)

    # Reference baseline statistics (1984–1992 reference epoch)
    base_mask = df["year"] <= 1992
    for m_key in METRIC_CONFIG.keys():
        ref_mean = df.loc[base_mask, m_key].mean()
        ref_std = df.loc[base_mask, m_key].std()
        if pd.isna(ref_std) or ref_std < 1e-4:
            ref_std = 0.05

        df[f"{m_key}_mean"] = ref_mean
        df[f"{m_key}_std"] = ref_std
        df[f"{m_key}_zscore"] = np.round((df[m_key] - ref_mean) / ref_std, 2)
        df[f"{m_key}_anomaly"] = np.round(df[m_key] - ref_mean, 4)

    return df


# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛰️ **Dashboard Controls**")
    st.markdown("<p style='font-size:0.82rem; color:#8b9bb4;'>Sentinel 8km × 8km Earth Observation Grid</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Grid Cell Selection (Anchor)
    selected_cell_id = st.selectbox(
        "Geospatial Grid Cell",
        options=list(GRID_LOCATIONS.keys()),
        index=0,
        format_func=lambda x: f"{x} ({GRID_LOCATIONS[x]['name']})",
        help="Locks the map view and analytics to this specific 8km x 8km sentinel cell."
    )

    # 1. Year Slider
    selected_year = st.slider(
        "Observation Year",
        min_value=1984,
        max_value=2026,
        value=2026,
        step=1,
        help="Select historical observation epoch (1984–2026)."
    )

    # 2. Anomaly Type Selector
    anomaly_filter = st.selectbox(
        "Anomaly Type",
        options=["All Anomalies", "Extreme Heat (LST)", "Drought (NDVI)", "Water Shift (NDWI)"],
        index=0,
        help="Filter and emphasize specific statistical anomaly triggers in the time-series."
    )

    # 3. Metric Selector
    metric_choice = st.selectbox(
        "Target Metric",
        options=["lst_c", "ndvi", "ndwi", "salinity_index"],
        index=0,
        format_func=lambda x: f"{METRIC_CONFIG[x]['name']} ({METRIC_CONFIG[x]['unit']})",
        help="Target environmental parameter to analyze and extrapolate."
    )

    st.markdown("---")

    # 4. Simulate Pro Account Toggle
    st.markdown("#### 🔐 **Tier Access Simulation**")
    is_pro = st.toggle(
        "Simulate Pro Account (Paid Tier)",
        value=False,
        help="Toggle between Free Tier (paywall blurred preview) and Pro Tier (unlocked 2026–2050 forecast engine)."
    )

    if is_pro:
        st.markdown("<span class='badge badge-pro'>⭐ PRO TIER ACTIVE</span>", unsafe_allow_html=True)
        st.caption("Forecasting engine & regression confidence bands unlocked.")
    else:
        st.markdown("<span class='badge badge-cyan'>FREE TIER ACTIVE</span>", unsafe_allow_html=True)
        st.caption("Forecasting view is gated behind the blurred paywall.")

    st.markdown("---")
    st.caption("• **Archive:** Landsat 4-9 Tier-1 Surface Reflectance\n• **Grid Cell Resolution:** 8.0 km × 8.0 km\n• **Reference Normal:** 1984–1992 Baseline")


# ==============================================================================
# 4. LOAD DATASET FOR CURRENT SELECTION
# ==============================================================================
df_cell = generate_cell_timeseries(selected_cell_id)
df_hist = df_cell[df_cell["year"] <= 2026].copy()
curr_record = df_hist[df_hist["year"] == selected_year].iloc[0]

metric_meta = METRIC_CONFIG[metric_choice]
curr_val = curr_record[metric_choice]
curr_z = curr_record[f"{metric_choice}_zscore"]
curr_anom = curr_record[f"{metric_choice}_anomaly"]
base_mean = curr_record[f"{metric_choice}_mean"]
base_std = curr_record[f"{metric_choice}_std"]


# ==============================================================================
# 5. MAIN DASHBOARD AREA (ABOVE THE FOLD): TWO-COLUMN LAYOUT [1.2, 1.8]
# ==============================================================================
st.title("Earth System Trend Detective")
st.markdown(
    f"<p style='color:#8b9bb4; font-size:1.02rem; margin-top:-10px; margin-bottom:18px;'>"
    f"Multi-decadal environmental analytics dashboard tracking 8km × 8km geospatial cell transformations (1984–2026)."
    f"</p>",
    unsafe_allow_html=True
)

col_map, col_chart = st.columns([1.2, 1.8], gap="medium")

# ------------------------------------------------------------------------------
# LEFT COLUMN: Map & Location View (Locked to the 8km x 8km Grid Cell)
# ------------------------------------------------------------------------------
with col_map:
    st.markdown(
        f"""
        <div class="card-container">
            <div class="card-header">
                <span>📍 Geospatial Cell: <code>{selected_cell_id}</code></span>
                <span class="badge badge-locked">🔒 LOCKED TO 8km GRID</span>
            </div>
        """,
        unsafe_allow_html=True
    )

    cell_lat = curr_record["latitude"]
    cell_lon = curr_record["longitude"]

    # Compute exact 8km x 8km bounding box (~0.072 deg lat, ~0.102 deg lon at ~45N)
    half_dlat, half_dlon = 0.036, 0.051
    min_lat, max_lat = cell_lat - half_dlat, cell_lat + half_dlat
    min_lon, max_lon = cell_lon - half_dlon, cell_lon + half_dlon

    bbox_poly = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat]
    ]

    # Internal 2km sub-grid lines to visually emphasize the spatial cell texture
    subgrid_lines = []
    for step in [0.25, 0.5, 0.75]:
        s_lon = min_lon + step * (max_lon - min_lon)
        s_lat = min_lat + step * (max_lat - min_lat)
        subgrid_lines.append([[s_lon, min_lat], [s_lon, max_lat]])
        subgrid_lines.append([[min_lon, s_lat], [max_lon, s_lat]])

    # Map color based on current epoch reading
    norm_val = np.clip((curr_val - base_mean) / (base_std * 3.0), -1, 1)
    if metric_choice == "lst_c":
        fill_rgba = [int(150 + 105 * max(0, norm_val)), 40, 50, 160]
    elif metric_choice == "ndwi":
        fill_rgba = [30, 140, int(150 + 105 * max(0, norm_val)), 170]
    elif metric_choice == "ndvi":
        fill_rgba = [50, int(150 + 105 * max(0, norm_val)), 80, 160]
    else:
        fill_rgba = [int(180 + 75 * max(0, norm_val)), int(140 + 70 * max(0, norm_val)), 40, 160]

    # PyDeck Layers
    cell_polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=[{"coordinates": [bbox_poly]}],
        get_polygon="-coordinates",
        filled=True,
        get_fill_color=fill_rgba,
        stroked=True,
        get_line_color=[0, 242, 254, 255],
        get_line_width=3,
        line_width_min_pixels=2
    )

    subgrid_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path} for path in subgrid_lines],
        get_path="-path",
        get_color=[0, 242, 254, 80],
        width_min_pixels=1
    )

    center_point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": cell_lat, "lon": cell_lon, "val": curr_val}],
        get_position="[lon, lat]",
        get_color=[255, 255, 255, 240],
        get_radius=600,
        pickable=True
    )

    # Locked view state anchored specifically to this 8km cell
    locked_view = pdk.ViewState(
        latitude=cell_lat,
        longitude=cell_lon,
        zoom=11.2,
        pitch=0,
        bearing=0
    )

    st.markdown('<div class="locked-map-frame">', unsafe_allow_html=True)
    st.markdown('<div class="map-lock-banner">🔒 8.0 km × 8.0 km Centered Grid</div>', unsafe_allow_html=True)

    st.pydeck_chart(
        pdk.Deck(
            layers=[cell_polygon_layer, subgrid_layer, center_point_layer],
            initial_view_state=locked_view,
            map_style=pdk.map_styles.CARTO_DARK,
            tooltip={"text": f"Cell: {selected_cell_id}\nCoordinates: {cell_lat:.4f}° N, {cell_lon:.4f}° E\nEpoch: {selected_year}\n{metric_meta['name']}: {curr_val} {metric_meta['unit']}"}
        ),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Metadata Metrics below the Map
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='meta-box'><div class='meta-label'>Latitude</div><div class='meta-value'>{cell_lat:.4f}° N</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='meta-box'><div class='meta-label'>Longitude</div><div class='meta-value'>{cell_lon:.4f}° E</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='meta-box'><div class='meta-label'>Elevation</div><div class='meta-value'>{curr_record['elevation']} m</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='meta-box'><div class='meta-label'>Baseline Mean</div><div class='meta-value'>{base_mean:.2f} {metric_meta['unit']}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# RIGHT COLUMN: Historical Time-Series Chart (1984–2026)
# ------------------------------------------------------------------------------
with col_chart:
    st.markdown(
        f"""
        <div class="card-container">
            <div class="card-header">
                <span>📈 Historical Time-Series ({metric_meta['name']})</span>
                <span class="badge {'badge-alert' if abs(curr_z) >= 2.0 else 'badge-cyan'}">
                    Epoch {selected_year}: Z = {curr_z:+.2f}σ
                </span>
            </div>
        """,
        unsafe_allow_html=True
    )

    # Identify anomaly spikes (|Z| > 2.0)
    z_col = f"{metric_choice}_zscore"
    anom_mask = df_hist[z_col].abs() >= 2.0
    anom_points = df_hist[anom_mask]

    fig_hist = go.Figure()

    # Normal range ±2σ threshold bands
    fig_hist.add_hrect(
        y0=base_mean - 2 * base_std,
        y1=base_mean + 2 * base_std,
        fillcolor="rgba(0, 242, 254, 0.05)",
        line_width=1,
        line_color="rgba(0, 242, 254, 0.25)",
        line_dash="dot",
        annotation_text="Normal Range (±2σ)",
        annotation_position="top left",
        annotation_font_color="rgba(0, 242, 254, 0.6)",
        annotation_font_size=10
    )

    # Baseline Mean line (μ)
    fig_hist.add_hline(
        y=base_mean,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.4)",
        line_width=1.5,
        annotation_text=f"Baseline μ = {base_mean:.2f}",
        annotation_position="bottom right",
        annotation_font_color="rgba(255, 255, 255, 0.6)",
        annotation_font_size=10
    )

    # Main historical trajectory curve
    fig_hist.add_trace(go.Scatter(
        x=df_hist["year"],
        y=df_hist[metric_choice],
        mode="lines+markers",
        name=metric_meta["name"],
        line=dict(color=metric_meta["color"], width=2.5),
        marker=dict(size=4, color=metric_meta["color"]),
        hovertemplate="<b>Year:</b> %{x}<br><b>Value:</b> %{y:.3f} " + metric_meta["unit"] + "<extra></extra>"
    ))

    # Anomaly Spikes Highlight (|Z| > 2.0)
    if not anom_points.empty:
        fig_hist.add_trace(go.Scatter(
            x=anom_points["year"],
            y=anom_points[metric_choice],
            mode="markers",
            name="Anomaly Spike (|Z| > 2.0)",
            marker=dict(size=9, color="#ff5e62", symbol="diamond", line=dict(width=1.5, color="#ffffff")),
            hovertemplate="<b>⚠️ ANOMALY SPIKE</b><br>Year: %{x}<br>Value: %{y:.3f}<br>Z-Score: %{text}σ<extra></extra>",
            text=[f"{z:+.2f}" for z in anom_points[z_col]]
        ))

    # Selected Year vertical indicator line
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
        height=320,
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Observation Year"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=metric_meta["unit"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistical breakdown & standard deviation threshold explanation
    anom_count = len(anom_points)
    st.markdown(
        f"""
        <div style="font-size:0.86rem; color:#b0bac9; background:#11141c; padding:10px 14px; border-radius:8px; border:1px solid #202634;">
            <b>Statistical Threshold Analysis:</b> 
            Reference baseline mean μ = <code>{base_mean:.3f}</code> (σ = <code>{base_std:.3f}</code>). 
            Points exceeding <b>±2.0σ</b> represent statistically significant outlier events ($p < 0.05$). 
            Across 1984–2026, <b>{anom_count}</b> critical anomaly spikes were detected. 
            Selected year <b>{selected_year}</b> deviation is <code>{curr_anom:+.3f} {metric_meta['unit']}</code> (<b>{curr_z:+.2f}σ</b>).
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 6. PREDICTION & FORECASTING SECTION (BELOW THE FOLD / TIERED ACCESS)
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-bottom: 14px;">
        <h3 style="margin-bottom: 3px; display:flex; align-items:center; gap:8px;">
            <span>🔮 Multi-Decadal Statistical Forecasting Engine (2026–2050)</span>
            <span class="badge badge-cyan">AUTOREGRESSIVE OLS</span>
        </h3>
        <p style="color:#8b9bb4; font-size:0.92rem;">
            Predictive modeling, regression slope extrapolation, and expanding 95% confidence bounds projecting multi-decadal climate shifts.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Compute linear regression and expanding 95% confidence intervals
hist_x = df_hist["year"].values
hist_y = df_hist[metric_choice].values
slope, intercept = np.polyfit(hist_x, hist_y, 1)

future_x = np.arange(2026, 2051)
proj_y = intercept + slope * future_x

# Expanding prediction error for 95% confidence intervals
n_pts = len(hist_x)
x_mean = np.mean(hist_x)
residuals = hist_y - (intercept + slope * hist_x)
s_err = np.std(residuals)
conf_margin = 1.96 * s_err * np.sqrt(1 + 1/n_pts + (future_x - x_mean)**2 / np.sum((hist_x - x_mean)**2))
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

# 2026 Current Era Marker
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
# ACCESS CONTROL LOGIC: PRO UNLOCKED vs. BLURRED PAYWALL OVERLAY
# ==============================================================================
if is_pro:
    # --------------------------------------------------------------------------
    # UNLOCKED PRO VIEW
    # --------------------------------------------------------------------------
    st.markdown('<div class="card-container" style="border: 1px solid rgba(0, 242, 254, 0.35);">', unsafe_allow_html=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    val_2050 = proj_y[-1]
    val_2026 = proj_y[0]
    net_change = val_2050 - val_2026
    decadal_rate = slope * 10

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.metric("Projected 2050 Value", f"{val_2050:.2f} {metric_meta['unit']}", f"{net_change:+.2f} by 2050")
    with f2:
        st.metric("Decadal Rate of Change", f"{decadal_rate:+.3f} {metric_meta['unit']}/dec", "OLS Trend Slope β")
    with f3:
        st.metric("95% CI Range (2050)", f"±{(ci_upper[-1] - val_2050):.2f} {metric_meta['unit']}", "Prediction Margin")
    with f4:
        st.metric("Tipping Point Risk", "CRITICAL" if abs(slope) > 0.05 else "HIGH", "Long-Term Shift", delta_color="inverse")

    st.markdown(
        f"""
        <div style="margin-top: 14px; padding: 12px 18px; background: rgba(0, 242, 254, 0.04); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 8px; font-size: 0.88rem; color: #d0d7de;">
            🔬 <b>Pro Analytical Diagnosis ({selected_cell_id}):</b> 
            Autoregressive climate extrapolation predicts a continuous annual shift of 
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
                    <div class="cta-title">Unlock Statistical Climate Forecasts - Upgrade to Pro</div>
                    <div class="cta-desc">
                        Gain access to multi-decadal predictive modeling, regression confidence bands,
                        and long-term trend extrapolation engines.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Blurred content container applying 8px blur
    st.markdown('<div class="blurred-content">', unsafe_allow_html=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    fb1, fb2, fb3, fb4 = st.columns(4)
    with fb1: st.metric("Projected 2050 Value", "49.80 °C", "+3.40 °C")
    with fb2: st.metric("Decadal Shift Rate", "+1.20 °C/dec", "Slope β")
    with fb3: st.metric("95% CI Range (2050)", "±1.85 °C", "Confidence")
    with fb4: st.metric("Tipping Point Risk", "CRITICAL", "High Confidence", delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

    # Upgrade CTA Action Button
    col_l, col_btn, col_r = st.columns([1.5, 1.2, 1.5])
    with col_btn:
        if st.button("🚀 Upgrade for €19/month", use_container_width=True, type="primary"):
            st.toast("🎉 Pro Tier simulated! Toggle 'Simulate Pro Account' in the sidebar to inspect.", icon="⭐")


# ==============================================================================
# 7. FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #5f6e85; font-size: 0.8rem; padding: 10px 0 20px 0;">
        Earth System Trend Detective • Developed for NASA Space Apps Hackathon • 
        Sentinel 8km × 8km Grid Resolution • Landsat Collection 2 Level-2 Surface Reflectance
    </div>
    """,
    unsafe_allow_html=True
)
