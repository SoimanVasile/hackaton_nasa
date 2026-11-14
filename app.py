import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Earth System Trend Detective", layout="wide")

RUST_API_URL = "http://localhost:8080"

REGIONS_CONFIG = {
    "sea_aquisition": {
        "title": "Aral Sea (Desiccation & Salinization)",
        "files": ["sea_aquisition.parquet", "data/sea_aquisition.parquet", "raw_landsat_grid.parquet"],
        "lat": 45.20, "lon": 59.80, "zoom": 6.2,
        "scale_km": 50, "mode": "water",
        "primary_idx": "ndwi", "primary_label": "Mean Water (NDWI)"
    },
    "urban_aquisition": {
        "title": "Las Vegas & Lake Mead (Urban Growth vs. Water Depletion)",
        "files": ["urban_aquisition.parquet", "data/urban_aquisition.parquet"],
        "lat": 36.285, "lon": -114.765, "zoom": 8.3,
        "scale_km": 20, "mode": "urban_water",
        "primary_idx": "ndwi", "primary_label": "Mean Water (NDWI)"
    },
    "snow_ice_aquisition": {
        "title": "Harvard & Columbia Glaciers, Alaska (Missing Ice)",
        "files": ["snow_ice_aquisition.parquet", "data/snow_ice_aquisition.parquet"],
        "lat": 61.12, "lon": -147.38, "zoom": 8.1,
        "scale_km": 15, "mode": "ice",
        "primary_idx": "ndsi", "primary_label": "Mean Ice (NDSI)"
    }
}

# Las Vegas Valley physical basin boundary
LV_VALLEY_POLYGON = np.array([
    [-115.335, 36.320], [-115.220, 36.310], [-115.085, 36.285],
    [-115.025, 36.235], [-115.015, 36.125], [-114.915, 36.115],
    [-114.935, 36.010], [-115.055, 35.965], [-115.175, 35.960],
    [-115.275, 36.005], [-115.330, 36.085], [-115.355, 36.165],
    [-115.340, 36.245],
])

# 1984 Anchor Polygon for the initial Las Vegas urban core
LV_1984_ANCHOR_POLYGON = np.array([
    [-115.235, 36.220], [-115.115, 36.230], [-115.040, 36.185],
    [-115.045, 36.090], [-115.130, 36.065], [-115.205, 36.085],
    [-115.240, 36.150],
])


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
    """Return first existing parquet path for the selected region."""
    for candidate in REGIONS_CONFIG[region_key]["files"]:
        if os.path.exists(candidate):
            return candidate
    return REGIONS_CONFIG[region_key]["files"][0]


@st.cache_data
def load_and_compute_region(region_key: str) -> pd.DataFrame:
    """Load parquet, lock the 1984 anchor mask, and compute 1985-2026 changes relative to 1984."""
    cfg = REGIONS_CONFIG[region_key]
    path = resolve_parquet_path(region_key)
    if not os.path.exists(path):
        st.error(f"Dataset file `{path}` not found. Please run `ingest_gee.py` first.")
        st.stop()

    df = pd.read_parquet(path)
    for col in ['ndwi', 'ndbi', 'ndsi', 'salinity_index', 'lst_c']:
        if col not in df.columns:
            df[col] = 0.0

    # Equalize horizontal spacing at high latitudes (e.g., Alaska ~61°N)
    lons = np.sort(df['longitude'].unique())
    lats = np.sort(df['latitude'].unique())
    if len(lons) > 2 and len(lats) > 2:
        dx = np.median(np.diff(lons)) * np.cos(np.radians(cfg["lat"]))
        dy = np.median(np.diff(lats))
        if dy / (dx + 1e-9) > 1.65:
            df = df[df['longitude'].isin(set(lons[::2]))].copy()

    df = df.drop_duplicates(subset=['latitude', 'longitude', 'year']).copy()

    # Reconstruct full (latitude, longitude, year) grid so GEE-masked pixels never leave holes
    coords = df[['latitude', 'longitude']].drop_duplicates()
    years = pd.DataFrame({'year': np.sort(df['year'].unique())})
    df = coords.merge(years, how='cross').merge(df, on=['latitude', 'longitude', 'year'], how='left')
    df = df.sort_values(['latitude', 'longitude', 'year']).reset_index(drop=True)

    same_prev = (df['latitude'] == df['latitude'].shift(1)) & (df['longitude'] == df['longitude'].shift(1))
    same_next = (df['latitude'] == df['latitude'].shift(-1)) & (df['longitude'] == df['longitude'].shift(-1))
    fill_cols = ['ndwi', 'ndbi', 'ndsi', 'salinity_index', 'lst_c']
    for _ in range(3):
        for col in fill_cols:
            df[col] = np.where(df[col].isna() & same_prev, df[col].shift(1), df[col])
            df[col] = np.where(df[col].isna() & same_next, df[col].shift(-1), df[col])

    if df[fill_cols].isna().any().any():
        for col in fill_cols:
            df[col] = df[col].fillna(df.groupby(['latitude', 'longitude'])[col].transform('median'))

    # 1984 Anchor Statistics
    anchor_1984 = df[df['year'] <= 1986].groupby(['latitude', 'longitude']).agg(
        anc_ndwi=('ndwi', 'max'),
        anc_ndbi=('ndbi', 'median'),
        anc_ndsi=('ndsi', 'median'),
        anc_lst=('lst_c', 'median')
    ).reset_index()

    base_stats = df[df['year'] <= 1992].groupby(['latitude', 'longitude']).agg(
        mean_ndwi=('ndwi', 'mean'), std_ndwi=('ndwi', 'std'),
        mean_ndbi=('ndbi', 'mean'), std_ndbi=('ndbi', 'std'),
        mean_ndsi=('ndsi', 'mean'), std_ndsi=('ndsi', 'std'),
        mean_lst=('lst_c', 'mean'), std_lst=('lst_c', 'std')
    ).reset_index()

    df = df.merge(base_stats, on=['latitude', 'longitude'], how='left')
    df = df.merge(anchor_1984, on=['latitude', 'longitude'], how='left')

    df['z_ndwi'] = ((df['ndwi'] - df['mean_ndwi']) / (df['std_ndwi'].fillna(0.04) + 0.03)).round(2)
    df['z_ndbi'] = ((df['ndbi'] - df['mean_ndbi']) / (df['std_ndbi'].fillna(0.04) + 0.03)).round(2)
    df['z_ndsi'] = ((df['ndsi'] - df['mean_ndsi']) / (df['std_ndsi'].fillna(0.05) + 0.03)).round(2)
    df['z_lst'] = ((df['lst_c'] - df['mean_lst']) / (df['std_lst'].fillna(2.0) + 0.5)).round(2)

    for col in ['ndwi', 'ndbi', 'ndsi', 'salinity_index']:
        df[col] = df[col].round(3)
    df['lst_c'] = df['lst_c'].round(1)

    df['base_urban'] = False
    df['is_water_now'] = False

    if cfg["mode"] == "ice":
        was_glacier = (df['anc_ndsi'] > 0.42) & (df['mean_lst'] < 12.0)
        is_still_glacier = df['ndsi'] > 0.38

        df['anom_ice'] = (df['year'] > 1984) & was_glacier & (~is_still_glacier)
        df['anom_water'] = False
        df['anom_urban'] = False
        df['anom_temp'] = False
        df['anom_combined'] = False
        df['is_water_now'] = (~is_still_glacier) & (df['ndwi'] > 0.20)
        df['severity'] = np.where(df['anom_ice'], df['z_ndsi'].abs(), 0.0).round(2)

    elif cfg["mode"] == "urban_water":
        # ----------------------------------------------------------------------
        # 1. LAKE MEAD (PURE NDWI — ZERO SALINITY INDEX)
        # ----------------------------------------------------------------------
        in_mead_basin = (
            (df['longitude'] > -114.91) &
            (df['latitude'] <= 36.465) &
            ~((df['latitude'] > 36.22) & (df['longitude'] < -114.48))
        )
        # 1984 high-water anchor outline using NDWI only
        water_anchor_1984 = in_mead_basin & ((df['anc_ndwi'] > -0.02) | (df['mean_ndwi'] > -0.01))

        # Keep main Lake Mead body blue wherever NDWI > -0.035; only dried shorelines turn orange
        still_water = df['ndwi'] > -0.035
        df['is_water_now'] = water_anchor_1984 & np.where(df['year'] == 1984, True, still_water)
        df['anom_water'] = (df['year'] > 1984) & water_anchor_1984 & (~df['is_water_now'])

        # ----------------------------------------------------------------------
        # 2. LAS VEGAS URBAN CORE & EXPANSION (PURE NDBI & GEOMETRY — ZERO SALINITY)
        # ----------------------------------------------------------------------
        in_valley_poly = points_in_polygon(df['longitude'].values, df['latitude'].values, LV_VALLEY_POLYGON)
        in_1984_core_poly = points_in_polygon(df['longitude'].values, df['latitude'].values, LV_1984_ANCHOR_POLYGON)

        old_henderson = ((df['longitude'] + 114.985)**2 + (df['latitude'] - 36.035)**2) < 0.032**2
        boulder_city = ((df['longitude'] + 114.835)**2 + (df['latitude'] - 35.978)**2) < 0.022**2

        # 1984 Urban Anchor Core (Indigo-Violet)
        df['base_urban'] = (~water_anchor_1984) & (in_1984_core_poly | old_henderson | boulder_city)

        # Post-1984 suburban expansion (Bright Magenta) driven by NDBI change and outward growth
        ndbi_roll = df.groupby(['latitude', 'longitude'])['ndbi'].transform(lambda s: s.rolling(3, min_periods=1).median())
        dist_from_1984_core = np.sqrt(((df['longitude'] + 115.14) * 0.81)**2 + (df['latitude'] - 36.16)**2)
        max_reach_for_year = 0.065 + 0.135 * ((df['year'] - 1984) / 42.0).clip(0.0, 1.0)

        ndbi_urban_shift = ((ndbi_roll - df['anc_ndbi']).abs() >= 0.012) | (ndbi_roll <= -0.04) | (df['z_ndbi'].abs() >= 0.8)
        raw_expansion = (
            in_valley_poly &
            (~water_anchor_1984) &
            (~df['base_urban']) &
            (df['year'] > 1984) &
            (dist_from_1984_core <= max_reach_for_year) &
            ndbi_urban_shift
        )
        df['anom_urban'] = raw_expansion.groupby([df['latitude'], df['longitude']]).cummax()

        # Higher thermal Z-threshold (2.6) so the background desert doesn't flood yellow
        df['anom_temp'] = (df['year'] > 1984) & (df['z_lst'] >= 2.6)
        df['anom_combined'] = (df['anom_water'] | df['anom_urban']) & df['anom_temp']
        df['anom_ice'] = False

        urban_sev = np.where(df['anom_urban'], np.maximum(df['z_ndbi'].abs(), 2.2), 0.0)
        water_sev = np.where(
            df['anom_water'],
            np.maximum(df['z_ndwi'].abs(), ((df['anc_ndwi'] - df['ndwi']) / 0.05).clip(lower=2.0)),
            0.0
        )
        df['severity'] = (water_sev + urban_sev + np.where(df['anom_temp'], df['z_lst'].clip(lower=0), 0.0)).round(2)

    else:
        # Classic Aral Sea model anchored to 1984
        was_water = df['anc_ndwi'] > 0.0
        df['is_water_now'] = df['ndwi'] > 0.0
        df['anom_water'] = (df['year'] > 1984) & was_water & ((df['z_ndwi'] <= -2.0) | (df['ndwi'] < -0.05))
        df['anom_urban'] = False
        df['anom_temp'] = (df['year'] > 1984) & (df['z_lst'] >= 1.8)
        df['anom_combined'] = df['anom_water'] & df['anom_temp']
        df['anom_ice'] = False
        df['severity'] = (
            np.where(df['anom_water'], df['z_ndwi'].abs(), 0.0) +
            np.where(df['anom_temp'], df['z_lst'].clip(lower=0), 0.0)
        ).round(2)

    return df


def fetch_region_data(region_key: str) -> pd.DataFrame:
    """Fetch processed dataset from Rust backend or fallback silently to local data."""
    try:
        resp = requests.get(f"{RUST_API_URL}/anomalies_all", params={"region": region_key}, timeout=0.6)
        resp.raise_for_status()
        return pd.DataFrame(resp.json()).rename(columns={"lat": "latitude", "lon": "longitude"})
    except Exception:
        return load_and_compute_region(region_key)


def compute_tangent_radius(df_year: pd.DataFrame, center_lat: float) -> tuple[int, int]:
    """Calculate circle radius = 0.50 * grid step so adjacent tiles touch tangentially."""
    lons = np.sort(df_year['longitude'].unique())
    lats = np.sort(df_year['latitude'].unique())
    dlon = float(np.median(np.diff(lons))) if len(lons) > 1 else 0.01
    dlat = float(np.median(np.diff(lats))) if len(lats) > 1 else 0.01

    dx_m = dlon * 111320.0 * np.cos(np.radians(center_lat))
    dy_m = dlat * 111132.0
    min_step_m = min(dx_m, dy_m)
    return max(int(round(min_step_m * 0.50)), 15), int(round(dy_m, -1))


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
            "Ice Lost / Bedrock & Fjord (< 0.25)"
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
        "Desiccated Bed / Arid Land (< -0.05)"
    ]
    return bands, domain, ["#38bdf8", "#f59e0b", "#ef4444"], (-0.45, 0.65)


def build_analytics_charts(ts_df: pd.DataFrame, reg_cfg: dict, chart_scope: str):
    """Render dual-axis historical chart and 10-year forecast chart with physical intervals."""
    primary_col = reg_cfg["primary_idx"]
    primary_label = reg_cfg["primary_label"]
    mode = reg_cfg["mode"]

    clean_ts = ts_df[['year', primary_col, 'lst_c']].dropna().sort_values('year').copy()
    min_year, max_year = int(clean_ts['year'].min()), int(clean_ts['year'].max())

    recent = clean_ts[clean_ts['year'] >= 2010]
    if len(recent) < 2:
        recent = clean_ts

    slope, intercept = np.polyfit(recent['year'], recent[primary_col], 1)
    residuals = recent[primary_col] - (slope * recent['year'] + intercept)
    sigma = float(np.std(residuals)) if len(residuals) > 2 else 0.03

    last_val = float(clean_ts.loc[clean_ts['year'] == max_year, primary_col].values[0])
    future_years = np.arange(max_year, max_year + 11)
    steps = future_years - max_year
    pred_vals = np.clip(last_val + slope * steps, -0.55, 0.88)
    spread = sigma * np.sqrt(1.0 + steps / 5.0)

    forecast_df = pd.DataFrame({
        "year": future_years,
        "forecast": np.round(pred_vals, 3),
        "ci_low": np.round(np.clip(pred_vals - spread, -0.58, 0.90), 3),
        "ci_high": np.round(np.clip(pred_vals + spread, -0.58, 0.90), 3)
    })

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown(f"#### 📈 Historical Trend (1984–2026) — {chart_scope}")
        bands_hist, domain, colors, y_lim = get_physical_regime_bands(mode, min_year, max_year)

        bg_hist = alt.Chart(bands_hist).mark_rect(opacity=0.14).encode(
            x=alt.X('xmin:Q', scale=alt.Scale(domain=[min_year, max_year], nice=False), axis=alt.Axis(format='d', title='Year', tickCount=9)),
            x2='xmax:Q',
            y=alt.Y('ymin:Q', scale=alt.Scale(domain=list(y_lim)), axis=alt.Axis(title=primary_label, titleColor='#38bdf8')),
            y2='ymax:Q',
            color=alt.Color('regime:N', scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient='bottom', columns=1, title="Physical Index Zones"))
        )
        idx_line = alt.Chart(clean_ts).mark_line(color='#38bdf8', strokeWidth=2.6, point=alt.OverlayMarkDef(size=28, color='#38bdf8')).encode(
            x=alt.X('year:Q', scale=alt.Scale(domain=[min_year, max_year], nice=False), axis=alt.Axis(format='d', title='Year')),
            y=alt.Y(f'{primary_col}:Q', scale=alt.Scale(domain=list(y_lim)), axis=alt.Axis(title=primary_label, titleColor='#38bdf8')),
            tooltip=[alt.Tooltip('year:Q', format='d', title='Year'), alt.Tooltip(f'{primary_col}:Q', format='.3f', title=primary_label), alt.Tooltip('lst_c:Q', format='.1f', title='Surface Temp (°C)')]
        )
        temp_min, temp_max = max(0.0, float(clean_ts['lst_c'].min()) - 4.0), float(clean_ts['lst_c'].max()) + 4.0
        temp_line = alt.Chart(clean_ts).mark_line(color='#fb923c', strokeWidth=2.0, strokeDash=[4, 3]).encode(
            x=alt.X('year:Q', scale=alt.Scale(domain=[min_year, max_year], nice=False)),
            y=alt.Y('lst_c:Q', scale=alt.Scale(domain=[temp_min, temp_max]), axis=alt.Axis(title='Land Surface Temp (°C)', titleColor='#fb923c', orient='right'))
        )
        st.altair_chart(alt.layer(bg_hist + idx_line, temp_line).resolve_scale(y='independent').properties(height=330), use_container_width=True)

    with g_col2:
        st.markdown(f"#### 🔮 10-Year Projection (2026–2036) — {primary_label}")
        bands_fut, domain, colors, y_lim = get_physical_regime_bands(mode, min_year, max_year + 10)

        bg_fut = alt.Chart(bands_fut).mark_rect(opacity=0.14).encode(
            x=alt.X('xmin:Q', scale=alt.Scale(domain=[min_year, max_year + 10], nice=False), axis=alt.Axis(format='d', title='Year (Historical → Projected)', tickCount=10)),
            x2='xmax:Q',
            y=alt.Y('ymin:Q', scale=alt.Scale(domain=list(y_lim)), axis=alt.Axis(title=primary_label)),
            y2='ymax:Q',
            color=alt.Color('regime:N', scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient='bottom', columns=1, title="Target Thresholds"))
        )
        hist_line = alt.Chart(clean_ts).mark_line(color='#38bdf8', strokeWidth=2.4).encode(
            x=alt.X('year:Q', scale=alt.Scale(domain=[min_year, max_year + 10], nice=False)),
            y=alt.Y(f'{primary_col}:Q', scale=alt.Scale(domain=list(y_lim)))
        )
        ci_band = alt.Chart(forecast_df).mark_area(color='#f97316', opacity=0.25).encode(x='year:Q', y='ci_low:Q', y2='ci_high:Q')
        proj_line = alt.Chart(forecast_df).mark_line(color='#f97316', strokeWidth=2.8, strokeDash=[6, 4], point=alt.OverlayMarkDef(size=35, color='#f97316')).encode(
            x='year:Q', y='forecast:Q',
            tooltip=[alt.Tooltip('year:Q', format='d', title='Projected Year'), alt.Tooltip('forecast:Q', format='.3f', title=f'Projected {primary_label}')]
        )
        now_rule = alt.Chart(pd.DataFrame({'year': [max_year]})).mark_rule(color='#e2e8f0', strokeDash=[3, 3], strokeWidth=1.5).encode(x='year:Q')
        st.altair_chart((bg_fut + hist_line + ci_band + proj_line + now_rule).properties(height=330), use_container_width=True)

    proj_2036 = float(forecast_df['forecast'].iloc[-1])
    st.info(
        f"**How to read these values:** Blue Line (Left Axis) = **{primary_label}** (`-1` to `+1`), "
        f"Dashed Orange Line (Right Axis) = **Land Surface Temp (°C)**. "
        f"**10-Year Linear Trend:** `{slope:+.4f} / yr` → **Projected 2036 Value:** `{proj_2036:.3f}`."
    )


# ==============================================================================
# 1. TOP CENTRAL BAR FOR REGION SELECTION
# ==============================================================================
_, center_col, _ = st.columns([1, 2.5, 1])
with center_col:
    selected_region = st.selectbox(
        "🌍 Select Satellite Monitoring Region",
        options=list(REGIONS_CONFIG.keys()),
        format_func=lambda k: REGIONS_CONFIG[k]["title"]
    )

reg_cfg = REGIONS_CONFIG[selected_region]
df_all = fetch_region_data(selected_region)

# ==============================================================================
# 2. SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.header("Analysis Controls")

    min_yr, max_yr = int(df_all['year'].min()), int(df_all['year'].max())
    selected_year = st.slider("Select Year", min_yr, max_yr, min(2024, max_yr))

    if reg_cfg["mode"] == "ice":
        anomaly_mode = st.selectbox(
            "Detected Anomaly Type",
            options=["ice"],
            format_func=lambda _: "Glacier Anomaly (Missing Ice)"
        )
    elif reg_cfg["mode"] == "urban_water":
        anomaly_mode = st.selectbox(
            "Detected Anomaly Type",
            options=["all_types", "combined", "water", "urban", "temp"],
            format_func=lambda x: {
                "all_types": "All Anomaly Types (Color-Coded)",
                "combined": "1. Compound Anomaly (Water/Urban + Heat)",
                "water": "2. Lake Mead Desiccation (NDWI vs. 1984)",
                "urban": "3. Urban Infrastructure (NDBI 1984 & Expansion)",
                "temp": "4. Thermal Anomaly (LST °C Spike)"
            }[x]
        )
    else:
        anomaly_mode = st.selectbox(
            "Detected Anomaly Type",
            options=["all_types", "combined", "water", "temp"],
            format_func=lambda x: {
                "all_types": "All 3 Anomaly Types (Color-Coded)",
                "combined": "1. Compound Anomaly (Water Loss + Heat)",
                "water": "2. Water Anomaly (NDWI Desiccation)",
                "temp": "3. Thermal Anomaly (LST °C Spike)"
            }[x]
        )

    # Salinity Index ONLY for Aral Sea
    color_options = ["status", "severity", "salinity_index"] if selected_region == "sea_aquisition" else ["status", "severity"]
    metric_choice = st.selectbox(
        "Map Color Layer",
        options=color_options,
        format_func=lambda x: {
            "status": "Hydrological & Urban Status",
            "severity": "Severity Heatmap",
            "salinity_index": "Salinity Heatmap (Aral Sea)"
        }[x]
    )

    only_anomalies = st.checkbox("Show only anomalous cells", value=False)

# ==============================================================================
# 3. YEAR FILTERING & PHYSICAL SURFACE CLASSIFICATION
# ==============================================================================
df_year = df_all[df_all['year'] == selected_year].copy()
cell_radius, cell_res_m = compute_tangent_radius(df_year, reg_cfg["lat"])

if anomaly_mode == "ice":
    df_year['active_anom'] = df_year['anom_ice']
elif anomaly_mode == "combined":
    df_year['active_anom'] = df_year['anom_combined']
elif anomaly_mode == "water":
    df_year['active_anom'] = df_year['anom_water']
elif anomaly_mode == "urban":
    df_year['active_anom'] = df_year['anom_urban']
elif anomaly_mode == "temp":
    df_year['active_anom'] = df_year['anom_temp']
else:
    df_year['active_anom'] = df_year['anom_water'] | df_year['anom_urban'] | df_year['anom_temp']

is_glacier_surface = (reg_cfg["mode"] == "ice") & (df_year['ndsi'] > 0.42)
is_liquid_water = df_year['is_water_now']

df_year['reason'] = np.select(
    [
        df_year['anom_ice'],
        df_year['anom_water'],
        df_year['anom_urban'],
        df_year['base_urban'],
        df_year['anom_combined'],
        df_year['anom_temp'],
        is_glacier_surface,
        is_liquid_water
    ],
    [
        "GLACIER ANOMALY (Missing Ice): Retreated vs. 1984 Anchor",
        "WATER ANOMALY: Receded Lake Mead shoreline (NDWI drop vs. 1984)",
        "URBAN EXPANSION: Post-1984 built-up infrastructure (NDBI)",
        "1984 URBAN ANCHOR: Original 1984 Las Vegas / Henderson core",
        "CRITICAL (Compound): Water/Urban shift + Thermal shock",
        "THERMAL ANOMALY: Land surface overheating",
        "Normal (Permanent Glacier / Snow cover)",
        "Normal (Liquid Water)"
    ],
    default="Normal (Arid Desert / Bedrock)"
)

df_display = df_year[df_year['active_anom']].copy() if only_anomalies else df_year.copy()

# ==============================================================================
# 4. SEVERITY-GRADED & 1984-ANCHORED RGB COLORING
# ==============================================================================
if metric_choice == "status":
    sev_norm = ((df_display['severity'] - 1.8) / 4.5).clip(0.0, 1.0)
    disp_glacier = (reg_cfg["mode"] == "ice") & (df_display['ndsi'] > 0.42)
    disp_water = df_display['is_water_now']

    if anomaly_mode == "ice":
        conds = [df_display['anom_ice'], disp_glacier, disp_water]
        df_display['r'] = np.select(conds, [(255 - sev_norm * 55).astype(int), 245, 15], default=145)
        df_display['g'] = np.select(conds, [(115 - sev_norm * 100).astype(int), 250, 110], default=135)
        df_display['b'] = np.select(conds, [(40 - sev_norm * 20).astype(int), 255, 245], default=120)

    elif anomaly_mode == "urban":
        conds = [df_display['anom_urban'], df_display['base_urban'], disp_water]
        df_display['r'] = np.select(conds, [(255 - sev_norm * 45).astype(int), 124, 25], default=185)
        df_display['g'] = np.select(conds, [(56 - sev_norm * 56).astype(int), 92, 135], default=170)
        df_display['b'] = np.select(conds, [(200 - sev_norm * 40).astype(int), 230, 255], default=145)

    elif anomaly_mode == "water":
        conds = [df_display['anom_water'], disp_water]
        df_display['r'] = np.select(conds, [(255 - sev_norm * 35).astype(int), 25], default=185)
        df_display['g'] = np.select(conds, [(165 - sev_norm * 95).astype(int), 135], default=170)
        df_display['b'] = np.select(conds, [0, 255], default=145)

    elif anomaly_mode == "temp":
        conds = [df_display['anom_temp'], disp_water]
        df_display['r'] = np.select(conds, [255, 25], default=185)
        df_display['g'] = np.select(conds, [(230 - sev_norm * 90).astype(int), 135], default=170)
        df_display['b'] = np.select(conds, [0, 255], default=145)

    elif anomaly_mode == "combined":
        conds = [df_display['anom_combined'], disp_water]
        df_display['r'] = np.select(conds, [(255 - sev_norm * 85).astype(int), 25], default=185)
        df_display['g'] = np.select(conds, [(40 - sev_norm * 40).astype(int), 135], default=170)
        df_display['b'] = np.select(conds, [(60 - sev_norm * 40).astype(int), 255], default=145)

    else:
        # In "all_types", Liquid Water (Blue), Receded Shoreline (Orange), 1984 Urban (Indigo-Violet),
        # and New Urban Expansion (Magenta) take priority over background desert heat
        conds = [
            disp_water,
            df_display['anom_water'],
            df_display['anom_urban'],
            df_display['base_urban'],
            df_display['anom_combined'],
            df_display['anom_temp']
        ]
        df_display['r'] = np.select(
            conds,
            [25, (255 - sev_norm * 30).astype(int), (255 - sev_norm * 45).astype(int), 124, (255 - sev_norm * 75).astype(int), 255],
            default=185
        )
        df_display['g'] = np.select(
            conds,
            [135, (155 - sev_norm * 80).astype(int), (56 - sev_norm * 56).astype(int), 92, (35 - sev_norm * 35).astype(int), (225 - sev_norm * 65).astype(int)],
            default=170
        )
        df_display['b'] = np.select(
            conds,
            [255, 0, (200 - sev_norm * 40).astype(int), 230, (60 - sev_norm * 35).astype(int), 0],
            default=145
        )

elif metric_choice == "severity":
    has_sev = df_display['severity'] > 0
    s_norm = (df_display['severity'] / 6.0).clip(0.0, 1.0)
    df_display['r'] = np.where(has_sev, 255, 45)
    df_display['g'] = np.where(has_sev, ((1.0 - s_norm) * 210).astype(int), 55)
    df_display['b'] = np.where(has_sev, 20, 72)

else:
    # Salinity Index ( strictly for Aral Sea )
    vals = df_display['salinity_index']
    vmin, vmax = vals.quantile(0.05), vals.quantile(0.95)
    norm = ((vals - vmin) / (vmax - vmin + 1e-6)).clip(0, 1)
    df_display['r'] = (norm * 255).astype(int)
    df_display['g'] = ((1 - norm) * 180).astype(int)
    df_display['b'] = 40

# ==============================================================================
# 5. KPI METRICS, VISUAL LEGEND, PYDECK MAP & BOTTOM SCALE BAR
# ==============================================================================
st.title(f"{reg_cfg['title']} — {selected_year}")

c1, c2, c3, c4 = st.columns(4)
c1.metric(reg_cfg["primary_label"], f"{df_year[reg_cfg['primary_idx']].mean():.3f}")
c2.metric("Mean Temperature", f"{df_year['lst_c'].mean():.1f} °C")
if reg_cfg["mode"] == "ice":
    c3.metric("Glacier Anomalies (Missing Ice)", f"{int(df_year['anom_ice'].sum())}")
    c4.metric("Fjord Water Cells", f"{int(is_liquid_water.sum())}")
elif reg_cfg["mode"] == "urban_water":
    c3.metric("Lake Mead Desiccation (vs. 1984)", f"{int(df_year['anom_water'].sum())}")
    c4.metric("Post-1984 Urban Expansion Cells", f"{int(df_year['anom_urban'].sum())}")
else:
    c3.metric("Water Anomalies (Desiccation)", f"{int(df_year['anom_water'].sum())}")
    c4.metric("Compound Anomalies (Water+Temp)", f"{int(df_year['anom_combined'].sum())}")

if metric_choice == "status":
    if reg_cfg["mode"] == "ice":
        st.markdown(
            """
            <div style="background:#161b22; padding:10px 16px; border-radius:6px; border:1px solid #30363d; margin-bottom:10px; display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:13px;">
                <b>Map Legend:</b>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f5faff;border:1px solid #888;margin-right:6px;"></span>Stable Glacier / Snow</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#0f6ef5;margin-right:6px;"></span>Liquid Water (Fjord)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#e63919;margin-right:6px;"></span><b>Missing Ice</b> (Glacier retreated vs. 1984)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#918778;margin-right:6px;"></span>Mountain Slope / Bedrock</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif reg_cfg["mode"] == "urban_water":
        st.markdown(
            """
            <div style="background:#161b22; padding:10px 16px; border-radius:6px; border:1px solid #30363d; margin-bottom:10px; display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:13px;">
                <b>Map Legend (Anchored to 1984 Baseline):</b>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#1987ff;margin-right:6px;"></span><b>Liquid Water (Lake Mead NDWI)</b></span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff8c00;margin-right:6px;"></span><b>Water Loss</b> (Receded vs. 1984 Shoreline)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#7c5ce6;margin-right:6px;"></span><b>1984 Urban Anchor Core</b> (Las Vegas / Henderson)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff38c8;margin-right:6px;"></span><b>Post-1984 Urban Expansion</b> (NDBI Growth)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ffd700;margin-right:6px;"></span><b>Thermal Anomaly</b></span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#b9aa91;margin-right:6px;"></span>Arid Desert</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background:#161b22; padding:10px 16px; border-radius:6px; border:1px solid #30363d; margin-bottom:10px; display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:13px;">
                <b>Map Legend:</b>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#1987ff;margin-right:6px;"></span><b>Liquid Water</b></span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff8c00;margin-right:6px;"></span><b>Water Anomaly</b> (Desiccated Zone)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ffd700;margin-right:6px;"></span><b>Thermal Anomaly</b> (Surface Overheating)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#dc143c;margin-right:6px;"></span><b>Compound Anomaly</b> (Desiccation + Heat Shock)</span>
                <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#b9aa91;margin-right:6px;"></span>Dry Land / Normal Shore</span>
            </div>
            """,
            unsafe_allow_html=True
        )
elif metric_choice == "severity":
    st.markdown(
        """
        <div style="background:#161b22; padding:10px 16px; border-radius:6px; border:1px solid #30363d; margin-bottom:10px; display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:13px;">
            <b>Severity Heatmap Legend:</b>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2d3748;margin-right:6px;"></span>Normal (Severity 0)</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ffd214;margin-right:6px;"></span>Moderate Severity (Z ≈ 2.0)</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff6b14;margin-right:6px;"></span>High Severity (Z ≈ 4.0)</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff1414;margin-right:6px;"></span>Extreme Severity (Z &gt; 6.0)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="background:#161b22; padding:10px 16px; border-radius:6px; border:1px solid #30363d; margin-bottom:10px; display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:13px;">
            <b>Salinity Legend (Aral Sea):</b>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#00b428;margin-right:6px;"></span>Low Salinity / Water</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#b46e28;margin-right:6px;"></span>Moderately Salinized Soil</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ff0028;margin-right:6px;"></span>Dense Salt Crust (Aralkum)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

layer = pdk.Layer(
    "ScatterplotLayer",
    id="earth-grid",
    data=df_display,
    get_position='[longitude, latitude]',
    get_fill_color='[r, g, b, 255]',
    get_radius=cell_radius,
    pickable=True,
    auto_highlight=True
)

event = st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=reg_cfg["lat"], longitude=reg_cfg["lon"], zoom=reg_cfg["zoom"]),
        tooltip={
            "text": (
                "{reason}\n"
                "Coords: {latitude}, {longitude}\n"
                "Severity: {severity}\n"
                "NDWI: {ndwi} (Z: {z_ndwi}) | NDBI: {ndbi} (Z: {z_ndbi})\n"
                "Temp: {lst_c} °C (Z: {z_lst})"
            )
        }
    ),
    on_select="rerun",
    selection_mode="single-object"
)

st.markdown(
    f"""
    <div style="background:#161b22; padding:8px 16px; border-radius:6px; border:1px solid #30363d; display:flex; justify-content:space-between; align-items:center; font-size:13px;">
        <div>
            <b>📏 Map Scale:</b>
            <span style="display:inline-block; width:85px; border-bottom:3px solid #e6edf3; border-left:2px solid #e6edf3; border-right:2px solid #e6edf3; height:6px; margin:0 8px;"></span>
            <b>~{reg_cfg['scale_km']} km</b> (Cell step: ~{cell_res_m} m)
        </div>
        <div>
            <b>Anomaly Severity:</b> Moderate (Z ≈ 2.0)
            <span style="display:inline-block; width:130px; height:10px; background:linear-gradient(90deg, #ffae34, #e62020, #8a0014); border-radius:4px; margin:0 6px;"></span>
            Extreme (Z &gt; 5.0)
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

selected_objects = event.selection.get("objects", {}).get("earth-grid", [])
if selected_objects:
    cell = selected_objects[0]
    st.info(
        f"**Cell Diagnostic:** {cell['reason']} | "
        f"**Coords:** `{cell['latitude']}, {cell['longitude']}` | "
        f"**NDWI:** `{cell['ndwi']}` (Z: `{cell['z_ndwi']}`) | "
        f"**NDBI:** `{cell['ndbi']}` (Z: `{cell['z_ndbi']}`) | "
        f"**Temp:** `{cell['lst_c']} °C` (Z: `{cell['z_lst']}`)"
    )

# ==============================================================================
# 6. DUAL-AXIS HISTORICAL CHART & 10-YEAR FORECAST WITH PHYSICAL INTERVALS
# ==============================================================================
primary_col = reg_cfg["primary_idx"]
if selected_objects:
    lat_s, lon_s = selected_objects[0]["latitude"], selected_objects[0]["longitude"]
    ts_df = df_all[(df_all["latitude"] == lat_s) & (df_all["longitude"] == lon_s)].sort_values("year")
    chart_scope = f"Selected Cell ({lat_s:.3f}, {lon_s:.3f})"
else:
    ts_df = df_all.groupby("year", as_index=False)[[primary_col, "lst_c"]].mean().sort_values("year")
    chart_scope = reg_cfg["title"]

st.markdown("---")
build_analytics_charts(ts_df, reg_cfg, chart_scope)