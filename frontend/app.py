import streamlit as st
import pandas as pd
import numpy as np
import requests
import pydeck as pdk

st.set_page_config(page_title="Aral Trend Detective", layout="wide")

RUST_API_URL = "http://localhost:8080"


def fetch_year_data(year: int) -> pd.DataFrame:
    try:
        resp = requests.get(f"{RUST_API_URL}/anomalies", params={"year": year}, timeout=5.0)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()).rename(columns={"lat": "latitude", "lon": "longitude"})
        st.sidebar.success("Conectat la Rust Engine (Live)")
        return df
    except Exception as e:
        st.error(f"Serverul Rust este oprit ({e}). Porneste backend-ul cu `./run.sh`!")
        st.stop()


with st.sidebar:
    st.header("Controale Analiză")
    selected_year = st.slider("Selectează Anul", 1984, 2026, 2000)

    anomaly_mode = st.selectbox(
        "Tip Anomalie Detectată",
        options=["all_types", "combined", "water", "temp"],
        format_func=lambda x: {
            "all_types": "Toate cele 3 tipuri (Colorate distinct)",
            "combined": "1. Anomalie Generală (Apă + Temp)",
            "water": "2. Anomalia Apei (Secare NDWI)",
            "temp": "3. Anomalie Termică (Șoc LST °C)"
        }[x]
    )

    metric_choice = st.selectbox(
        "Mod colorare fundal hartă",
        options=["status", "ndwi", "lst_c", "salinity_index", "severity"],
        format_func=lambda x: {
            "status": "Status Hidologic & Anomalii",
            "ndwi": "Heatmap NDWI (Indice Apă)",
            "lst_c": "Heatmap Temperatură Sol (°C)",
            "salinity_index": "Heatmap Salinitate",
            "severity": "Heatmap Severitate"
        }[x]
    )

    only_anomalies = st.checkbox("Afișează doar celulele cu anomalie", value=False)

df_year = fetch_year_data(selected_year)

if anomaly_mode == "combined":
    df_year['active_anom'] = df_year['anom_combined']
elif anomaly_mode == "water":
    df_year['active_anom'] = df_year['anom_water']
elif anomaly_mode == "temp":
    df_year['active_anom'] = df_year['anom_temp']
else:
    df_year['active_anom'] = df_year['anom_water'] | df_year['anom_temp']

df_year['motiv'] = np.select(
    [
        df_year['anom_combined'],
        df_year['anom_water'],
        df_year['anom_temp'],
        df_year['ndwi'] > 0
    ],
    [
        "CRITIC (General): Secare apa + Soc termic",
        "ANOMALIE APA: Retragere severa fata de 1984-1992",
        "ANOMALIE TERMICA: Supraincalzire sol",
        "Normal (Acoperit cu apa)"
    ],
    default="Normal (Uscat istoric)"
)

df_display = df_year[df_year['active_anom']].copy() if only_anomalies else df_year.copy()

if metric_choice == "status":
    conds = [
        df_display['anom_combined'] & df_display['active_anom'],
        df_display['anom_water'] & df_display['active_anom'],
        df_display['anom_temp'] & df_display['active_anom'],
        df_display['ndwi'] > 0.0
    ]
    df_display['r'] = np.select(conds, [255, 255, 255, 30], default=195)
    df_display['g'] = np.select(conds, [20, 140, 215, 144], default=175)
    df_display['b'] = np.select(conds, [60, 0, 0, 255], default=145)
else:
    vals = df_display[metric_choice]
    vmin, vmax = vals.quantile(0.05), vals.quantile(0.95)
    norm = ((vals - vmin) / (vmax - vmin + 1e-6)).clip(0, 1)
    if metric_choice == "ndwi":
        df_display['r'] = ((1 - norm) * 210).astype(int)
        df_display['g'] = (140 + norm * 40).astype(int)
        df_display['b'] = (norm * 255).astype(int)
    else:
        df_display['r'] = (norm * 255).astype(int)
        df_display['g'] = ((1 - norm) * 180).astype(int)
        df_display['b'] = 40

st.title(f"Monitorizare Marea Aral — {selected_year}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Medie Apă (NDWI)", f"{df_year['ndwi'].mean():.3f}")
c2.metric("Temperatură Medie", f"{df_year['lst_c'].mean():.1f} °C")
c3.metric("Anomalii Apă (Secare)", f"{int(df_year['anom_water'].sum())}")
c4.metric("Anomalii Generale (Apă+Temp)", f"{int(df_year['anom_combined'].sum())}")

st.caption("Legendă Status: **Albastru** = Apă | **Portocaliu** = Anomalie Apă (Secare) | **Galben** = Anomalie Termică | **Roșu** = Anomalie Generală (Apă + Temperatură)")

layer = pdk.Layer(
    "ScatterplotLayer",
    id="aral-grid",
    data=df_display,
    get_position='[longitude, latitude]',
    get_fill_color='[r, g, b, 255]',
    get_radius=1550,
    pickable=True,
    auto_highlight=True
)

event = st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=45.2, longitude=59.8, zoom=6.2),
        tooltip={
            "text": "{motiv}\nCoordonate: {latitude}, {longitude}\nNDWI: {ndwi} (Z: {z_ndwi})\nTemp: {lst_c} °C (Z: {z_lst})"
        }
    ),
    on_select="rerun",
    selection_mode="single-object"
)

selected_objects = event.selection.get("objects", {}).get("aral-grid", [])
if selected_objects:
    cell = selected_objects[0]
    st.info(f"**Diagnostic Celulă:** {cell['motiv']} | **Coordonate:** `{cell['latitude']}, {cell['longitude']}` | **NDWI:** `{cell['ndwi']}` (Z: `{cell['z_ndwi']}`) | **Temp:** `{cell['lst_c']} °C` (Z: `{cell['z_lst']}`)")
    
