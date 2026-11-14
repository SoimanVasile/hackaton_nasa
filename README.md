# 🛰️ Aral Trend Detective (Earth System Detective)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apache Arrow](https://img.shields.io/badge/Apache%20Arrow-60.1-red.svg)](https://arrow.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B.svg)](https://streamlit.io/)

A high-performance geospatial analytics engine and 3D visual intelligence platform developed for the **NASA Hackathon**. 

**Aral Trend Detective** ingests over **40 years of continuous satellite observations (1984–2026)** across the Aral Sea basin, utilizing a custom **zero-copy Rust server** built on Apache Arrow/Parquet and an interactive **Streamlit + PyDeck** geospatial dashboard to track and diagnose ecological desiccation, thermal stress, and salinization in real time.

---

## Key Features

* * Ultra-High Throughput Rust Engine (`nasa_query`):**
  * Built with **Axum**, **Tokio**, and **Apache Arrow / Parquet (`arrow-rs` v60)**.
  * In-memory columnar storage (`GridDataset`) managing **446,494 observations** across **10,302 unique 8km × 8km grid cells**.
  * Pre-computed spatial and temporal indices providing **sub-millisecond ($O(1)$) lookups** without per-request allocations.
  * Native SIMD-friendly vector operations over primitive contiguous float slices (`&[f64]`).

* ** Rigorous Statistical & Anomaly Engine:**
  * **Ordinary Least Squares (OLS) Linear Regression:** Calculates long-term trendline slopes ($\beta$) to measure decades of gradual surface warming and drying.
  * **Z-Score Normalization:** Evaluates deviations against the historical reference period (**1984–1992**, before acute Aral desiccation accelerated).
  * **Multi-Hazard Anomaly Classification:**
    * **Water Desiccation (`anom_water`):** Identifies severe NDWI drops ($Z_{\text{ndwi}} \le -2.0$) or historical water loss.
    * **Thermal Shock (`anom_temp`):** Flags extreme Land Surface Temperature spikes ($Z_{\text{lst}} \ge 1.8$).
    * **Combined Crisis (`anom_combined`):** Pinpoints zones suffering simultaneous desiccation and heat shock.
    * **Compound Severity:** Computes cumulative stress index ($|Z_{\text{ndwi}}| + \max(0, Z_{\text{lst}})$).

* ** 3D Geospatial Interactive Map (`frontend`):**
  * Rendered with **Streamlit** and **PyDeck (Deck.gl)**.
  * Dynamic timeline slider from **1984 to 2026**.
  * Multi-mode color coding: status diagnostics, water index (NDWI), temperature (LST °C), and soil salinity.
  * Single-cell click inspection tooltips for localized diagnostics.

* ** Harmonized Multi-Sensor Data Ingestion (`backend`):**
  * Automated Google Earth Engine (GEE) pipeline harmonizing **Landsat 4, 5, 7, 8, and 9 Collection 2 Level-2 Surface Reflectance / Surface Temperature**.

---

##  Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │          Google Earth Engine (GEE)           │
                       │    Landsat 4/5/7/8/9 Collection 2 Tier 1     │
                       └──────────────────────┬───────────────────────┘
                                              │ Ingestion & Harmonization
                                              ▼
                               ┌─────────────────────────────┐
                               │  data/raw_landsat_grid      │
                               │     (.parquet columnar)     │
                               └──────────────┬──────────────┘
                                              │ Fast Zero-Copy Mmap / Scan
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                        Rust Engine (nasa_query :8080)                           │
    │  - Apache Arrow In-Memory Columnar Arrays                                       │
    │  - Spatial Grid Pre-Index: (lat, lon) -> Row Slices                             │
    │  - Reference Baseline (1984–1992): μ, σ, OLS Slope, Z-Scores                    │
    │  - High-Throughput REST API (Axum + Tokio + Tower CORS)                         │
    └────────────────────────────────────────┬────────────────────────────────────────┘
                                             │ JSON (<5ms latency)
                                             ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                      Streamlit + PyDeck Dashboard (:8501)                       │
    │  - 3D Scatterplot Layer (10,302 interactive cells)                              │
    │  - Anomaly Heatmaps, Severity Scoring, Historical Time-Series Visualizer        │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

---

##  Quick Start (One Command)

### Prerequisites

* [Rust & Cargo](https://rustup.rs/) (v1.85+)
* Python 3.10+

### Run All Services

Clone the repository and run the unified startup script:

```bash
git clone https://github.com/SoimanVasile/hackaton_nasa.git
cd hackaton_nasa

# Start both the Rust engine (:8080) and Streamlit dashboard (:8501)
./run.sh
```

The script will:
1. Automatically compile the optimized release binary of the Rust engine.
2. Launch the Rust Axum backend on `http://localhost:8080`.
3. Wait for the engine to initialize and verify its health.
4. Launch the Streamlit frontend on `http://localhost:8501`.

Open **`http://localhost:8501`** in your browser to explore the dashboard.

---

##  REST API Reference

The Rust backend serves on `http://localhost:8080` (CORS enabled):

### 1. `GET /anomalies`
Calculates statistical anomalies and threshold triggers.

* **Mode A: Full Regional Map for a Given Year (Used by Frontend)**
  ```bash
  curl -s "http://localhost:8080/anomalies?year=2020" | jq .
  ```
  *Response:* Array of 10,500+ cells containing coordinates, metrics, z-scores, and anomaly flags (`anom_water`, `anom_temp`, `anom_combined`, `severity`).

* **Mode B: Single-Point Historical Anomaly**
  ```bash
  curl -s "http://localhost:8080/anomalies?lat=43.5&lon=58.0&metric=ndvi" | jq .
  ```
  *Response:* Historical anomalies and z-scores for the given coordinate over time.

---

### 2. `GET /time-series`
Returns full 43-year historical time series and Ordinary Least Squares (OLS) linear regression trendlines for a given point.

```bash
curl -s "http://localhost:8080/time-series?lat=43.3886&lon=58.0132&metric=ndvi" | jq .
```

*Example Response:*
```json
{
  "requested_lat": 43.3886,
  "requested_lon": 58.0132,
  "latitude": 43.3886,
  "longitude": 58.0132,
  "target_metric": "ndvi",
  "count": 43,
  "statistics": [
    {
      "metric": "ndvi",
      "mean": 0.061368,
      "std_dev": 0.011804,
      "slope": 0.0002695,
      "intercept": -0.47903
    }
  ],
  "timeseries": [ ... ]
}
```

---

### 3. `GET /box` (or `GET /grid`)
Retrieves all grid observations for a specific year, with optional bounding-box spatial cropping.

```bash
# Full dataset bounding box for 2020:
curl -s "http://localhost:8080/box?year=2020" | jq .

# Crop to a specific sub-bounding box (min_lon, min_lat, max_lon, max_lat):
curl -s "http://localhost:8080/box?year=2020&bbox=58.0,43.0,59.0,44.0" | jq .
```

---

##  Scientific & Mathematical Formulation

| Metric | Formulation | Description |
| :--- | :--- | :--- |
| **NDWI** | $\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$ | Normalized Difference Water Index (McFeeters). Values $> 0$ indicate open water. |
| **LST (°C)** | Radiometric calibration of thermal infrared bands | Land Surface Temperature in degrees Celsius. |
| **NDVI** | $\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$ | Normalized Difference Vegetation Index. |
| **Salinity Index** | $\frac{\text{SWIR1} - \text{NIR}}{\text{SWIR1} + \text{NIR}}$ | Soil and dry lakebed salinization index. |
| **Z-Score** | $Z = \frac{X - \mu_{\text{baseline}}}{\sigma_{\text{baseline}} + \epsilon}$ | Normalized anomaly relative to reference baseline (1984–1992). |
| **Linear Trend** | $\beta = \frac{\sum (t_i - \bar{t})(y_i - \bar{y})}{\sum (t_i - \bar{t})^2}$ | Ordinary Least Squares (OLS) slope of annual change over 43 years. |

---

##  Repository Structure

```
hackaton_nasa/
├── backend/
│   └── ingest_gee.py            # Google Earth Engine data ingestion & preprocessing
├── data/
│   └── raw_landsat_grid.parquet # 446k rows of harmonized Landsat satellite data
├── frontend/
│   └── app.py                   # Streamlit + PyDeck 3D dashboard
├── nasa_query/                  # High-performance Rust backend engine
│   ├── Cargo.toml               # Rust dependencies (Axum, Arrow, Parquet, Tokio)
│   └── src/
│       ├── analytics.rs         # Mathematical engine (OLS slope, std dev, z-score)
│       ├── anomalies.rs         # Anomaly classification & grid evaluation
│       ├── error.rs             # Typed HTTP error handling
│       ├── grid.rs              # Bounding-box spatial queries
│       ├── main.rs              # Axum server configuration & routing
│       ├── query_parquet.rs     # Columnar Apache Arrow dataset & spatial index
│       └── timeseries.rs        # Point time-series calculation
├── run.sh                       # Unified launcher script (starts Rust + Streamlit)
├── requirements.txt             # Python frontend dependencies
└── README.md
```

---

##  Authors & Team

Developed with pride for the **NASA Space Apps Hackathon** by **Soiman Vasile** and team.

* **License:** [MIT License](LICENSE)
