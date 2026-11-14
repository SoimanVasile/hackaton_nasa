use std::sync::Arc;

use axum::{
    extract::{Query, State},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::analytics::{compute_median_mad, compute_robust_z_score, compute_statistics};
use crate::error::AppError;
use crate::query_parquet::GridDataset;

#[derive(Deserialize)]
pub struct AnomaliesQuery {
    #[serde(alias = "lat")]
    pub latitude: Option<f64>,
    #[serde(alias = "lon")]
    pub longitude: Option<f64>,
    pub metric: Option<String>,
    pub year: Option<i32>,
}

#[derive(Serialize)]
pub struct AnomalyPoint {
    pub timestamp: String,
    pub year: i32,
    pub value: f64,
    pub anomaly: f64,
    pub z_score: f64,
}

#[derive(Serialize)]
pub struct AnomaliesResponse {
    pub requested_lat: f64,
    pub requested_lon: f64,
    pub latitude: f64,
    pub longitude: f64,
    pub metric: String,
    pub mean: f64,
    pub std_dev: f64,
    pub slope: f64,
    pub intercept: f64,
    pub count: usize,
    pub anomalies: Vec<AnomalyPoint>,
}

#[derive(Serialize)]
pub struct GridAnomalyRecord {
    pub lat: f64,
    pub lon: f64,
    pub year: i32,
    pub ndwi: f64,
    pub lst_c: f64,
    pub salinity_index: f64,
    pub z_ndwi: f64,
    pub z_lst: f64,
    pub anom_water: bool,
    pub anom_temp: bool,
    pub anom_combined: bool,
    pub anom_desiccation: bool,
    pub severity: f64,
}

pub enum AnomaliesOutput {
    Point(AnomaliesResponse),
    Grid(Vec<GridAnomalyRecord>),
}

impl IntoResponse for AnomaliesOutput {
    fn into_response(self) -> Response {
        match self {
            AnomaliesOutput::Point(p) => Json(p).into_response(),
            AnomaliesOutput::Grid(g) => Json(g).into_response(),
        }
    }
}

pub async fn anomalies(
    State(dataset): State<Arc<GridDataset>>,
    Query(query): Query<AnomaliesQuery>,
) -> Result<AnomaliesOutput, AppError> {
    // 1. Grid-wide anomalies for a specific year (when coordinates are omitted)
    if query.latitude.is_none() && query.longitude.is_none() {
        let yr = query.year.ok_or_else(|| {
            AppError::BadRequest(
                "Provide either 'year' (e.g. /anomalies?year=2020) or 'latitude' & 'longitude' (e.g. /anomalies?lat=43.5&lon=58.0)".to_string(),
            )
        })?;

        let year_indices = dataset
            .get_year_indices(yr)
            .ok_or_else(|| AppError::NotFound(format!("No observations found for year {yr}")))?;

        let mut records = Vec::with_capacity(year_indices.len());

        for &idx in year_indices {
            let lat = dataset.latitudes[idx];
            let lon = dataset.longitudes[idx];
            let coord_bits = (lat.to_bits(), lon.to_bits());

            let ndwi_val = (dataset.ndwi[idx] * 1000.0).round() / 1000.0;
            let lst_val = (dataset.lst_c[idx] * 10.0).round() / 10.0;
            let sal_val = (dataset.salinity_index[idx] * 1000.0).round() / 1000.0;

            let base = dataset.location_baselines.get(&coord_bits);
            let (z_ndwi, z_lst, anom_water, anom_temp, anom_combined, anom_desiccation, severity) = if let Some(b) = base {
                let z_n = (compute_robust_z_score(dataset.ndwi[idx], b.base_median_ndwi, b.base_mad_ndwi) * 100.0).round() / 100.0;
                let z_l = (compute_robust_z_score(dataset.lst_c[idx], b.base_median_lst, b.base_mad_lst) * 100.0).round() / 100.0;
                let z_salinity = compute_robust_z_score(
                    dataset.salinity_index[idx],
                    b.base_median_salinity,
                    b.base_mad_salinity,
                );
                let prior_water = dataset.location_index.get(&coord_bits).is_some_and(|indices| {
                    indices.iter().any(|&history_idx| {
                        dataset.ndwi[history_idx] > 0.0
                            && (dataset.years[history_idx] < yr
                                || (1984..=1992).contains(&dataset.years[history_idx]))
                    })
                });
                let ever_water = b.was_water || prior_water || z_n < -1.5;
                let lost_water = ever_water && dataset.ndwi[idx] <= 0.0;
                let a_d = dataset.ndwi[idx] < -0.05 || z_salinity > 3.0;
                let a_w = (z_n < -2.0) || lost_water;
                let a_t = z_l > 2.0;
                let a_c = lost_water
                    && (z_l > 1.5 || dataset.lst_c[idx] > b.base_median_lst);
                let sev = ((z_n.abs() + z_l.max(0.0)) * 100.0).round() / 100.0;
                (z_n, z_l, a_w, a_t, a_c, a_d, sev)
            } else {
                (0.0, 0.0, false, false, false, dataset.ndwi[idx] < -0.05, 0.0)
            };

            records.push(GridAnomalyRecord {
                lat,
                lon,
                year: yr,
                ndwi: ndwi_val,
                lst_c: lst_val,
                salinity_index: sal_val,
                z_ndwi,
                z_lst,
                anom_water,
                anom_temp,
                anom_combined,
                anom_desiccation,
                severity,
            });
        }

        return Ok(AnomaliesOutput::Grid(records));
    }

    // 2. Point time-series anomaly query
    let (req_lat, req_lon) = match (query.latitude, query.longitude) {
        (Some(lat), Some(lon)) => {
            if !lat.is_finite()
                || !lon.is_finite()
                || !(-90.0..=90.0).contains(&lat)
                || !(-180.0..=180.0).contains(&lon)
            {
                return Err(AppError::BadRequest(
                    "latitude must be in -90..90 and longitude in -180..180".to_string(),
                ));
            }
            (lat, lon)
        }
        _ => {
            return Err(AppError::BadRequest(
                "Both latitude and longitude are required when querying point anomalies".to_string(),
            ));
        }
    };

    let metric_name = query.metric.unwrap_or_else(|| "ndvi".to_string()).to_lowercase();

    let (nearest_lat, nearest_lon) = dataset
        .find_nearest_coord(req_lat, req_lon)
        .ok_or_else(|| AppError::NotFound("Dataset contains no coordinates".to_string()))?;

    let indices = dataset
        .get_row_indices(nearest_lat, nearest_lon)
        .ok_or_else(|| AppError::NotFound("No observations found for coordinates".to_string()))?;

    let mut sorted_indices = indices.to_vec();
    sorted_indices.sort_by_key(|&idx| dataset.years[idx]);

    let x_years: Vec<f64> = sorted_indices.iter().map(|&i| dataset.years[i] as f64).collect();
    let y_vals = dataset
        .get_metric_values(&metric_name, &sorted_indices)
        .ok_or_else(|| {
            AppError::BadRequest(format!(
                "Invalid metric '{}'. Supported metrics: ndwi, lst_c, ndvi, salinity_index",
                metric_name
            ))
        })?;

    let stats = compute_statistics(&x_years, &y_vals).ok_or_else(|| {
        AppError::Internal(anyhow::anyhow!("Insufficient data to compute statistics"))
    })?;
    let (robust_median, robust_mad) = compute_median_mad(&y_vals).unwrap_or((stats.mean, 0.0));

    let mut anomalies = Vec::new();
    for (i, &idx) in sorted_indices.iter().enumerate() {
        let yr = dataset.years[idx];
        if let Some(target_yr) = query.year {
            if yr != target_yr {
                continue;
            }
        }

        let val = y_vals[i];
        let anomaly = val - stats.mean;
        let z_score = compute_robust_z_score(val, robust_median, robust_mad);

        anomalies.push(AnomalyPoint {
            timestamp: dataset.timestamps[idx].clone(),
            year: yr,
            value: val,
            anomaly,
            z_score,
        });
    }

    Ok(AnomaliesOutput::Point(AnomaliesResponse {
        requested_lat: req_lat,
        requested_lon: req_lon,
        latitude: nearest_lat,
        longitude: nearest_lon,
        metric: metric_name,
        mean: stats.mean,
        std_dev: stats.std_dev,
        slope: stats.slope,
        intercept: stats.intercept,
        count: anomalies.len(),
        anomalies,
    }))
}
