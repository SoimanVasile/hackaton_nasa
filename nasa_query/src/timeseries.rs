use std::sync::Arc;

use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::analytics::{compute_median_mad, compute_robust_z_score, compute_statistics};
use crate::error::AppError;
use crate::query_parquet::{GridDataset, Metrics};

#[derive(Deserialize)]
pub struct TimeseriesQuery {
    #[serde(alias = "lat")]
    pub latitude: f64,
    #[serde(alias = "lon")]
    pub longitude: f64,
    pub metric: Option<String>,
}

#[derive(Serialize)]
pub struct TimeseriesPoint {
    pub timestamp: String,
    pub year: i32,
    pub values: Metrics,
    pub anomaly: Metrics,
    pub z_score: Metrics,
}

#[derive(Serialize)]
pub struct MetricTrend {
    pub metric: String,
    pub mean: f64,
    pub std_dev: f64,
    pub slope: f64,
    pub intercept: f64,
}

#[derive(Serialize)]
pub struct TimeseriesResponse {
    pub requested_lat: f64,
    pub requested_lon: f64,
    pub latitude: f64,
    pub longitude: f64,
    pub target_metric: Option<String>,
    pub count: usize,
    pub statistics: Vec<MetricTrend>,
    pub timeseries: Vec<TimeseriesPoint>,
}

pub async fn timeseries(
    State(dataset): State<Arc<GridDataset>>,
    Query(query): Query<TimeseriesQuery>,
) -> Result<Json<TimeseriesResponse>, AppError> {
    if !query.latitude.is_finite()
        || !query.longitude.is_finite()
        || !(-90.0..=90.0).contains(&query.latitude)
        || !(-180.0..=180.0).contains(&query.longitude)
    {
        return Err(AppError::BadRequest(
            "latitude must be in -90..90 and longitude in -180..180".to_string(),
        ));
    }

    let (nearest_lat, nearest_lon) = dataset
        .find_nearest_coord(query.latitude, query.longitude)
        .ok_or_else(|| AppError::NotFound("Dataset contains no coordinates".to_string()))?;

    let indices = dataset
        .get_row_indices(nearest_lat, nearest_lon)
        .ok_or_else(|| AppError::NotFound("No observations found for coordinates".to_string()))?;

    // Sort indices by year/timestamp chronologically
    let mut sorted_indices = indices.to_vec();
    sorted_indices.sort_by_key(|&idx| dataset.years[idx]);

    let x_years: Vec<f64> = sorted_indices.iter().map(|&i| dataset.years[i] as f64).collect();

    // Compute stats for all 4 metrics
    let metrics_list = ["ndvi", "ndwi", "lst_c", "salinity_index"];
    let mut statistics = Vec::new();

    let mut ndvi_vals = Vec::with_capacity(sorted_indices.len());
    let mut ndwi_vals = Vec::with_capacity(sorted_indices.len());
    let mut lst_vals = Vec::with_capacity(sorted_indices.len());
    let mut sal_vals = Vec::with_capacity(sorted_indices.len());

    for &idx in &sorted_indices {
        ndvi_vals.push(dataset.ndvi[idx]);
        ndwi_vals.push(dataset.ndwi[idx]);
        lst_vals.push(dataset.lst_c[idx]);
        sal_vals.push(dataset.salinity_index[idx]);
    }

    let ndvi_stats = compute_statistics(&x_years, &ndvi_vals).unwrap_or(crate::analytics::StatisticalSummary {
        count: 0,
        mean: 0.0,
        std_dev: 0.0,
        slope: 0.0,
        intercept: 0.0,
    });
    let ndwi_stats = compute_statistics(&x_years, &ndwi_vals).unwrap_or(crate::analytics::StatisticalSummary {
        count: 0,
        mean: 0.0,
        std_dev: 0.0,
        slope: 0.0,
        intercept: 0.0,
    });
    let lst_stats = compute_statistics(&x_years, &lst_vals).unwrap_or(crate::analytics::StatisticalSummary {
        count: 0,
        mean: 0.0,
        std_dev: 0.0,
        slope: 0.0,
        intercept: 0.0,
    });
    let sal_stats = compute_statistics(&x_years, &sal_vals).unwrap_or(crate::analytics::StatisticalSummary {
        count: 0,
        mean: 0.0,
        std_dev: 0.0,
        slope: 0.0,
        intercept: 0.0,
    });
    let ndvi_robust = compute_median_mad(&ndvi_vals).unwrap_or((ndvi_stats.mean, 0.0));
    let ndwi_robust = compute_median_mad(&ndwi_vals).unwrap_or((ndwi_stats.mean, 0.0));
    let lst_robust = compute_median_mad(&lst_vals).unwrap_or((lst_stats.mean, 0.0));
    let sal_robust = compute_median_mad(&sal_vals).unwrap_or((sal_stats.mean, 0.0));

    for m in &metrics_list {
        let stats = match *m {
            "ndvi" => ndvi_stats,
            "ndwi" => ndwi_stats,
            "lst_c" => lst_stats,
            "salinity_index" => sal_stats,
            _ => continue,
        };
        statistics.push(MetricTrend {
            metric: m.to_string(),
            mean: stats.mean,
            std_dev: stats.std_dev,
            slope: stats.slope,
            intercept: stats.intercept,
        });
    }

    // Filter statistics if a specific metric was requested
    let filtered_statistics = if let Some(ref target) = query.metric {
        let normalized = target.to_lowercase();
        statistics
            .into_iter()
            .filter(|s| s.metric == normalized || (s.metric == "lst_c" && normalized == "lst"))
            .collect()
    } else {
        statistics
    };

    let timeseries: Vec<TimeseriesPoint> = sorted_indices
        .iter()
        .map(|&idx| {
            let values = dataset.get_metrics_at(idx);
            let anomaly = Metrics {
                ndwi: values.ndwi - ndwi_stats.mean,
                lst_c: values.lst_c - lst_stats.mean,
                ndvi: values.ndvi - ndvi_stats.mean,
                salinity_index: values.salinity_index - sal_stats.mean,
            };
            let z_score = Metrics {
                ndwi: compute_robust_z_score(values.ndwi, ndwi_robust.0, ndwi_robust.1),
                lst_c: compute_robust_z_score(values.lst_c, lst_robust.0, lst_robust.1),
                ndvi: compute_robust_z_score(values.ndvi, ndvi_robust.0, ndvi_robust.1),
                salinity_index: compute_robust_z_score(values.salinity_index, sal_robust.0, sal_robust.1),
            };

            TimeseriesPoint {
                timestamp: dataset.timestamps[idx].clone(),
                year: dataset.years[idx],
                values,
                anomaly,
                z_score,
            }
        })
        .collect();

    Ok(Json(TimeseriesResponse {
        requested_lat: query.latitude,
        requested_lon: query.longitude,
        latitude: nearest_lat,
        longitude: nearest_lon,
        target_metric: query.metric,
        count: timeseries.len(),
        statistics: filtered_statistics,
        timeseries,
    }))
}
