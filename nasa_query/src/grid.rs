use std::sync::Arc;

use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::analytics::compute_robust_z_score;
use crate::error::AppError;
use crate::query_parquet::{GridDataset, Metrics};

#[derive(Deserialize)]
pub struct BoxQuery {
    pub year: i32,
    pub metric: Option<String>,
    pub min_lat: Option<f64>,
    pub max_lat: Option<f64>,
    pub min_lon: Option<f64>,
    pub max_lon: Option<f64>,
    /// Bounding box formatted as "min_lon,min_lat,max_lon,max_lat" (standard GIS order)
    pub bbox: Option<String>,
}

#[derive(Serialize)]
pub struct GridCell {
    pub timestamp: String,
    pub latitude: f64,
    pub longitude: f64,
    pub values: Metrics,
    pub anomaly: Metrics,
    pub z_score: Metrics,
}

#[derive(Serialize)]
pub struct BoundingBox {
    pub min_latitude: f64,
    pub max_latitude: f64,
    pub min_longitude: f64,
    pub max_longitude: f64,
}

#[derive(Serialize)]
pub struct BoxResponse {
    pub year: i32,
    pub metric_filter: Option<String>,
    pub bbox: BoundingBox,
    pub count: usize,
    pub cells: Vec<GridCell>,
}

pub async fn box_handler(
    State(dataset): State<Arc<GridDataset>>,
    Query(query): Query<BoxQuery>,
) -> Result<Json<BoxResponse>, AppError> {
    let year_indices = dataset
        .get_year_indices(query.year)
        .ok_or_else(|| {
            AppError::NotFound(format!(
                "No observations available for year {}. Available years: 1984–2026",
                query.year
            ))
        })?;

    // Determine query bounding box (defaults to entire dataset box if omitted)
    let (global_min_lat, global_max_lat, global_min_lon, global_max_lon) = dataset.bounding_box();

    let mut q_min_lat = query.min_lat.unwrap_or(global_min_lat);
    let mut q_max_lat = query.max_lat.unwrap_or(global_max_lat);
    let mut q_min_lon = query.min_lon.unwrap_or(global_min_lon);
    let mut q_max_lon = query.max_lon.unwrap_or(global_max_lon);

    if let Some(ref bbox_str) = query.bbox {
        let parts: Vec<&str> = bbox_str.split(',').map(|s| s.trim()).collect();
        if parts.len() == 4 {
            if let (Ok(min_lon), Ok(min_lat), Ok(max_lon), Ok(max_lat)) = (
                parts[0].parse::<f64>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
                parts[3].parse::<f64>(),
            ) {
                q_min_lon = min_lon;
                q_min_lat = min_lat;
                q_max_lon = max_lon;
                q_max_lat = max_lat;
            } else {
                return Err(AppError::BadRequest(
                    "Invalid bbox format. Expected 'min_lon,min_lat,max_lon,max_lat'".to_string(),
                ));
            }
        } else {
            return Err(AppError::BadRequest(
                "Invalid bbox parameter. Expected 4 comma-separated numbers: 'min_lon,min_lat,max_lon,max_lat'".to_string(),
            ));
        }
    }

    if q_min_lat > q_max_lat || q_min_lon > q_max_lon {
        return Err(AppError::BadRequest(
            "Invalid bounding box: min coordinates must not exceed max coordinates".to_string(),
        ));
    }

    let mut cells = Vec::with_capacity(year_indices.len());

    for &idx in year_indices {
        let lat = dataset.latitudes[idx];
        let lon = dataset.longitudes[idx];

        if lat < q_min_lat || lat > q_max_lat || lon < q_min_lon || lon > q_max_lon {
            continue;
        }

        let values = dataset.get_metrics_at(idx);
        let coord_bits = (lat.to_bits(), lon.to_bits());

        let (anomaly, z_score) = if let Some(baseline) = dataset.location_baselines.get(&coord_bits) {
            let a = Metrics {
                ndwi: values.ndwi - baseline.mean.ndwi,
                lst_c: values.lst_c - baseline.mean.lst_c,
                ndvi: values.ndvi - baseline.mean.ndvi,
                salinity_index: values.salinity_index - baseline.mean.salinity_index,
            };
            let z = Metrics {
                ndwi: compute_robust_z_score(values.ndwi, baseline.median.ndwi, baseline.mad.ndwi),
                lst_c: compute_robust_z_score(values.lst_c, baseline.median.lst_c, baseline.mad.lst_c),
                ndvi: compute_robust_z_score(values.ndvi, baseline.median.ndvi, baseline.mad.ndvi),
                salinity_index: compute_robust_z_score(values.salinity_index, baseline.median.salinity_index, baseline.mad.salinity_index),
            };
            (a, z)
        } else {
            (Metrics::new(), Metrics::new())
        };

        cells.push(GridCell {
            timestamp: dataset.timestamps[idx].clone(),
            latitude: lat,
            longitude: lon,
            values,
            anomaly,
            z_score,
        });
    }

    Ok(Json(BoxResponse {
        year: query.year,
        metric_filter: query.metric,
        bbox: BoundingBox {
            min_latitude: q_min_lat,
            max_latitude: q_max_lat,
            min_longitude: q_min_lon,
            max_longitude: q_max_lon,
        },
        count: cells.len(),
        cells,
    }))
}
