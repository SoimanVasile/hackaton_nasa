use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{anyhow, Context};
use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub ndwi: f64,
    pub lst_c: f64,
    pub ndvi: f64,
    pub salinity_index: f64,
}

#[allow(dead_code)]
impl Metrics {
    pub fn new() -> Self {
        Self {
            ndwi: 0.0,
            lst_c: 0.0,
            ndvi: 0.0,
            salinity_index: 0.0,
        }
    }

    pub fn get(&self, metric: &str) -> Option<f64> {
        match metric.to_lowercase().as_str() {
            "ndwi" => Some(self.ndwi),
            "lst_c" | "lst" => Some(self.lst_c),
            "ndvi" => Some(self.ndvi),
            "salinity_index" | "salinity" => Some(self.salinity_index),
            _ => None,
        }
    }
}

/// Helper function to safely extract and downcast a Float64Array from a RecordBatch.
pub fn extract_metric_values<'a>(batch: &'a RecordBatch, column_name: &str) -> anyhow::Result<&'a Float64Array> {
    batch
        .column_by_name(column_name)
        .ok_or_else(|| anyhow!("column '{column_name}' not found in RecordBatch"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow!("column '{column_name}' could not be downcast to Float64Array"))
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocationBaseline {
    pub mean: Metrics,
    pub std_dev: Metrics,
    pub base_mean_ndwi: f64,
    pub base_std_ndwi: f64,
    pub base_mean_lst: f64,
    pub base_std_lst: f64,
    pub was_water: bool,
}

/// Columnar in-memory dataset optimized for zero-copy spatial queries and time-series extraction.
pub struct GridDataset {
    pub timestamps: Vec<String>,
    pub years: Vec<i32>,
    pub latitudes: Vec<f64>,
    pub longitudes: Vec<f64>,
    pub ndwi: Vec<f64>,
    pub lst_c: Vec<f64>,
    pub ndvi: Vec<f64>,
    pub salinity_index: Vec<f64>,

    /// Map (lat.to_bits(), lon.to_bits()) -> Vec<row_index>
    pub location_index: HashMap<(u64, u64), Vec<usize>>,
    /// Pre-indexed unique coordinates for O(K) nearest neighbor search (K ≈ 10,000)
    pub unique_coordinates: Vec<(f64, f64)>,
    /// Map year -> Vec<row_index> for instant O(1) year/box queries
    pub year_index: HashMap<i32, Vec<usize>>,
    /// Precomputed baseline (mean, std_dev) per location for instant full-grid anomaly calculation
    pub location_baselines: HashMap<(u64, u64), LocationBaseline>,
}

impl GridDataset {
    pub fn load_from_parquet(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open parquet file at '{}'", path.as_ref().display()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()
            .context("Failed to build ParquetRecordBatchReader")?;

        let mut timestamps = Vec::new();
        let mut years = Vec::new();
        let mut latitudes = Vec::new();
        let mut longitudes = Vec::new();
        let mut ndwi = Vec::new();
        let mut lst_c = Vec::new();
        let mut ndvi = Vec::new();
        let mut salinity_index = Vec::new();

        for maybe_batch in reader {
            let batch = maybe_batch.context("Error reading RecordBatch from Parquet")?;
            let num_rows = batch.num_rows();

            // Downcast columns with proper error messages
            let ts_col = batch
                .column_by_name("timestamp")
                .ok_or_else(|| anyhow!("missing 'timestamp' column"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("'timestamp' column is not a StringArray"))?;

            let yr_col = batch
                .column_by_name("year")
                .ok_or_else(|| anyhow!("missing 'year' column"))?
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("'year' column is not an Int64Array"))?;

            let lat_col = extract_metric_values(&batch, "latitude")?;
            let lon_col = extract_metric_values(&batch, "longitude")?;
            let ndwi_col = extract_metric_values(&batch, "ndwi")?;
            let lst_col = extract_metric_values(&batch, "lst_c")?;
            let ndvi_col = extract_metric_values(&batch, "ndvi")?;
            let sal_col = extract_metric_values(&batch, "salinity_index")?;

            for i in 0..num_rows {
                // Safeguard against null entries in Arrow buffers
                if ts_col.is_null(i) || lat_col.is_null(i) || lon_col.is_null(i) {
                    continue;
                }

                timestamps.push(ts_col.value(i).to_string());
                years.push(yr_col.value(i) as i32);
                latitudes.push(lat_col.value(i));
                longitudes.push(lon_col.value(i));
                ndwi.push(ndwi_col.value(i));
                lst_c.push(lst_col.value(i));
                ndvi.push(ndvi_col.value(i));
                salinity_index.push(sal_col.value(i));
            }
        }

        // Build spatial index and year index at startup
        let mut location_index: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
        let mut year_index: HashMap<i32, Vec<usize>> = HashMap::new();

        for (idx, ((&lat, &lon), &yr)) in latitudes.iter().zip(longitudes.iter()).zip(years.iter()).enumerate() {
            location_index
                .entry((lat.to_bits(), lon.to_bits()))
                .or_default()
                .push(idx);
            year_index
                .entry(yr)
                .or_default()
                .push(idx);
        }

        let unique_coordinates: Vec<(f64, f64)> = location_index
            .keys()
            .map(|&(lat_bits, lon_bits)| (f64::from_bits(lat_bits), f64::from_bits(lon_bits)))
            .collect();

        // Precompute baselines for each unique location for instant full-grid anomaly calculations
        let mut location_baselines: HashMap<(u64, u64), LocationBaseline> = HashMap::with_capacity(location_index.len());
        for (&coord_bits, row_idxs) in &location_index {
            let n = row_idxs.len() as f64;
            if n == 0.0 {
                continue;
            }
            let (mut sum_ndwi, mut sum_lst, mut sum_ndvi, mut sum_sal) = (0.0, 0.0, 0.0, 0.0);
            for &i in row_idxs {
                sum_ndwi += ndwi[i];
                sum_lst += lst_c[i];
                sum_ndvi += ndvi[i];
                sum_sal += salinity_index[i];
            }
            let mean = Metrics {
                ndwi: sum_ndwi / n,
                lst_c: sum_lst / n,
                ndvi: sum_ndvi / n,
                salinity_index: sum_sal / n,
            };

            let denom = if row_idxs.len() > 1 { (row_idxs.len() - 1) as f64 } else { 1.0 };
            let (mut var_ndwi, mut var_lst, mut var_ndvi, mut var_sal) = (0.0, 0.0, 0.0, 0.0);
            for &i in row_idxs {
                var_ndwi += (ndwi[i] - mean.ndwi).powi(2);
                var_lst += (lst_c[i] - mean.lst_c).powi(2);
                var_ndvi += (ndvi[i] - mean.ndvi).powi(2);
                var_sal += (salinity_index[i] - mean.salinity_index).powi(2);
            }
            let std_dev = Metrics {
                ndwi: (var_ndwi / denom).sqrt(),
                lst_c: (var_lst / denom).sqrt(),
                ndvi: (var_ndvi / denom).sqrt(),
                salinity_index: (var_sal / denom).sqrt(),
            };

            // Historical reference baseline 1984-1992 (Aral Sea baseline period)
            let early_rows: Vec<usize> = row_idxs.iter().copied().filter(|&i| years[i] <= 1992).collect();
            let (base_mean_ndwi, base_std_ndwi, base_mean_lst, base_std_lst) = if !early_rows.is_empty() {
                let e_n = early_rows.len() as f64;
                let e_sum_ndwi: f64 = early_rows.iter().map(|&i| ndwi[i]).sum();
                let e_sum_lst: f64 = early_rows.iter().map(|&i| lst_c[i]).sum();
                let m_ndwi = e_sum_ndwi / e_n;
                let m_lst = e_sum_lst / e_n;
                let e_denom = if early_rows.len() > 1 { (early_rows.len() - 1) as f64 } else { 1.0 };
                let var_e_ndwi: f64 = early_rows.iter().map(|&i| (ndwi[i] - m_ndwi).powi(2)).sum();
                let var_e_lst: f64 = early_rows.iter().map(|&i| (lst_c[i] - m_lst).powi(2)).sum();
                let s_ndwi = if early_rows.len() > 1 { (var_e_ndwi / e_denom).sqrt() } else { 0.05 };
                let s_lst = if early_rows.len() > 1 { (var_e_lst / e_denom).sqrt() } else { 2.0 };
                (m_ndwi, s_ndwi, m_lst, s_lst)
            } else {
                (mean.ndwi, std_dev.ndwi, mean.lst_c, std_dev.lst_c)
            };

            let was_water = base_mean_ndwi > 0.0;

            location_baselines.insert(coord_bits, LocationBaseline {
                mean,
                std_dev,
                base_mean_ndwi,
                base_std_ndwi,
                base_mean_lst,
                base_std_lst,
                was_water,
            });
        }

        Ok(Self {
            timestamps,
            years,
            latitudes,
            longitudes,
            ndwi,
            lst_c,
            ndvi,
            salinity_index,
            location_index,
            unique_coordinates,
            year_index,
            location_baselines,
        })
    }

    /// Computes dataset global bounding box: (min_lat, max_lat, min_lon, max_lon)
    pub fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        for &(lat, lon) in &self.unique_coordinates {
            if lat < min_lat { min_lat = lat; }
            if lat > max_lat { max_lat = lat; }
            if lon < min_lon { min_lon = lon; }
            if lon > max_lon { max_lon = lon; }
        }
        (min_lat, max_lat, min_lon, max_lon)
    }

    /// Returns row indices for a specific year in O(1) time.
    pub fn get_year_indices(&self, year: i32) -> Option<&[usize]> {
        self.year_index.get(&year).map(|v| v.as_slice())
    }

    /// Finds the nearest grid point using Euclidean distance over the pre-indexed unique coordinates.
    pub fn find_nearest_coord(&self, lat: f64, lon: f64) -> Option<(f64, f64)> {
        self.unique_coordinates
            .iter()
            .min_by(|(la1, lo1), (la2, lo2)| {
                let d1 = (la1 - lat).powi(2) + (lo1 - lon).powi(2);
                let d2 = (la2 - lat).powi(2) + (lo2 - lon).powi(2);
                d1.total_cmp(&d2)
            })
            .copied()
    }

    /// Returns row indices for a specific coordinate in O(1) time.
    pub fn get_row_indices(&self, lat: f64, lon: f64) -> Option<&[usize]> {
        self.location_index
            .get(&(lat.to_bits(), lon.to_bits()))
            .map(|v| v.as_slice())
    }

    /// Extracts metric values for the given row indices without copying unnecessary columns.
    pub fn get_metric_values(&self, metric: &str, row_indices: &[usize]) -> Option<Vec<f64>> {
        let col = match metric.to_lowercase().as_str() {
            "ndwi" => &self.ndwi,
            "lst_c" | "lst" => &self.lst_c,
            "ndvi" => &self.ndvi,
            "salinity_index" | "salinity" => &self.salinity_index,
            _ => return None,
        };

        Some(row_indices.iter().map(|&idx| col[idx]).collect())
    }

    /// Returns Metrics struct for a specific row index.
    pub fn get_metrics_at(&self, idx: usize) -> Metrics {
        Metrics {
            ndwi: self.ndwi[idx],
            lst_c: self.lst_c[idx],
            ndvi: self.ndvi[idx],
            salinity_index: self.salinity_index[idx],
        }
    }
}

/// Convenience alias for dataset loading matching the original signature.
pub fn load_observations(path: impl AsRef<Path>) -> anyhow::Result<GridDataset> {
    GridDataset::load_from_parquet(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parquet_loading_and_spatial_query() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = [
            manifest_dir.join("raw_landsat_grid.parquet"),
            manifest_dir.join("../data/raw_landsat_grid.parquet"),
            PathBuf::from("data/raw_landsat_grid.parquet"),
        ]
        .into_iter()
        .find(|p| p.exists());

        let Some(path) = path else {
            return;
        };
        let dataset = load_observations(&path).expect("Failed to load dataset");
        assert_eq!(dataset.timestamps.len(), 446494);
        assert_eq!(dataset.unique_coordinates.len(), 10302);

        // Test nearest coord
        let coord = dataset.find_nearest_coord(35.0, -115.0).expect("nearest coord found");
        let indices = dataset.get_row_indices(coord.0, coord.1).expect("indices found");
        assert!(!indices.is_empty());

        let ndvi = dataset.get_metric_values("ndvi", indices).expect("metric values extracted");
        assert_eq!(ndvi.len(), indices.len());
    }
}

