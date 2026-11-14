use std::{path::PathBuf, sync::Arc};
use axum::{routing::get, Router};
use tower_http::cors::CorsLayer;

mod analytics;
mod anomalies;
mod error;
mod grid;
mod query_parquet;
mod timeseries;

use anomalies::anomalies;
use grid::box_handler;
use query_parquet::load_observations;
use timeseries::timeseries;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let data_path = std::env::var_os("NASA_PARQUET_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let candidates = [
                manifest_dir.join("../data/raw_landsat_grid.parquet"),
                PathBuf::from("data/raw_landsat_grid.parquet"),
                manifest_dir.join("raw_landsat_grid.parquet"),
                PathBuf::from("../data/raw_landsat_grid.parquet"),
                PathBuf::from("raw_landsat_grid.parquet"),
            ];
            for candidate in &candidates {
                if candidate.exists() {
                    return candidate.clone();
                }
            }
            candidates[0].clone()
        });

    println!("Loading dataset from: {}", data_path.display());
    let dataset = Arc::new(load_observations(data_path)?);
    println!(
        "Successfully loaded {} observations across {} unique grid cells",
        dataset.timestamps.len(),
        dataset.unique_coordinates.len()
    );

    let app = Router::new()
        .route("/anomalies", get(anomalies))
        .route("/time-series", get(timeseries))
        .route("/timeseries", get(timeseries))
        .route("/box", get(box_handler))
        .route("/grid", get(box_handler))
        .layer(tower_http::catch_panic::CatchPanicLayer::new())
        .layer(CorsLayer::permissive())
        .with_state(dataset);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("API server listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}
