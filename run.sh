#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="$SCRIPT_DIR/nasa_query/target/release/nasa"
VENV_STREAMLIT="$SCRIPT_DIR/venv/bin/streamlit"

# 1. Build Rust release binary if missing
if [ ! -f "$BIN_PATH" ]; then
    echo "[run.sh] Building Rust release binary..."
    cargo build --release --manifest-path "$SCRIPT_DIR/nasa_query/Cargo.toml"
fi

# 2. Start Rust Engine in the background
echo "[run.sh] Starting Rust geospatial engine (Axum) on port 8080..."
"$BIN_PATH" &
RUST_PID=$!

cleanup() {
    echo ""
    echo "[run.sh] Shutting down Rust engine (PID: $RUST_PID)..."
    kill "$RUST_PID" 2>/dev/null || true
    wait "$RUST_PID" 2>/dev/null || true
    echo "[run.sh] All services stopped."
}
trap cleanup EXIT INT TERM

# 3. Wait for Rust engine to be healthy and responding
echo "[run.sh] Waiting for Rust engine to initialize dataset..."
for i in {1..30}; do
    if curl -s -f -o /dev/null "http://127.0.0.1:8080/time-series?lat=43.5&lon=58.0"; then
        echo "[run.sh] Rust engine is live at http://localhost:8080"
        break
    fi
    sleep 0.3
done

# 4. Start the Streamlit Frontend Server
echo "[run.sh] Starting Streamlit dashboard server on http://localhost:8501..."
if [ -f "$VENV_STREAMLIT" ]; then
    "$VENV_STREAMLIT" run "$SCRIPT_DIR/frontend/app.py" --server.headless true
elif command -v streamlit >/dev/null 2>&1; then
    streamlit run "$SCRIPT_DIR/frontend/app.py" --server.headless true
else
    echo "[run.sh] Error: streamlit not found in venv or PATH. Install it with: pip install -r requirements.txt"
    exit 1
fi
