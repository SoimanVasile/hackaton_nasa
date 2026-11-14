#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="$SCRIPT_DIR/nasa_query/target/release/nasa"

if [ ! -f "$BIN_PATH" ]; then
    echo "Building Rust release binary..."
    cargo build --release --manifest-path "$SCRIPT_DIR/nasa_query/Cargo.toml"
fi

exec "$BIN_PATH"
