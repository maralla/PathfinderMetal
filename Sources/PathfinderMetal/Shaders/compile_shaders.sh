#!/usr/bin/env bash
set -euo pipefail

# This script compiles all Metal shaders in this Shaders/ directory and places
# the resulting .metallib files into the curated Resources/Shaders folders.
# - Raster:   Resources/Shaders/*.metallib
# - Compute:  Resources/Shaders/d3d11/*.metallib

SHADERS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$(cd "$SHADERS_DIR/.." && pwd)"                      # .../swift/Sources/PathfinderMetal
OUT_BASE="$MODULE_DIR/Resources/Shaders"
OUT_COMPUTE="$OUT_BASE/d3d11"
TMP_DIR="$MODULE_DIR/.metalbuild"

SRC_RASTER="$SHADERS_DIR"
SRC_COMPUTE="$SHADERS_DIR/d3d11"

mkdir -p "$OUT_BASE" "$OUT_COMPUTE" "$TMP_DIR"

# Clean previous .metallib outputs
find "$OUT_BASE" -type f -name '*.metallib' -delete || true

compile_one() {
  local input_metal="$1"
  local out_lib="$2"
  local base
  base="$(basename "$input_metal" .metal)"
  local air="$TMP_DIR/$base.air"
  echo "Compiling: $input_metal -> $out_lib"
  xcrun metal -c "$input_metal" -o "$air"
  xcrun metallib "$air" -o "$out_lib"
}

# Compile raster shaders (*.metal in Shaders/)
shopt -s nullglob
for f in "$SRC_RASTER"/*.metal; do
  base="$(basename "$f" .metal)"
  compile_one "$f" "$OUT_BASE/$base.metallib"
done

# Compile compute shaders (*.metal in Shaders/d3d11/)
for f in "$SRC_COMPUTE"/*.metal; do
  base="$(basename "$f" .metal)"
  compile_one "$f" "$OUT_COMPUTE/$base.metallib"
done
shopt -u nullglob

echo "Done. metallib outputs in:"
echo "  $OUT_BASE"
echo "  $OUT_COMPUTE" 