#!/usr/bin/env bash
# Script to pull and run Vaxign-ML Docker container for antigenicity prediction
set -e

INPUT_FASTA=${1:-"data/input.fa"}
OUTPUT_DIR=${2:-"data/output"}
ORGANISM=${3:-"bacteria"}

if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed or not in PATH."
    exit 1
fi

echo "[*] Pulling Vaxign-ML Docker image..."
docker pull e4ong1031/vaxign-ml:v1.0

INPUT_DIR=$(dirname "$(realpath "$INPUT_FASTA")")
INPUT_FILE=$(basename "$INPUT_FASTA")

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_REAL=$(realpath "$OUTPUT_DIR")

echo "[*] Running Vaxign-ML container on $INPUT_FILE..."
docker run --rm \
    -v "$INPUT_DIR:/input" \
    -v "$OUTPUT_DIR_REAL:/output" \
    e4ong1031/vaxign-ml:v1.0 \
    /input/"$INPUT_FILE" /output "$ORGANISM"

echo "[+] Vaxign-ML completed. Results saved in $OUTPUT_DIR"
