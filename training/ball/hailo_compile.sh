#!/usr/bin/env bash
# ONNX -> HEF for the Raspberry Pi's Hailo-8L, with the Hailo Dataflow
# Compiler (WSL2 Ubuntu 22.04, Python 3.10, see README.md).
#
#   ./hailo_compile.sh ball.onnx calib.npy      ->  ball_h8l.hef
set -euo pipefail

ONNX=${1:?uso: hailo_compile.sh ball.onnx calib.npy}
CALIB=${2:?uso: hailo_compile.sh ball.onnx calib.npy}
ARCH=${HAILO_ARCH:-hailo8l}
NAME=ball_tracknet
HERE=$(cd "$(dirname "$0")" && pwd)

hailo parser onnx "$ONNX" --hw-arch "$ARCH" --net-name "$NAME" -y
hailo optimize "$NAME.har" --hw-arch "$ARCH" --calib-set-path "$CALIB" \
    --model-script "$HERE/ball.alls" --output-har-path "${NAME}_optimized.har"
hailo compiler "${NAME}_optimized.har" --hw-arch "$ARCH" --output-dir .
mv "$NAME.hef" ball_h8l.hef
echo "Pronto: ball_h8l.hef -> copialo sul Raspberry in weights/"
