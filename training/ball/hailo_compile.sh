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

# Each step kept on disk: rerun after a stop (memory, power) and the steps
# already done are skipped. Delete the .har files to start over.
# A step left from another network is not one already done: a .har older
# than the ONNX (or the optimized one older than the calibration set) is of
# a previous compilation, and would compile the old network again.
if [ -f "$NAME.har" ] && [ "$ONNX" -nt "$NAME.har" ]; then
    echo "ONNX più recente dei file intermedi: ricomincio da capo."
    rm -f "$NAME.har" "${NAME}_optimized.har" "${NAME}_optimized_tmp.har" "${NAME}_compiled.har"
fi
if [ -f "${NAME}_optimized.har" ] && { [ "$CALIB" -nt "${NAME}_optimized.har" ] || [ "$NAME.har" -nt "${NAME}_optimized.har" ]; }; then
    echo "Calibrazione o rete più recenti della quantizzazione: la rifaccio."
    rm -f "${NAME}_optimized.har" "${NAME}_optimized_tmp.har"
fi
[ -f "$NAME.har" ] || hailo parser onnx "$ONNX" --hw-arch "$ARCH" --net-name "$NAME" -y
# The quantization holds activations of the whole calibration set: with
# little memory (WSL takes half of the PC's) a large swap is what lets it end.
[ -f "${NAME}_optimized.har" ] || hailo optimize "$NAME.har" --hw-arch "$ARCH" --calib-set-path "$CALIB" \
    --model-script "$HERE/ball.alls" --output-har-path "${NAME}_optimized_tmp.har"
[ -f "${NAME}_optimized_tmp.har" ] && mv "${NAME}_optimized_tmp.har" "${NAME}_optimized.har"
hailo compiler "${NAME}_optimized.har" --hw-arch "$ARCH" --output-dir .
mv "$NAME.hef" ball_h8l.hef
echo "Pronto: ball_h8l.hef -> copialo sul Raspberry in weights/"
