#!/usr/bin/env python3
"""The calibration set for the Hailo compiler: real inputs of the network.

    python make_calib.py palla_1.zip --count 64 --out calib.npy

The compiler quantizes the network to 8 bits on the ranges of these inputs:
stacks of three consecutive frames from the exported zips (512 x 288,
9 channels 0-255, as the chip receives them), taken evenly over the clips.
64 are what the compiler uses at optimization level 1 (each is 5 MB as
float: 512 would not fit in a PC's WSL with 4-8 GB).
"""
from __future__ import annotations

import argparse

import numpy as np

from train_tracknet import HEIGHT, WIDTH, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zips", nargs="+")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--out", default="calib.npy")
    args = parser.parse_args()

    samples, images = load(args.zips)
    every = [s for key in sorted(samples) for s in samples[key]]
    if not every:
        raise SystemExit("Nessun fotogramma annotato negli zip.")
    pick = np.linspace(0, len(every) - 1, min(args.count, len(every))).round().astype(int)
    calib = np.zeros((len(pick), HEIGHT, WIDTH, 9), np.uint8)       # NHWC, as the compiler wants
    for n, i in enumerate(pick):
        calib[n] = np.concatenate([images(f) for f in every[i][0]], axis=2)
    np.save(args.out, calib.astype(np.float32))
    print(f"Salvato {args.out}: {len(pick)} ingressi {calib.shape[1:]}")


if __name__ == "__main__":
    main()
