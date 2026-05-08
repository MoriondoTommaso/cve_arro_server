#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import zarr


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a .npy file to a local Zarr directory.")
    parser.add_argument(
        "input_file",
        help="Path to the input .npy file",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Path to the output .zarr directory (default: ./<input_stem>.zarr)",
    )
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    return Path.cwd() / f"{input_path.stem}.zarr"


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_dir) if args.output_dir else default_output_dir(input_path)

    data = np.load(input_path, mmap_mode="r")
    zarr.save(output_path, data)

    print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
