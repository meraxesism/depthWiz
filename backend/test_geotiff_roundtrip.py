"""Standalone GeoTIFF metadata and read/write round-trip check.

Usage:
    python -m backend.test_geotiff_roundtrip input.tif [output.tif]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from .geotiff_io import read_geotiff, write_dsm


def main() -> int:
    """Read a real sample, print metadata, write a DSM, and verify georeferencing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}-roundtrip.tif")
    raster = read_geotiff(args.input, require_georeferencing=True)
    metadata = raster.metadata
    print(f"CRS: {metadata.crs}")
    print(f"Native bounds: {metadata.bounds}")
    print(f"WGS84 bounds: {metadata.wgs84_bounds}")
    print(f"Bands: {metadata.count}")
    print(f"dtype: {metadata.dtype}; bit depth: {metadata.bit_depth}")
    print(f"Shape: {raster.data.shape}; RGB model input: {raster.rgb8.shape}")
    elevations = raster.data[0].astype(np.float32)
    write_dsm(output, elevations, metadata)
    with rasterio.open(args.input) as original, rasterio.open(output) as written:
        original_bounds = original.bounds
        assert written.crs == metadata.crs
        assert written.transform == metadata.transform
        assert written.bounds == original_bounds
    print(f"Round-trip georeferencing preserved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
