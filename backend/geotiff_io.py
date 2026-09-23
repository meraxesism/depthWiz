"""GeoTIFF reading, RGB preparation, validation, and DSM writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import ColorInterp

LOGGER = logging.getLogger(__name__)


class GeoTIFFError(ValueError):
    """Base error for invalid or unsupported raster inputs."""


class NotGeoreferencedError(GeoTIFFError):
    """Raised when a raster lacks both a CRS and a usable affine transform."""


class UnsupportedRasterError(GeoTIFFError):
    """Raised when a raster cannot provide the required pixel representation."""


@dataclass(frozen=True)
class RasterMetadata:
    """Metadata extracted from a rasterio dataset."""

    path: Path
    width: int
    height: int
    count: int
    dtype: str
    bit_depth: int
    crs: CRS | None
    transform: Affine
    bounds: tuple[float, float, float, float]
    wgs84_bounds: tuple[float, float, float, float] | None
    color_interpretation: tuple[str, ...]
    nodata: float | None


@dataclass(frozen=True)
class GeoTIFFData:
    """Original raster values, normalized RGB model input, and metadata."""

    data: np.ndarray
    rgb8: np.ndarray
    metadata: RasterMetadata


def _metadata(dataset: rasterio.io.DatasetReader, path: Path) -> RasterMetadata:
    """Build metadata using rasterio dataset properties and CRS transforms."""
    crs = dataset.crs
    bounds = tuple(float(value) for value in dataset.bounds)
    wgs84_bounds = None
    if crs is not None:
        wgs84_bounds = tuple(
            float(value)
            for value in transform_bounds(crs, CRS.from_epsg(4326), *bounds, densify_pts=21)
        )
    dtype = np.dtype(dataset.dtypes[0])
    interpretations = tuple(str(value) for value in dataset.colorinterp)
    return RasterMetadata(
        path=path,
        width=dataset.width,
        height=dataset.height,
        count=dataset.count,
        dtype=dtype.name,
        bit_depth=dtype.itemsize * 8,
        crs=crs,
        transform=dataset.transform,
        bounds=bounds,
        wgs84_bounds=wgs84_bounds,
        color_interpretation=interpretations,
        nodata=float(dataset.nodata) if dataset.nodata is not None else None,
    )


def _validate_georeferencing(dataset: rasterio.io.DatasetReader) -> None:
    """Reject rasters without CRS or a non-identity affine transform."""
    if dataset.crs is None:
        raise NotGeoreferencedError("GeoTIFF has no CRS; a georeferenced raster is required")
    if dataset.transform == Affine.identity():
        raise NotGeoreferencedError("GeoTIFF has no usable geotransform; affine transform is identity")


def _rgb_band_indexes(dataset: rasterio.io.DatasetReader) -> tuple[int, int, int]:
    """Select RGB bands from rasterio color interpretation, with an explicit fallback."""
    roles = {ColorInterp.red: None, ColorInterp.green: None, ColorInterp.blue: None}
    for index, interpretation in enumerate(dataset.colorinterp, start=1):
        if interpretation in roles:
            roles[interpretation] = index
    if all(value is not None for value in roles.values()):
        return (roles[ColorInterp.red], roles[ColorInterp.green], roles[ColorInterp.blue])  # type: ignore[return-value]
    if dataset.count >= 3:
        LOGGER.warning(
            "Raster %s has %d bands but no unambiguous RGB color interpretation; using bands 1, 2, 3",
            dataset.name,
            dataset.count,
        )
        return (1, 2, 3)
    raise UnsupportedRasterError("Raster must contain RGB bands or at least three fallback bands")


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize one channel to uint8 using its dtype range or finite data range."""
    values = channel.astype(np.float32, copy=False)
    if np.issubdtype(channel.dtype, np.integer):
        info = np.iinfo(channel.dtype)
        low, high = float(info.min), float(info.max)
    else:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(channel.shape, dtype=np.uint8)
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros(channel.shape, dtype=np.uint8)
    normalized = (values - low) / (high - low) * 255.0
    return np.clip(np.nan_to_num(normalized, nan=0.0), 0.0, 255.0).astype(np.uint8)


def normalize_rgb8(data: np.ndarray, dataset: rasterio.io.DatasetReader) -> np.ndarray:
    """Convert selected raster bands to HxWx3 uint8 while retaining source values separately."""
    if data.ndim != 3:
        raise UnsupportedRasterError(f"Expected band-first raster data, got shape {data.shape}")
    if data.shape[0] == 1:
        gray = _normalize_channel(data[0])
        return np.repeat(gray[..., None], 3, axis=2)
    indexes = _rgb_band_indexes(dataset)
    return np.stack([_normalize_channel(data[index - 1]) for index in indexes], axis=2)


def read_geotiff(path: str | Path, *, require_georeferencing: bool = True) -> GeoTIFFData:
    """Read a GeoTIFF with rasterio, preserving original values and producing RGB model input."""
    raster_path = Path(path)
    try:
        with rasterio.open(raster_path) as dataset:
            if require_georeferencing:
                _validate_georeferencing(dataset)
            data = dataset.read()
            if data.size == 0:
                raise UnsupportedRasterError("GeoTIFF contains no pixel data")
            metadata = _metadata(dataset, raster_path)
            rgb8 = normalize_rgb8(data, dataset)
    except rasterio.errors.RasterioIOError as error:
        raise GeoTIFFError(f"Could not read raster {raster_path}: {error}") from error
    return GeoTIFFData(data=data, rgb8=rgb8, metadata=metadata)


def write_dsm(
    output_path: str | Path,
    elevations: np.ndarray,
    source: RasterMetadata,
    *,
    nodata: float = -9999.0,
) -> Path:
    """Write float32 elevation values using the source CRS, transform, dimensions, and bounds."""
    output = Path(output_path)
    values = np.asarray(elevations, dtype=np.float32)
    if values.shape != (source.height, source.width):
        raise ValueError(f"DSM shape {values.shape} does not match source grid {(source.height, source.width)}")
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": source.height,
        "width": source.width,
        "count": 1,
        "dtype": "float32",
        "crs": source.crs,
        "transform": source.transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dataset:
        dataset.write(np.where(np.isfinite(values), values, nodata), 1)
    return output


def raster_bounds_from_grid(metadata: RasterMetadata) -> tuple[float, float, float, float]:
    """Return bounds computed from the source grid as a consistency check."""
    return tuple(float(value) for value in array_bounds(metadata.height, metadata.width, metadata.transform))
