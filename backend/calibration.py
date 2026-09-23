"""SRTM/GCP scale calibration for relative depth rasters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import os
import tempfile
from typing import Literal, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject

from .geotiff_io import GeoTIFFError, RasterMetadata

GCP = tuple[float, float, float]
FitMethod = Literal["linear", "huber"]


class CalibrationError(ValueError):
    """Raised when calibration inputs cannot produce a valid fit."""


@dataclass(frozen=True)
class CalibrationMetrics:
    """Regression coefficients and residual diagnostics."""

    method: str
    slope: float
    intercept: float
    r2: float
    residuals: np.ndarray
    sample_count: int


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated elevations and diagnostics returned to callers."""

    elevations: np.ndarray
    metrics: CalibrationMetrics
    sample_relative_depth: np.ndarray
    sample_elevation: np.ndarray


def fetch_srtm(bounds_wgs84: Sequence[float], output_path: str | Path, *, product: str | None = None) -> Path:
    """Fetch SRTM from OpenTopography's REST API and validate it with rasterio.

    The installed ``elevation`` package was tested against a real bbox and
    failed before download because it shells out to missing Windows GNU Make.
    OpenTopography returned a valid GeoTIFF directly, so this implementation
    avoids that CLI dependency while preserving the same SRTM source.
    """
    if len(bounds_wgs84) != 4:
        raise CalibrationError("SRTM bounds must be (left, bottom, right, top)")
    left, bottom, right, top = (float(value) for value in bounds_wgs84)
    if not left < right or not bottom < top:
        raise CalibrationError("SRTM bounds must have right > left and top > bottom")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    demtype = product or os.environ.get("DEPTHWIZ_SRTM_PRODUCT", "SRTMGL3")
    api_key = os.environ.get("OPENTOPO_API_KEY", "demoapikeyot2022")
    query = urlencode({
        "demtype": demtype,
        "west": left,
        "south": bottom,
        "east": right,
        "north": top,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    })
    url = f"https://portal.opentopography.org/API/globaldem?{query}"
    try:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if len(payload) < 1024 or payload[:4] not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM+\x00"}:
            raise CalibrationError("OpenTopography returned a non-TIFF or empty SRTM response")
        output.write_bytes(payload)
    except Exception as error:
        _fetch_aws_srtm1((left, bottom, right, top), output, error)
    if not output.exists() or output.stat().st_size == 0:
        raise CalibrationError("SRTM REST download produced no raster")
    try:
        with rasterio.open(output) as dataset:
            if dataset.count < 1 or dataset.crs is None:
                raise CalibrationError("Downloaded SRTM response is not a usable georeferenced raster")
    except rasterio.errors.RasterioIOError as error:
        raise CalibrationError(f"Downloaded SRTM response is not readable by rasterio: {error}") from error
    return output


def _fetch_aws_srtm1(
    bounds_wgs84: tuple[float, float, float, float],
    output: Path,
    primary_error: Exception,
) -> None:
    """Download AWS SRTM1 HGT.GZ tiles and mosaic them with rasterio."""
    left, bottom, right, top = bounds_wgs84
    west_tiles = range(int(np.floor(left)), int(np.floor(right)) + 1)
    south_tiles = range(int(np.floor(bottom)), int(np.floor(top)) + 1)
    temporary = tempfile.TemporaryDirectory(prefix="depthwiz-aws-srtm-")
    sources = []
    try:
        for tile_lon in west_tiles:
            for tile_lat in south_tiles:
                lon_prefix = "E" if tile_lon >= 0 else "W"
                lat_prefix = "N" if tile_lat >= 0 else "S"
                tile_name = f"{lat_prefix}{abs(tile_lat):02d}{lon_prefix}{abs(tile_lon):03d}"
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_prefix}{abs(tile_lat):02d}/{tile_name}.hgt.gz"
                try:
                    with urlopen(url, timeout=120) as response:
                        compressed = response.read()
                    hgt = gzip.decompress(compressed)
                except Exception as error:
                    raise CalibrationError(
                        f"SRTM REST failed ({primary_error}); AWS tile {tile_name} also failed: {error}"
                    ) from error
                side = int(round((len(hgt) // 2) ** 0.5))
                if side * side * 2 != len(hgt) or side < 2:
                    raise CalibrationError(f"AWS tile {tile_name} has invalid HGT dimensions")
                array = np.frombuffer(hgt, dtype=">i2").reshape(side, side)
                path = Path(temporary.name) / f"{tile_name}.tif"
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=side,
                    height=side,
                    count=1,
                    dtype="int16",
                    crs="EPSG:4326",
                    transform=from_origin(tile_lon, tile_lat + 1, 1 / (side - 1), 1 / (side - 1)),
                    nodata=-32768,
                ) as dataset:
                    dataset.write(array, 1)
                sources.append(rasterio.open(path))
        mosaic, transform = merge(sources, bounds=(left, bottom, right, top))
        profile = sources[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress="deflate",
        )
        with rasterio.open(output, "w", **profile) as dataset:
            dataset.write(mosaic)
    finally:
        for source in sources:
            source.close()
        temporary.cleanup()


def reproject_srtm_to_grid(
    srtm_path: str | Path,
    target: RasterMetadata,
    *,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    """Reproject SRTM pixels onto the exact target CRS, transform, width, and height."""
    destination = np.full((target.height, target.width), dst_nodata, dtype=np.float32)
    with rasterio.open(srtm_path) as source:
        if source.crs is None:
            raise CalibrationError("Downloaded SRTM raster has no CRS")
        source_array = source.read(1).astype(np.float32, copy=False)
        reproject(
            source=source_array,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=target.transform,
            dst_crs=target.crs,
            dst_nodata=dst_nodata,
            resampling=Resampling.bilinear,
        )
    return destination


def _sample_indices(
    shape: tuple[int, int],
    *,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    """Select reproducible flat pixel indices with a configurable upper bound."""
    total = shape[0] * shape[1]
    indices = np.arange(total, dtype=np.int64)
    if max_samples is not None and max_samples <= 0:
        raise CalibrationError("max_samples must be positive when supplied")
    if max_samples is not None and max_samples < total:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)
    return indices


def paired_samples(
    relative_depth: np.ndarray,
    reference_elevation: np.ndarray,
    *,
    gcps: Sequence[GCP] = (),
    max_samples: int | None = 10000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite depth/elevation pairs and append valid pixel-indexed GCP values."""
    depth = np.asarray(relative_depth, dtype=np.float64)
    elevation = np.asarray(reference_elevation, dtype=np.float64)
    if depth.shape != elevation.shape:
        raise CalibrationError(f"Depth shape {depth.shape} does not match reference shape {elevation.shape}")
    indices = _sample_indices(depth.shape, max_samples=max_samples, seed=seed)
    flat_depth = depth.ravel()[indices]
    flat_elevation = elevation.ravel()[indices]
    valid = np.isfinite(flat_depth) & np.isfinite(flat_elevation)
    samples_depth = list(flat_depth[valid])
    samples_elevation = list(flat_elevation[valid])
    height, width = depth.shape
    for pixel_x, pixel_y, known_elevation in gcps:
        x, y = int(round(pixel_x)), int(round(pixel_y))
        if 0 <= x < width and 0 <= y < height and np.isfinite(depth[y, x]) and np.isfinite(known_elevation):
            samples_depth.append(float(depth[y, x]))
            samples_elevation.append(float(known_elevation))
    if len(samples_depth) < 2:
        raise CalibrationError("At least two finite calibration pairs are required")
    return np.asarray(samples_depth), np.asarray(samples_elevation)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = slope*x + intercept using numpy least squares."""
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise CalibrationError("Calibration depth samples have no usable variation")
    return float(coefficients[0]), float(coefficients[1])


def _huber_fit(x: np.ndarray, y: np.ndarray, iterations: int = 30, delta: float = 1.345) -> tuple[float, float]:
    """Fit a Huber-weighted line using numpy iteratively reweighted least squares."""
    slope, intercept = _linear_fit(x, y)
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    for _ in range(iterations):
        residual = y - (slope * x + intercept)
        scale = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), 1e-9)
        normalized = np.abs(residual) / (delta * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / normalized)
        sqrt_weights = np.sqrt(weights)
        weighted_design = design * sqrt_weights[:, None]
        coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, y * sqrt_weights, rcond=None)
        if rank < 2:
            raise CalibrationError("Huber calibration became rank deficient")
        next_slope, next_intercept = float(coefficients[0]), float(coefficients[1])
        if abs(next_slope - slope) < 1e-10 and abs(next_intercept - intercept) < 1e-8:
            break
        slope, intercept = next_slope, next_intercept
    return slope, intercept


def fit_calibration(relative_depth: np.ndarray, elevations: np.ndarray, *, method: FitMethod = "linear") -> CalibrationMetrics:
    """Fit linear or Huber calibration and calculate R² plus per-sample residuals."""
    x = np.asarray(relative_depth, dtype=np.float64).ravel()
    y = np.asarray(elevations, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        raise CalibrationError("At least two finite calibration pairs are required")
    if method == "linear":
        slope, intercept = _linear_fit(x, y)
    elif method == "huber":
        slope, intercept = _huber_fit(x, y)
    else:
        raise CalibrationError(f"Unsupported calibration method: {method}")
    residuals = y - (slope * x + intercept)
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / total if total > 0 else 0.0
    return CalibrationMetrics(method=method, slope=slope, intercept=intercept, r2=float(r2), residuals=residuals, sample_count=x.size)


def calibrate_relative_depth(
    relative_depth: np.ndarray,
    reference_elevation: np.ndarray,
    *,
    gcps: Sequence[GCP] = (),
    max_samples: int | None = 10000,
    seed: int = 0,
    method: FitMethod = "linear",
) -> CalibrationResult:
    """Fit calibration on paired samples and apply it to the full relative-depth grid."""
    sample_depth, sample_elevation = paired_samples(
        relative_depth,
        reference_elevation,
        gcps=gcps,
        max_samples=max_samples,
        seed=seed,
    )
    metrics = fit_calibration(sample_depth, sample_elevation, method=method)
    elevations = metrics.slope * np.asarray(relative_depth, dtype=np.float64) + metrics.intercept
    elevations[~np.isfinite(relative_depth)] = np.nan
    return CalibrationResult(
        elevations=elevations.astype(np.float32),
        metrics=metrics,
        sample_relative_depth=sample_depth,
        sample_elevation=sample_elevation,
    )


def temporary_srtm_for_bounds(bounds_wgs84: Sequence[float], *, product: str | None = None) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Download SRTM into a managed temporary directory for one calibration request."""
    directory = tempfile.TemporaryDirectory(prefix="depthwiz-srtm-")
    path = Path(directory.name) / "srtm.tif"
    try:
        fetch_srtm(bounds_wgs84, path, product=product)
    except Exception:
        directory.cleanup()
        raise
    return path, directory
