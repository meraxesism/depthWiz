"""Visual and numerical comparison of real satellite inputs through Depth Anything V2."""

from __future__ import annotations

import math
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

from .calibration import calibrate_relative_depth, fetch_srtm, reproject_srtm_to_grid
from .depth_model import create_depth_anything_v2, resize_depth
from .geotiff_io import read_geotiff

ZOOM = 14
OUTPUT_DIR = Path(os.environ.get("DEPTHWIZ_DIAGNOSTICS_DIR", "C:/temp/depthwiz-depth-diagnostics"))
LOCATIONS = {
    "golden_foothills": (39.755, -105.22),
    "golden_foothills_2": (39.68, -105.35),
    "golden_foothills_3": (39.85, -105.35),
    "denver_urban": (39.7392, -104.9903),
    "denver_urban_2": (39.68, -104.95),
    "denver_urban_3": (39.77, -104.88),
    "kansas_sparse": (39.115, -100.65),
    "kansas_sparse_2": (38.95, -100.90),
    "kansas_sparse_3": (39.30, -100.20),
    "west_virginia_forested": (39.05, -79.50),
    "west_virginia_forested_2": (38.80, -79.60),
    "west_virginia_forested_3": (39.30, -78.90),
}


def _tile_xy(latitude: float, longitude: float, zoom: int) -> tuple[int, int]:
    """Convert WGS84 coordinates to slippy-map tile coordinates."""
    scale = 2**zoom
    x = int((longitude + 180.0) / 360.0 * scale)
    y = int((1.0 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2.0 * scale)
    return x, y


def _tile_wgs84(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Return longitude/latitude at the top-left of a slippy-map tile."""
    scale = 2**zoom
    longitude = x / scale * 360.0 - 180.0
    latitude = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / scale))))
    return longitude, latitude


def download_imagery(latitude: float, longitude: float) -> Image.Image:
    """Download a 2x2 real Esri World Imagery tile mosaic around a location."""
    center_x, center_y = _tile_xy(latitude, longitude, ZOOM)
    mosaic = Image.new("RGB", (512, 512))
    for row in range(2):
        for column in range(2):
            x, y = center_x + column - 1, center_y + row - 1
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{ZOOM}/{y}/{x}"
            request = Request(url, headers={"User-Agent": "DepthWizard diagnostics"})
            with urlopen(request, timeout=60) as response:
                tile = Image.open(response).convert("RGB")
            mosaic.paste(tile, (column * 256, row * 256))
    return mosaic


def write_georeferenced_imagery(imagery: Image.Image, latitude: float, longitude: float, path: Path) -> None:
    """Write the 2x2 tile mosaic with its exact WGS84 tile extent."""
    center_x, center_y = _tile_xy(latitude, longitude, ZOOM)
    left, top = _tile_wgs84(center_x - 1, center_y - 1, ZOOM)
    right, bottom = _tile_wgs84(center_x + 1, center_y + 1, ZOOM)
    pixels = np.asarray(imagery, dtype=np.uint8).transpose(2, 0, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(left, bottom, right, top, 512, 512),
    ) as dataset:
        dataset.write(pixels)


def save_depth_visualization(depth: np.ndarray, name: str) -> None:
    """Save normalized grayscale and blue-to-red heatmap views of a depth grid."""
    finite = np.nan_to_num(depth, nan=float(np.nanmean(depth)))
    minimum, maximum = float(np.min(finite)), float(np.max(finite))
    normalized = (finite - minimum) / max(maximum - minimum, 1e-6)
    grayscale = (normalized * 255).astype(np.uint8)
    Image.fromarray(grayscale, mode="L").resize((512, 512), Image.Resampling.NEAREST).save(OUTPUT_DIR / f"{name}-grayscale.png")
    red = (normalized * 255).astype(np.uint8)
    blue = ((1.0 - normalized) * 255).astype(np.uint8)
    green = (1.0 - np.abs(normalized - 0.5) * 2.0) * 180
    heatmap = np.stack((red, green.astype(np.uint8), blue), axis=2)
    Image.fromarray(heatmap, mode="RGB").resize((512, 512), Image.Resampling.NEAREST).save(OUTPUT_DIR / f"{name}-heatmap.png")


def structure_metrics(depth: np.ndarray) -> dict[str, float]:
    """Measure range, variation, and neighboring-pixel structure in a depth grid."""
    dx = np.diff(depth, axis=1)
    dy = np.diff(depth, axis=0)
    return {
        "minimum": float(np.min(depth)),
        "maximum": float(np.max(depth)),
        "range": float(np.ptp(depth)),
        "stddev": float(np.std(depth)),
        "mean_abs_neighbor_delta": float((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2.0),
        "smoothness_correlation_x": float(np.corrcoef(depth[:, :-1].ravel(), depth[:, 1:].ravel())[0, 1]),
        "smoothness_correlation_y": float(np.corrcoef(depth[:-1, :].ravel(), depth[1:, :].ravel())[0, 1]),
    }


def calibration_metrics(depth: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return fit error normalized by reference relief and a mean-baseline skill score."""
    calibration = calibrate_relative_depth(depth, reference, max_samples=None)
    residuals = calibration.metrics.residuals
    rmse = float(np.sqrt(np.mean(residuals**2)))
    valid_reference = calibration.sample_elevation[np.isfinite(calibration.sample_elevation)]
    reference_range = max(float(np.ptp(valid_reference)), 1e-6)
    baseline_rmse = float(np.sqrt(np.mean((valid_reference - np.mean(valid_reference)) ** 2)))
    return {
        "r2": calibration.metrics.r2,
        "rmse": rmse,
        "nrmse_by_reference_range": rmse / reference_range,
        "mean_baseline_rmse": baseline_rmse,
        "skill_vs_mean_baseline": 1.0 - rmse / max(baseline_rmse, 1e-6),
        "valid_sample_fraction": float(calibration.metrics.sample_count / depth.size),
    }


def main() -> int:
    """Run both real-location tests and write visual diagnostics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = {"small": os.environ.get("DEPTHWIZ_DEPTH_MODEL_PATH")}
    base_path = os.environ.get("DEPTHWIZ_DEPTH_BASE_MODEL_PATH")
    if base_path:
        model_paths["base"] = base_path
    if not model_paths["small"]:
        raise RuntimeError("Set DEPTHWIZ_DEPTH_MODEL_PATH to the Small quantized ONNX model")
    scene_data = {}
    for name, (latitude, longitude) in LOCATIONS.items():
        imagery = download_imagery(latitude, longitude)
        imagery.save(OUTPUT_DIR / f"{name}-imagery.jpg", quality=92)
        source_path = OUTPUT_DIR / f"{name}-imagery.tif"
        write_georeferenced_imagery(imagery, latitude, longitude, source_path)
        raster = read_geotiff(source_path)
        srtm_path = OUTPUT_DIR / f"{name}-srtm.tif"
        if not srtm_path.exists():
            fetch_srtm(raster.metadata.wgs84_bounds, srtm_path)
        reference_64 = resize_depth(reproject_srtm_to_grid(srtm_path, raster.metadata), (64, 64))
        scene_data[name] = (np.asarray(imagery, dtype=np.uint8), reference_64)
    for model_name, model_path in model_paths.items():
        if not model_path:
            continue
        estimator = create_depth_anything_v2(model_path)
        for name, (input_rgb, reference_64) in scene_data.items():
            depth = resize_depth(estimator(input_rgb), (64, 64))
            save_depth_visualization(depth, f"{model_name}-{name}")
            print(model_name, name, {**structure_metrics(depth), **calibration_metrics(depth, reference_64)})
    print(f"visualizations: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
