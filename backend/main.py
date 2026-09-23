"""FastAPI application for DepthWizard raster processing."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
from affine import Affine
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .calibration import CalibrationError, calibrate_relative_depth, reproject_srtm_to_grid, temporary_srtm_for_bounds
from .depth_model import DepthEstimator, DepthModelError, create_depth_anything_v2, load_configured_estimator, run_depth_estimator, save_relative_array
from .geotiff_io import GeoTIFFError, GeoTIFFData, NotGeoreferencedError, read_geotiff, write_dsm

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("DEPTHWIZ_OUTPUT_DIR", BASE_DIR / "outputs"))
UPLOAD_DIR = Path(os.environ.get("DEPTHWIZ_UPLOAD_DIR", BASE_DIR / "uploads"))
MAX_UPLOAD_BYTES = int(os.environ.get("DEPTHWIZ_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class UploadFormatError(ValueError):
    """Raised when an upload extension or magic header is unsupported."""


def _validate_header(path: Path, suffix: str, header: bytes) -> None:
    """Validate common PNG, JPEG, and TIFF magic bytes before rasterio opens a file."""
    if suffix == ".png" and not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise UploadFormatError("File extension is PNG but the file signature is invalid")
    if suffix in {".jpg", ".jpeg"} and not header.startswith(b"\xff\xd8\xff"):
        raise UploadFormatError("File extension is JPEG but the file signature is invalid")
    if suffix in {".tif", ".tiff"} and header[:4] not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        raise UploadFormatError("File extension is TIFF but the file signature is invalid")


def _metadata_response(data: GeoTIFFData) -> dict[str, Any]:
    """Serialize raster metadata without exposing non-JSON rasterio objects."""
    metadata = data.metadata
    return {
        "width": metadata.width,
        "height": metadata.height,
        "resolution": [abs(metadata.transform.a), abs(metadata.transform.e)],
        "bounds_native_crs": list(metadata.bounds),
        "bounds_wgs84": list(metadata.wgs84_bounds) if metadata.wgs84_bounds else None,
        "crs": metadata.crs.to_string() if metadata.crs else None,
        "transform": list(metadata.transform),
        "band_count": metadata.count,
        "dtype": metadata.dtype,
        "bit_depth": metadata.bit_depth,
        "color_interpretation": list(metadata.color_interpretation),
    }


def _metrics_response(metrics: Any) -> dict[str, Any]:
    """Serialize fit metrics with bounded residual detail for API responses."""
    residuals = np.asarray(metrics.residuals, dtype=np.float64)
    return {
        "method": metrics.method,
        "slope": metrics.slope,
        "intercept": metrics.intercept,
        "r2": metrics.r2,
        "sample_count": metrics.sample_count,
        "residual_rmse": float(np.sqrt(np.mean(residuals**2))),
        "residual_mean_absolute": float(np.mean(np.abs(residuals))),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
    }


def _remove_file(path: Path) -> None:
    """Remove a temporary upload after FastAPI has completed the response."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def create_app(estimator: DepthEstimator | None = None) -> FastAPI:
    """Create the FastAPI app with an optional injected production depth estimator."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="DepthWizard Backend", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[value.strip() for value in os.environ.get("DEPTHWIZ_ALLOW_ORIGINS", "http://localhost:5173").split(",")],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
    loaded_estimator = estimator

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        """Report service availability and whether a depth estimator is configured."""
        configured = estimator is not None or bool(os.environ.get("DEPTHWIZ_DEPTH_ESTIMATOR"))
        return {"ok": True, "service": "depthwiz-python", "depth_model_configured": configured}

    @app.get("/api/health")
    async def api_health() -> dict[str, str | bool]:
        """Provide the frontend-compatible health route."""
        return await health()

    @app.post("/process")
    async def process(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
        """Process an upload into an absolute GeoTIFF or relative NumPy product."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"Unsupported format {suffix or '<none>'}; use PNG, JPG, or TIFF")
        upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
        size = 0
        try:
            with upload_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit")
                    destination.write(chunk)
            with upload_path.open("rb") as source:
                _validate_header(upload_path, suffix, source.read(16))
            try:
                raster = read_geotiff(upload_path, require_georeferencing=False)
            except (GeoTIFFError, NotGeoreferencedError) as error:
                raise HTTPException(status_code=400, detail=f"Invalid or corrupt raster: {error}") from error
            georeferenced = raster.metadata.crs is not None and raster.metadata.transform != Affine.identity()
            try:
                nonlocal loaded_estimator
                if loaded_estimator is None:
                    loaded_estimator = (
                        create_depth_anything_v2()
                        if os.environ.get("DEPTHWIZ_DEPTH_MODEL_PATH")
                        else load_configured_estimator()
                    )
                active_estimator = loaded_estimator
                relative = run_depth_estimator(active_estimator, raster.rgb8)
            except DepthModelError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            if not georeferenced:
                output = save_relative_array(OUTPUT_DIR / f"{upload_path.stem}-relative.npy", relative)
                return {
                    "mode": "relative",
                    "output": {"path": str(output), "url": f"/outputs/{output.name}", "format": "npy"},
                    "metadata": _metadata_response(raster),
                }
            if raster.metadata.wgs84_bounds is None:
                raise HTTPException(status_code=422, detail="Georeferenced raster has no transformable WGS84 bounds")
            srtm_path = None
            temporary_directory = None
            try:
                try:
                    srtm_path, temporary_directory = temporary_srtm_for_bounds(raster.metadata.wgs84_bounds)
                    reference = reproject_srtm_to_grid(srtm_path, raster.metadata)
                    calibrated = calibrate_relative_depth(relative, reference)
                except CalibrationError as error:
                    raise HTTPException(status_code=502, detail=f"Calibration failed: {error}") from error
                output = write_dsm(OUTPUT_DIR / f"{upload_path.stem}-dsm.tif", calibrated.elevations, raster.metadata)
                return {
                    "mode": "absolute",
                    "output": {"path": str(output), "url": f"/outputs/{output.name}", "format": "GeoTIFF"},
                    "metadata": _metadata_response(raster),
                    "calibration": _metrics_response(calibrated.metrics),
                }
            finally:
                if temporary_directory is not None:
                    temporary_directory.cleanup()
        except HTTPException:
            raise
        except UploadFormatError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Could not store upload: {error}") from error
        finally:
            background_tasks.add_task(_remove_file, upload_path)

    return app


app = create_app()
