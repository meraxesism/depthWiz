"""Exercise the real Depth Anything V2 + SRTM + calibration API path."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from .depth_model import create_depth_anything_v2
from .main import create_app


def main() -> int:
    """Run direct model inference and one complete georeferenced /process request."""
    model_path = os.environ.get("DEPTHWIZ_DEPTH_MODEL_PATH")
    if not model_path:
        raise RuntimeError("Set DEPTHWIZ_DEPTH_MODEL_PATH before running this test")
    estimator = create_depth_anything_v2(model_path)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :]
    depth = estimator(image)
    assert depth.shape == image.shape[:2]
    assert np.isfinite(depth).all()
    print(f"direct model depth: shape={depth.shape}, min={depth.min():.6f}, max={depth.max():.6f}")

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "real-input.tif"
        values = np.zeros((64, 64, 3), dtype=np.uint8)
        values[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :]
        values[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None]
        with rasterio.open(
            source,
            "w",
            driver="GTiff",
            width=64,
            height=64,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(-105.3, 39.8, 0.003125, 0.0015625),
        ) as dataset:
            dataset.write(values.transpose(2, 0, 1))
        app = create_app()
        client = TestClient(app)
        with source.open("rb") as handle:
            response = client.post("/process", files={"file": ("real-input.tif", handle, "image/tiff")})
        print(f"full process status: {response.status_code}")
        print(response.json())
        response.raise_for_status()
        payload = response.json()
        assert payload["mode"] == "absolute"
        assert payload["calibration"]["sample_count"] > 1
        assert Path(payload["output"]["path"]).exists()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
