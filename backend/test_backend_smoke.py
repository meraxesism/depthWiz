"""Runtime smoke tests for the backend components."""

from pathlib import Path
import tempfile

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from .calibration import calibrate_relative_depth
from .geotiff_io import read_geotiff, write_dsm
from .main import create_app


def main() -> int:
    """Exercise GeoTIFF round-trip, exact calibration, and relative API processing."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.tif"
        transform = from_origin(10, 20, 0.5, 0.5)
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        with rasterio.open(
            source,
            "w",
            driver="GTiff",
            width=4,
            height=3,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dataset:
            dataset.write(values, 1)
        raster = read_geotiff(source)
        output = root / "out.tif"
        write_dsm(output, values + 100, raster.metadata)
        with rasterio.open(output) as dataset:
            assert dataset.crs == raster.metadata.crs
            assert dataset.transform == raster.metadata.transform
            assert dataset.dtypes[0] == "float32"

        reference = 3 * values + 10
        result = calibrate_relative_depth(values, reference, max_samples=None)
        assert abs(result.metrics.slope - 3) < 1e-6
        assert abs(result.metrics.intercept - 10) < 1e-6
        assert result.metrics.r2 > 0.999

        def estimator(image: np.ndarray) -> np.ndarray:
            return np.mean(image, axis=2).astype(np.float32)

        app = create_app(estimator=estimator)
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        image_path = root / "image.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), "RGB").save(image_path)
        with image_path.open("rb") as image_file:
            response = client.post(
                "/process",
                files={"file": ("image.png", image_file, "image/png")},
            )
        assert response.status_code == 200, response.text
        assert response.json()["mode"] == "relative"
    print("backend smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
