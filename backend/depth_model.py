"""Depth-model boundary for the FastAPI backend.

The repository currently contains a browser-side ONNX model, not Python model
weights or a Python inference package. This module therefore accepts a real
injected estimator rather than silently substituting luminance for depth.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, Protocol

import numpy as np


class DepthModelError(RuntimeError):
    """Raised when relative-depth inference cannot be completed."""


class DepthEstimator(Protocol):
    """Callable contract for a production depth model adapter."""

    def __call__(self, image_rgb8: np.ndarray) -> np.ndarray:
        """Return a 2-D relative-depth array matching the input image dimensions."""


EstimatorFactory = Callable[[np.ndarray], np.ndarray]


class DepthAnythingV2Onnx:
    """Run the published ONNX-community Depth Anything V2 Small model.

    The model signature and preprocessing values mirror its published
    ``preprocessor_config.json``: RGB values are divided by 255 and normalized
    with ImageNet mean/std; ONNX Runtime executes ``pixel_values`` and returns
    ``predicted_depth``.
    """

    def __init__(self, model_path: str | Path, *, input_size: int = 518) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise DepthModelError("onnxruntime is required for Depth Anything V2 ONNX inference") from error
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or inputs[0].name != "pixel_values":
            raise DepthModelError("Depth Anything V2 ONNX model has no pixel_values input")
        if not outputs or outputs[0].name != "predicted_depth":
            raise DepthModelError("Depth Anything V2 ONNX model has no predicted_depth output")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_size = input_size

    @staticmethod
    def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize HxWx3 arrays with rasterio's bilinear resampling API."""
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject

        result = np.empty((height, width, image.shape[2]), dtype=np.float32)
        source_transform = from_bounds(0, 0, image.shape[1], image.shape[0], image.shape[1], image.shape[0])
        target_transform = from_bounds(0, 0, width, height, width, height)
        for channel in range(image.shape[2]):
            reproject(
                source=image[:, :, channel].astype(np.float32),
                destination=result[:, :, channel],
                src_transform=source_transform,
                src_crs="EPSG:3857",
                dst_transform=target_transform,
                dst_crs="EPSG:3857",
                resampling=Resampling.bilinear,
            )
        return result

    def __call__(self, image_rgb8: np.ndarray) -> np.ndarray:
        """Infer a relative-depth grid from an HxWx3 uint8 RGB array."""
        image = np.asarray(image_rgb8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise DepthModelError(f"Expected HxWx3 RGB input, got {image.shape}")
        resized = self._resize(image, self._input_size, self._input_size) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((resized - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
        prediction = np.asarray(self._session.run([self._output_name], {self._input_name: tensor})[0])[0]
        return resize_depth(prediction.astype(np.float32), image.shape[:2])


def create_depth_anything_v2(model_path: str | Path | None = None) -> DepthAnythingV2Onnx:
    """Create the real Depth Anything V2 Small ONNX adapter from an explicit path."""
    path = Path(model_path or os.environ.get("DEPTHWIZ_DEPTH_MODEL_PATH", ""))
    if not path.is_file():
        raise DepthModelError(
            "Depth Anything V2 model file not found; set DEPTHWIZ_DEPTH_MODEL_PATH to model_quantized.onnx"
        )
    return DepthAnythingV2Onnx(path)


def load_configured_estimator() -> DepthEstimator:
    """Load an estimator from DEPTHWIZ_DEPTH_ESTIMATOR='module:function'."""
    reference = os.environ.get("DEPTHWIZ_DEPTH_ESTIMATOR")
    if not reference or ":" not in reference:
        raise DepthModelError(
            "No Python depth estimator configured. Set DEPTHWIZ_DEPTH_ESTIMATOR="
            "module:function or inject an estimator into create_app()."
        )
    module_name, function_name = reference.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        estimator = getattr(module, function_name)
    except (ImportError, AttributeError) as error:
        raise DepthModelError(f"Could not load configured depth estimator {reference}: {error}") from error
    if not callable(estimator):
        raise DepthModelError(f"Configured depth estimator {reference} is not callable")
    return estimator


def run_depth_estimator(estimator: DepthEstimator, image_rgb8: np.ndarray) -> np.ndarray:
    """Run an injected model and validate its output without changing its values."""
    try:
        output = np.asarray(estimator(image_rgb8), dtype=np.float32)
    except Exception as error:
        raise DepthModelError(f"Depth model inference failed: {error}") from error
    if output.ndim != 2:
        raise DepthModelError(f"Depth model must return a 2-D array, got shape {output.shape}")
    if output.shape != image_rgb8.shape[:2]:
        raise DepthModelError(
            f"Depth model output shape {output.shape} does not match image shape {image_rgb8.shape[:2]}"
        )
    if not np.isfinite(output).any():
        raise DepthModelError("Depth model returned no finite values")
    return output


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize model output with rasterio when a model returns a different grid."""
    if depth.shape == shape:
        return depth.astype(np.float32, copy=False)
    try:
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject
    except ImportError as error:
        raise DepthModelError("rasterio is required to resize depth output") from error
    source_transform = from_bounds(0, 0, depth.shape[1], depth.shape[0], depth.shape[1], depth.shape[0])
    target_transform = from_bounds(0, 0, shape[1], shape[0], shape[1], shape[0])
    resized = np.full(shape, np.nan, dtype=np.float32)
    reproject(
        source=depth,
        destination=resized,
        src_transform=source_transform,
        src_crs="EPSG:3857",
        dst_transform=target_transform,
        dst_crs="EPSG:3857",
        resampling=Resampling.bilinear,
    )
    return resized


def save_relative_array(output_path: str | Path, relative_depth: np.ndarray) -> Path:
    """Save a non-georeferenced relative depth product as a NumPy array."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(relative_depth, dtype=np.float32))
    return path
