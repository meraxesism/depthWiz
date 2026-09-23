# DepthWizard Python backend

This backend uses the requested rasterio, NumPy, SciPy-compatible numerical boundary, elevation, and FastAPI stack.

## Install

```powershell
python -m pip install -r requirements-backend.txt
```

`python-multipart` is included because FastAPI's `UploadFile` form parsing requires it at runtime. It is a FastAPI upload prerequisite, not an application-specific library.

The `elevation` package is a wrapper around external SRTM/GDAL tooling. Its
`clip()` implementation invokes GNU Make and requires `gdal_translate`,
`gdalbuildvrt`, `unzip`, and `gunzip` on `PATH`. Run
`python -c "import elevation; print(elevation.util.selfcheck(elevation.TOOLS))"`
to verify these prerequisites before using automatic SRTM calibration.

## Run

```powershell
$env:DEPTHWIZ_DEPTH_MODEL_PATH = "C:\\models\\depth-anything-v2-small\\model_quantized.onnx"
python -m uvicorn backend.main:app --reload --port 8787
```

The estimator must be a real callable with this contract:

```python
def estimate(image_rgb8: numpy.ndarray) -> numpy.ndarray:
    # image_rgb8: H x W x 3 uint8
    # return: H x W finite relative-depth array
    ...
```

The adapter uses the published `onnx-community/depth-anything-v2-small`
ONNX artifact. Download `onnx/model_quantized.onnx` from that repository and
set `DEPTHWIZ_DEPTH_MODEL_PATH`. Without a model path, the API returns HTTP
503 instead of substituting brightness for depth. A custom
`DEPTHWIZ_DEPTH_ESTIMATOR=module:function` remains supported for testing.

## Endpoints

- `GET /health`
- `POST /process` with multipart field `file` containing PNG, JPG, or TIFF
- `GET /outputs/{name}` for generated `.npy` or GeoTIFF products

Georeferenced inputs request an OpenTopography GeoTIFF directly. If its API
quota or credentials fail, the backend falls back to the AWS SRTM1
`*.hgt.gz` tiles used by the elevation package, decompressing with Python and
building a rasterio mosaic without GNU Make or GDAL CLI tools. It then
reprojects the DEM onto the input grid, fits calibration, and writes a
float32 DSM preserving CRS and transform. Non-georeferenced inputs produce
only a relative `.npy` result.

## Round-trip test

```powershell
python -m backend.test_geotiff_roundtrip path\to\sample.tif
```

The script prints CRS, native/WGS84 bounds, band count, dtype, bit depth, and verifies CRS, transform, and bounds after writing the float32 output.

## Real-scene depth diagnostics

```powershell
$env:DEPTHWIZ_DEPTH_MODEL_PATH = "C:\\models\\depth-anything-v2-small\\model_quantized.onnx"
$env:DEPTHWIZ_DEPTH_BASE_MODEL_PATH = "C:\\models\\depth-anything-v2-base\\model_quantized.onnx"
python -m backend.test_depth_comparison
```

This downloads public Esri World Imagery tiles for three samples each of
Golden foothills, Denver urban, Kansas sparse, and West Virginia forested
scenes; runs Small and the optional Base model from the full-resolution
mosaics; saves 64x64 grayscale/heatmap views; aligns SRTM; and prints R²,
range-normalized RMSE, and skill versus a mean-elevation baseline. Nodata
pixels are excluded from both fitting and normalization. The diagnostic uses
Pillow for JPEG tile decoding and visualization.
