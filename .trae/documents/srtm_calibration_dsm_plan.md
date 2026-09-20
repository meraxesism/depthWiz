# SRTM Auto-Calibration, Per-Terrain Regression & DSM Export — Implementation Plan

## Repository Research — Current vs. Spec

### Existing (preserved untouched unless explicitly noted)

- **Input handling** in [App.tsx handleImagery](file:///c:/Users/Aditya/Documents/GitHub/depthWiz/src/App.tsx#L92-L142) routes GeoTIFF vs. PNG/JPG differently:
  - GeoTIFF → reads **band-0 raw values as direct metric elevations** (treats every `.tif/.tiff` as an elevation DEM already) → `rasterHeights = rawRaster.values`, `CalibrationProduct { source: 'GeoTIFF raster', isMetric: true }`.
  - PNG/JPG → `buildRelativeDepthFromImage` (luma placeholder) → user clicks **Run Depth Anything V2** → `estimateRelativeDepth` → 128² relative 0–1 heights → synthetic 8–188 m display scale.
- **Reference DEM calibration** exists only via manual user-uploaded `.tif` in [handleReference](file:///c:/Users/Aditya/Documents/GitHub/depthWiz/src/App.tsx#L162-L172), wired through [pipeline.ts calibrateRelativeDepth](file:///c:/Users/Aditya/Documents/GitHub/depthWiz/src/pipeline.ts#L191-L206) (a 35/65 value-blend, NOT a least-squares regression).
- **Mesh rendering** in [App.tsx render-useEffect](file:///c:/Users/Aditya/Documents/GitHub/depthWiz/src/App.tsx#L57-L90) uses `DEFAULT_GRID_SIZE=128`, Cesium `Primitive`; texture drapes when `textureUrl != null`.
- **Server** in [server/index.ts](file:///c:/Users/Aditya/Documents/GitHub/depthWiz/server/index.ts) currently has only `/api/health` and `/api/upload` (multer memoryStorage; no actual server-side processing).
- **No external DEM fetching** exists. No SRTM, no OpenTopography, no tile downloads.
- **No export/deliverable** exists — the DSM is only rendered; user cannot download it.
- **No terrain classification** exists. Every pixel uses the same calibration blend.

### Spec Gaps → New Capabilities to Add

| # | Spec requirement | Current status | Action |
|---|---|---|---|
| 1 | **2 kinds of GeoTIFF**: RGB/Satellite/Drone *imagery* GeoTIFF (goes through DA V2 + SRTM calibration) vs. pure elevation DEM GeoTIFF (use elevations directly) | App treats **all** `.tif` as DEM elevations | Auto-detect DEM-vs-Imagery GeoTIFF via band count, bit depth, nodata and value-range heuristics; let user override in UI. |
| 2 | **SRTM reference tile auto-fetch** from geographic bounds | Does not exist | Add `/api/srtm?west=&south=&east=&north=` Express endpoint; fetch from OpenTopography public API (or AWS SRTM 1° tiles as fallback); parse response TIFF → resample to grid → `DepthProduct`. |
| 3 | **Pixel-for-pixel resample** of reference to match image | Manual blend only (`% reference.values.length` wrap fallback) | Proper bilinear resampling to the exact `(gridW, gridH)` mesh grid. |
| 4 | **Least-squares linear regression** `elev = a·depth + b` | Only 35/65 weighted blend | Add `fitLinearRegression(depth[], elev[])` → `{a, b, r²}`; add per-terrain variant (see gap #6). |
| 5 | **Evaluation fit quality** (`R²`, residuals) | No fit, no metrics | Report fit metrics per terrain class and global summary. |
| 6 | **Per-terrain-type calibration** (urban / sparse / hilly / forested) via classification | None | Heuristic terrain-classifier from relative-depth + reference: std-dev of elevations, rugosity, mean local slope → 4 classes; fit and apply regression **per class**. |
| 7 | **Non-georeferenced PNG/JPG skip calibration** → relative only | Already correct (manual reference DEM still available) | Preserve as-is; add status bar reminder. |
| 8 | **DSM deliverable** export in standard geospatial format | No export | Add **Export GeoTIFF** button: write band-0 elevations + GeoKey CRS/bounds via `geotiff.writeArray` → `Blob` → trigger download. Also export relative-only GeoTIFF for the PNG/JPG case. |
| 9 | **Input concept descriptions** (InSAR / GeoTIFF-georef / monocular) | Not shown | Add compact inline help/callouts so UI reflects the 3 input descriptions from spec. |
| 10 | **First-person flightpath polish** | Flythrough orbital exists; free look OK | Preserve; add altitude-smoothing in flythrough. |

---

## Files and Modules

| File | Change type | Purpose |
|---|---|---|
| `server/index.ts` | Modify — **append new routes only; do not alter `/api/health` or `/api/upload`** | Add `/api/srtm` (fetch and resample SRTM 30m DEM tile for given WGS-84 bounds). Add optional OpenTopography key passthrough via `process.env.OPENTOPO_API_KEY`. |
| `src/pipeline.ts` | Modify — **preserve all existing exports; add new ones after** | Add: `bilinearResample(source, srcW, srcH, dstW, dstH)`, `fitLinearRegression(xs, ys, mask?) → {a,b,r²,predicted}`, `classifyTerrain(heights, w, h) → Urban|Sparse|Hilly|Forested[] per pixel`, `calibrateByTerrainClass(relative, reference, w, h, bounds?, classifier?)`, helper `writeGeoTiffFloat32(heights, w, h, bounds?, crs?) → Promise<Blob>`. **Do not modify signatures of existing `build*`, `calibrateRelativeDepth`, `read*`, `computeMeanSlope`, `DEFAULT_GRID_SIZE`.** |
| `src/depthAnything.ts` | Keep identical (no signature change). | Already matches new grid size. |
| `src/App.tsx` | Modify — **preserve all existing handlers and useEffect; extend branches only.** | Add: (a) GeoTIFF classification pass (DEM vs. RGB imagery) → route to "direct elevations" (existing) or "DA V2 → SRTM fit" (new); (b) `autoCalibrateSrtm()` button + async flow: geoMetadata.bounds → GET `/api/srtm` → `bilinearResample` → `calibrateByTerrainClass` → render; (c) **Export DSM** button → pipeline `writeGeoTiffFloat32` → download; (d) fit-quality status (R², per-class counts); (e) add inline input-mode explainer (spec section 1 descriptions); (f) add optional user toggle to force GeoTIFF-as-imagery or GeoTIFF-as-DEM. |
| `src/App.css` | Append-only CSS additions | Legend fit-quality pill, per-terrain classifier legend, new control-row buttons for auto-calibrate + export, input description callouts. |
| `package.json` | No new dependencies required | `geotiff` already has `writeArray`; `@huggingface/transformers` present; server is `express`. |

---

## Implementation Steps (dependency order)

### 1. Server: `/api/srtm` endpoint (append-only)
- Append to `server/index.ts` after existing routes (do **not** modify `/api/health`, `/api/upload`).
- Flow: `GET /api/srtm?west&south&east&north` (WGS-84 degrees). Use OpenTopography `SRTMGL3` (30 m) endpoint: `https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&west=...&south=...&east=...&north=...&outputFormat=GTiff&API_Key=...`. If `OPENTOPO_API_KEY` env var absent, fall back to **NASA EarthData AWS SRTM 1° GeoTIFFs** (`https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_XX_YY.zip` URL structure). Read TIFF with `fromArrayBuffer` → `readRasters({samples:[0]})` → send back `{ ok, values, width, height, bounds, crs: 'EPSG:4326' }` JSON so browser code can reuse existing `DepthProduct` flow.
- Robustness: clamp bounds to `[-180,-90,180,90]`, `east>west && north>south` guard, 4 KB min size check, 30-second timeout with friendly error body.

### 2. `pipeline.ts` — new exports (append after `computeMeanSlope`)
- **`bilinearResample(src: Float64Array|number[], srcW, srcH, dstW, dstH, srcNoDataValue?: number): number[]`** — standard bilinear with edge clamp.
- **`fitLinearRegression(xs: number[], ys: number[], mask?: boolean[]): { a: number; b: number; r2: number; predicted: number[] }`** — normal equations `X = [xs, 1]`, solve via 2×2 closed form (no matrix dep needed). Compute `r2 = 1 − SS_res / SS_tot`.
- **`classifyTerrain(heights: number[], w: number, h: number): ('Urban'|'Sparse'|'Hilly'|'Forested')[]`** — per-pixel 3×3 window: compute `localStd`, `localSlopeMean`, `localRugosity = z_i − bilinear(z)`; simple rule-tree:
  - `localStd > T1 && localSlopeMean > 20°` → **Hilly**
  - `rugosity quantile 75 > T2` → **Forested** (high micro-roughness)
  - `localStd moderate + many step edges (Sobel on heights > T3)` → **Urban** (building edges)
  - else → **Sparse** / flat-open
  Thresholds tuned via quantiles so they auto-adapt to any DEM range.
- **`calibrateByTerrainClass(relative: DepthProduct, reference: DepthProduct, w, h): CalibrationProduct & { fit: { global: {a,b,r2}; perClass: Record<TerrainClass, {count,a,b,r2}> }; terrainClass: TerrainClass[]; }`** — Bilinear-resample reference to (w,h), build `referenceMask = Number.isFinite(refV)`, run global fit, then per-class masked fit, prefer per-class predicted when class-fit `r2 > global_r2 * 0.8`, else fall back global.
- **`writeGeoTiffFloat32(heightValues: number[], w: number, h: number, bounds?: [west,south,east,north], crs?: string): Promise<Blob>`** — use `geotiff.fromArray([values], w, h, { width, height, samplesPerPixel: 1, bitsPerSample: 32, sampleFormat: 3, noData: -9999 })` (Float32). If geographic bounds + CRS are present, write ModelTiepoint/ModelPixelScale and EPSG:4326 GeoKeys. Return `Blob` of the bytes.

### 3. `App.tsx` — new UI/flows (append features; **do not remove existing behavior**)
Keep the 5 existing top-level state pieces untouched. Add state for: `srtmRef: DepthProduct|null`, `terrainClass[]|null`, `fitMetrics|null` (`{ global, perClass }`), `geoTiffKind: 'auto'|'as-dem'|'as-imagery'`.

#### 3.a GeoTIFF classification (inside `handleImagery`, only for `isGeo=true`)
- After we parse tiffImage → read band count `samplesPerPixel = image.getSamplesPerPixel()`, band-0 stats `(min, max, mean)`.
- Heuristic `geoTiffKind auto = (samplesPerPixel >= 3 || (bitsPerSample == 8 && 0 <= min && max <= 255)) ? 'as-imagery' : 'as-dem'` (bandcount ≥ 3 = RGB satellite/drone imagery go through DA V2; single-band Float32/Int16 = DEM).
- User override: add toggle button "Treat this GeoTIFF as: ▢ DEM ▢ Imagery".
- Route **as-imagery** GeoTIFF: `setTextureUrl( buildRasterTexture blob )`, run relative depth via DA V2 (same as PNG/JPG path), enable **Auto-calibrate with SRTM** button (because we have geographic bounds now).
- Route **as-dem** GeoTIFF: the existing path we already fixed (rawRaster elevations direct, preserves current behavior).

#### 3.b New handler `autoCalibrateSrtm()`
- Requires: `geoMetadata.geographic === true` (bounds are safe degrees).
- `GET /api/srtm?west&south&east&north` → parse JSON response.
- `bilinearResample(reference.values, ref.w, ref.h, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)`.
- `calibrateByTerrainClass(relativeDepth, resampledRef, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)` → merged calibrated values.
- Write to state: `setRasterHeights(calibrated.values)`, `setCalibration({...calibrated, source:'SRTM auto-calibrated', isMetric:true})`, `setRasterStats({min,max,mean, slope: computeMeanSlope(...)})`, `setFitMetrics(fit)`, `setTerrainClass(classifier)`.

#### 3.c Export button `exportDSM()`
- Button disabled unless `rasterHeights != null`.
- `writeGeoTiffFloat32(rasterHeights, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, geoMetadata?.geographic ? geoMetadata.bounds : undefined, geoMetadata?.crs)`.
- `URL.createObjectURL(blob)` → `<a download>` click → "depthwiz-dsm-YYYYMMDD-HHMMSS.tif".
- For relative-only calibrations: same export but include CRS only when known.

#### 3.d Fit-quality and input-descriptions UI (append control panel sections)
- Add a 3-item input-description pill-row ("InSAR · avoided", "GeoTIFF · with CRS", "Monocular · relative only") with compact spec-sheet wording so UI matches Section 1 description.
- After auto-calibration succeeded, render a compact legend: `R²={r2.toFixed(3)}` and per-terrain badges with counts (`Urban N=…`, etc).

### 4. Append-only `App.css`
- Add `.fit-row { }`, `.terrain-legend { }`, `.input-descriptions { }`, `.export-button { }`, `.kind-toggle-row { }` CSS blocks to the **end** of existing file — don't alter earlier selectors.

### 5. Validation (post-execution checklist)
- `npm run lint` → oxlint pass.
- `npx tsc -b` → strict-ts pass.
- `npm run build` → Vite build succeeds.
- `npm run dev:full` starts; `/api/health` returns `{ ok }`; `/api/srtm` with a valid small bounds tile (e.g., Golden, CO `w=-105.3, s=39.7, e=-105.1, n=39.8`) returns a values JSON response.
- Regression: the **original flow** (PNG/JPG → DA V2 button → `8 + value*180m`) still produces mesh and doesn't error. DEM GeoTIFF upload continues to render at raw elevations as before.
- Export: download a GeoTIFF; inspect with QGIS/GDAL `gdalinfo` shows correct Float32 band and (if applicable) EPSG:4326 GeoKeys + ModelTiepoint + ModelPixelScale.
- Fit metrics: `r2` after SRTM auto-cal is > 0.5 on typical medium-relief tile; per-class fit better than global in ≥1 class.

---

## Dependencies and Considerations

- **Single source of truth**: keep `DEFAULT_GRID_SIZE = 128` as the single constant; do not hard-code 128 elsewhere.
- **No new npm deps** requested. `geotiff 3.x` already ships `fromArray`/`writeArray`; Express + CORS already set up for server.
- **API key**: OpenTopography rate-limits; env `OPENTOPO_API_KEY` is optional; code must degrade to CGIAR-CSI AWS tiles gracefully when key absent or server returns 429.
- **GeoTIFF write spec**: Float32, noData = `-9999`, EPSG:4326 only (match what SRTM endpoint supplies). Projected CRS GeoTIFF export leaves out GeoKeys and just writes heights with bounding comment in file-name or download toast.
- **Preservation**: Every existing public function signature in `pipeline.ts` stays identical. Existing state in `App.tsx` is not renamed.

---

## Risks

| Risk | Handling / Fallback |
|---|---|
| OpenTopography API down / rate-limited / key absent | Secondary CGIAR-CSI 5° tile URL; tertiary: in-app message "SRTM service unavailable; use manual reference DEM upload" (existing path still works). |
| GeoTIFF classifies incorrectly (e.g. single-band 8-bit hillshade mis-routed to "as-imagery" → DA V2 wastes time) | User override `kind-toggle` button; auto-detect default wins only when confident. |
| SRTM tile contains `nodata = -32768` around oceans | `mask[]` in regressor ignores these pixels; per-pixel `Number.isFinite(v) && v > -1e5` guard. |
| Per-terrain class thresholds too brittle | Thresholds derived from **quantiles of local stats on each tile** (not magic hard-coded numbers). Quantile-based always adapts. |
| Exported GeoTIFF misaligns with original bounds | Use exact `ModelPixelScale = (east-west)/w × (north-south)/h`; ModelTiepoint at corner `(west, north, 0)` — standard GIS convention. |
| Regression instability when reference values vary little (super-flat tiles) | Add tiny Tikhonov ridge `λ=1e-6` to normal equations; `r2 < 0.15` → keep relative product and warn in UI instead of overwriting heights. |
