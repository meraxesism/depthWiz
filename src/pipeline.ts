import { fromArrayBuffer, fromBlob } from "geotiff";

export type DepthProduct = {
  values: number[];
  minimum: number;
  maximum: number;
  mean: number;
};

export type CalibrationProduct = DepthProduct & {
  source: "reference DEM" | "relative only" | "GeoTIFF raster" | "SRTM auto-calibrated";
  isMetric: boolean;
};

export type TextureProduct = {
  blob: Blob;
  samplesPerPixel: 1 | 3;
  width: number;
  height: number;
};

const summarize = (values: number[]): DepthProduct => {
  const valid = values.filter(Number.isFinite);
  const minimum = valid.length ? Math.min(...valid) : 0;
  const maximum = valid.length ? Math.max(...valid) : 0;
  const mean = valid.length
    ? valid.reduce((sum, value) => sum + value, 0) / valid.length
    : 0;
  return { values, minimum, maximum, mean };
};

const clampByte = (value: number): number =>
  Math.max(0, Math.min(255, Math.round(value)));

export const buildRasterTexture = async (
  file: Blob,
  width = 512,
  height = 512,
): Promise<TextureProduct> => {
  const image = await (await fromBlob(file)).getImage();
  const samples = (image.getSamplesPerPixel?.() ?? 1) as number;
  const tileWidth = image.getWidth();
  const tileHeight = image.getHeight();
  const samplesPerPixel: 1 | 3 = samples >= 3 ? 3 : 1;
  const source = (await image.readRasters({
    interleave: true,
    samples:
      samplesPerPixel === 3
        ? [0, 1, 2]
        : undefined,
    width,
    height,
  })) as Float32Array | Uint16Array | Int16Array | Uint8Array;
  const bands = samplesPerPixel;
  const stats: { min: number; max: number }[] = Array.from(
    { length: bands },
    () => ({ min: Infinity, max: -Infinity }),
  );
  for (let i = 0; i < source.length; i += 1) {
    const band = i % bands;
    const value = Number(source[i]);
    if (!Number.isFinite(value)) continue;
    if (value < stats[band].min) stats[band].min = value;
    if (value > stats[band].max) stats[band].max = value;
  }
  const ranges = stats.map((s) =>
    Number.isFinite(s.min) && Number.isFinite(s.max) && s.max > s.min
      ? s.max - s.min
      : 1,
  );
  const mins = stats.map((s) => (Number.isFinite(s.min) ? s.min : 0));
  const canvas =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(width, height)
      : (() => {
          const c = document.createElement("canvas");
          c.width = width;
          c.height = height;
          return c;
        })();
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas unavailable for raster texture");
  const output = context.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const srcOffset = (y * width + x) * bands;
      const dstOffset = (y * width + x) * 4;
      if (bands === 1) {
        const raw = Number(source[srcOffset]);
        const normalized = Number.isFinite(raw)
          ? (raw - mins[0]) / ranges[0]
          : 0;
        const gray = clampByte(normalized * 255);
        output.data[dstOffset] = gray;
        output.data[dstOffset + 1] = gray;
        output.data[dstOffset + 2] = gray;
      } else {
        const r = Number(source[srcOffset]);
        const g = Number(source[srcOffset + 1]);
        const b = Number(source[srcOffset + 2]);
        output.data[dstOffset] = Number.isFinite(r)
          ? clampByte(((r - mins[0]) / ranges[0]) * 255)
          : 0;
        output.data[dstOffset + 1] = Number.isFinite(g)
          ? clampByte(((g - mins[1]) / ranges[1]) * 255)
          : 0;
        output.data[dstOffset + 2] = Number.isFinite(b)
          ? clampByte(((b - mins[2]) / ranges[2]) * 255)
          : 0;
      }
      output.data[dstOffset + 3] = 255;
    }
  }
  context.putImageData(output, 0, 0);
  const blob =
    typeof (canvas as OffscreenCanvas).convertToBlob === "function"
      ? await (canvas as OffscreenCanvas).convertToBlob({ type: "image/png" })
      : await new Promise<Blob>((resolve, reject) =>
          (canvas as HTMLCanvasElement).toBlob(
            (b) => (b ? resolve(b) : reject(new Error("Blob failed"))),
            "image/png",
          ),
        );
  void tileWidth;
  void tileHeight;
  return { blob, samplesPerPixel, width, height };
};

export const readRawRaster = async (
  file: Blob,
  width = 128,
  height = 128,
): Promise<DepthProduct> => {
  const image = await (await fromBlob(file)).getImage();
  const raster = (await image.readRasters({
    interleave: true,
    samples: [0],
    width,
    height,
  })) as Float32Array | Uint16Array | Int16Array | Uint8Array;
  return summarize(Array.from(raster, Number));
};

export const buildRelativeDepthFromRaster = async (
  file: Blob,
  width = 128,
  height = 128,
): Promise<DepthProduct> => {
  const product = await readRawRaster(file, width, height);
  const range = Math.max(1, product.maximum - product.minimum);
  const normalized = product.values.map((value) =>
    Number.isFinite(value) ? (value - product.minimum) / range : 0,
  );
  return summarize(normalized);
};

export const readReferenceElevation = async (
  file: Blob,
  width = 128,
  height = 128,
): Promise<DepthProduct> => {
  return readRawRaster(file, width, height);
};

export const buildRelativeDepthFromImage = async (
  file: Blob,
  width = 128,
  height = 128,
): Promise<DepthProduct> => {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas unavailable");
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const pixels = context.getImageData(0, 0, width, height).data;
  const values = Array.from({ length: width * height }, (_, index) => {
    const offset = index * 4;
    return (
      (pixels[offset] * 0.299 +
        pixels[offset + 1] * 0.587 +
        pixels[offset + 2] * 0.114) /
      255
    );
  });
  return summarize(values);
};

export const calibrateRelativeDepth = (
  relative: DepthProduct,
  reference: DepthProduct | null,
): CalibrationProduct => {
  if (!reference)
    return { ...relative, source: "relative only", isMetric: false };
  const referenceRange = Math.max(1, reference.maximum - reference.minimum);
  const values = relative.values.map((value, index) => {
    const referenceValue = reference.values[index % reference.values.length];
    const referenceNormalized =
      (referenceValue - reference.minimum) / referenceRange;
    const blended = value * 0.35 + referenceNormalized * 0.65;
    return reference.minimum + blended * referenceRange;
  });
  return { ...summarize(values), source: "reference DEM", isMetric: true };
};

export const DEFAULT_GRID_SIZE = 128;

const DEG_TO_RAD = Math.PI / 180;
const EARTH_RADIUS_METERS = 6_378_137;

export const computeMeanSlope = (
  heights: number[],
  width: number,
  height: number,
  bounds?: [number, number, number, number],
  isMetric = false,
): number => {
  if (width < 2 || height < 2 || heights.length < width * height) return 0;
  let metersPerStepX = 1;
  let metersPerStepY = 1;
  if (bounds) {
    const [west, south, east, north] = bounds;
    const degreesX = Math.max(1e-9, east - west);
    const degreesY = Math.max(1e-9, north - south);
    const centerLat = (south + north) * 0.5 * DEG_TO_RAD;
    const metersPerDegX = EARTH_RADIUS_METERS * Math.cos(centerLat) * DEG_TO_RAD;
    const metersPerDegY = EARTH_RADIUS_METERS * DEG_TO_RAD;
    metersPerStepX = (degreesX * metersPerDegX) / (width - 1);
    metersPerStepY = (degreesY * metersPerDegY) / (height - 1);
  } else if (isMetric) {
    metersPerStepX = 30;
    metersPerStepY = 30;
  } else {
    const range = Math.max(
      1e-6,
      Math.max(...heights.filter(Number.isFinite)) -
        Math.min(...heights.filter(Number.isFinite)),
    );
    metersPerStepX = Math.max(range / Math.max(1, width), 1);
    metersPerStepY = Math.max(range / Math.max(1, height), 1);
  }
  let sum = 0;
  let count = 0;
  for (let row = 0; row < height - 1; row += 1) {
    for (let col = 0; col < width - 1; col += 1) {
      const idx = row * width + col;
      const dzx = heights[idx + 1] - heights[idx];
      const dzy = heights[idx + width] - heights[idx];
      if (!Number.isFinite(dzx) || !Number.isFinite(dzy)) continue;
      const slopeX = Math.atan(Math.abs(dzx) / metersPerStepX) / DEG_TO_RAD;
      const slopeY = Math.atan(Math.abs(dzy) / metersPerStepY) / DEG_TO_RAD;
      sum += Math.min(90, Math.sqrt(slopeX * slopeX + slopeY * slopeY));
      count += 1;
    }
  }
  return count ? sum / count : 0;
};

export type TerrainClass = "Urban" | "Sparse" | "Hilly" | "Forested";

export type LinearFit = { a: number; b: number; r2: number; predicted: number[] };

export type ClassFit = { count: number; a: number; b: number; r2: number };

export type TerrainCalibration = CalibrationProduct & {
  fit: {
    global: { a: number; b: number; r2: number };
    perClass: Record<TerrainClass, ClassFit>;
  };
  terrainClass: TerrainClass[];
};

export const bilinearResample = (
  src: number[] | ArrayLike<number>,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
  srcNoDataValue?: number,
): number[] => {
  const dst = new Array<number>(dstW * dstH);
  const sx = srcW > 1 ? (srcW - 1) / (dstW - 1) : 0;
  const sy = srcH > 1 ? (srcH - 1) / (dstH - 1) : 0;
  for (let y = 0; y < dstH; y += 1) {
    for (let x = 0; x < dstW; x += 1) {
      const fx = x * sx;
      const fy = y * sy;
      const x0 = Math.min(srcW - 2, Math.max(0, Math.floor(fx)));
      const y0 = Math.min(srcH - 2, Math.max(0, Math.floor(fy)));
      const x1 = x0 + 1;
      const y1 = y0 + 1;
      const tx = fx - x0;
      const ty = fy - y0;
      const get = (xi: number, yi: number): number | null => {
        const v = Number(src[yi * srcW + xi]);
        if (!Number.isFinite(v)) return null;
        if (srcNoDataValue !== undefined && Math.abs(v - srcNoDataValue) < 1e-6) return null;
        if (v < -1e5 || v > 1e7) return null;
        return v;
      };
      const z00 = get(x0, y0);
      const z10 = get(x1, y0);
      const z01 = get(x0, y1);
      const z11 = get(x1, y1);
      const samples: number[] = [];
      if (z00 !== null) samples.push(z00);
      if (z10 !== null) samples.push(z10);
      if (z01 !== null) samples.push(z01);
      if (z11 !== null) samples.push(z11);
      if (samples.length === 4) {
        const top = samples[0] * (1 - tx) + samples[1] * tx;
        const bottom = samples[2] * (1 - tx) + samples[3] * tx;
        dst[y * dstW + x] = top * (1 - ty) + bottom * ty;
      } else if (samples.length) {
        dst[y * dstW + x] = samples.reduce((s, v) => s + v, 0) / samples.length;
      } else {
        dst[y * dstW + x] = NaN;
      }
    }
  }
  return dst;
};

export const fitLinearRegression = (
  xs: number[],
  ys: number[],
  mask?: boolean[],
): LinearFit => {
  const n = Math.min(xs.length, ys.length);
  const LAMBDA = 1e-6;
  let count = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = 0; i < n; i += 1) {
    if (mask && !mask[i]) continue;
    const x = Number(xs[i]);
    const y = Number(ys[i]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    count += 1;
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
  }
  const predicted = new Array<number>(xs.length).fill(0);
  if (count < 2) return { a: 0, b: count ? sy / count : 0, r2: 0, predicted };
  const denom = count * (sxx + LAMBDA) - sx * sx;
  let a = 0;
  let b = sy / count;
  if (Math.abs(denom) > 1e-12) {
    a = (count * sxy - sx * sy) / denom;
    b = (sy - a * sx) / count;
  }
  let ssRes = 0;
  const meanY = sy / count;
  let ssTot = 0;
  for (let i = 0; i < n; i += 1) {
    if (mask && !mask[i]) continue;
    const x = Number(xs[i]);
    const y = Number(ys[i]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const yHat = a * x + b;
    predicted[i] = yHat;
    ssRes += (y - yHat) * (y - yHat);
    ssTot += (y - meanY) * (y - meanY);
  }
  const r2 = ssTot > 0 ? Math.max(0, Math.min(1, 1 - ssRes / ssTot)) : 0;
  return { a, b, r2, predicted };
};

const quantile = (sorted: number[], q: number): number => {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  const f = pos - lo;
  return sorted[lo] * (1 - f) + sorted[hi] * f;
};

export const classifyTerrain = (
  heights: number[],
  w: number,
  h: number,
): TerrainClass[] => {
  const n = w * h;
  const classes: TerrainClass[] = new Array(n).fill("Sparse");
  if (w < 3 || h < 3 || heights.length < n) return classes;
  const localStd = new Float64Array(n);
  const localRugosity = new Float64Array(n);
  const localSlope = new Float64Array(n);
  const sobelMag = new Float64Array(n);
  const valid = Number.isFinite;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const xMin = Math.max(0, x - 1);
      const xMax = Math.min(w - 1, x + 1);
      const yMin = Math.max(0, y - 1);
      const yMax = Math.min(h - 1, y + 1);
      const vals: number[] = [];
      for (let yy = yMin; yy <= yMax; yy += 1) {
        for (let xx = xMin; xx <= xMax; xx += 1) {
          const v = Number(heights[yy * w + xx]);
          if (valid(v)) vals.push(v);
        }
      }
      const i = y * w + x;
      const z = Number(heights[i]);
      if (vals.length >= 3) {
        const m = vals.length;
        const mean = vals.reduce((s, v) => s + v, 0) / m;
        let variance = 0;
        for (let k = 0; k < m; k += 1) variance += (vals[k] - mean) * (vals[k] - mean);
        localStd[i] = Math.sqrt(variance / m);
        const bilinear = mean;
        localRugosity[i] = valid(z) ? Math.abs(z - bilinear) : 0;
        const zc = valid(z) ? z : mean;
        const zxMinus = valid(heights[y * w + Math.max(0, x - 1)]) ? heights[y * w + Math.max(0, x - 1)] : zc;
        const zxPlus = valid(heights[y * w + Math.min(w - 1, x + 1)]) ? heights[y * w + Math.min(w - 1, x + 1)] : zc;
        const zyMinus = valid(heights[Math.max(0, y - 1) * w + x]) ? heights[Math.max(0, y - 1) * w + x] : zc;
        const zyPlus = valid(heights[Math.min(h - 1, y + 1) * w + x]) ? heights[Math.min(h - 1, y + 1) * w + x] : zc;
        const dx = (zxPlus - zxMinus) / 2;
        const dy = (zyPlus - zyMinus) / 2;
        const gx = zxPlus + 2 * zc + zxMinus - (zyPlus + 2 * zc + zyMinus);
        const gy = zxPlus + 2 * zc + zyPlus - (zxMinus + 2 * zc + zyMinus);
        localSlope[i] = (Math.atan(Math.sqrt(dx * dx + dy * dy)) / DEG_TO_RAD);
        sobelMag[i] = Math.sqrt(gx * gx + gy * gy);
      } else {
        localStd[i] = 0;
        localRugosity[i] = 0;
        localSlope[i] = 0;
        sobelMag[i] = 0;
      }
    }
  }
  const sortedStd = Array.from(localStd).sort((p, q) => p - q);
  const sortedRug = Array.from(localRugosity).sort((p, q) => p - q);
  const sortedSobel = Array.from(sobelMag).sort((p, q) => p - q);
  const tStdHill = quantile(sortedStd, 0.6);
  const tRugForest = quantile(sortedRug, 0.75);
  const tSobelUrban = quantile(sortedSobel, 0.7);
  const tSlopeHill = 20;
  const range = (sortedStd[sortedStd.length - 1] ?? 0) - (sortedStd[0] ?? 0);
  const stdModerate = range > 0 ? quantile(sortedStd, 0.35) : 0;
  for (let i = 0; i < n; i += 1) {
    if (localStd[i] > tStdHill && localSlope[i] > tSlopeHill) {
      classes[i] = "Hilly";
    } else if (localRugosity[i] > tRugForest) {
      classes[i] = "Forested";
    } else if (localStd[i] >= stdModerate && sobelMag[i] > tSobelUrban) {
      classes[i] = "Urban";
    } else {
      classes[i] = "Sparse";
    }
  }
  return classes;
};

export const calibrateByTerrainClass = (
  relative: DepthProduct,
  reference: DepthProduct,
  w: number,
  h: number,
  referenceSrcW?: number,
  referenceSrcH?: number,
): TerrainCalibration => {
  const refW = referenceSrcW ?? w;
  const refH = referenceSrcH ?? h;
  const resampled =
    refW === w && refH === h
      ? reference.values.slice()
      : bilinearResample(reference.values, refW, refH, w, h);
  const mask = new Array<boolean>(w * h);
  for (let i = 0; i < mask.length; i += 1) {
    const rv = Number(resampled[i]);
    const dv = Number(relative.values[i]);
    mask[i] = Number.isFinite(rv) && rv > -1e5 && rv < 1e7 && Number.isFinite(dv);
  }
  const globalFit = fitLinearRegression(relative.values, resampled, mask);
  const terrainClass = classifyTerrain(resampled, w, h);
  const byClass: Record<TerrainClass, { xs: number[]; ys: number[]; mask: boolean[] }> = {
    Urban: { xs: [], ys: [], mask: [] },
    Sparse: { xs: [], ys: [], mask: [] },
    Hilly: { xs: [], ys: [], mask: [] },
    Forested: { xs: [], ys: [], mask: [] },
  };
  for (let i = 0; i < terrainClass.length; i += 1) {
    const cls = terrainClass[i];
    byClass[cls].xs.push(Number(relative.values[i]));
    byClass[cls].ys.push(Number(resampled[i]));
    byClass[cls].mask.push(Boolean(mask[i]));
  }
  const perClass: Record<TerrainClass, ClassFit> = {
    Urban: { count: 0, a: 0, b: 0, r2: 0 },
    Sparse: { count: 0, a: 0, b: 0, r2: 0 },
    Hilly: { count: 0, a: 0, b: 0, r2: 0 },
    Forested: { count: 0, a: 0, b: 0, r2: 0 },
  };
  const classFitPredicted = new Map<TerrainClass, number[]>();
  for (const cls of Object.keys(byClass) as TerrainClass[]) {
    const bucket = byClass[cls];
    const fit = fitLinearRegression(bucket.xs, bucket.ys, bucket.mask);
    perClass[cls] = {
      count: bucket.mask.filter(Boolean).length,
      a: fit.a,
      b: fit.b,
      r2: fit.r2,
    };
    classFitPredicted.set(cls, fit.predicted);
  }
  const calibrated = new Array<number>(w * h);
  const classCursor: Record<TerrainClass, number> = { Urban: 0, Sparse: 0, Hilly: 0, Forested: 0 };
  for (let i = 0; i < calibrated.length; i += 1) {
    const cls = terrainClass[i];
    const cursor = classCursor[cls];
    classCursor[cls] = cursor + 1;
    const d = Number(relative.values[i]);
    const classR2 = perClass[cls].r2;
    const useClass = classR2 >= globalFit.r2 * 0.8 && perClass[cls].count >= 8;
    let value: number;
    if (useClass) {
      const preds = classFitPredicted.get(cls);
      value = preds && Number.isFinite(preds[cursor]) ? preds[cursor]! : perClass[cls].a * d + perClass[cls].b;
    } else if (Number.isFinite(globalFit.predicted[i])) {
      value = globalFit.predicted[i];
    } else {
      value = globalFit.a * d + globalFit.b;
    }
    calibrated[i] = Number.isFinite(value) ? value : reference.mean;
  }
  const stats = summarize(calibrated);
  return {
    ...stats,
    source: "reference DEM",
    isMetric: true,
    fit: {
      global: { a: globalFit.a, b: globalFit.b, r2: globalFit.r2 },
      perClass,
    },
    terrainClass,
  };
};

export const writeGeoTiffFloat32 = async (
  heightValues: number[],
  w: number,
  h: number,
  bounds?: [number, number, number, number],
  crs?: string,
): Promise<Blob> => {
  const NODATA = -9999;
  const clean = new Float32Array(w * h);
  for (let i = 0; i < w * h; i += 1) {
    const v = Number(heightValues[i]);
    clean[i] = Number.isFinite(v) ? v : NODATA;
  }
  const geoKeys: Record<string, number> | undefined = bounds && crs && /EPSG:4326|WGS 84|Geographic/i.test(crs)
    ? { GTModelTypeGeoKey: 2, GTRasterTypeGeoKey: 1, GeographicTypeGeoKey: 4326 }
    : undefined;
  const modelTiepoint = bounds ? [0, 0, 0, bounds[0], bounds[3], 0] : undefined;
  const modelPixelScale = bounds ? [(bounds[2] - bounds[0]) / w, (bounds[3] - bounds[1]) / h, 0] : undefined;
  const tiff = await (fromArrayBuffer as unknown as (
    data: Float32Array[],
    width: number,
    height: number,
    metadata?: Record<string, unknown>,
  ) => ReturnType<typeof fromBlob>)([clean], w, h, {
    width: w,
    height: h,
    samplesPerPixel: 1,
    bitsPerSample: 32,
    sampleFormat: 3,
    noData: String(NODATA),
    ...(geoKeys ? { geoKeys } : {}),
    ...(modelTiepoint ? { modelTiepoint } : {}),
    ...(modelPixelScale ? { modelPixelScale } : {}),
  } as Record<string, unknown>);
  const bytes = await (tiff as unknown as { toArrayBuffer(): Promise<ArrayBuffer> }).toArrayBuffer();
  return new Blob([bytes], { type: "image/tiff" });
};
