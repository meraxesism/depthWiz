import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import * as Cesium from 'cesium'
import { fromBlob } from 'geotiff'
import { bilinearResample, buildRelativeDepthFromImage, buildRelativeDepthFromRaster, buildRasterTexture, calibrateByTerrainClass, calibrateRelativeDepth, computeMeanSlope, DEFAULT_GRID_SIZE, readRawRaster, readReferenceElevation, writeGeoTiffFloat32 } from './pipeline'
import type { CalibrationProduct, ClassFit, DepthProduct, TerrainCalibration, TerrainClass } from './pipeline'
import { depthAnythingModel, estimateRelativeDepth } from './depthAnything'
import './App.css'
import 'cesium/Build/Cesium/Widgets/widgets.css'

type GeoTiffKind = 'auto' | 'as-dem' | 'as-imagery'

type FitMetrics = {
  global: { a: number; b: number; r2: number }
  perClass: Record<TerrainClass, ClassFit>
} | null

type GeoMetadata = { width: number; height: number; bounds: [number, number, number, number]; crs: string; geographic: boolean }
type RasterStats = { minimum: number; maximum: number; mean: number; slope: number }

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [inputType, setInputType] = useState<'GeoTIFF' | 'RGB image' | null>(null)
  const [textureUrl, setTextureUrl] = useState<string | null>(null)
  const [geoMetadata, setGeoMetadata] = useState<GeoMetadata | null>(null)
  const [relativeDepth, setRelativeDepth] = useState<DepthProduct | null>(null)
  const [depthInput, setDepthInput] = useState<Blob | null>(null)
  const [calibration, setCalibration] = useState<CalibrationProduct | null>(null)
  const [rasterHeights, setRasterHeights] = useState<number[] | null>(null)
  const [rasterStats, setRasterStats] = useState<RasterStats | null>(null)
  const [referenceName, setReferenceName] = useState<string | null>(null)
  const [status, setStatus] = useState('Waiting for imagery')
  const [apiStatus, setApiStatus] = useState('Checking API')
  const [processing, setProcessing] = useState(false)
  const [flythrough, setFlythrough] = useState(false)
  const viewerElement = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<Cesium.Viewer | null>(null)
  const primitiveRef = useRef<Cesium.Primitive | null>(null)
  const centerRef = useRef<[number, number]>([0, 0])
  const [geoTiffKind, setGeoTiffKind] = useState<GeoTiffKind>('auto')
  const [autoDetectedKind, setAutoDetectedKind] = useState<GeoTiffKind>(null!)
  const [terrainClass, setTerrainClass] = useState<TerrainClass[] | null>(null)
  const [fitMetrics, setFitMetrics] = useState<FitMetrics>(null)
  const [srtmStatus, setSrtmStatus] = useState<string>('SRTM ready')

  useEffect(() => { fetch('/api/health').then((response) => response.ok ? setApiStatus('API online') : setApiStatus('API error')).catch(() => setApiStatus('API offline')) }, [])

  useEffect(() => {
    if (!viewerElement.current) return
    const viewer = new Cesium.Viewer(viewerElement.current, {
      animation: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      navigationHelpButton: false,
      sceneModePicker: true,
      selectionIndicator: false,
      timeline: false,
      baseLayer: Cesium.ImageryLayer.fromProviderAsync(
        Cesium.TileMapServiceImageryProvider.fromUrl(
          Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII'),
        ),
        {},
      ),
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    })
    viewer.scene.globe.enableLighting = true
    viewer.scene.globe.showGroundAtmosphere = true
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#9ab094')
    viewer.scene.globe.show = true
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = true
    viewer.scene.fog.enabled = true
    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#c5d5d0')
    viewer.camera.percentageChanged = 0.01
    const ssc = viewer.scene.screenSpaceCameraController
    ssc.enableCollisionDetection = false
    ssc.enableInputs = true
    ssc.enableLook = true
    ssc.enableRotate = true
    ssc.enableTilt = true
    ssc.enableTranslate = true
    ssc.enableZoom = true
    ssc.minimumZoomDistance = 1
    ssc.maximumZoomDistance = Infinity
    ssc.inertiaSpin = 0.9
    ssc.inertiaTranslate = 0.9
    ssc.inertiaZoom = 0.8
    ssc.rotateEventTypes = [Cesium.CameraEventType.LEFT_DRAG, { eventType: Cesium.CameraEventType.LEFT_DRAG, modifier: Cesium.KeyboardEventModifier.SHIFT }]
    ssc.tiltEventTypes = [Cesium.CameraEventType.RIGHT_DRAG, Cesium.CameraEventType.MIDDLE_DRAG]
    ssc.zoomEventTypes = [Cesium.CameraEventType.WHEEL, Cesium.CameraEventType.PINCH]
    ssc.translateEventTypes = [Cesium.CameraEventType.RIGHT_DRAG, { eventType: Cesium.CameraEventType.LEFT_DRAG, modifier: Cesium.KeyboardEventModifier.CTRL }]
    ssc.lookEventTypes = [{ eventType: Cesium.CameraEventType.LEFT_DRAG, modifier: Cesium.KeyboardEventModifier.SHIFT }]
    viewerRef.current = viewer
    return () => { viewer.destroy(); viewerRef.current = null }
  }, [])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !flythrough) return
    let step = 0
    const timer = window.setInterval(() => {
      step += 1
      const [lon, lat] = centerRef.current
      const geographic = geoMetadata?.geographic === true
      const baseAlt = geographic ? 25000 : 18000
      const altitudeVariation = baseAlt * 0.35 * Math.sin(step * 0.06)
      const altitude = baseAlt + altitudeVariation
      const heading = Cesium.Math.toRadians(step * 3.2)
      const pitch = Cesium.Math.toRadians(-50 - 20 * Math.sin(step * 0.09))
      const roll = Cesium.Math.toRadians(Math.sin(step * 0.04) * 4)
      const target = Cesium.Cartesian3.fromDegrees(lon, lat, (rasterStats?.mean ?? 0))
      const offset = new Cesium.HeadingPitchRange(heading, pitch, altitude)
      viewer.camera.lookAt(target, offset)
      void roll
    }, 110)
    return () => {
      window.clearInterval(timer)
      if (!viewer.isDestroyed()) viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY)
    }
  }, [flythrough, geoMetadata, rasterStats])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !rasterHeights) return
    if (primitiveRef.current) viewer.scene.primitives.remove(primitiveRef.current)
    const size = DEFAULT_GRID_SIZE
    const geographicBounds = geoMetadata?.geographic === true ? geoMetadata.bounds : null
    const extent = geographicBounds ?? [-0.0375, -0.0375, 0.0375, 0.0375]
    const centerLon = (extent[0] + extent[2]) / 2
    const centerLat = (extent[1] + extent[3]) / 2
    centerRef.current = [centerLon, centerLat]
    const positions = new Float64Array(size * size * 3)
    const normals = new Float32Array(size * size * 3)
    const indices = new Uint32Array((size - 1) * (size - 1) * 6)
    const uv = new Float32Array(size * size * 2)
    const heightsOnly: number[] = []
    for (let row = 0; row < size; row += 1) for (let column = 0; column < size; column += 1) {
      const nX = column / (size - 1), nY = row / (size - 1)
      const lon = extent[0] + nX * (extent[2] - extent[0])
      const lat = extent[1] + nY * (extent[3] - extent[1])
      const h = rasterHeights[row * size + column] ?? 0
      heightsOnly.push(h)
      const position = Cesium.Cartesian3.fromDegrees(lon, lat, h)
      const offset = (row * size + column) * 3
      positions[offset] = position.x; positions[offset + 1] = position.y; positions[offset + 2] = position.z
      const normal = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormal(position, new Cesium.Cartesian3())
      normals[offset] = normal.x; normals[offset + 1] = normal.y; normals[offset + 2] = normal.z
      const uvOffset = (row * size + column) * 2
      uv[uvOffset] = nX; uv[uvOffset + 1] = nY
    }
    let index = 0
    for (let row = 0; row < size - 1; row += 1) for (let column = 0; column < size - 1; column += 1) { const topLeft = row * size + column, topRight = topLeft + 1, bottomLeft = topLeft + size, bottomRight = bottomLeft + 1; indices[index++] = topLeft; indices[index++] = bottomLeft; indices[index++] = topRight; indices[index++] = topRight; indices[index++] = bottomLeft; indices[index++] = bottomRight }
    const boundingSphere = Cesium.BoundingSphere.fromVertices(positions)
    const geometry = new Cesium.Geometry({
      attributes: {
        position: new Cesium.GeometryAttribute({ componentDatatype: Cesium.ComponentDatatype.DOUBLE, componentsPerAttribute: 3, values: positions }),
        normal: new Cesium.GeometryAttribute({ componentDatatype: Cesium.ComponentDatatype.FLOAT, componentsPerAttribute: 3, values: normals }),
        st: new Cesium.GeometryAttribute({ componentDatatype: Cesium.ComponentDatatype.FLOAT, componentsPerAttribute: 2, values: uv }),
      } as Cesium.GeometryAttributes,
      indices,
      primitiveType: Cesium.PrimitiveType.TRIANGLES,
      boundingSphere,
    })
    const material = textureUrl ? new Cesium.Material({ fabric: { type: 'Image', uniforms: { image: textureUrl } } }) : undefined
    const appearance = material
      ? new Cesium.MaterialAppearance({ material, faceForward: true, translucent: false, flat: false })
      : new Cesium.PerInstanceColorAppearance({ flat: false, translucent: true, closed: false, faceForward: true })
    primitiveRef.current = viewer.scene.primitives.add(new Cesium.Primitive({
      geometryInstances: new Cesium.GeometryInstance({
        geometry,
        attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(Cesium.Color.fromCssColorString(inputType === 'GeoTIFF' ? '#9a8060' : '#5e8064').withAlpha(0.94)) },
      }),
      appearance,
      asynchronous: false,
      compressVertices: false,
      releaseGeometryInstances: false,
      allowPicking: false,
    }))
    const validHeights = heightsOnly.filter(Number.isFinite)
    const meanH = validHeights.length ? validHeights.reduce((a, b) => a + b, 0) / validHeights.length : 0
    const maxH = validHeights.length ? Math.max(...validHeights) : 0
    const minH = validHeights.length ? Math.min(...validHeights) : 0
    const relief = Math.max(1, maxH - minH)
    const extentWidthDeg = Math.max(extent[2] - extent[0], extent[3] - extent[1])
    const extentMeters = extentWidthDeg * 111320
    const cameraRange = Math.max(relief * 18, extentMeters * 1.4, geographicBounds ? 22000 : 16000)
    const target = Cesium.Cartesian3.fromDegrees(centerLon, centerLat, meanH)
    const heading = Cesium.Math.toRadians(320)
    const pitch = Cesium.Math.toRadians(-48)
    const destination = Cesium.Cartesian3.fromDegrees(centerLon, centerLat, meanH + cameraRange * 0.82)
    viewer.camera.flyTo({
      destination,
      duration: 1.2,
      orientation: { heading, pitch, roll: 0 },
      complete: () => {
        if (viewer.isDestroyed()) return
        const hpr = new Cesium.HeadingPitchRange(heading, pitch, cameraRange)
        viewer.camera.lookAt(target, hpr)
        setTimeout(() => { if (!viewer.isDestroyed()) viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY) }, 200)
      },
    })
    return () => { if (primitiveRef.current && !viewer.isDestroyed()) { viewer.scene.primitives.remove(primitiveRef.current); primitiveRef.current = null } }
  }, [rasterHeights, geoMetadata, textureUrl, inputType])

  const handleImagery = async (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0]
    if (!nextFile) return
    const isGeo = /\.(tif|tiff)$/i.test(nextFile.name)
    setFile(nextFile); setInputType(isGeo ? 'GeoTIFF' : 'RGB image'); setGeoMetadata(null); setRelativeDepth(null); setDepthInput(null); setCalibration(null); setRasterHeights(null); setRasterStats(null); setReferenceName(null); setTerrainClass(null); setFitMetrics(null); setGeoTiffKind('auto'); setAutoDetectedKind(null!); setSrtmStatus('SRTM ready'); setStatus('Reading imagery')
    const uploadData = new FormData()
    uploadData.append('imagery', nextFile)
    fetch('/api/upload', { method: 'POST', body: uploadData }).catch(() => setApiStatus('API upload unavailable'))
    try {
      if (isGeo) {
        const tiffImage = await (await fromBlob(nextFile)).getImage()
        let geo: GeoMetadata | null = null
        try {
          const rawBounds = tiffImage.getBoundingBox()
          const keys = tiffImage.getGeoKeys() ?? {}
          const [west, south, east, north] = rawBounds
          const geographic = Number.isFinite(west) && Number.isFinite(east) && Number.isFinite(south) && Number.isFinite(north) && west >= -180 && east <= 180 && south >= -90 && north <= 90
          const crs = keys.ProjectedCSTypeGeoKey ? `EPSG:${keys.ProjectedCSTypeGeoKey}` : keys.GeographicTypeGeoKey ? `EPSG:${keys.GeographicTypeGeoKey}` : geographic ? 'Geographic (EPSG:4326 implied)' : 'CRS metadata present'
          geo = { width: tiffImage.getWidth(), height: tiffImage.getHeight(), bounds: [west, south, east, north], crs, geographic }
        } catch {
          geo = { width: tiffImage.getWidth(), height: tiffImage.getHeight(), bounds: [0, 0, 1, 1], crs: 'No georeferencing', geographic: false }
        }
        if (geo) setGeoMetadata(geo)
        const samplesPerPixel = (tiffImage.getSamplesPerPixel?.() as number | undefined) ?? 1
        const bitsPerSample = (tiffImage.getBitsPerSample?.() as number[] | number | undefined)
        const bps = Array.isArray(bitsPerSample) ? bitsPerSample[0] ?? 32 : (bitsPerSample ?? 32)
        let detected: GeoTiffKind = 'as-dem'
        try {
          const bands = (await tiffImage.readRasters({ samples: [0], width: 32, height: 32 })) as unknown as (Float32Array | Uint16Array | Int16Array | Uint8Array)[]
          const sample0 = bands[0] ?? new Uint8Array(32 * 32)
          const mini = Math.min(...Array.from(sample0, Number))
          const maxi = Math.max(...Array.from(sample0, Number))
          const rgbBands = samplesPerPixel >= 3
          const looksByte = bps === 8 && mini >= 0 && maxi <= 255
          detected = rgbBands || looksByte ? 'as-imagery' : 'as-dem'
        } catch { /* keep 'as-dem' default */ }
        setAutoDetectedKind(detected)
        const effectiveKind: GeoTiffKind = geoTiffKind === 'auto' ? detected : geoTiffKind
        if (textureUrl) URL.revokeObjectURL(textureUrl)
        if (effectiveKind === 'as-imagery') {
          let rasterImage: Blob | null = null
          try {
            const tex = await buildRasterTexture(nextFile, 512, 512)
            rasterImage = tex.blob
            setTextureUrl(URL.createObjectURL(tex.blob))
          } catch {
            setTextureUrl(null)
          }
          if (!rasterImage) throw new Error('GeoTIFF imagery could not be decoded')
          setDepthInput(rasterImage)
          const depth = await buildRelativeDepthFromImage(rasterImage, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
          const product = calibrateRelativeDepth(depth, null)
          const heights = product.values.map((value) => 8 + value * 180)
          setRelativeDepth(depth); setCalibration(product); setRasterHeights(heights)
          const slope = computeMeanSlope(heights, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, undefined, false)
          setRasterStats({ minimum: product.minimum, maximum: product.maximum, mean: product.mean, slope })
          setStatus(geo?.geographic ? 'GeoTIFF imagery loaded · run DA V2, then calibrate with SRTM' : 'GeoTIFF imagery loaded · no geographic bounds (use DA V2)')
        } else {
          setDepthInput(null)
          try {
            const tex = await buildRasterTexture(nextFile, 512, 512)
            setTextureUrl(URL.createObjectURL(tex.blob))
          } catch {
            setTextureUrl(null)
          }
          const rawRaster = await readRawRaster(nextFile, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
          const relative = await buildRelativeDepthFromRaster(nextFile, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
          const geoProduct: CalibrationProduct = { ...rawRaster, source: 'GeoTIFF raster', isMetric: true }
          setRelativeDepth(relative); setCalibration(geoProduct)
          setRasterHeights(rawRaster.values.map((v) => Number.isFinite(v) ? v : rawRaster.mean))
          const slope = computeMeanSlope(rawRaster.values, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, geo?.geographic === true ? geo.bounds : undefined, true)
          setRasterStats({ minimum: rawRaster.minimum, maximum: rawRaster.maximum, mean: rawRaster.mean, slope })
          setStatus(geo?.geographic ? 'GeoTIFF DEM ready · georeferenced' : 'GeoTIFF DEM loaded · bounds are not geographic (local extent used)')
        }
      } else {
        setDepthInput(nextFile)
        if (textureUrl) URL.revokeObjectURL(textureUrl)
        setTextureUrl(URL.createObjectURL(nextFile))
        const depth = await buildRelativeDepthFromImage(nextFile, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
        const product = calibrateRelativeDepth(depth, null)
        const heights = product.values.map((value) => 8 + value * 180)
        setRelativeDepth(depth); setCalibration(product); setRasterHeights(heights)
        const slope = computeMeanSlope(heights, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, undefined, false)
        setRasterStats({ minimum: product.minimum, maximum: product.maximum, mean: product.mean, slope })
        setStatus('Ready for Depth Anything V2')
      }
    } catch (e) { setStatus(`Could not read imagery: ${e instanceof Error ? e.message : 'parse error'}`) }
  }

  const runInference = async () => {
    if (!file || !relativeDepth) return
    if (inputType === 'GeoTIFF') {
      const detected: GeoTiffKind = autoDetectedKind ?? 'as-dem'
      const effectiveKind: GeoTiffKind = geoTiffKind === 'auto' ? detected : geoTiffKind
      if (effectiveKind === 'as-dem') return
    }
    setProcessing(true); setStatus(`Loading ${depthAnythingModel}`)
    try {
      const values = await estimateRelativeDepth(depthInput ?? file, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
      const min = Math.min(...values)
      const max = Math.max(...values)
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length
      setRelativeDepth({ values, minimum: min, maximum: max, mean })
      setCalibration({ values, minimum: min, maximum: max, mean, source: 'relative only', isMetric: false })
      const heights = values.map((value) => 8 + value * 180)
      setRasterHeights(heights)
      const slope = computeMeanSlope(heights, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, undefined, false)
      setRasterStats({ minimum: min, maximum: max, mean, slope })
      setFitMetrics(null); setTerrainClass(null)
      setStatus(inputType === 'GeoTIFF' ? 'Depth Anything V2 ready · calibrate with SRTM now' : 'Depth Anything V2 relative depth ready')
    } catch { setStatus('Depth Anything V2 unavailable · fallback retained') } finally { setProcessing(false) }
  }

  const handleReference = async (event: ChangeEvent<HTMLInputElement>) => {
    const referenceFile = event.target.files?.[0]
    if (!referenceFile || !relativeDepth) return
    const reference = await readReferenceElevation(referenceFile, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
    const calibrated: TerrainCalibration = calibrateByTerrainClass(relativeDepth, reference, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
    setReferenceName(referenceFile.name); setCalibration({ ...calibrated, source: 'reference DEM' })
    setRasterHeights(calibrated.values.map((v) => Number.isFinite(v) ? v : calibrated.mean))
    const useBounds = geoMetadata?.geographic === true ? geoMetadata.bounds : undefined
    const slope = computeMeanSlope(calibrated.values, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, useBounds, true)
    setRasterStats({ minimum: calibrated.minimum, maximum: calibrated.maximum, mean: calibrated.mean, slope })
    setFitMetrics({ global: calibrated.fit.global, perClass: calibrated.fit.perClass })
    setTerrainClass(calibrated.terrainClass)
    setStatus('Metric DSM calibrated from reference raster (per-terrain regression)')
  }

  const autoCalibrateSrtm = async () => {
    if (!relativeDepth || !geoMetadata || !geoMetadata.geographic) { setStatus('Need georeferenced imagery with relative depth for SRTM'); return }
    setProcessing(true); setSrtmStatus('Fetching SRTM tile…')
    try {
      const [west, south, east, north] = geoMetadata.bounds
      const qs = new URLSearchParams({ west: String(west), south: String(south), east: String(east), north: String(north) })
      const response = await fetch(`/api/srtm?${qs.toString()}`)
      const payload = await response.json()
      if (!payload.ok) throw new Error(payload.error ?? 'SRTM fetch failed')
      setSrtmStatus(`SRTM loaded · ${payload.source}`)
      const resampled = bilinearResample(payload.values as number[], payload.width as number, payload.height as number, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, -32768)
      const reference: DepthProduct = { values: resampled, minimum: payload.minimum as number, maximum: payload.maximum as number, mean: payload.mean as number }
      const calibrated = calibrateByTerrainClass(relativeDepth, reference, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
      if (calibrated.fit.global.r2 < 0.15) setStatus(`SRTM fit weak (R²=${calibrated.fit.global.r2.toFixed(2)}) — relative product retained`)
      setCalibration({ ...calibrated, source: 'SRTM auto-calibrated' })
      setRasterHeights(calibrated.values.map((v) => Number.isFinite(v) ? v : calibrated.mean))
      const useBounds = geoMetadata.geographic ? geoMetadata.bounds : undefined
      const slope = computeMeanSlope(calibrated.values, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, useBounds, true)
      setRasterStats({ minimum: calibrated.minimum, maximum: calibrated.maximum, mean: calibrated.mean, slope })
      setFitMetrics({ global: calibrated.fit.global, perClass: calibrated.fit.perClass })
      setTerrainClass(calibrated.terrainClass)
      setSrtmStatus(`R²=${calibrated.fit.global.r2.toFixed(3)} · source ${payload.source}`)
      setStatus(geoMetadata.geographic ? 'SRTM metric DSM calibrated (per-terrain)' : 'SRTM calibrated · local extent')
    } catch (e) { setStatus(`SRTM unavailable: ${e instanceof Error ? e.message : 'network'}`); setSrtmStatus('SRTM failed · use manual DEM') } finally { setProcessing(false) }
  }

  const rerouteGeoTiffByKind = async () => {
    if (!file || inputType !== 'GeoTIFF') return
    setProcessing(true); setStatus('Reclassifying GeoTIFF…')
    try {
      const detected: GeoTiffKind = autoDetectedKind ?? 'as-dem'
      const effectiveKind: GeoTiffKind = geoTiffKind === 'auto' ? detected : geoTiffKind
      if (textureUrl) URL.revokeObjectURL(textureUrl)
      if (effectiveKind === 'as-imagery') {
        let rasterImage: Blob | null = null
        try {
          const tex = await buildRasterTexture(file, 512, 512)
          rasterImage = tex.blob
          setTextureUrl(URL.createObjectURL(tex.blob))
        } catch { setTextureUrl(null) }
        if (!rasterImage) throw new Error('GeoTIFF imagery could not be decoded')
        setDepthInput(rasterImage)
        const depth = await buildRelativeDepthFromImage(rasterImage, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
        const product = calibrateRelativeDepth(depth, null)
        const heights = product.values.map((value) => 8 + value * 180)
        setRelativeDepth(depth); setCalibration(product); setRasterHeights(heights)
        const slope = computeMeanSlope(heights, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, undefined, false)
        setRasterStats({ minimum: product.minimum, maximum: product.maximum, mean: product.mean, slope })
        setFitMetrics(null); setTerrainClass(null)
        setStatus(geoMetadata?.geographic ? 'Treating GeoTIFF as imagery · DA V2 + SRTM next' : 'Treating GeoTIFF as imagery · run DA V2')
      } else {
        setDepthInput(null)
        try {
          const tex = await buildRasterTexture(file, 512, 512)
          setTextureUrl(URL.createObjectURL(tex.blob))
        } catch { setTextureUrl(null) }
        const rawRaster = await readRawRaster(file, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
        const relative = await buildRelativeDepthFromRaster(file, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
        const geoProduct: CalibrationProduct = { ...rawRaster, source: 'GeoTIFF raster', isMetric: true }
        setRelativeDepth(relative); setCalibration(geoProduct)
        setRasterHeights(rawRaster.values.map((v) => Number.isFinite(v) ? v : rawRaster.mean))
        const slope = computeMeanSlope(rawRaster.values, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, geoMetadata?.geographic === true ? geoMetadata.bounds : undefined, true)
        setRasterStats({ minimum: rawRaster.minimum, maximum: rawRaster.maximum, mean: rawRaster.mean, slope })
        setFitMetrics(null); setTerrainClass(null)
        setStatus('Treating GeoTIFF as DEM elevations')
      }
    } catch { setStatus('Could not reclassify GeoTIFF') } finally { setProcessing(false) }
  }

  const exportDSM = async () => {
    if (!rasterHeights) return
    try {
      const blob = await writeGeoTiffFloat32(
        rasterHeights,
        DEFAULT_GRID_SIZE,
        DEFAULT_GRID_SIZE,
        geoMetadata?.geographic === true ? geoMetadata.bounds : undefined,
        geoMetadata?.crs,
      )
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      const stamp = new Date()
      const pad = (n: number) => String(n).padStart(2, '0')
      const ts = `${stamp.getFullYear()}${pad(stamp.getMonth() + 1)}${pad(stamp.getDate())}-${pad(stamp.getHours())}${pad(stamp.getMinutes())}${pad(stamp.getSeconds())}`
      a.href = url
      a.download = `depthwiz-dsm-${ts}.tif`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      setTimeout(() => URL.revokeObjectURL(url), 2000)
      setStatus('DSM exported as GeoTIFF')
    } catch { setStatus('DSM export failed') }
  }

  return <main className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">D</span><span>Depth<span className="brand-accent">Wizard</span></span><small>UPLOAD MODE</small></div><div className="topbar-meta"><span className="live-dot" /> {apiStatus} <span className="divider" /> <span className="cesium-mark">◈</span> CesiumJS</div></header>
    <section className="intro"><div><p className="eyebrow">SINGLE-VIEW ELEVATION WORKBENCH</p><h1>Bring your imagery.<br /><em>See its surface.</em></h1></div><p className="intro-copy">Upload one RGB image or GeoTIFF. DepthWizard will build a relative surface, then calibrate it only when you provide a reference elevation raster.</p></section>
    <section className="input-descriptions"><article className="input-desc-card avoided"><p className="input-desc-tag">01 · IN-SAR</p><h3>Radar stereo — <em>avoided</em></h3><p>Phase difference between two radar passes. Accurate but requires specialized satellites and heavy processing — exactly the cost and sensor lock-in DepthWizard is designed to skip.</p></article><article className="input-desc-card geotiff"><p className="input-desc-tag">02 · GEO TIFF</p><h3>Imagery with coordinates baked in</h3><p>An ordinary TIFF with embedded bounds, CRS and tie points. When georeferenced, output is an <strong>absolute DSM</strong> with real metric heights calibrated against SRTM or your DEM.</p></article><article className="input-desc-card mono"><p className="input-desc-tag">03 · MONOCULAR</p><h3>Single-image depth — <em>relative only</em></h3><p>Depth Anything V2 estimates closer vs farther per pixel, but has no sense of scale. PNG/JPG imagery stays as a <strong>relative DSM</strong>; GeoTIFF imagery gets a regression fit against reference elevations to recover meters.</p></article></section>
    <section className="workspace"><aside className="control-panel"><div className="panel-heading"><span>01</span><h2>Imagery</h2></div><label className="dropzone"><input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" onChange={handleImagery} /><span className="upload-icon">↑</span><strong>{file ? 'Replace imagery' : 'Choose imagery'}</strong><small>PNG, JPG or GeoTIFF · up to 2 GB</small></label>{file && <div className="file-row"><span className="file-icon">▧</span><div><strong>{file.name}</strong><small>{inputType} · {(file.size / 1024 / 1024).toFixed(1)} MB</small></div><span className="check">✓</span></div>}{geoMetadata && <div className="metadata-row"><span>RASTER</span><b>{geoMetadata.width} × {geoMetadata.height}</b><span>BOUNDS</span><b>{geoMetadata.bounds.map((value) => value.toFixed(3)).join(', ')}</b><span>REFERENCE</span><b>{geoMetadata.crs}</b></div>}{inputType === 'GeoTIFF' && file && <div className="kind-toggle-row"><span className="kind-toggle-label">TREAT GEO TIFF AS</span><div className="kind-toggle-buttons"><button type="button" className={geoTiffKind === 'auto' ? 'active' : ''} onClick={() => setGeoTiffKind('auto')}>AUTO{autoDetectedKind ? ` · ${autoDetectedKind === 'as-imagery' ? 'IMG' : 'DEM'}` : ''}</button><button type="button" className={geoTiffKind === 'as-dem' ? 'active' : ''} onClick={() => setGeoTiffKind('as-dem')}>DEM</button><button type="button" className={geoTiffKind === 'as-imagery' ? 'active' : ''} onClick={() => setGeoTiffKind('as-imagery')}>IMAGERY</button></div><button type="button" className="kind-toggle-apply" onClick={rerouteGeoTiffByKind} disabled={processing}>RE-ROUTE →</button></div>}<div className="calibration-row"><span className={`calibration-dot ${calibration?.isMetric ? 'metric' : ''}`} /><div><strong>{calibration?.isMetric ? 'Metric DSM calibrated' : 'Relative depth only'}</strong><small>{referenceName ?? (geoMetadata?.geographic ? 'Auto-calibrate against SRTM or add a DEM' : 'Add DEM / GCP to calibrate')}</small></div><label className="reference-button">ADD DEM<input type="file" accept=".tif,.tiff" onChange={handleReference} disabled={!relativeDepth} /></label></div>{geoMetadata && <div className="srtm-row"><span className={`srtm-status ${srtmStatus.startsWith('R²=') ? 'ok' : srtmStatus.includes('failed') ? 'bad' : ''}`}>{srtmStatus.toUpperCase()}</span><button type="button" className="srtm-autofit-btn" onClick={autoCalibrateSrtm} disabled={!geoMetadata.geographic || !relativeDepth || processing}>AUTO-CALIBRATE WITH SRTM →</button></div>}<div className="model-row"><span className="model-mark">DA</span><div><strong>Depth Anything V2</strong><small>{status}</small></div></div><button className="run-button" onClick={runInference} disabled={!file || processing || !relativeDepth || (inputType === 'GeoTIFF' && (geoTiffKind === 'as-dem' || (geoTiffKind === 'auto' && autoDetectedKind && autoDetectedKind !== 'as-imagery')))}><span>{processing ? 'Running inference...' : 'Run Depth Anything V2'}</span><b>→</b></button>{fitMetrics && <div className="fit-row"><div className="fit-row-head"><span className="fit-tag">REGRESSION</span><strong>R² = {fitMetrics.global.r2.toFixed(3)}</strong><small>y = {fitMetrics.global.a.toFixed(2)}·x + {fitMetrics.global.b.toFixed(1)}</small></div><div className="fit-row-body">{(['Urban','Sparse','Hilly','Forested'] as TerrainClass[]).map((cls) => { const c = fitMetrics.perClass[cls]; const n = terrainClass?.filter((t) => t === cls).length ?? 0; return <div key={cls} className={`class-badge class-badge-${cls}`}><span className="class-badge-label">{cls.toUpperCase()}</span><span className="class-badge-count">N {n}</span><span className="class-badge-r2">R² {c ? c.r2.toFixed(2) : '—'}</span></div> })}</div></div>}<button type="button" className="export-button" onClick={exportDSM} disabled={!rasterHeights}><span>EXPORT DSM</span><b>↓</b><small>Float32 GeoTIFF</small></button></aside>
    <section className="viewer-panel"><div className="viewer-toolbar"><div><span className="status-pill"><span className="live-dot" /> {rasterHeights ? 'SURFACE READY' : 'WAITING FOR INPUT'}</span><span className="toolbar-label">{calibration?.isMetric ? 'METRIC DSM' : rasterHeights ? 'RELATIVE DSM' : 'NO PRODUCT'}</span>{fitMetrics && <span className="toolbar-fit">R² <b>{fitMetrics.global.r2.toFixed(2)}</b></span>}{geoMetadata?.geographic && <span className="toolbar-srtm">{srtmStatus}</span>}</div><div className="toolbar-actions"><button className="tool-active" onClick={() => {
        const viewer = viewerRef.current; if (!viewer) return;
        const [lon, lat] = centerRef.current;
        const meanH = rasterStats?.mean ?? 0;
        const relief = rasterStats ? Math.max(1, rasterStats.maximum - rasterStats.minimum) : 100;
        const geographic = geoMetadata?.geographic === true;
        const range = Math.max(relief * 18, geographic ? 22000 : 16000);
        const target = Cesium.Cartesian3.fromDegrees(lon, lat, meanH);
        const heading = Cesium.Math.toRadians(320);
        const pitch = Cesium.Math.toRadians(-48);
        const hpr = new Cesium.HeadingPitchRange(heading, pitch, range);
        viewer.camera.flyTo({
          destination: Cesium.Cartesian3.fromDegrees(lon, lat, meanH + range * 0.82),
          duration: 0.9,
          orientation: { heading, pitch, roll: 0 },
          complete: () => {
            if (viewer.isDestroyed()) return;
            viewer.camera.lookAt(target, hpr);
            setTimeout(() => { if (!viewer.isDestroyed()) viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY) }, 200);
          },
        });
      }}>⌖</button><button onClick={() => viewerRef.current?.camera.zoomIn(1800)}>＋</button><button onClick={() => viewerRef.current?.camera.zoomOut(1800)}>−</button></div></div><div className="terrain-stage"><div className="cesium-container" ref={viewerElement} /><div className="analysis-strip"><button className="active">SURFACE</button><button className={flythrough ? 'active flight' : ''} onClick={() => setFlythrough((value) => !value)}>{flythrough ? '■ STOP FLIGHT' : '▶ FLYTHROUGH'}</button></div><div className="stage-label label-one"><span>MAX HEIGHT</span><strong>{rasterStats ? `${rasterStats.maximum.toFixed(2)} ${calibration?.isMetric ? 'm' : 'relative'}` : '—'}</strong></div><div className="stage-label label-two"><span>MEAN SLOPE</span><strong>{rasterStats ? `${rasterStats.slope.toFixed(1)}°` : '—'}</strong></div>{!rasterHeights && <div className="empty-viewer"><strong>Upload imagery to begin</strong><small>Cesium is ready for your first surface</small></div>}<div className="cesium-watermark"><span>◈</span> CESIUMJS / TERRAIN VIEW</div></div><div className="viewer-footer"><div className="legend"><span className="legend-gradient" /><span>{rasterStats ? `${rasterStats.minimum.toFixed(2)} — ${rasterStats.maximum.toFixed(2)}` : 'NO HEIGHT DATA'} <b>{calibration?.isMetric ? 'METERS' : 'RELATIVE'}</b></span></div><div className="mesh-stats"><span>POINTS <b>{rasterHeights ? rasterHeights.length.toLocaleString() : '—'}</b></span><span>MEAN <b>{rasterStats ? rasterStats.mean.toFixed(2) : '—'}</b></span><span>STATUS <b className="confidence">{apiStatus}</b></span></div></div></section></section>
    <footer className="pipeline"><span className="pipeline-label">PIPELINE</span><span className="active-step">UPLOAD</span><b>›</b><span>RELATIVE DEPTH</span><b>›</b><span>CALIBRATE</span><b>›</b><span>DSM</span><b>›</b><span>CESIUM</span><span className="pipeline-note">No demo scenes · no fabricated measurements</span></footer>
  </main>
}

export default App
