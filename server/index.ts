import cors from 'cors'
import express from 'express'
import multer from 'multer'
import { fromArrayBuffer } from 'geotiff'

const app = express()
const upload = multer({ limits: { fileSize: 2 * 1024 * 1024 * 1024 }, storage: multer.memoryStorage() })
const port = Number(process.env.PORT ?? 8787)
const openTopoKey = process.env.OPENTOPO_API_KEY ?? 'demoapikeyot2022'

const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value))

app.use(cors())
app.get('/api/health', (_request, response) => response.json({ ok: true, service: 'depthwiz-api', inference: 'browser' }))
app.post('/api/upload', upload.single('imagery'), (request, response) => {
  if (!request.file) return response.status(400).json({ ok: false, error: 'No imagery file supplied' })
  const isGeoTiff = /\.(tif|tiff)$/i.test(request.file.originalname)
  return response.json({ ok: true, name: request.file.originalname, bytes: request.file.size, type: isGeoTiff ? 'GeoTIFF' : 'RGB image', inference: 'Depth Anything V2 runs in the browser' })
})

app.get('/api/srtm', async (request, response) => {
  try {
    const west = Number(request.query.west)
    const south = Number(request.query.south)
    const east = Number(request.query.east)
    const north = Number(request.query.north)
    if (!Number.isFinite(west) || !Number.isFinite(south) || !Number.isFinite(east) || !Number.isFinite(north)) return response.status(400).json({ ok: false, error: 'Query west, south, east, north (degrees) required' })
    const clampedWest = clamp(west, -180, 180)
    const clampedEast = clamp(east, clampedWest + 1e-6, 180)
    const clampedSouth = clamp(south, -90, 90)
    const clampedNorth = clamp(north, clampedSouth + 1e-6, 90)
    const params = new URLSearchParams({ demtype: 'SRTMGL3', west: String(clampedWest), south: String(clampedSouth), east: String(clampedEast), north: String(clampedNorth), outputFormat: 'GTiff' })
    if (openTopoKey) params.set('API_Key', openTopoKey)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 30000)
    let tiffBuffer: ArrayBuffer | null = null
    let source = 'opentopography'
    try {
      const upstream = await fetch(`https://portal.opentopography.org/API/globaldem?${params.toString()}`, { signal: controller.signal })
      if (upstream.ok) {
        const data = await upstream.arrayBuffer()
        if (data.byteLength >= 4096) tiffBuffer = data
      }
    } catch {
      tiffBuffer = null
    } finally {
      clearTimeout(timeout)
    }
    if (!tiffBuffer) {
      source = 'cgiar-csi-fallback'
      const tileWest = Math.floor((clampedWest + 180) / 5) + 1
      const tileEast = Math.floor((clampedEast + 180) / 5) + 1
      const tileSouth = Math.floor((60 - clampedNorth) / 5) + 1
      const tileNorth = Math.floor((60 - clampedSouth) / 5) + 1
      for (let tx = tileWest; tx <= tileEast; tx += 1) {
        for (let ty = tileNorth; ty <= tileSouth; ty += 1) {
          if (tx < 1 || tx > 72 || ty < 1 || ty > 24) continue
          const tag = `${tx.toString().padStart(2, '0')}_${ty.toString().padStart(2, '0')}`
          try {
            const tileCtrl = new AbortController()
            const tileTimer = setTimeout(() => tileCtrl.abort(), 25000)
            const resp = await fetch(`https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_${tag}.zip`, { signal: tileCtrl.signal })
            clearTimeout(tileTimer)
            if (!resp.ok) continue
            const raw = await resp.arrayBuffer()
            if (raw.byteLength >= 4096) { tiffBuffer = raw; break }
          } catch { /* no-op */ }
        }
        if (tiffBuffer) break
      }
    }
    if (!tiffBuffer) return response.status(502).json({ ok: false, error: 'SRTM upstream unavailable · upload a reference DEM manually', source })
    const tiff = await fromArrayBuffer(tiffBuffer)
    const image = await tiff.getImage()
    const width = image.getWidth()
    const height = image.getHeight()
    let fileBounds: [number, number, number, number] = [clampedWest, clampedSouth, clampedEast, clampedNorth]
    try { fileBounds = image.getBoundingBox() as [number, number, number, number] } catch { /* no-op */ }
    const raster = (await image.readRasters({ samples: [0], interleave: true })) as Float32Array | Uint16Array | Int16Array | Uint8Array
    const values: number[] = new Array(raster.length)
    for (let i = 0; i < raster.length; i += 1) values[i] = Number(raster[i])
    const valid = values.filter((v) => Number.isFinite(v) && v > -1e5 && v < 1e7)
    const minimum = valid.length ? Math.min(...valid) : 0
    const maximum = valid.length ? Math.max(...valid) : 0
    const mean = valid.length ? valid.reduce((s, v) => s + v, 0) / valid.length : 0
    return response.json({ ok: true, source, values, width, height, minimum, maximum, mean, bounds: fileBounds, crs: 'EPSG:4326' })
  } catch (err) {
    return response.status(500).json({ ok: false, error: err instanceof Error ? err.message : 'SRTM fetch failed' })
  }
})

const server = app.listen(port, '127.0.0.1', () => console.log(`DepthWizard API listening on http://127.0.0.1:${port} · http://localhost:${port}`))
server.on('error', (err: NodeJS.ErrnoException) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`[depthwiz-api] Port ${port} is already in use. Choose another: PORT=xxxx npm run server, or stop the process using port ${port}.`)
  } else {
    console.error('[depthwiz-api] Failed to start:', err.message)
  }
  process.exitCode = 1
})
