import type { ServerResponse } from 'node:http'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import * as cesiumModule from 'vite-plugin-cesium'
import type { Plugin } from 'vite'

const cesium = (cesiumModule as unknown as { default: () => Plugin }).default

const apiDownFallback = (res: ServerResponse | undefined, status: number, payload: object) => {
  const body = JSON.stringify(payload)
  if (res && typeof (res as unknown as { end?: unknown }).end === 'function' && !(res as unknown as { headersSent: boolean }).headersSent) {
    const srv = res as unknown as { statusCode: number, setHeader: (a: string, b: string | number) => void, end: (buf: Uint8Array | string) => void }
    srv.statusCode = status
    srv.setHeader('content-type', 'application/json; charset=utf-8')
    srv.setHeader('content-length', String(Buffer.byteLength(body)))
    srv.end(body)
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), cesium()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
        secure: false,
        ws: false,
        configure: (proxy) => {
          let warned = false
          proxy.on('error', (err, _req, _res) => {
            const code = (err as NodeJS.ErrnoException).code
            if (code === 'ECONNREFUSED' || code === 'ECONNRESET') {
              if (!warned) {
                // eslint-disable-next-line no-console
                console.warn('[vite] DepthWiz API is OFFLINE (port 8787). Run "npm run dev:full" to launch both servers. Uploads & SRTM calibration will use offline mode.')
                warned = true
              }
              const reqAny = _req as unknown as { originalUrl?: string, url?: string }
              const resAny = _res as unknown as undefined | ServerResponse
              const url = reqAny.originalUrl ?? reqAny.url ?? ''
              if (url.startsWith('/api/health')) {
                apiDownFallback(resAny, 503, { ok: false, service: 'depthwiz-api', offline: true, message: 'Run npm run dev:full' })
              } else if (url.startsWith('/api/upload')) {
                apiDownFallback(resAny, 503, { ok: false, error: 'DepthWiz API is offline. Run: npm run dev:full', offline: true })
              } else if (url.startsWith('/api/srtm')) {
                apiDownFallback(resAny, 503, { ok: false, error: 'SRTM unavailable · API offline. Upload a reference DEM manually or run: npm run dev:full', offline: true, source: 'offline' })
              } else {
                apiDownFallback(resAny, 503, { ok: false, error: 'API offline', offline: true })
              }
              return
            }
            try { apiDownFallback(_res as unknown as ServerResponse | undefined, 502, { ok: false, error: err.message ?? 'Proxy error' }) } catch { /* no-op */ }
          })
        },
      },
    },
  },
})
