import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In production the built bundle is served by nginx, which proxies /api to the
// backend service (see nginx.conf). For `npm run dev` outside Docker, Vite
// proxies /api to a locally running backend instead, and VITE_API_BASE can
// override where api.js sends requests.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.VITE_DEV_API_TARGET || 'http://localhost:8000', changeOrigin: true },
    },
  },
  // `npm run preview` serves the production build the same way nginx does:
  // static bundle + /api proxied to the backend.
  preview: {
    port: 4173,
    proxy: {
      '/api': { target: process.env.VITE_DEV_API_TARGET || 'http://localhost:8000', changeOrigin: true },
    },
  },
})
