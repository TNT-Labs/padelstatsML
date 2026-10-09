import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs: the same build works at the root of a Pi and under
  // a path of a shared domain (shopbeautylab.it/padel/). The app has no
  // client-side routes, so every page load is at the application root.
  base: './',
  server: {
    port: 5173,
    proxy: {
      // Every route is under /api; the old /matches and /health entries
      // predate that and proxied nothing.
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
