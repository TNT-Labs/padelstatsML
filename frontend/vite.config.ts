import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Every route is under /api; the old /matches and /health entries
      // predate that and proxied nothing.
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
