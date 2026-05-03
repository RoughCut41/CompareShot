import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: true,
  },
  // face-api.js performs feature detection at runtime to pick its TF backend.
  // Excluding it from pre-bundling avoids issues with the dynamic import path.
  optimizeDeps: {
    exclude: ['@vladmandic/face-api'],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Keep face-api in its own chunk so it's only fetched on Smart Align
          'face-api': ['@vladmandic/face-api'],
        },
      },
    },
  },
});
