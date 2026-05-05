import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { copyFileSync, existsSync, mkdirSync } from 'fs';

// Copy ONNX Runtime Web's WASM files to /public/ort so they're served from
// the same origin as the app. This avoids CORS issues with cross-origin WASM.
function copyOrtWasm() {
  return {
    name: 'copy-ort-wasm',
    buildStart() {
      const srcDir = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist');
      const destDir = path.resolve(__dirname, 'public/ort');
      if (!existsSync(destDir)) mkdirSync(destDir, { recursive: true });
      const files = [
        'ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd-threaded.jsep.wasm',
        'ort-wasm-simd-threaded.mjs',
        'ort-wasm-simd-threaded.jsep.mjs',
      ];
      for (const f of files) {
        const src = path.join(srcDir, f);
        const dest = path.join(destDir, f);
        if (existsSync(src)) {
          try {
            copyFileSync(src, dest);
          } catch (err) {
            console.warn('[copy-ort-wasm] failed to copy', f, err);
          }
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), copyOrtWasm()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: true,
    headers: {
      // Required for SharedArrayBuffer (multi-threaded WASM in ORT Web)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          ort: ['onnxruntime-web'],
        },
      },
    },
  },
});
