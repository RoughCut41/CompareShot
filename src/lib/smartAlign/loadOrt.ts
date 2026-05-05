/**
 * Lazy-loads ONNX Runtime Web. Configures the runtime to find its WASM files
 * at /ort/ (copied at build time by the vite plugin).
 *
 * Inference is run on CPU via WASM with SIMD + multi-threading. We don't use
 * WebGPU because availability varies across browsers; WASM works everywhere.
 */
import * as ortNS from 'onnxruntime-web';

type OrtModule = typeof ortNS;

let ortPromise: Promise<OrtModule> | null = null;

export async function loadOrt(): Promise<OrtModule> {
  if (ortPromise) return ortPromise;
  ortPromise = (async () => {
    const ort = await import('onnxruntime-web');
    // WASM files are served from the same origin (copied by vite plugin)
    ort.env.wasm.wasmPaths = '/ort/';
    // Enable multi-threading if SharedArrayBuffer is available
    if (typeof SharedArrayBuffer !== 'undefined') {
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency ?? 4);
    } else {
      ort.env.wasm.numThreads = 1;
    }
    ort.env.wasm.simd = true;
    return ort;
  })().catch((err) => {
    ortPromise = null;
    throw err;
  });
  return ortPromise;
}
