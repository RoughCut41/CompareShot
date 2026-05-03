/**
 * Lazy-loads OpenCV.js. Cached after first successful load.
 * Returns the global `cv` namespace once it's fully initialized (cv.Mat ready).
 *
 * Strategy: try a list of CDNs in order, then poll for cv.Mat availability.
 * Polling is the only approach that works reliably across all OpenCV.js
 * build variants (some expose cv as a thenable factory, some need
 * onRuntimeInitialized, some are ready synchronously).
 */

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cv?: any;
  }
}

const CDNS = [
  'https://docs.opencv.org/4.8.0/opencv.js',
  'https://cdnjs.cloudflare.com/ajax/libs/opencv.js/4.8.0/opencv.js',
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let cvPromise: Promise<any> | null = null;

function loadScript(url: string, timeoutMs = 30000): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-cv-src="${url}"]`);
    if (existing) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = url;
    script.async = true;
    script.dataset.cvSrc = url;
    const timer = window.setTimeout(() => {
      script.remove();
      reject(new Error(`Timeout loading ${url}`));
    }, timeoutMs);
    script.onload = () => {
      window.clearTimeout(timer);
      resolve();
    };
    script.onerror = () => {
      window.clearTimeout(timer);
      script.remove();
      reject(new Error(`Failed to load ${url}`));
    };
    document.head.appendChild(script);
  });
}

/**
 * Wait until window.cv exists AND has cv.Mat available.
 *
 * Some OpenCV.js builds set cv synchronously when the script loads (cv.Mat is
 * ready immediately). Others set cv as a "Module" object that fires
 * onRuntimeInitialized only after the WASM has loaded. A few builds expose cv
 * as a thenable factory. We sidestep all of this by:
 *   1. Setting onRuntimeInitialized as a hint (in case the build uses it)
 *   2. Polling cv.Mat in a loop
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function waitForCvRuntime(timeoutMs = 30000): Promise<any> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    let resolved = false;

    const finish = () => {
      if (resolved) return;
      if (typeof window.cv !== 'undefined' && window.cv.Mat) {
        resolved = true;
        resolve(window.cv);
      }
    };

    // Hint for builds that use the Emscripten init callback
    if (typeof window.cv !== 'undefined' && !window.cv.Mat) {
      try {
        window.cv['onRuntimeInitialized'] = finish;
      } catch {
        /* some builds make cv read-only — ignore */
      }
    }

    const tick = () => {
      if (resolved) return;
      finish();
      if (resolved) return;
      if (Date.now() - start > timeoutMs) {
        reject(new Error('OpenCV runtime init timeout'));
        return;
      }
      window.setTimeout(tick, 100);
    };
    tick();
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadOpenCV(): Promise<any> {
  if (cvPromise) return cvPromise;
  cvPromise = (async () => {
    let lastErr: Error | null = null;
    for (const url of CDNS) {
      try {
        await loadScript(url);
        const cv = await waitForCvRuntime();
        return cv;
      } catch (err) {
        lastErr = err instanceof Error ? err : new Error(String(err));
        console.warn('[CompareShot] OpenCV load failed for', url, '— trying next');
      }
    }
    throw lastErr ?? new Error('All OpenCV CDNs failed');
  })().catch((err) => {
    cvPromise = null; // allow retry on next call
    throw err;
  });
  return cvPromise;
}
