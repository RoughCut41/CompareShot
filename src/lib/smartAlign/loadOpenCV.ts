/**
 * Lazy-loads OpenCV.js from the local same-origin URL. Cached after first load.
 * The opencv.js file is downloaded at build time by scripts/download-models.mjs
 * and served from /public, which avoids COEP/CORS issues that block external CDNs.
 */

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cv?: any;
  }
}

const OPENCV_URL = '/opencv.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let cvPromise: Promise<any> | null = null;

function loadScript(url: string, timeoutMs = 60000): Promise<void> {
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function waitForCvRuntime(timeoutMs = 60000): Promise<any> {
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
    await loadScript(OPENCV_URL);
    return await waitForCvRuntime();
  })().catch((err) => {
    cvPromise = null; // allow retry on next call
    throw err;
  });
  return cvPromise;
}
