/**
 * Lazy-loads OpenCV.js from jsDelivr. Cached after first successful load.
 * Returns the global `cv` namespace once it's fully initialized (cv.Mat ready).
 */

// We declare a minimal type for the global to keep things clean.
// The actual API surface we use is small.
declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cv?: any;
  }
}

const CV_CDN = 'https://docs.opencv.org/4.8.0/opencv.js';
// Fallback: cdnjs
const CV_FALLBACK = 'https://cdnjs.cloudflare.com/ajax/libs/opencv.js/4.8.0/opencv.js';

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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function waitForCvRuntime(timeoutMs = 30000): Promise<any> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      if (typeof window.cv !== 'undefined' && window.cv.Mat) {
        resolve(window.cv);
        return;
      }
      if (typeof window.cv !== 'undefined' && typeof window.cv.then === 'function') {
        // Some builds expose cv as a promise
        window.cv
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .then((mod: any) => {
            window.cv = mod;
            resolve(mod);
          })
          .catch(reject);
        return;
      }
      if (typeof window.cv !== 'undefined' && !window.cv.Mat) {
        // Standard build needs onRuntimeInitialized
        window.cv['onRuntimeInitialized'] = () => resolve(window.cv);
        return;
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error('OpenCV runtime init timeout'));
        return;
      }
      window.setTimeout(check, 100);
    };
    check();
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadOpenCV(): Promise<any> {
  if (cvPromise) return cvPromise;
  cvPromise = (async () => {
    try {
      await loadScript(CV_CDN);
    } catch {
      await loadScript(CV_FALLBACK);
    }
    return waitForCvRuntime();
  })().catch((err) => {
    cvPromise = null; // allow retry on next call
    throw err;
  });
  return cvPromise;
}
