/**
 * Downloads ML models and libraries needed at runtime from GitHub Releases.
 *
 * - YOLO pose model is placed in public/models/ and loaded via ONNX Runtime Web.
 * - OpenCV.js is placed in public/ (root) and loaded as a script tag at runtime.
 *   It must be at root because the loader uses /opencv.js as a same-origin URL
 *   to bypass COEP cross-origin script blocking.
 *
 * GitHub Releases redirects to a signed S3 URL. We need to:
 *  1) Send a User-Agent header (some GitHub endpoints return 404 without one)
 *  2) Follow redirects manually so we can debug
 *
 * Runs automatically before `npm run build` and `npm run dev`.
 */
import { existsSync, mkdirSync, statSync, createWriteStream } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = resolve(__dirname, '..', 'public');
const MODELS_DIR = resolve(PUBLIC_DIR, 'models');

/**
 * Each entry specifies:
 *  - name: filename to save as
 *  - url:  download URL (GitHub Release asset)
 *  - destDir: where to put it locally
 *  - minSizeBytes: sanity check — error if downloaded file is smaller
 */
const ASSETS = [
  {
    name: 'yolo11s-pose.onnx',
    url: 'https://github.com/RoughCut41/CompareShot/releases/download/v0.1.0-models/yolo11s-pose.onnx',
    destDir: MODELS_DIR,
    minSizeBytes: 5 * 1024 * 1024,
  },
  {
    name: 'opencv.js',
    url: 'https://github.com/RoughCut41/CompareShot/releases/download/v0.1.0-models/opencv.js',
    destDir: PUBLIC_DIR,
    minSizeBytes: 5 * 1024 * 1024,
  },
];

// Ensure all destination directories exist
for (const asset of ASSETS) {
  if (!existsSync(asset.destDir)) {
    mkdirSync(asset.destDir, { recursive: true });
  }
}

async function fetchWithRedirects(url, maxRedirects = 5) {
  let currentUrl = url;
  for (let i = 0; i < maxRedirects; i++) {
    console.log(`[assets] GET ${currentUrl}`);
    const res = await fetch(currentUrl, {
      redirect: 'manual',
      headers: {
        'User-Agent': 'CompareShot-Build/1.0',
        Accept: 'application/octet-stream',
      },
    });
    console.log(`[assets]   → status ${res.status}`);
    if (res.status >= 200 && res.status < 300) {
      return res;
    }
    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get('location');
      if (!location) {
        throw new Error(`Redirect ${res.status} without Location header`);
      }
      console.log(`[assets]   → redirect to ${location}`);
      currentUrl = location;
      continue;
    }
    const body = await res.text().catch(() => '');
    throw new Error(
      `Download failed: ${res.status} ${res.statusText}${body ? ` — ${body.slice(0, 200)}` : ''}`
    );
  }
  throw new Error(`Too many redirects (>${maxRedirects})`);
}

async function downloadAsset(asset) {
  const dest = resolve(asset.destDir, asset.name);
  if (existsSync(dest)) {
    const size = statSync(dest).size;
    if (size >= asset.minSizeBytes) {
      console.log(
        `[assets] ${asset.name} already exists (${(size / 1024 / 1024).toFixed(1)} MB) — skipping`
      );
      return;
    }
    console.log(
      `[assets] ${asset.name} exists but looks incomplete (${size} bytes) — re-downloading`
    );
  }
  console.log(`[assets] Downloading ${asset.name} …`);
  const res = await fetchWithRedirects(asset.url);
  if (!res.body) {
    throw new Error('Response has no body');
  }
  await pipeline(res.body, createWriteStream(dest));
  const size = statSync(dest).size;
  console.log(`[assets] Saved ${asset.name} (${(size / 1024 / 1024).toFixed(1)} MB)`);
  if (size < asset.minSizeBytes) {
    throw new Error(`Downloaded file too small (${size} bytes) — likely an error page`);
  }
}

(async () => {
  try {
    for (const a of ASSETS) {
      await downloadAsset(a);
    }
    console.log('[assets] All assets ready.');
  } catch (err) {
    console.error('[assets] Download failed:', err);
    process.exit(1);
  }
})();
