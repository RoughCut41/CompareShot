/**
 * Downloads ONNX models needed by Smart Align from GitHub Releases.
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
const MODELS_DIR = resolve(__dirname, '..', 'public', 'models');

const MODELS = [
  {
    name: 'yolo11s-pose.onnx',
    url: 'https://github.com/RoughCut41/CompareShot/releases/download/v0.1.0-models/yolo11s-pose.onnx',
    minSizeBytes: 5 * 1024 * 1024,
  },
];

if (!existsSync(MODELS_DIR)) {
  mkdirSync(MODELS_DIR, { recursive: true });
}

async function fetchWithRedirects(url, maxRedirects = 5) {
  let currentUrl = url;
  for (let i = 0; i < maxRedirects; i++) {
    console.log(`[models] GET ${currentUrl}`);
    const res = await fetch(currentUrl, {
      redirect: 'manual',
      headers: {
        'User-Agent': 'CompareShot-Build/1.0',
        Accept: 'application/octet-stream',
      },
    });
    console.log(`[models]   → status ${res.status}`);
    if (res.status >= 200 && res.status < 300) {
      return res;
    }
    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get('location');
      if (!location) {
        throw new Error(`Redirect ${res.status} without Location header`);
      }
      console.log(`[models]   → redirect to ${location}`);
      currentUrl = location;
      continue;
    }
    // Not OK and not redirect
    const body = await res.text().catch(() => '');
    throw new Error(`Download failed: ${res.status} ${res.statusText}${body ? ` — ${body.slice(0, 200)}` : ''}`);
  }
  throw new Error(`Too many redirects (>${maxRedirects})`);
}

async function downloadModel(model) {
  const dest = resolve(MODELS_DIR, model.name);
  if (existsSync(dest)) {
    const size = statSync(dest).size;
    if (size >= model.minSizeBytes) {
      console.log(
        `[models] ${model.name} already exists (${(size / 1024 / 1024).toFixed(1)} MB) — skipping`
      );
      return;
    }
    console.log(
      `[models] ${model.name} exists but looks incomplete (${size} bytes) — re-downloading`
    );
  }

  console.log(`[models] Downloading ${model.name} …`);
  const res = await fetchWithRedirects(model.url);
  if (!res.body) {
    throw new Error('Response has no body');
  }
  await pipeline(res.body, createWriteStream(dest));
  const size = statSync(dest).size;
  console.log(`[models] Saved ${model.name} (${(size / 1024 / 1024).toFixed(1)} MB)`);
  if (size < model.minSizeBytes) {
    throw new Error(`Downloaded file too small (${size} bytes) — likely an error page`);
  }
}

(async () => {
  try {
    for (const m of MODELS) {
      await downloadModel(m);
    }
    console.log('[models] All models ready.');
  } catch (err) {
    console.error('[models] Download failed:', err);
    process.exit(1);
  }
})();
