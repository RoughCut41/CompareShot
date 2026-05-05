/**
 * Downloads ONNX models needed by Smart Align.
 *
 * Runs automatically before `npm run build` and `npm run dev` (via package.json
 * scripts). Skips download if the file already exists locally.
 *
 * Models are pulled from Hugging Face onnx-community, a curated hub of
 * verified ONNX exports of popular models.
 */
import { existsSync, mkdirSync, statSync, createWriteStream } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = resolve(__dirname, '..', 'public', 'models');

const MODELS = [
  {
    name: 'yolo11n-pose.onnx',
    url: 'https://huggingface.co/onnx-community/yolo11n-pose/resolve/main/onnx/model.onnx',
    minSizeBytes: 5 * 1024 * 1024, // sanity check: should be > 5 MB
  },
];

if (!existsSync(MODELS_DIR)) {
  mkdirSync(MODELS_DIR, { recursive: true });
}

async function downloadModel(model) {
  const dest = resolve(MODELS_DIR, model.name);
  if (existsSync(dest)) {
    const size = statSync(dest).size;
    if (size >= model.minSizeBytes) {
      console.log(`[models] ${model.name} already exists (${(size / 1024 / 1024).toFixed(1)} MB) — skipping`);
      return;
    }
    console.log(`[models] ${model.name} exists but looks incomplete (${size} bytes) — re-downloading`);
  }

  console.log(`[models] Downloading ${model.name} from ${model.url} …`);
  const res = await fetch(model.url, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }
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
