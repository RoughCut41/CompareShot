/**
 * Sanity-checks that ONNX models exist in public/models/.
 *
 * Models are committed directly to the repo (in public/models/) rather than
 * downloaded at build time. This is more robust than relying on external
 * hosting (Hugging Face requires auth tokens for some files, jsDelivr has
 * caching delays, etc.). Trade-off: ~20 MB in the repo, which is fine.
 */
import { existsSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = resolve(__dirname, '..', 'public', 'models');

const REQUIRED = [
  { name: 'yolo11n-pose.onnx', minSizeBytes: 5 * 1024 * 1024 },
];

let ok = true;
for (const m of REQUIRED) {
  const p = resolve(MODELS_DIR, m.name);
  if (!existsSync(p)) {
    console.error(`[models] MISSING: ${p}`);
    ok = false;
    continue;
  }
  const size = statSync(p).size;
  if (size < m.minSizeBytes) {
    console.error(`[models] TOO SMALL: ${p} (${size} bytes)`);
    ok = false;
    continue;
  }
  console.log(`[models] OK: ${m.name} (${(size / 1024 / 1024).toFixed(1)} MB)`);
}

if (!ok) {
  console.error('[models] One or more models are missing or invalid.');
  console.error('[models] See README for download instructions, or run the curl commands.');
  process.exit(1);
}
console.log('[models] All models present.');
