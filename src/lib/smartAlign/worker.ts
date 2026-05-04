/**
 * Smart Align Web Worker (Classic worker, supports importScripts).
 *
 * Loads OpenCV.js in a separate thread so the main UI never freezes during
 * heavy computation. The main thread sends down-scaled ImageData arrays +
 * image metadata; the worker returns AlignTransform objects ready to apply.
 */

/// <reference lib="webworker" />

declare const self: WorkerGlobalScope &
  typeof globalThis & {
    importScripts: (...urls: string[]) => void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cv?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    postMessage: (msg: any) => void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    onmessage: ((e: MessageEvent) => any) | null;
  };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const cv: any;

// ---- Message protocol ----

interface AlignTransform {
  zoom: number;
  panX: number;
  panY: number;
  rotation: number;
}

interface SlotPayload {
  slotIndex: number;
  imageData: ImageData;
  preScale: number;
  naturalWidth: number;
  naturalHeight: number;
  containerW: number;
  containerH: number;
}

interface RunMessage {
  type: 'run';
  slots: SlotPayload[];
}

interface ProgressMessage {
  type: 'progress';
  label: string;
}

interface AlignResultPayload {
  slotIndex: number;
  status: 'reference' | 'aligned' | 'failed' | 'skipped';
  reason?: string;
  transform?: AlignTransform;
}

interface DoneMessage {
  type: 'done';
  mode: 'feature';
  referenceSlotIndex: number;
  results: AlignResultPayload[];
}

interface ErrorMessage {
  type: 'error';
  message: string;
}

function post(msg: ProgressMessage | DoneMessage | ErrorMessage) {
  self.postMessage(msg);
}

function progress(label: string) {
  post({ type: 'progress', label });
}

// ---- OpenCV loader inside the worker ----

const OPENCV_URLS = [
  'https://docs.opencv.org/4.8.0/opencv.js',
  'https://cdnjs.cloudflare.com/ajax/libs/opencv.js/4.8.0/opencv.js',
];

let cvReadyPromise: Promise<void> | null = null;

function loadOpenCV(): Promise<void> {
  if (cvReadyPromise) return cvReadyPromise;
  cvReadyPromise = (async () => {
    let lastErr: Error | null = null;
    for (const url of OPENCV_URLS) {
      try {
        progress(`Loading OpenCV from ${new URL(url).hostname}…`);
        self.importScripts(url);
        progress('Initializing OpenCV runtime…');
        await waitForCvMat();
        return;
      } catch (err) {
        lastErr = err instanceof Error ? err : new Error(String(err));
      }
    }
    throw lastErr ?? new Error('All OpenCV CDNs failed');
  })();
  return cvReadyPromise;
}

function waitForCvMat(timeoutMs = 60000): Promise<void> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const tick = () => {
      const c = self.cv;
      if (c && c.Mat) {
        resolve();
        return;
      }
      if (c && !c.Mat && typeof c.onRuntimeInitialized !== 'function') {
        c.onRuntimeInitialized = () => resolve();
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error('OpenCV runtime init timeout'));
        return;
      }
      setTimeout(tick, 50);
    };
    tick();
  });
}

// ---- Image processing helpers ----

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function imageDataToMat(data: ImageData): any {
  // cv.imread() expects an HTMLImageElement/HTMLCanvasElement, which doesn't
  // exist in workers. cv.matFromImageData() takes raw ImageData and works
  // in any context (main thread or worker).
  return cv.matFromImageData(data);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function laplacianVariance(gray: any): number {
  const lap = new cv.Mat();
  cv.Laplacian(gray, lap, cv.CV_64F);
  const mean = new cv.Mat();
  const stddev = new cv.Mat();
  cv.meanStdDev(lap, mean, stddev);
  const v = stddev.doubleAt(0, 0);
  lap.delete();
  mean.delete();
  stddev.delete();
  return v * v;
}

function normalize(values: number[]): number[] {
  const mn = Math.min(...values);
  const mx = Math.max(...values);
  if (mx - mn < 1e-9) return values.map(() => 0.5);
  return values.map((v) => (v - mn) / (mx - mn));
}

function computeCoverScale(naturalW: number, naturalH: number, cw: number, ch: number) {
  const w = cw > 0 ? cw : 960;
  const h = ch > 0 ? ch : 1625;
  return Math.max(w / naturalW, h / naturalH);
}

interface FeatureSet {
  payload: SlotPayload;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  mat: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  gray: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  kp: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  desc: any;
  sharpness: number;
  kpCount: number;
}

// ---- Similarity Transform via RANSAC (replacement for cv.estimateAffinePartial2D) ----

interface Similarity {
  a: number;
  b: number;
  tx: number;
  ty: number;
}

function fitSimilarityFromTwoPairs(
  src: number[],
  dst: number[],
  i: number,
  j: number
): Similarity | null {
  const x1 = src[i * 2], y1 = src[i * 2 + 1];
  const x2 = src[j * 2], y2 = src[j * 2 + 1];
  const u1 = dst[i * 2], v1 = dst[i * 2 + 1];
  const u2 = dst[j * 2], v2 = dst[j * 2 + 1];
  const dx = x2 - x1;
  const dy = y2 - y1;
  const det = dx * dx + dy * dy;
  if (det < 1e-9) return null;
  const du = u2 - u1;
  const dv = v2 - v1;
  const a = (dx * du + dy * dv) / det;
  const b = (dx * dv - dy * du) / det;
  const tx = u1 - a * x1 + b * y1;
  const ty = v1 - b * x1 - a * y1;
  return { a, b, tx, ty };
}

function fitSimilarityLeastSquares(
  src: number[],
  dst: number[],
  inlierIdx: number[]
): Similarity | null {
  const n = inlierIdx.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, su = 0, sv = 0;
  let sxx_yy = 0, sxu_yv = 0, sxv_yu = 0;
  for (const k of inlierIdx) {
    const x = src[k * 2], y = src[k * 2 + 1];
    const u = dst[k * 2], v = dst[k * 2 + 1];
    sx += x;
    sy += y;
    su += u;
    sv += v;
    sxx_yy += x * x + y * y;
    sxu_yv += x * u + y * v;
    sxv_yu += x * v - y * u;
  }
  const denom = n * sxx_yy - sx * sx - sy * sy;
  if (Math.abs(denom) < 1e-9) return null;
  const a = (n * sxu_yv - sx * su - sy * sv) / denom;
  const b = (n * sxv_yu - sx * sv + sy * su) / denom;
  const tx = (su - a * sx + b * sy) / n;
  const ty = (sv - b * sx - a * sy) / n;
  return { a, b, tx, ty };
}

function estimateSimilarityRANSAC(
  src: number[],
  dst: number[],
  iterations = 300,
  threshold = 3
): Similarity | null {
  const n = src.length / 2;
  if (n < 2) return null;

  let bestModel: Similarity | null = null;
  let bestInliers: number[] = [];

  for (let iter = 0; iter < iterations; iter++) {
    const i = Math.floor(Math.random() * n);
    let j = Math.floor(Math.random() * n);
    if (i === j) j = (j + 1) % n;

    const model = fitSimilarityFromTwoPairs(src, dst, i, j);
    if (!model) continue;

    const inliers: number[] = [];
    for (let k = 0; k < n; k++) {
      const x = src[k * 2], y = src[k * 2 + 1];
      const u = dst[k * 2], v = dst[k * 2 + 1];
      const pu = model.a * x - model.b * y + model.tx;
      const pv = model.b * x + model.a * y + model.ty;
      const du = pu - u;
      const dv = pv - v;
      if (du * du + dv * dv < threshold * threshold) {
        inliers.push(k);
      }
    }

    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
      bestModel = model;
    }
  }

  if (!bestModel || bestInliers.length < 4) return null;
  const refined = fitSimilarityLeastSquares(src, dst, bestInliers);
  return refined ?? bestModel;
}

// ---- Main alignment routine ----

async function runAlignment(slots: SlotPayload[]): Promise<DoneMessage> {
  if (slots.length < 2) {
    return {
      type: 'done',
      mode: 'feature',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({ slotIndex: s.slotIndex, status: 'skipped' })),
    };
  }

  await loadOpenCV();

  progress('Detecting features…');
  const akaze = new cv.AKAZE();
  const features: FeatureSet[] = [];
  try {
    for (let i = 0; i < slots.length; i++) {
      progress(`Detecting features ${i + 1}/${slots.length}…`);
      const payload = slots[i];
      const mat = imageDataToMat(payload.imageData);
      const gray = new cv.Mat();
      cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
      const kp = new cv.KeyPointVector();
      const desc = new cv.Mat();
      const noMask = new cv.Mat();
      akaze.detectAndCompute(gray, noMask, kp, desc);
      noMask.delete();
      features.push({
        payload,
        mat,
        gray,
        kp,
        desc,
        sharpness: laplacianVariance(gray),
        kpCount: kp.size(),
      });
    }
  } finally {
    akaze.delete();
  }

  progress('Choosing reference…');
  const matchSums = new Array(features.length).fill(0);
  const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
  try {
    for (let i = 0; i < features.length; i++) {
      for (let j = i + 1; j < features.length; j++) {
        const knn = new cv.DMatchVectorVector();
        try {
          matcher.knnMatch(features[i].desc, features[j].desc, knn, 2);
          let good = 0;
          for (let k = 0; k < knn.size(); k++) {
            const pair = knn.get(k);
            if (pair.size() >= 2) {
              const m1 = pair.get(0);
              const m2 = pair.get(1);
              if (m1.distance < 0.75 * m2.distance) good++;
            }
          }
          matchSums[i] += good;
          matchSums[j] += good;
        } catch {
          /* skip */
        } finally {
          knn.delete();
        }
      }
    }
  } finally {
    matcher.delete();
  }

  const sN = normalize(features.map((f) => f.sharpness));
  const kN = normalize(features.map((f) => f.kpCount));
  const mN = normalize(matchSums);
  const scores = features.map((_, i) => 0.55 * mN[i] + 0.3 * sN[i] + 0.15 * kN[i]);
  let refLocal = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[refLocal]) refLocal = i;
  }
  const reference = features[refLocal];
  const referenceSlotIndex = reference.payload.slotIndex;

  const results: AlignResultPayload[] = [];
  for (let i = 0; i < features.length; i++) {
    if (i === refLocal) {
      results.push({ slotIndex: features[i].payload.slotIndex, status: 'reference' });
      continue;
    }
    progress(`Aligning ${i + 1}/${features.length}…`);
    const cur = features[i];
    const matcher2 = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const knn = new cv.DMatchVectorVector();
    try {
      matcher2.knnMatch(cur.desc, reference.desc, knn, 2);
      const good: { q: number; t: number }[] = [];
      for (let j = 0; j < knn.size(); j++) {
        const pair = knn.get(j);
        if (pair.size() >= 2) {
          const m1 = pair.get(0);
          const m2 = pair.get(1);
          if (m1.distance < 0.75 * m2.distance) {
            good.push({ q: m1.queryIdx, t: m1.trainIdx });
          }
        }
      }
      if (good.length < 8) {
        results.push({
          slotIndex: cur.payload.slotIndex,
          status: 'failed',
          reason: 'Not enough matching features',
        });
        continue;
      }
      const srcPts: number[] = [];
      const dstPts: number[] = [];
      for (const gm of good) {
        const sp = cur.kp.get(gm.q).pt;
        const dp = reference.kp.get(gm.t).pt;
        srcPts.push(sp.x / cur.payload.preScale, sp.y / cur.payload.preScale);
        dstPts.push(dp.x / reference.payload.preScale, dp.y / reference.payload.preScale);
      }

      // Estimate similarity transform via custom RANSAC
      const sim = estimateSimilarityRANSAC(srcPts, dstPts);
      if (!sim) {
        results.push({
          slotIndex: cur.payload.slotIndex,
          status: 'failed',
          reason: 'Transform estimation failed',
        });
        continue;
      }

      const a = sim.a;
      const b = sim.b;
      const txImg = sim.tx;
      const tyImg = sim.ty;
      const scaleImg = Math.sqrt(a * a + b * b);
      const rotRad = Math.atan2(b, a);
      let rotDeg = (rotRad * 180) / Math.PI;
      if (rotDeg > 30) rotDeg = 30;
      else if (rotDeg < -30) rotDeg = -30;

      const refCover = computeCoverScale(
        reference.payload.naturalWidth,
        reference.payload.naturalHeight,
        reference.payload.containerW,
        reference.payload.containerH
      );
      const curCover = computeCoverScale(
        cur.payload.naturalWidth,
        cur.payload.naturalHeight,
        cur.payload.containerW,
        cur.payload.containerH
      );
      const zoom = (refCover * scaleImg) / curCover;
      const cxCur = cur.payload.naturalWidth / 2;
      const cyCur = cur.payload.naturalHeight / 2;
      const mappedX = a * cxCur - b * cyCur + txImg;
      const mappedY = b * cxCur + a * cyCur + tyImg;
      const refCxImg = reference.payload.naturalWidth / 2;
      const refCyImg = reference.payload.naturalHeight / 2;
      let panX = (mappedX - refCxImg) * refCover;
      let panY = (mappedY - refCyImg) * refCover;
      let finalZoom = Math.max(0.2, Math.min(5, zoom));

      // "Auto-fill": bump up the zoom so that no black borders are visible.
      // At zoom Z, the displayed image extends Z * containerW/2 from the rendered
      // center horizontally (and same for vertical with containerH/2). After
      // applying pan, the visible-without-black-border condition is:
      //   |panX| + containerW/2 <= Z * containerW/2
      //   |panY| + containerH/2 <= Z * containerH/2
      // so Z must be at least max(1 + 2|panX|/cw, 1 + 2|panY|/ch).
      // We also have to consider rotation — rotating expands the bounding box by
      // roughly |cos| + |sin| in each axis, so we add a small safety margin.
      const cw = cur.payload.containerW > 0 ? cur.payload.containerW : 960;
      const ch = cur.payload.containerH > 0 ? cur.payload.containerH : 1625;
      const rotRadAbs = Math.abs((rotDeg * Math.PI) / 180);
      const rotExpand = Math.abs(Math.cos(rotRadAbs)) + Math.abs(Math.sin(rotRadAbs));
      const minZoomX = (1 + (2 * Math.abs(panX)) / cw) * rotExpand;
      const minZoomY = (1 + (2 * Math.abs(panY)) / ch) * rotExpand;
      const minZoom = Math.max(minZoomX, minZoomY);
      if (finalZoom < minZoom) {
        // Scale up: we want to multiply zoom by k = minZoom/finalZoom. To keep the
        // *content* aligned (the feature points still landing at the same on-screen
        // position), we also scale pan by k.
        const k = minZoom / finalZoom;
        finalZoom = minZoom;
        panX *= k;
        panY *= k;
      }
      // Re-clamp in case minZoom was extreme (very large pan)
      finalZoom = Math.min(finalZoom, 5);

      results.push({
        slotIndex: cur.payload.slotIndex,
        status: 'aligned',
        transform: {
          zoom: finalZoom,
          panX,
          panY,
          rotation: rotDeg,
        },
      });
    } finally {
      matcher2.delete();
      knn.delete();
    }
  }

  for (const f of features) {
    f.mat.delete();
    f.gray.delete();
    f.kp.delete();
    f.desc.delete();
  }

  return {
    type: 'done',
    mode: 'feature',
    referenceSlotIndex,
    results,
  };
}

// ---- Worker entry ----

self.onmessage = async (e: MessageEvent<RunMessage>) => {
  const msg = e.data;
  if (msg.type !== 'run') return;
  try {
    const done = await runAlignment(msg.slots);
    post(done);
  } catch (err) {
    post({
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    });
  }
};

export {};
