/**
 * Feature-based Smart Align using OpenCV.js (AKAZE + RANSAC).
 *
 * Pipeline:
 *  1. Detect AKAZE keypoints + descriptors on every input image (grayscale).
 *  2. Compute sharpness (Laplacian variance) per image — used for scoring.
 *  3. Cross-match every pair, count "good" Lowe-ratio matches per image — used
 *     for the centrality score.
 *  4. Pick the reference: weighted blend of normalised centrality (55%),
 *     sharpness (30%), keypoint count (15%).
 *  5. For every non-reference image, estimate a similarity transform
 *     (translation + uniform scale + rotation) via RANSAC against the reference.
 *  6. Convert the image-pixel transform into the container coordinate system
 *     used by ImageState (panX/panY in container px, zoom multiplier on top of
 *     cover-fit, rotation in degrees).
 */
import { ImageState } from '@/lib/types';
import { decodeImage } from '@/lib/exportRenderer';
import { loadOpenCV } from './loadOpenCV';
import {
  AlignResult,
  AlignTransform,
  AlignableSlot,
  ProgressCallback,
  SmartAlignReport,
} from './types';

// -------- Helpers --------

function computeCoverScale(naturalW: number, naturalH: number, containerW: number, containerH: number) {
  // object-fit: cover — pick the larger ratio so the image fully covers the container.
  // If the container hasn't been measured yet (ResizeObserver pending), assume
  // the container has the export aspect ratio (960×1625) — same shape the
  // export pipeline assumes.
  const cw = containerW > 0 ? containerW : 960;
  const ch = containerH > 0 ? containerH : 1625;
  return Math.max(cw / naturalW, ch / naturalH);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function imageToMat(cv: any, img: HTMLImageElement, maxDim = 1600) {
  // Downscale very large images before feature detection — AKAZE on a 48 MP
  // photo takes seconds and yields no better matches than 1600 px.
  const ratio = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * ratio);
  const h = Math.round(img.naturalHeight * ratio);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);
  const mat = cv.imread(canvas);
  return { mat, scale: ratio };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function laplacianVariance(cv: any, gray: any): number {
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

// -------- Per-image feature extraction --------

interface ImageFeatures {
  slotIndex: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  mat: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  gray: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  kp: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  desc: any;
  /** Scale factor applied during preprocessing (image was downscaled by this much) */
  preScale: number;
  /** Sharpness (Laplacian variance) on the preprocessed grayscale image */
  sharpness: number;
  /** Number of detected keypoints */
  kpCount: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function extractFeatures(cv: any, slots: AlignableSlot[]): Promise<ImageFeatures[]> {
  const akaze = new cv.AKAZE();
  const features: ImageFeatures[] = [];
  try {
    for (const slot of slots) {
      const img = await decodeImage(slot.state.url);
      const { mat, scale } = imageToMat(cv, img);
      const gray = new cv.Mat();
      cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
      const kp = new cv.KeyPointVector();
      const desc = new cv.Mat();
      const noMask = new cv.Mat();
      akaze.detectAndCompute(gray, noMask, kp, desc);
      noMask.delete();
      features.push({
        slotIndex: slot.slotIndex,
        mat,
        gray,
        kp,
        desc,
        preScale: scale,
        sharpness: laplacianVariance(cv, gray),
        kpCount: kp.size(),
      });
    }
  } finally {
    akaze.delete();
  }
  return features;
}

// -------- Reference selection via Smart-Score --------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function pickReference(cv: any, features: ImageFeatures[]): number {
  const n = features.length;
  if (n === 1) return 0;

  const matchSums = new Array(n).fill(0);
  const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
  try {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
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
          // descriptor-size mismatch or similar — skip pair
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
  const W_CENTRAL = 0.55;
  const W_SHARP = 0.3;
  const W_KP = 0.15;
  const scores = features.map((_, i) => W_CENTRAL * mN[i] + W_SHARP * sN[i] + W_KP * kN[i]);

  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }
  return bestIdx;
}

// -------- Pairwise alignment --------

interface PairAlignResult {
  /** Image-pixel similarity transform from current → reference */
  a: number;
  b: number;
  txImg: number;
  tyImg: number;
  matches: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function alignPair(cv: any, current: ImageFeatures, reference: ImageFeatures): PairAlignResult | null {
  const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
  const knn = new cv.DMatchVectorVector();
  let srcMat: any = null;
  let dstMat: any = null;
  let inliers: any = null;
  let M: any = null;
  try {
    matcher.knnMatch(current.desc, reference.desc, knn, 2);
    const good: { q: number; t: number }[] = [];
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() >= 2) {
        const m1 = pair.get(0);
        const m2 = pair.get(1);
        if (m1.distance < 0.75 * m2.distance) {
          good.push({ q: m1.queryIdx, t: m1.trainIdx });
        }
      }
    }
    if (good.length < 8) return null;

    // Build src/dst point arrays in *original-pixel* coordinates by undoing the preScale
    const srcPts: number[] = [];
    const dstPts: number[] = [];
    for (const gm of good) {
      const sp = current.kp.get(gm.q).pt;
      const dp = reference.kp.get(gm.t).pt;
      srcPts.push(sp.x / current.preScale, sp.y / current.preScale);
      dstPts.push(dp.x / reference.preScale, dp.y / reference.preScale);
    }
    srcMat = cv.matFromArray(srcPts.length / 2, 1, cv.CV_32FC2, srcPts);
    dstMat = cv.matFromArray(dstPts.length / 2, 1, cv.CV_32FC2, dstPts);
    inliers = new cv.Mat();
    M = cv.estimateAffinePartial2D(srcMat, dstMat, inliers, cv.RANSAC, 3, 2000, 0.99);
    if (!M || M.empty()) return null;

    return {
      a: M.doubleAt(0, 0),
      b: M.doubleAt(0, 1),
      txImg: M.doubleAt(0, 2),
      tyImg: M.doubleAt(1, 2),
      matches: good.length,
    };
  } finally {
    matcher.delete();
    knn.delete();
    srcMat?.delete?.();
    dstMat?.delete?.();
    inliers?.delete?.();
    M?.delete?.();
  }
}

// -------- Image-pixel transform → container-pixel transform --------

/**
 * Convert an image-pixel similarity transform (current → reference, in original
 * source pixels) into the container-pixel form used by ImageState.
 *
 * The CSS rendering applies, in order:
 *   1. cover-fit: image is drawn at coverScale into the container, centered
 *   2. transform: translate(panX, panY) rotate(deg) scale(zoom)
 *
 * For the warped image to land where the reference's *center* lands, we need
 * the displayed center of the current image to coincide with the displayed
 * center of the reference. Conveniently, both refs and currents are
 * cover-rendered into containers of the same on-screen size — so the
 * "cover-center" in container coords is just (containerW/2, containerH/2)
 * for both. We therefore only need the translation between centers:
 *
 *   image-pixel center of current image → mapped via transform → image-pixel
 *   coordinate in the reference frame. Difference between this mapped point
 *   and the reference's natural center (in reference image pixels) gives the
 *   offset we want to translate, scaled to container space.
 */
function imagePixelToContainerTransform(
  current: ImageState,
  reference: ImageState,
  pair: PairAlignResult
): AlignTransform {
  const { a, b, txImg, tyImg } = pair;

  // Decompose similarity: scale & rotation
  const scaleImg = Math.sqrt(a * a + b * b);
  const rotRad = Math.atan2(b, a);
  let rotDeg = (rotRad * 180) / Math.PI;
  // Sanity-clamp: smartphones held by hand introduce a few degrees of tilt at most.
  // Anything larger is almost certainly RANSAC noise on a tricky scene.
  if (rotDeg > 30) rotDeg = 30;
  else if (rotDeg < -30) rotDeg = -30;

  // Displayed cover-scales
  const refCover = computeCoverScale(
    reference.naturalWidth,
    reference.naturalHeight,
    reference._containerW,
    reference._containerH
  );
  const curCover = computeCoverScale(
    current.naturalWidth,
    current.naturalHeight,
    current._containerW,
    current._containerH
  );

  // The total zoom we need to apply on top of the current's cover-fit so the
  // current image has the same on-screen "image-pixels-per-container-pixel"
  // ratio as the reference, scaled by `scaleImg` (which says how much we need
  // to up/downscale current relative to reference to make features overlap).
  const zoom = (refCover * scaleImg) / curCover;

  // Map the natural center of the *current* image through the similarity
  // transform → coordinate in reference image space.
  const cxCur = current.naturalWidth / 2;
  const cyCur = current.naturalHeight / 2;
  const mappedX = a * cxCur - b * cyCur + txImg;
  const mappedY = b * cxCur + a * cyCur + tyImg;

  // Reference image natural center
  const refCxImg = reference.naturalWidth / 2;
  const refCyImg = reference.naturalHeight / 2;

  // Offset in *reference-image* pixels
  const offsetXrefPx = mappedX - refCxImg;
  const offsetYrefPx = mappedY - refCyImg;

  // Convert that offset into reference-container pixels using the reference
  // cover scale. (The reference is drawn at scale=1 in its own container, and
  // both current and reference live in containers of the same size in the UI.)
  const panX = offsetXrefPx * refCover;
  const panY = offsetYrefPx * refCover;

  // Clamp zoom to a sane range — extreme values are almost always RANSAC noise.
  const clampedZoom = Math.max(0.2, Math.min(5, zoom));

  return {
    zoom: clampedZoom,
    panX,
    panY,
    rotation: rotDeg,
  };
}

// -------- Public entry point --------

export async function featureAlign(
  slots: AlignableSlot[],
  onProgress?: ProgressCallback
): Promise<SmartAlignReport> {
  if (slots.length < 2) {
    return {
      mode: 'feature',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({ slotIndex: s.slotIndex, status: 'skipped' })),
    };
  }

  onProgress?.('Loading…');
  const cv = await loadOpenCV();

  onProgress?.('Analyzing…');
  const features = await extractFeatures(cv, slots);

  try {
    const refIdxLocal = pickReference(cv, features);
    const reference = features[refIdxLocal];
    const referenceSlotIndex = reference.slotIndex;
    const refState = slots[refIdxLocal].state;

    const results: AlignResult[] = [];
    for (let i = 0; i < features.length; i++) {
      const f = features[i];
      if (i === refIdxLocal) {
        results.push({ slotIndex: f.slotIndex, status: 'reference' });
        continue;
      }
      onProgress?.(`Aligning ${i + 1}/${features.length}…`);
      const pair = alignPair(cv, f, reference);
      if (!pair) {
        results.push({
          slotIndex: f.slotIndex,
          status: 'failed',
          reason: 'Not enough matching features',
        });
        continue;
      }
      const curState = slots[i].state;
      const transform = imagePixelToContainerTransform(curState, refState, pair);
      results.push({
        slotIndex: f.slotIndex,
        status: 'aligned',
        transform,
      });
    }

    return {
      mode: 'feature',
      referenceSlotIndex,
      results,
    };
  } finally {
    // Cleanup all OpenCV mats to avoid wasm memory leaks
    for (const f of features) {
      f.mat.delete();
      f.gray.delete();
      f.kp.delete();
      f.desc.delete();
    }
  }
}
