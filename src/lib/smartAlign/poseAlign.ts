/**
 * Pose-based Smart Align using YOLO11s-Pose via ONNX Runtime Web.
 *
 * Algorithm (after iteration & external review):
 *  1. Compute scaleMatchZoom per image so that head-on-screen sizes match
 *     (relative scaling sacred — heads must end up the same size).
 *  2. Find a globalCropZoom in [1.0, 1.15] via binary search such that ALL
 *     images have a reachable common eye-Y position without black borders.
 *  3. Final zoom per image = scaleMatchZoom × globalCropZoom.
 *     This preserves scale ratios — heads stay equal size by construction.
 *  4. Target Y = median of all images' natural eye-Y positions, clamped into
 *     the feasible interval. Reference is informational only.
 *  5. Y-pan to hit target Y exactly, hard-clamped to reserve as final safety.
 *  6. X-pan analogous but with lower priority (larger tolerance).
 *  7. No more "auto-fill" zoom-bumping — globalCropZoom replaces it cleanly.
 */
import type { Tensor } from 'onnxruntime-web';
import { decodeImage } from '@/lib/exportRenderer';
import { ImageState } from '@/lib/types';
import { loadOrt } from './loadOrt';
import { loadYoloPose, KP } from './loadYoloPose';
import {
  AlignResult,
  AlignTransform,
  AlignableSlot,
  ProgressCallback,
  SmartAlignReport,
} from './types';

const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.25;
const KEYPOINT_VIS_THRESHOLD = 0.5;
const MAX_GLOBAL_CROP = 1.15; // hard ceiling — won't crop more than 15%
const SAFETY_PX = 1; // tiny margin against rounding-induced black borders

interface Keypoint { x: number; y: number; v: number; }
interface Person { box: { x: number; y: number; w: number; h: number }; score: number; keypoints: Keypoint[]; }

interface PoseData {
  slotIndex: number;
  anchorXNorm: number;
  anchorYNorm: number;
  anchorScale: number;
  anchorMode: 'multi-anchor' | 'shoulders-only' | 'eye-to-eye';
  detectionScore: number;
  sharpness: number;
  naturalWidth: number;
  naturalHeight: number;
}

function computeCoverScale(naturalW: number, naturalH: number, containerW: number, containerH: number) {
  const cw = containerW > 0 ? containerW : 960;
  const ch = containerH > 0 ? containerH : 1625;
  return Math.max(cw / naturalW, ch / naturalH);
}

async function imageSharpness(img: HTMLImageElement, maxDim = 400): Promise<number> {
  const ratio = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(8, Math.round(img.naturalWidth * ratio));
  const h = Math.max(8, Math.round(img.naturalHeight * ratio));
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  let sum = 0, sumSq = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const c = y * w + x;
      const lap = 4 * gray[c] - gray[c - 1] - gray[c + 1] - gray[c - w] - gray[c + w];
      sum += lap; sumSq += lap * lap; n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

function normalize(values: number[]): number[] {
  const mn = Math.min(...values);
  const mx = Math.max(...values);
  if (mx - mn < 1e-9) return values.map(() => 0.5);
  return values.map((v) => (v - mn) / (mx - mn));
}

function median(arr: number[]): number {
  const s = arr.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

interface PreprocessedImage { data: Float32Array; scale: number; padX: number; padY: number; }

function preprocess(img: HTMLImageElement): PreprocessedImage {
  const W = img.naturalWidth, H = img.naturalHeight;
  const scale = INPUT_SIZE / Math.max(W, H);
  const newW = Math.round(W * scale), newH = Math.round(H * scale);
  const padX = Math.floor((INPUT_SIZE - newW) / 2);
  const padY = Math.floor((INPUT_SIZE - newH) / 2);
  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE; canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(img, padX, padY, newW, newH);
  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const float = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    float[i] = imageData[i * 4] / 255;
    float[planeSize + i] = imageData[i * 4 + 1] / 255;
    float[2 * planeSize + i] = imageData[i * 4 + 2] / 255;
  }
  return { data: float, scale, padX, padY };
}

function postprocess(output: Float32Array, numDetections: number, pre: PreprocessedImage, origW: number, origH: number): Person[] {
  const people: Person[] = [];
  const kpStartChannel = 5, numKeypoints = 17;
  for (let i = 0; i < numDetections; i++) {
    const score = output[4 * numDetections + i];
    if (score < CONFIDENCE_THRESHOLD) continue;
    const cxModel = output[0 * numDetections + i];
    const cyModel = output[1 * numDetections + i];
    const wModel = output[2 * numDetections + i];
    const hModel = output[3 * numDetections + i];
    const cxOrig = (cxModel - pre.padX) / pre.scale;
    const cyOrig = (cyModel - pre.padY) / pre.scale;
    const wOrig = wModel / pre.scale, hOrig = hModel / pre.scale;
    const keypoints: Keypoint[] = [];
    for (let k = 0; k < numKeypoints; k++) {
      const xCh = kpStartChannel + k * 3;
      const yCh = kpStartChannel + k * 3 + 1;
      const vCh = kpStartChannel + k * 3 + 2;
      const xModel = output[xCh * numDetections + i];
      const yModel = output[yCh * numDetections + i];
      const v = output[vCh * numDetections + i];
      keypoints.push({ x: (xModel - pre.padX) / pre.scale, y: (yModel - pre.padY) / pre.scale, v });
    }
    people.push({ box: { x: cxOrig - wOrig / 2, y: cyOrig - hOrig / 2, w: wOrig, h: hOrig }, score, keypoints });
  }
  return nms(people, 0.5).filter((p) => p.box.x + p.box.w > 0 && p.box.y + p.box.h > 0 && p.box.x < origW && p.box.y < origH);
}

function nms(people: Person[], iouThreshold: number): Person[] {
  const sorted = people.slice().sort((a, b) => b.score - a.score);
  const kept: Person[] = [];
  for (const p of sorted) {
    let suppressed = false;
    for (const k of kept) {
      const x1 = Math.max(p.box.x, k.box.x), y1 = Math.max(p.box.y, k.box.y);
      const x2 = Math.min(p.box.x + p.box.w, k.box.x + k.box.w);
      const y2 = Math.min(p.box.y + p.box.h, k.box.y + k.box.h);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const union = p.box.w * p.box.h + k.box.w * k.box.h - inter;
      if (union > 0 && inter / union > iouThreshold) { suppressed = true; break; }
    }
    if (!suppressed) kept.push(p);
  }
  return kept;
}

interface RawAnchor {
  anchorX: number; anchorY: number; anchorScale: number;
  mode: 'multi-anchor' | 'shoulders-only' | 'eye-to-eye';
}

function extractAnchor(person: Person): RawAnchor | null {
  const kp = person.keypoints;
  const lEye = kp[KP.LEFT_EYE], rEye = kp[KP.RIGHT_EYE];
  const nose = kp[KP.NOSE];
  const lSh = kp[KP.LEFT_SHOULDER], rSh = kp[KP.RIGHT_SHOULDER];
  const lHip = kp[11], rHip = kp[12];

  const eyesGood = lEye.v > KEYPOINT_VIS_THRESHOLD && rEye.v > KEYPOINT_VIS_THRESHOLD;
  const shouldersGood = lSh.v > KEYPOINT_VIS_THRESHOLD && rSh.v > KEYPOINT_VIS_THRESHOLD;
  const noseGood = nose.v > KEYPOINT_VIS_THRESHOLD;
  const hipsGood = lHip && rHip && lHip.v > KEYPOINT_VIS_THRESHOLD && rHip.v > KEYPOINT_VIS_THRESHOLD;

  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

  if (eyesGood && shouldersGood) {
    const eyeMidX = (lEye.x + rEye.x) / 2, eyeMidY = (lEye.y + rEye.y) / 2;
    const shMidX = (lSh.x + rSh.x) / 2, shMidY = (lSh.y + rSh.y) / 2;
    const measures: number[] = [];
    measures.push(dist({ x: eyeMidX, y: eyeMidY }, { x: shMidX, y: shMidY }));
    measures.push(dist(lSh, rSh) / 1.5);
    measures.push(dist(lEye, rEye) / 0.4);
    if (noseGood) measures.push(dist(nose, { x: shMidX, y: shMidY }) / 0.85);
    if (hipsGood) {
      const hipMidX = (lHip.x + rHip.x) / 2, hipMidY = (lHip.y + rHip.y) / 2;
      measures.push(dist({ x: eyeMidX, y: eyeMidY }, { x: hipMidX, y: hipMidY }) / 3.5);
    }
    return { anchorX: eyeMidX, anchorY: eyeMidY, anchorScale: Math.max(1, median(measures)), mode: 'multi-anchor' };
  }
  if (shouldersGood) {
    const shMidX = (lSh.x + rSh.x) / 2, shMidY = (lSh.y + rSh.y) / 2;
    return { anchorX: shMidX, anchorY: shMidY, anchorScale: Math.max(1, dist(lSh, rSh)), mode: 'shoulders-only' };
  }
  if (eyesGood) {
    const eyeMidX = (lEye.x + rEye.x) / 2, eyeMidY = (lEye.y + rEye.y) / 2;
    return { anchorX: eyeMidX, anchorY: eyeMidY, anchorScale: Math.max(1, dist(lEye, rEye)), mode: 'eye-to-eye' };
  }
  return null;
}

export async function detectPoseForSlots(
  slots: AlignableSlot[],
  onProgress?: ProgressCallback
): Promise<(PoseData | null)[]> {
  onProgress?.('Loading pose model…');
  const session = await loadYoloPose();
  const ort = await loadOrt();

  const results: (PoseData | null)[] = [];
  for (let i = 0; i < slots.length; i++) {
    onProgress?.(`Detecting person ${i + 1}/${slots.length}…`);
    const slot = slots[i];
    const img = await decodeImage(slot.state.url);
    try {
      const pre = preprocess(img);
      const inputTensor = new ort.Tensor('float32', pre.data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const inputName = session.inputNames[0], outputName = session.outputNames[0];
      const feeds: Record<string, Tensor> = { [inputName]: inputTensor };
      const outputMap = await session.run(feeds);
      const output = outputMap[outputName];
      const data = output.data as Float32Array;
      const numDetections = output.dims[2];
      const people = postprocess(data, numDetections, pre, img.naturalWidth, img.naturalHeight);
      console.log('[CompareShot] Slot', slot.slotIndex, '— person candidates:', people.length);
      if (people.length === 0) { results.push(null); continue; }
      let best = people[0], bestArea = best.box.w * best.box.h;
      for (let k = 1; k < people.length; k++) {
        const a = people[k].box.w * people[k].box.h;
        if (a > bestArea) { best = people[k]; bestArea = a; }
      }
      const anchor = extractAnchor(best);
      if (!anchor) { console.warn('[CompareShot] Slot', slot.slotIndex, '— no usable keypoints'); results.push(null); continue; }
      const W = img.naturalWidth, H = img.naturalHeight;
      const longEdge = Math.max(W, H);
      const anchorXNorm = anchor.anchorX / W;
      const anchorYNorm = anchor.anchorY / H;
      const anchorScaleNorm = anchor.anchorScale / longEdge;
      console.log(
        '[CompareShot] Slot', slot.slotIndex,
        '— mode:', anchor.mode,
        'natW×H:', W + '×' + H,
        'normPos: (' + anchorXNorm.toFixed(3) + ',' + anchorYNorm.toFixed(3) + ')',
        'normScale:', anchorScaleNorm.toFixed(4)
      );
      const sharpness = await imageSharpness(img);
      results.push({
        slotIndex: slot.slotIndex,
        anchorXNorm, anchorYNorm,
        anchorScale: anchorScaleNorm,
        anchorMode: anchor.mode,
        detectionScore: best.score,
        sharpness,
        naturalWidth: W,
        naturalHeight: H,
      });
    } catch (err) {
      console.warn('[CompareShot] Pose detection failed for slot', slot.slotIndex, err);
      results.push(null);
    }
  }
  return results;
}

function pickReference(poses: PoseData[]): number {
  if (poses.length === 0) return 0;
  const sN = normalize(poses.map((f) => f.sharpness));
  const hN = normalize(poses.map((f) => f.anchorScale));
  const dN = normalize(poses.map((f) => f.detectionScore));
  const scores = poses.map((_, i) => 0.55 * hN[i] + 0.3 * sN[i] + 0.15 * dN[i]);
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  return best;
}

// ----- Per-image computed values used by the global solver -----

interface SlotCalc {
  poseIdx: number;          // index into withPose array
  slotIndex: number;
  state: ImageState;
  pose: PoseData;
  cover: number;
  scaleMatchZoom: number;   // multiplier needed for this image's head to match target size
  containerW: number;
  containerH: number;
}

/**
 * For a given globalCropZoom, compute each slot's feasible Y interval (where
 * the eye-anchor can land on the container without producing a black border).
 * Returns the intersection range across all slots (could be empty).
 */
function computeFeasibleYInterval(
  calcs: SlotCalc[],
  globalCropZoom: number
): { lo: number; hi: number; perSlot: { baseY: number; reserve: number }[] } {
  const perSlot = calcs.map((c) => {
    const z = c.scaleMatchZoom * globalCropZoom;
    const drawnH = c.pose.naturalHeight * c.cover * z;
    const reserve = Math.max(0, (drawnH - c.containerH) / 2 - SAFETY_PX);
    const baseY = (c.pose.anchorYNorm - 0.5) * c.pose.naturalHeight * c.cover * z;
    return { baseY, reserve };
  });
  let lo = -Infinity, hi = Infinity;
  for (const s of perSlot) {
    lo = Math.max(lo, s.baseY - s.reserve);
    hi = Math.min(hi, s.baseY + s.reserve);
  }
  return { lo, hi, perSlot };
}

/**
 * Binary-search the smallest globalCropZoom in [1.0, MAX_GLOBAL_CROP] such
 * that all slots' Y-intervals overlap. Returns MAX_GLOBAL_CROP if no overlap
 * is achievable within the cap (caller will then accept Y-imperfection).
 */
function solveGlobalCropZoom(calcs: SlotCalc[]): number {
  // Already feasible at zoom 1?
  let { lo, hi } = computeFeasibleYInterval(calcs, 1.0);
  if (lo <= hi) return 1.0;

  // Try the cap — if even at the cap we can't overlap, return cap (best effort)
  ({ lo, hi } = computeFeasibleYInterval(calcs, MAX_GLOBAL_CROP));
  if (lo > hi) return MAX_GLOBAL_CROP;

  // Binary search
  let low = 1.0, high = MAX_GLOBAL_CROP;
  for (let iter = 0; iter < 24; iter++) {
    const mid = (low + high) / 2;
    const r = computeFeasibleYInterval(calcs, mid);
    if (r.lo <= r.hi) high = mid;
    else low = mid;
  }
  return high;
}

export async function poseAlign(
  slots: AlignableSlot[],
  detected: (PoseData | null)[],
  onProgress?: ProgressCallback
): Promise<SmartAlignReport> {
  const withPose: PoseData[] = [];
  for (const f of detected) if (f) withPose.push(f);
  if (withPose.length === 0) {
    return {
      mode: 'face',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({ slotIndex: s.slotIndex, status: 'failed', reason: 'No person detected' })),
    };
  }

  onProgress?.('Choosing reference…');
  const refLocal = pickReference(withPose);
  const reference = withPose[refLocal];
  const referenceSlotIndex = reference.slotIndex;

  // ---- Step 1: Compute scaleMatchZoom for each slot ----
  // Target on-screen size = MEDIAN of all images' natural on-screen sizes.
  // Median (not reference) is more robust against outliers.
  const calcs: SlotCalc[] = withPose.map((p, i) => {
    const slot = slots.find((s) => s.slotIndex === p.slotIndex)!;
    const cover = computeCoverScale(
      slot.state.naturalWidth,
      slot.state.naturalHeight,
      slot.state._containerW,
      slot.state._containerH
    );
    return {
      poseIdx: i,
      slotIndex: p.slotIndex,
      state: slot.state,
      pose: p,
      cover,
      scaleMatchZoom: 1, // filled in below
      containerW: slot.state._containerW > 0 ? slot.state._containerW : 960,
      containerH: slot.state._containerH > 0 ? slot.state._containerH : 1625,
    };
  });

  const onScreenSizes = calcs.map(
    (c) => c.pose.anchorScale * Math.max(c.pose.naturalWidth, c.pose.naturalHeight) * c.cover
  );
  const targetOnScreenSize = median(onScreenSizes);
  for (let i = 0; i < calcs.length; i++) {
    calcs[i].scaleMatchZoom = targetOnScreenSize / Math.max(1e-6, onScreenSizes[i]);
  }

  console.log('[CompareShot] Target on-screen size (median):', targetOnScreenSize.toFixed(1));
  for (const c of calcs) {
    console.log(
      '[CompareShot] Slot', c.slotIndex,
      'onScreen:', (c.pose.anchorScale * Math.max(c.pose.naturalWidth, c.pose.naturalHeight) * c.cover).toFixed(1),
      'scaleMatchZoom:', c.scaleMatchZoom.toFixed(3)
    );
  }

  // ---- Step 2: Solve globalCropZoom ----
  const globalCropZoom = solveGlobalCropZoom(calcs);
  console.log('[CompareShot] globalCropZoom:', globalCropZoom.toFixed(4));

  // ---- Step 3: Compute final Y target ----
  const { lo: globalLoY, hi: globalHiY, perSlot } = computeFeasibleYInterval(calcs, globalCropZoom);
  const naturalEyeYs = calcs.map((c, i) => perSlot[i].baseY);
  const preferredY = median(naturalEyeYs);
  const targetY = clamp(preferredY, globalLoY, globalHiY);
  const yIntervalEmpty = globalLoY > globalHiY;
  console.log(
    '[CompareShot] Y-interval feasible:', !yIntervalEmpty,
    'lo:', globalLoY.toFixed(1), 'hi:', globalHiY.toFixed(1),
    'preferredY:', preferredY.toFixed(1),
    'targetY:', targetY.toFixed(1)
  );

  // ---- Step 4: Compute final X target (lower priority — same approach but more lenient) ----
  // For X we just use the median of natural anchor X positions, then clamp per-slot.
  const naturalEyeXs = calcs.map((c) => {
    const z = c.scaleMatchZoom * globalCropZoom;
    return (c.pose.anchorXNorm - 0.5) * c.pose.naturalWidth * c.cover * z;
  });
  const targetX = median(naturalEyeXs);

  // ---- Step 5: Build per-image transforms ----
  const results: AlignResult[] = [];
  for (let i = 0; i < slots.length; i++) {
    const slot = slots[i];
    if (slot.slotIndex === referenceSlotIndex) {
      results.push({ slotIndex: slot.slotIndex, status: 'reference' });
      continue;
    }
    const pose = detected[i];
    if (!pose) {
      results.push({ slotIndex: slot.slotIndex, status: 'failed', reason: 'No person detected — will fall back to features' });
      continue;
    }
    onProgress?.(`Aligning ${i + 1}/${slots.length}…`);
    const c = calcs.find((x) => x.slotIndex === slot.slotIndex)!;
    const idxInCalcs = calcs.indexOf(c);

    const finalZoom = c.scaleMatchZoom * globalCropZoom;
    const drawnW = c.pose.naturalWidth * c.cover * finalZoom;
    const drawnH = c.pose.naturalHeight * c.cover * finalZoom;
    const reserveX = Math.max(0, (drawnW - c.containerW) / 2 - SAFETY_PX);
    const reserveY = Math.max(0, (drawnH - c.containerH) / 2 - SAFETY_PX);

    const baseX = (c.pose.anchorXNorm - 0.5) * c.pose.naturalWidth * c.cover * finalZoom;
    const baseY = perSlot[idxInCalcs].baseY;

    let panX = targetX - baseX;
    let panY = targetY - baseY;

    panX = clamp(panX, -reserveX, reserveX);
    panY = clamp(panY, -reserveY, reserveY);

    console.log(
      '[CompareShot] Transform slot', slot.slotIndex,
      'finalZoom:', finalZoom.toFixed(3),
      '(scaleMatch:', c.scaleMatchZoom.toFixed(3), '× globalCrop:', globalCropZoom.toFixed(3) + ')',
      'panX:', panX.toFixed(1), 'panY:', panY.toFixed(1),
      'reserveX:', reserveX.toFixed(0), 'reserveY:', reserveY.toFixed(0)
    );

    results.push({
      slotIndex: slot.slotIndex,
      status: 'aligned',
      transform: { zoom: finalZoom, panX, panY, rotation: 0 },
    });
  }

  // ---- Step 6: Also apply zoom & pan to the "reference" slot ----
  // The reference is informational only; mathematically it's just another slot
  // and should also receive scaleMatchZoom × globalCropZoom + pan-to-targetY.
  // But to keep the "reference" UX label meaningful (so the user sees ONE
  // image with status "reference" and untouched-looking framing), we leave it
  // at the system-chosen identity transform. This is a UX choice; if you
  // want the reference also normalized, change this block.
  // Leaving as-is keeps backward compatibility with the existing UI.

  return { mode: 'face', referenceSlotIndex, results };
}

export type { PoseData };
