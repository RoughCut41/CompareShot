/**
 * Face-based Smart Align using MediaPipe FaceLandmarker.
 *
 * Strategy:
 *  - Run MediaPipe at multiple scales for robustness on small faces.
 *  - Pick the largest face per image (closest person).
 *  - Two-point alignment using:
 *      Eye midpoint   — translation anchor (where the eyes should land)
 *      Eye-to-chin    — scale metric (head size)
 *    These are anatomically very stable across photos of the same person:
 *    eye landmarks are sub-pixel precise (iris detection), chin is structural,
 *    and the eye-to-chin distance barely changes with mimicry or lighting.
 *  - Auto-fill: bump zoom so no black borders appear.
 */
import type { FaceLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { decodeImage } from '@/lib/exportRenderer';
import { ImageState } from '@/lib/types';
import { loadFaceLandmarker } from './loadFaceLandmarker';
import {
  AlignResult,
  AlignTransform,
  AlignableSlot,
  ProgressCallback,
  SmartAlignReport,
} from './types';

// MediaPipe landmark indices for our two-point alignment:
//   Iris (sub-pixel precise eye centers): 468 = left iris, 473 = right iris
//   Chin tip: 152
//
// Iris landmarks come from MediaPipe's iris-tracking sub-model and are
// extraordinarily stable. We use them directly as the eye centers.
const LEFT_IRIS = 468;
const RIGHT_IRIS = 473;
const CHIN_TIP = 152;

interface FaceData {
  slotIndex: number;
  img: HTMLImageElement;
  /** Eye midpoint in image pixels — translation anchor */
  eyeMidX: number;
  eyeMidY: number;
  /** Eye-to-chin distance in image pixels — scale metric */
  eyeToChin: number;
  detectionScore: number;
  sharpness: number;
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
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const c = y * w + x;
      const lap = 4 * gray[c] - gray[c - 1] - gray[c + 1] - gray[c - w] - gray[c + w];
      sum += lap;
      sumSq += lap * lap;
      n++;
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

// -------- Detection at multiple scales --------

function renderToScale(img: HTMLImageElement, maxDim: number): { canvas: HTMLCanvasElement; scale: number } {
  const ratio = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * ratio));
  const h = Math.max(1, Math.round(img.naturalHeight * ratio));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);
  return { canvas, scale: ratio };
}

interface RawDetection {
  /** All 478 landmarks, scaled back to original image pixels */
  landmarks: { x: number; y: number }[];
  /** Approximate bounding box of the face (used for "largest face" selection) */
  box: { x: number; y: number; w: number; h: number };
  /** Pseudo-confidence proxy: face-area-to-image ratio (MediaPipe in IMAGE mode
   *  doesn't expose a per-detection score directly, so we approximate). */
  score: number;
}

/**
 * Run MediaPipe at multiple scales and merge detections. Mostly relevant for
 * very high-resolution images where downscaling helps the detector.
 */
function detectAtMultipleScales(
  landmarker: FaceLandmarker,
  img: HTMLImageElement
): RawDetection[] {
  // MediaPipe handles input scaling internally, but on extremely large images
  // a manual pre-scale to ~1600 px helps detection latency without hurting
  // accuracy. We run two scales: 1600 and 2400.
  const scales = [1600, 2400];

  const all: RawDetection[] = [];
  for (const maxDim of scales) {
    const { canvas, scale } = renderToScale(img, maxDim);
    try {
      const result = landmarker.detect(canvas);
      if (!result.faceLandmarks || result.faceLandmarks.length === 0) continue;
      for (const face of result.faceLandmarks) {
        // MediaPipe normalizes coordinates to [0, 1] — convert to source-image px
        const lms = face.map((p: NormalizedLandmark) => ({
          x: (p.x * canvas.width) / scale,
          y: (p.y * canvas.height) / scale,
        }));
        // Compute bounding box from landmarks
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const p of lms) {
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
        const w = maxX - minX;
        const h = maxY - minY;
        const score = (w * h) / (img.naturalWidth * img.naturalHeight);
        all.push({
          landmarks: lms,
          box: { x: minX, y: minY, w, h },
          score,
        });
      }
    } catch (err) {
      console.warn('[CompareShot] MediaPipe detection failed at scale', maxDim, err);
    }
  }
  return all;
}

/** Suppress duplicate detections across scales by IoU. */
function dedupeDetections(dets: RawDetection[]): RawDetection[] {
  const sorted = dets.slice().sort((a, b) => b.box.w * b.box.h - a.box.w * a.box.h);
  const kept: RawDetection[] = [];
  for (const d of sorted) {
    let dup = false;
    for (const k of kept) {
      const x1 = Math.max(d.box.x, k.box.x);
      const y1 = Math.max(d.box.y, k.box.y);
      const x2 = Math.min(d.box.x + d.box.w, k.box.x + k.box.w);
      const y2 = Math.min(d.box.y + d.box.h, k.box.y + k.box.h);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const union = d.box.w * d.box.h + k.box.w * k.box.h - inter;
      if (union > 0 && inter / union > 0.4) {
        dup = true;
        break;
      }
    }
    if (!dup) kept.push(d);
  }
  return kept;
}

export async function detectFacesForSlots(
  slots: AlignableSlot[],
  onProgress?: ProgressCallback
): Promise<(FaceData | null)[]> {
  onProgress?.('Loading face models…');
  const landmarker = await loadFaceLandmarker();

  const results: (FaceData | null)[] = [];
  for (let i = 0; i < slots.length; i++) {
    onProgress?.(`Detecting faces ${i + 1}/${slots.length}…`);
    const slot = slots[i];
    const img = await decodeImage(slot.state.url);

    try {
      const raw = detectAtMultipleScales(landmarker, img);
      const merged = dedupeDetections(raw);
      console.log('[CompareShot] Slot', slot.slotIndex, '— face candidates:', merged.length);

      if (merged.length === 0) {
        results.push(null);
        continue;
      }

      // Pick the largest face (closest person)
      let best = merged[0];
      let bestArea = best.box.w * best.box.h;
      for (let k = 1; k < merged.length; k++) {
        const a = merged[k].box.w * merged[k].box.h;
        if (a > bestArea) {
          best = merged[k];
          bestArea = a;
        }
      }

      // Iris landmarks are MediaPipe's most stable points — they come from the
      // iris-tracking sub-model and are sub-pixel precise.
      const leftIris = best.landmarks[LEFT_IRIS];
      const rightIris = best.landmarks[RIGHT_IRIS];
      const chin = best.landmarks[CHIN_TIP];

      const eyeMidX = (leftIris.x + rightIris.x) / 2;
      const eyeMidY = (leftIris.y + rightIris.y) / 2;
      const eyeToChin = Math.sqrt(
        (chin.x - eyeMidX) ** 2 + (chin.y - eyeMidY) ** 2
      );

      const sharpness = await imageSharpness(img);

      results.push({
        slotIndex: slot.slotIndex,
        img,
        eyeMidX,
        eyeMidY,
        eyeToChin: Math.max(1, eyeToChin),
        detectionScore: best.score,
        sharpness,
      });
    } catch (err) {
      console.warn('[CompareShot] Face detection failed for slot', slot.slotIndex, err);
      results.push(null);
    }
  }
  return results;
}

// -------- Reference selection --------

function pickFaceReference(faces: FaceData[]): number {
  if (faces.length === 0) return 0;
  const sN = normalize(faces.map((f) => f.sharpness));
  // Use eye-to-chin (head size) as the centrality / closeness proxy
  const hN = normalize(faces.map((f) => f.eyeToChin));
  const dN = normalize(faces.map((f) => f.detectionScore));
  const scores = faces.map((_, i) => 0.55 * hN[i] + 0.3 * sN[i] + 0.15 * dN[i]);
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

// -------- Per-image transform --------

function faceDataToTransform(
  current: FaceData,
  reference: FaceData,
  currentState: ImageState,
  referenceState: ImageState
): AlignTransform {
  // Match the eye-to-chin distance: this is our "head size" measurement
  const imageScale = reference.eyeToChin / Math.max(1, current.eyeToChin);

  const refCover = computeCoverScale(
    referenceState.naturalWidth,
    referenceState.naturalHeight,
    referenceState._containerW,
    referenceState._containerH
  );
  const curCover = computeCoverScale(
    currentState.naturalWidth,
    currentState.naturalHeight,
    currentState._containerW,
    currentState._containerH
  );

  const zoom = (refCover * imageScale) / curCover;

  // Eye midpoint offsets from each image's natural center
  const refOffsetX_refImg = reference.eyeMidX - referenceState.naturalWidth / 2;
  const refOffsetY_refImg = reference.eyeMidY - referenceState.naturalHeight / 2;
  const refOffsetX_container = refOffsetX_refImg * refCover;
  const refOffsetY_container = refOffsetY_refImg * refCover;

  const curOffsetX_curImg = current.eyeMidX - currentState.naturalWidth / 2;
  const curOffsetY_curImg = current.eyeMidY - currentState.naturalHeight / 2;
  const curProjectedX = curOffsetX_curImg * refCover * imageScale;
  const curProjectedY = curOffsetY_curImg * refCover * imageScale;

  let panX = refOffsetX_container - curProjectedX;
  let panY = refOffsetY_container - curProjectedY;

  let finalZoom = Math.max(0.2, Math.min(5, zoom));

  // Auto-fill against black borders
  const cw = currentState._containerW > 0 ? currentState._containerW : 960;
  const ch = currentState._containerH > 0 ? currentState._containerH : 1625;
  const minZoomX = 1 + (2 * Math.abs(panX)) / cw;
  const minZoomY = 1 + (2 * Math.abs(panY)) / ch;
  const minZoom = Math.max(minZoomX, minZoomY);
  if (finalZoom < minZoom) {
    const k = minZoom / finalZoom;
    finalZoom = minZoom;
    panX *= k;
    panY *= k;
  }
  finalZoom = Math.min(finalZoom, 5);

  return { zoom: finalZoom, panX, panY, rotation: 0 };
}

// -------- Public entry point --------

export async function faceAlign(
  slots: AlignableSlot[],
  detected: (FaceData | null)[],
  onProgress?: ProgressCallback
): Promise<SmartAlignReport> {
  const withFaces: FaceData[] = [];
  for (const f of detected) {
    if (f) withFaces.push(f);
  }

  if (withFaces.length === 0) {
    return {
      mode: 'face',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({
        slotIndex: s.slotIndex,
        status: 'failed',
        reason: 'No face detected',
      })),
    };
  }

  onProgress?.('Choosing reference…');
  const refLocal = pickFaceReference(withFaces);
  const reference = withFaces[refLocal];
  const referenceSlotIndex = reference.slotIndex;
  const refState = slots.find((s) => s.slotIndex === referenceSlotIndex)!.state;

  const results: AlignResult[] = [];
  for (let i = 0; i < slots.length; i++) {
    const slot = slots[i];
    if (slot.slotIndex === referenceSlotIndex) {
      results.push({ slotIndex: slot.slotIndex, status: 'reference' });
      continue;
    }
    const face = detected[i];
    if (!face) {
      results.push({
        slotIndex: slot.slotIndex,
        status: 'failed',
        reason: 'No face detected — will fall back to features',
      });
      continue;
    }
    onProgress?.(`Aligning ${i + 1}/${slots.length}…`);
    const transform = faceDataToTransform(face, reference, slot.state, refState);
    results.push({
      slotIndex: slot.slotIndex,
      status: 'aligned',
      transform,
    });
  }

  return {
    mode: 'face',
    referenceSlotIndex,
    results,
  };
}

export type { FaceData };
