/**
 * Face-based Smart Align using MediaPipe FaceLandmarker.
 *
 * Robustness strategy:
 *  - Render every input image to a fresh canvas before detection. This
 *    normalizes EXIF orientation (which raw HTMLImageElement reading can get
 *    wrong inside MediaPipe's WASM) and removes color-profile gotchas.
 *  - Try multiple detection passes per image:
 *      1) Whole image at 1500px max edge (best for normal-distance shots)
 *      2) Whole image at 800px max edge (sometimes BlazeFace needs smaller)
 *      3) Center-crop (50% of image around center) at 1000px (helps when the
 *         subject is small but centered, which is the common smartphone case)
 *    First successful detection wins.
 *  - Track all 478 MediaPipe landmarks. Use iris-center landmarks (468, 473)
 *    for the eye anchor and chin tip (152) for the head-size metric.
 *
 * Two-point alignment:
 *    Eye midpoint (translation anchor) + eye-to-chin distance (scale).
 *    Anatomically very stable across photos of the same person.
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

// MediaPipe landmark indices
const LEFT_IRIS = 468;
const RIGHT_IRIS = 473;
const CHIN_TIP = 152;

interface FaceData {
  slotIndex: number;
  img: HTMLImageElement;
  /** Eye midpoint in original-image pixels — translation anchor */
  eyeMidX: number;
  eyeMidY: number;
  /** Eye-to-chin distance in original-image pixels — scale metric */
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

// -------- Multi-strategy detection --------

interface DetectionAttempt {
  /** Friendly name for logging */
  name: string;
  /** The canvas we hand to MediaPipe */
  canvas: HTMLCanvasElement;
  /** Map a canvas-pixel coordinate back to original-image-pixel coordinate */
  canvasToImage: (x: number, y: number) => { x: number; y: number };
}

/**
 * Build the list of detection attempts for an image. Each attempt renders the
 * source onto a canvas at a particular size/region; downstream we map
 * coordinates back to original-image pixels.
 */
function buildAttempts(img: HTMLImageElement): DetectionAttempt[] {
  const W = img.naturalWidth;
  const H = img.naturalHeight;

  const makeFullScale = (maxDim: number, name: string): DetectionAttempt => {
  // Render the full image into a SQUARE canvas with black letterboxing.
  // This makes the ROI square — required for MediaPipe to project landmarks
  // accurately — while still containing the entire source image.
  const longest = Math.max(W, H);
  const ratio = Math.min(1, maxDim / longest);
  const targetSize = Math.max(1, Math.round(longest * ratio));
  const drawW = Math.max(1, Math.round(W * ratio));
  const drawH = Math.max(1, Math.round(H * ratio));
  const offsetX = (targetSize - drawW) / 2;
  const offsetY = (targetSize - drawH) / 2;
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(img, 0, 0, W, H, offsetX, offsetY, drawW, drawH);
  return {
    name,
    canvas,
    canvasToImage: (x, y) => ({
      x: (x - offsetX) / ratio,
      y: (y - offsetY) / ratio,
    }),
  };
};

const makeCenterCrop = (cropFraction: number, maxDim: number, name: string): DetectionAttempt => {
  // Square crop centered on the image — MediaPipe's landmark projection emits
  // a warning for non-square ROIs ("NORM_RECT without IMAGE_DIMENSIONS is only
  // supported for the square ROI"), and produces inaccurate landmarks. Using
  // a square crop avoids this entirely.
  const side = Math.min(W, H) * cropFraction;
  const cropX = (W - side) / 2;
  const cropY = (H - side) / 2;
  const ratio = Math.min(1, maxDim / side);
  const csize = Math.max(1, Math.round(side * ratio));
  const canvas = document.createElement('canvas');
  canvas.width = csize;
  canvas.height = csize;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, cropX, cropY, side, side, 0, 0, csize, csize);
  return {
    name,
    canvas,
    canvasToImage: (x, y) => ({
      x: cropX + x / ratio,
      y: cropY + y / ratio,
    }),
  };
};

  return [
    makeFullScale(1500, 'full@1500'),
    makeFullScale(800, 'full@800'),
    makeCenterCrop(0.7, 1000, 'center70@1000'),
  ];
}

interface DetectedFace {
  /** Landmarks already mapped back to original-image pixels */
  landmarks: { x: number; y: number }[];
  box: { x: number; y: number; w: number; h: number };
  score: number;
}

function runDetection(landmarker: FaceLandmarker, attempt: DetectionAttempt): DetectedFace[] {
  const out: DetectedFace[] = [];
  try {
    const result = landmarker.detect(attempt.canvas);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      return out;
    }
    for (const face of result.faceLandmarks) {
      // MediaPipe gives normalized [0..1] coords relative to the canvas
      const lms = face.map((p: NormalizedLandmark) => {
        const cx = p.x * attempt.canvas.width;
        const cy = p.y * attempt.canvas.height;
        return attempt.canvasToImage(cx, cy);
      });
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of lms) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
      }
      out.push({
        landmarks: lms,
        box: { x: minX, y: minY, w: maxX - minX, h: maxY - minY },
        score: 1,
      });
    }
  } catch (err) {
    console.warn('[CompareShot] detection threw on', attempt.name, err);
  }
  return out;
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

    const attempts = buildAttempts(img);
    let detected: DetectedFace[] = [];
    let usedAttempt = '';
    for (const att of attempts) {
      detected = runDetection(landmarker, att);
      console.log(
        '[CompareShot] Slot', slot.slotIndex,
        '— attempt', att.name,
        'on', att.canvas.width, '×', att.canvas.height,
        '→ found', detected.length
      );
      if (detected.length > 0) {
        usedAttempt = att.name;
        break;
      }
    }

    if (detected.length === 0) {
      console.warn('[CompareShot] Slot', slot.slotIndex, '— no face after all attempts');
      results.push(null);
      continue;
    }
    console.log('[CompareShot] Slot', slot.slotIndex, '— used:', usedAttempt);

    // Pick the largest face (closest person)
    let best = detected[0];
    let bestArea = best.box.w * best.box.h;
    for (let k = 1; k < detected.length; k++) {
      const a = detected[k].box.w * detected[k].box.h;
      if (a > bestArea) {
        best = detected[k];
        bestArea = a;
      }
    }

    const leftIris = best.landmarks[LEFT_IRIS];
    const rightIris = best.landmarks[RIGHT_IRIS];
    const chin = best.landmarks[CHIN_TIP];

    if (!leftIris || !rightIris || !chin) {
      console.warn('[CompareShot] Slot', slot.slotIndex, '— landmarks incomplete');
      results.push(null);
      continue;
    }

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
  }
  return results;
}

// -------- Reference selection --------

function pickFaceReference(faces: FaceData[]): number {
  if (faces.length === 0) return 0;
  const sN = normalize(faces.map((f) => f.sharpness));
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
