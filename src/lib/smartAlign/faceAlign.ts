/**
 * Face-based Smart Align using face-api.js (Tiny Face Detector + 68 Landmarks).
 *
 * Strategy:
 *  - Detect faces at multiple scales for sensitivity on small faces.
 *  - Pick the largest face per image (closest person).
 *  - Use stable landmark-based metrics:
 *      head size  = eye midpoint → chin distance
 *      anchor     = eye midpoint
 *    These are geometrically consistent across detections, unlike the
 *    detection bounding box which fluctuates with lighting / hair / confidence.
 *  - Align all images so that eye midpoints land at the same on-screen
 *    position and eye-to-chin distance matches the reference.
 *  - Auto-fill: bump zoom so no black borders appear.
 */
import { decodeImage } from '@/lib/exportRenderer';
import { ImageState } from '@/lib/types';
import { loadFaceApi } from './loadFaceApi';
import {
  AlignResult,
  AlignTransform,
  AlignableSlot,
  ProgressCallback,
  SmartAlignReport,
} from './types';

interface FaceData {
  slotIndex: number;
  img: HTMLImageElement;
  /** Eye midpoint in image pixels — translation anchor */
  headCenterX: number;
  headCenterY: number;
  /** Eye-to-chin distance in image pixels — used for scale matching */
  headHeight: number;
  detectionScore: number;
  sharpness: number;
}

function computeCoverScale(naturalW: number, naturalH: number, containerW: number, containerH: number) {
  const cw = containerW > 0 ? containerW : 960;
  const ch = containerH > 0 ? containerH : 1625;
  return Math.max(cw / naturalW, ch / naturalH);
}

function avgLandmarks(lm: { x: number; y: number }[], indices: number[]): { x: number; y: number } {
  let x = 0, y = 0;
  for (const i of indices) {
    x += lm[i].x;
    y += lm[i].y;
  }
  return { x: x / indices.length, y: y / indices.length };
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
  box: { x: number; y: number; w: number; h: number };
  landmarks: { x: number; y: number }[];
  score: number;
}

async function detectAtMultipleScales(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  faceapi: any,
  img: HTMLImageElement
): Promise<RawDetection[]> {
  const scales = [
    { maxDim: 1024, inputSize: 416 },
    { maxDim: 1600, inputSize: 608 },
    { maxDim: 2400, inputSize: 800 },
  ];

  const all: RawDetection[] = [];
  for (const s of scales) {
    const { canvas, scale } = renderToScale(img, s.maxDim);
    const opts = new faceapi.TinyFaceDetectorOptions({
      inputSize: s.inputSize,
      scoreThreshold: 0.3,
    });
    try {
      const detections = await faceapi.detectAllFaces(canvas, opts).withFaceLandmarks();
      for (const d of detections) {
        const box = d.detection.box;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const lms = d.landmarks.positions.map((p: any) => ({
          x: p.x / scale,
          y: p.y / scale,
        }));
        all.push({
          box: {
            x: box.x / scale,
            y: box.y / scale,
            w: box.width / scale,
            h: box.height / scale,
          },
          landmarks: lms,
          score: d.detection.score,
        });
      }
    } catch {
      /* skip this scale */
    }
  }
  return all;
}

function dedupeDetections(dets: RawDetection[]): RawDetection[] {
  const sorted = dets.slice().sort((a, b) => b.score - a.score);
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
  const faceapi = await loadFaceApi();

  const results: (FaceData | null)[] = [];
  for (let i = 0; i < slots.length; i++) {
    onProgress?.(`Detecting faces ${i + 1}/${slots.length}…`);
    const slot = slots[i];
    const img = await decodeImage(slot.state.url);

    try {
      const raw = await detectAtMultipleScales(faceapi, img);
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

      const lm = best.landmarks;

      // Stable face metrics — based on landmark positions only, not the detection
      // box (which fluctuates depending on the detector's confidence and lighting).
      //
      // Eye centers: average of left eye landmarks (36..41) and right eye (42..47).
      // These are extremely consistent across detections of the same face.
      const leftEyeCenter = avgLandmarks(lm, [36, 37, 38, 39, 40, 41]);
      const rightEyeCenter = avgLandmarks(lm, [42, 43, 44, 45, 46, 47]);
      const eyeMidX = (leftEyeCenter.x + rightEyeCenter.x) / 2;
      const eyeMidY = (leftEyeCenter.y + rightEyeCenter.y) / 2;
      // Chin tip: landmark 8 (very stable)
      const chinX = lm[8].x;
      const chinY = lm[8].y;

      // Use eye-to-chin distance as our "head size" metric. This is geometrically
      // stable across detections — unlike the detection box which can include
      // varying amounts of hair.
      const eyeToChin = Math.sqrt((chinX - eyeMidX) ** 2 + (chinY - eyeMidY) ** 2);
      const headHeight = Math.max(1, eyeToChin);

      // Anchor point for translation: the eye midpoint. Aligning eyes is what
      // the human visual system perceives as "lined up".
      const headCenterX = eyeMidX;
      const headCenterY = eyeMidY;

      const sharpness = await imageSharpness(img);

      results.push({
        slotIndex: slot.slotIndex,
        img,
        headCenterX,
        headCenterY,
        headHeight,
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
  const hN = normalize(faces.map((f) => f.headHeight));
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
  const imageScale = reference.headHeight / Math.max(1, current.headHeight);

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

  const refOffsetX_refImg = reference.headCenterX - referenceState.naturalWidth / 2;
  const refOffsetY_refImg = reference.headCenterY - referenceState.naturalHeight / 2;
  const refOffsetX_container = refOffsetX_refImg * refCover;
  const refOffsetY_container = refOffsetY_refImg * refCover;

  const curOffsetX_curImg = current.headCenterX - currentState.naturalWidth / 2;
  const curOffsetY_curImg = current.headCenterY - currentState.naturalHeight / 2;
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
