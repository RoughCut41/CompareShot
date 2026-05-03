/**
 * Face-based Smart Align using face-api.js (Tiny Face Detector + 68 Landmarks).
 *
 * Strategy:
 *  - Detect the largest face in every input image
 *  - From the 68 landmarks, extract:
 *      • inter-ocular distance (eye-to-eye, in image pixels)
 *      • nose tip position (image pixels)
 *  - Score each image: sharpness (Laplacian via Canvas), face area (proxy
 *    for centrality / closeness), and face-detection confidence.
 *  - Pick the highest-scoring image as the reference.
 *  - For every other image, compute a transform that:
 *      • scales so the eye-to-eye distance matches the reference
 *      • translates so the nose tip lands at the reference's nose tip
 *      • applies no rotation (face landmarks are noisy enough that rotating
 *        based on them often makes things worse — users can rotate manually)
 *  - The final transform is converted to container-pixel form, identical to
 *    the feature pipeline.
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
  noseX: number; // image px
  noseY: number;
  eyeDistance: number; // image px
  faceArea: number; // image px²
  detectionScore: number;
  sharpness: number;
}

function computeCoverScale(naturalW: number, naturalH: number, containerW: number, containerH: number) {
  const cw = containerW > 0 ? containerW : 960;
  const ch = containerH > 0 ? containerH : 1625;
  return Math.max(cw / naturalW, ch / naturalH);
}

/**
 * Compute Laplacian-style variance of an image via a quick canvas-based
 * 3x3 Laplacian convolution on a downsampled grayscale version. Used as a
 * sharpness proxy; absolute value isn't important, only relative ordering.
 */
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

  // grayscale
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // Laplacian via 4-neighbor kernel on inner pixels
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

// -------- Detection --------

async function detectFaces(slots: AlignableSlot[]): Promise<(FaceData | null)[]> {
  const faceapi = await loadFaceApi();
  const detectorOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 416,
    scoreThreshold: 0.5,
  });

  const results: (FaceData | null)[] = [];
  for (const slot of slots) {
    const img = await decodeImage(slot.state.url);
    try {
      const detection = await faceapi
        .detectSingleFace(img, detectorOptions)
        .withFaceLandmarks();

      if (!detection) {
        results.push(null);
        continue;
      }

      // 68-point landmark indices (face-api.js convention):
      //   left eye:  36..41   right eye: 42..47
      //   nose tip:  30
      const lm = detection.landmarks.positions;
      const leftEye = avgPoint(lm, [36, 37, 38, 39, 40, 41]);
      const rightEye = avgPoint(lm, [42, 43, 44, 45, 46, 47]);
      const nose = lm[30];
      const eyeDx = rightEye.x - leftEye.x;
      const eyeDy = rightEye.y - leftEye.y;
      const eyeDistance = Math.sqrt(eyeDx * eyeDx + eyeDy * eyeDy);

      const box = detection.detection.box;
      const faceArea = box.width * box.height;
      const sharpness = await imageSharpness(img);

      results.push({
        slotIndex: slot.slotIndex,
        img,
        noseX: nose.x,
        noseY: nose.y,
        eyeDistance,
        faceArea,
        detectionScore: detection.detection.score,
        sharpness,
      });
    } catch {
      results.push(null);
    }
  }
  return results;
}

function avgPoint(points: { x: number; y: number }[], indices: number[]) {
  let sx = 0;
  let sy = 0;
  for (const i of indices) {
    sx += points[i].x;
    sy += points[i].y;
  }
  return { x: sx / indices.length, y: sy / indices.length };
}

// -------- Reference selection --------

function pickFaceReference(faces: FaceData[]): number {
  if (faces.length === 0) return 0;
  const sN = normalize(faces.map((f) => f.sharpness));
  const aN = normalize(faces.map((f) => f.faceArea));
  const dN = normalize(faces.map((f) => f.detectionScore));
  // Sharpness 30, face area (≈ centrality/closeness) 55, detection confidence 15
  const scores = faces.map((_, i) => 0.3 * sN[i] + 0.55 * aN[i] + 0.15 * dN[i]);
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
  // Image-pixel scale to apply: makes current's eye distance match reference's
  // (in their respective image-pixel spaces).
  const imageScale = reference.eyeDistance / Math.max(1, current.eyeDistance);

  // Cover scales for the on-screen rendering
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

  // Final on-screen zoom multiplier (relative to current's natural cover-fit)
  const zoom = (refCover * imageScale) / curCover;

  // We want: when current is rendered with this zoom and a translation,
  // its nose tip ends up at the same on-screen position as the reference's
  // nose tip (which is just centered in the reference container plus an
  // offset from its natural center).
  //
  // Reference nose offset from its natural center, expressed in reference-
  // container pixels:
  const refNoseOffsetX_refImg = reference.noseX - referenceState.naturalWidth / 2;
  const refNoseOffsetY_refImg = reference.noseY - referenceState.naturalHeight / 2;
  const refNoseOffsetX_refContainer = refNoseOffsetX_refImg * refCover;
  const refNoseOffsetY_refContainer = refNoseOffsetY_refImg * refCover;

  // Current nose offset from its natural center, in current image pixels.
  // After we apply zoom on top of cover, this offset projected onto the
  // current container becomes: offset * curCover * zoom = offset * refCover * imageScale.
  const curNoseOffsetX_curImg = current.noseX - currentState.naturalWidth / 2;
  const curNoseOffsetY_curImg = current.noseY - currentState.naturalHeight / 2;
  const curNoseProjectedX = curNoseOffsetX_curImg * refCover * imageScale;
  const curNoseProjectedY = curNoseOffsetY_curImg * refCover * imageScale;

  // To align the noses, panX must shift the projected current nose to the
  // reference nose location:
  const panX = refNoseOffsetX_refContainer - curNoseProjectedX;
  const panY = refNoseOffsetY_refContainer - curNoseProjectedY;

  const clampedZoom = Math.max(0.2, Math.min(5, zoom));
  return { zoom: clampedZoom, panX, panY, rotation: 0 };
}

// -------- Public entry point --------

export async function faceAlign(
  slots: AlignableSlot[],
  detected: (FaceData | null)[],
  onProgress?: ProgressCallback
): Promise<SmartAlignReport> {
  // Filter to slots that actually have a detected face
  const withFaces: FaceData[] = [];
  const indexInWithFaces = new Map<number, number>(); // slotIndex → position in withFaces
  for (let i = 0; i < slots.length; i++) {
    const f = detected[i];
    if (f) {
      indexInWithFaces.set(slots[i].slotIndex, withFaces.length);
      withFaces.push(f);
    }
  }

  if (withFaces.length < 2) {
    // Fall back: caller will likely retry with feature align — but for safety,
    // mark everything skipped.
    return {
      mode: 'face',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({
        slotIndex: s.slotIndex,
        status: 'skipped',
        reason: 'No face detected',
      })),
    };
  }

  onProgress?.('Choosing reference…');
  const refLocal = pickFaceReference(withFaces);
  const reference = withFaces[refLocal];
  const referenceSlotIndex = reference.slotIndex;

  // Find reference state
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
        reason: 'No face detected',
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

export async function detectFacesForSlots(
  slots: AlignableSlot[],
  onProgress?: ProgressCallback
): Promise<(FaceData | null)[]> {
  onProgress?.('Loading face models…');
  await loadFaceApi();
  onProgress?.('Detecting faces…');
  return detectFaces(slots);
}
