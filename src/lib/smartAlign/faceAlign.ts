/**
 * Face-based Smart Align using face-api.js (Tiny Face Detector + 68 Landmarks).
 *
 * Strategy:
 *  - Detect ALL faces in every input image (more sensitive settings than
 *    detectSingleFace, so small faces like in smartphone landscape shots get
 *    caught)
 *  - For each image, pick the LARGEST face (closest person) as the anchor
 *  - From the 68 landmarks, extract the head bounding box: top-of-head from
 *    the detection box top, chin from landmark 8 (jaw bottom), face-center
 *    from the average of all landmarks
 *  - Score each image: sharpness + head size (proxy for centrality / closeness)
 *    + detection confidence. Pick the highest as the reference.
 *  - For every other image: scale so that head height matches reference, and
 *    translate so that the head center lands at the reference's head center.
 *  - No rotation (landmarks are too noisy for stable rotation estimation).
 *  - Apply auto-fill so no black borders appear.
 *  - For images WHERE NO FACE WAS DETECTED, return null transform — caller
 *    falls back to feature alignment for those slots.
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
  /** Center of the head (image px) — used for translation alignment */
  headCenterX: number;
  headCenterY: number;
  /** Head height in image pixels — distance from top-of-head to chin */
  headHeight: number;
  /** Detection score (0..1) */
  detectionScore: number;
  /** Sharpness proxy (relative ordering only) */
  sharpness: number;
}

function computeCoverScale(naturalW: number, naturalH: number, containerW: number, containerH: number) {
  const cw = containerW > 0 ? containerW : 960;
  const ch = containerH > 0 ? containerH : 1625;
  return Math.max(cw / naturalW, ch / naturalH);
}

/**
 * Quick Laplacian-style sharpness on a downsampled grayscale image. Used only
 * for relative comparison between images — absolute value is meaningless.
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

// -------- Detection --------

/**
 * Detect faces in all input images. Returns null for slots where no face is
 * found (caller decides what to do).
 *
 * Uses detectAllFaces (instead of detectSingleFace) with a sensitive
 * configuration: lower scoreThreshold and a larger inputSize, so smaller
 * faces (typical for smartphone landscape shots) are still detected. From
 * the resulting list we pick the largest face per image (closest person).
 */
export async function detectFacesForSlots(
  slots: AlignableSlot[],
  onProgress?: ProgressCallback
): Promise<(FaceData | null)[]> {
  onProgress?.('Loading face models…');
  const faceapi = await loadFaceApi();

  onProgress?.('Detecting faces…');
  const detectorOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 608, // higher resolution → small faces still get detected
    scoreThreshold: 0.3, // lower threshold → more sensitive
  });

  const results: (FaceData | null)[] = [];
  for (const slot of slots) {
    const img = await decodeImage(slot.state.url);
    try {
      // detectAllFaces returns every face; we then pick the largest.
      const detections = await faceapi.detectAllFaces(img, detectorOptions).withFaceLandmarks();

      if (!detections || detections.length === 0) {
        results.push(null);
        continue;
      }

      // Pick the face with the largest bounding box (closest person)
      let largest = detections[0];
      let largestArea = largest.detection.box.width * largest.detection.box.height;
      for (let i = 1; i < detections.length; i++) {
        const d = detections[i];
        const area = d.detection.box.width * d.detection.box.height;
        if (area > largestArea) {
          largest = d;
          largestArea = area;
        }
      }

      // Extract head geometry
      const lm = largest.landmarks.positions;
      const box = largest.detection.box;

      // Top-of-head: use the detection box top edge (face-api's box covers the
      // visible head reasonably well; landmarks alone don't include hair).
      const headTop = box.top;

      // Chin: landmark 8 is the bottom of the jaw
      const chinY = lm[8].y;

      // Head height: from top of detection box to chin
      const headHeight = Math.max(1, chinY - headTop);

      // Head center: average of all 68 landmark positions for stability
      let cx = 0;
      let cy = 0;
      for (const p of lm) {
        cx += p.x;
        cy += p.y;
      }
      cx /= lm.length;
      cy /= lm.length;

      // We want the head center to be vertically at the midpoint between top-of-head
      // and chin, not at the landmark centroid (which leans toward the lower face).
      // Use the geometric vertical midpoint of the head.
      const headCenterY = (headTop + chinY) / 2;
      const headCenterX = cx; // horizontal midpoint from landmarks is fine

      const sharpness = await imageSharpness(img);

      results.push({
        slotIndex: slot.slotIndex,
        img,
        headCenterX,
        headCenterY,
        headHeight,
        detectionScore: largest.detection.score,
        sharpness,
      });
    } catch {
      results.push(null);
    }
  }
  return results;
}

// -------- Reference selection --------

function pickFaceReference(faces: FaceData[]): number {
  if (faces.length === 0) return 0;
  const sN = normalize(faces.map((f) => f.sharpness));
  // Use head height (in image pixels) as the proxy for closeness/centrality —
  // a closer person has a larger head.
  const hN = normalize(faces.map((f) => f.headHeight));
  const dN = normalize(faces.map((f) => f.detectionScore));
  // Weights: closeness (head size) most important, then sharpness, then confidence.
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
  // Image-pixel scale to apply: makes the current head height equal the
  // reference head height (in their respective image-pixel spaces).
  const imageScale = reference.headHeight / Math.max(1, current.headHeight);

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

  // Reference head-center offset from its natural center, in reference-container px:
  const refOffsetX_refImg = reference.headCenterX - referenceState.naturalWidth / 2;
  const refOffsetY_refImg = reference.headCenterY - referenceState.naturalHeight / 2;
  const refOffsetX_container = refOffsetX_refImg * refCover;
  const refOffsetY_container = refOffsetY_refImg * refCover;

  // Current head-center offset from its natural center, in current image px.
  // After applying zoom on top of cover, this projects onto the container as
  //   offset * curCover * zoom = offset * refCover * imageScale.
  const curOffsetX_curImg = current.headCenterX - currentState.naturalWidth / 2;
  const curOffsetY_curImg = current.headCenterY - currentState.naturalHeight / 2;
  const curProjectedX = curOffsetX_curImg * refCover * imageScale;
  const curProjectedY = curOffsetY_curImg * refCover * imageScale;

  // To align the head centers, panX must shift the projected current head to
  // the reference head location:
  let panX = refOffsetX_container - curProjectedX;
  let panY = refOffsetY_container - curProjectedY;

  let finalZoom = Math.max(0.2, Math.min(5, zoom));

  // Auto-fill: bump zoom so no black borders appear after pan + rotation (no rotation here).
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

/**
 * Align using face-api results. Slots without a detected face return a
 * 'no-face' result so the caller can choose to feature-align them as a fallback.
 */
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
      // Mark for caller-driven fallback. The caller (smartAlign in index.ts)
      // will detect this status and run feature alignment for this slot.
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

// Re-export FaceData type for use in detection
export type { FaceData };
