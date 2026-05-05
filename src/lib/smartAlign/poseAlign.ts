/**
 * Pose-based Smart Align using YOLO11n-Pose via ONNX Runtime Web.
 *
 * Strategy:
 *  - Run YOLO11-Pose on every input image at 640×640 input resolution.
 *  - Pick the LARGEST detected person per image (closest to camera).
 *  - Use stable keypoint pairs as alignment anchors:
 *      Translation: midpoint of left_eye + right_eye
 *      Scale:       eye-to-shoulder-midpoint distance
 *    This is more robust than eye-to-chin (which face-detectors gave us)
 *    because shoulders are always present even when face is partially
 *    occluded, and the eye-shoulder distance is anatomically stable.
 *  - Fall back gracefully: if eyes are low-confidence but shoulders are
 *    high-confidence, use shoulder midpoint as anchor instead.
 *  - Auto-fill: bump zoom so no black borders appear.
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

interface Keypoint {
  x: number; // pixel coords in original image
  y: number;
  v: number; // visibility / confidence
}

interface Person {
  box: { x: number; y: number; w: number; h: number };
  score: number;
  keypoints: Keypoint[]; // 17 entries
}

interface PoseData {
  slotIndex: number;
  /** Translation anchor in original image pixels */
  anchorX: number;
  anchorY: number;
  /** Scale metric in original image pixels — anatomical reference distance */
  anchorScale: number;
  /** Which anchor strategy was used (for diagnostics) */
  anchorMode: 'eyes-to-shoulders' | 'shoulders-only' | 'eye-to-eye';
  /** Detection confidence */
  detectionScore: number;
  /** Image sharpness (relative comparison only) */
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
  let sum = 0, sumSq = 0, n = 0;
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

// -------- Image preprocessing for YOLO --------

interface PreprocessedImage {
  data: Float32Array;
  /** Mapping from model coords (0..640) back to original image pixels */
  scale: number;
  padX: number;
  padY: number;
}

/**
 * Preprocess image for YOLO11: letterbox to 640×640 maintaining aspect ratio,
 * normalize to 0..1, transpose to NCHW.
 */
function preprocess(img: HTMLImageElement): PreprocessedImage {
  const W = img.naturalWidth;
  const H = img.naturalHeight;

  // Letterbox: scale so longest edge = INPUT_SIZE, then pad with gray
  const scale = INPUT_SIZE / Math.max(W, H);
  const newW = Math.round(W * scale);
  const newH = Math.round(H * scale);
  const padX = Math.floor((INPUT_SIZE - newW) / 2);
  const padY = Math.floor((INPUT_SIZE - newH) / 2);

  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgb(114, 114, 114)'; // YOLO standard letterbox color
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(img, padX, padY, newW, newH);

  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  // Convert RGBA → RGB and HWC → CHW, normalize to 0..1
  const float = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    const r = imageData[i * 4] / 255;
    const g = imageData[i * 4 + 1] / 255;
    const b = imageData[i * 4 + 2] / 255;
    float[i] = r;
    float[planeSize + i] = g;
    float[2 * planeSize + i] = b;
  }

  return { data: float, scale, padX, padY };
}

// -------- YOLO output postprocessing --------

/**
 * Parse YOLO11-Pose output tensor [1, 56, 8400] into Person[] with
 * coordinates mapped back to the original image space.
 *
 * Output channels per detection:
 *   0..3: bbox (cx, cy, w, h) in 640×640 model space
 *   4:    person confidence
 *   5..55: 17 × (x, y, visibility) — keypoints in 640×640 model space
 */
function postprocess(
  output: Float32Array,
  numDetections: number,
  pre: PreprocessedImage,
  origW: number,
  origH: number
): Person[] {
  const people: Person[] = [];
  // Output is [1, 56, 8400] flattened. Channels are stored contiguously:
  //   output[c * numDetections + i] = value of channel c for detection i

  const kpStartChannel = 5;
  const numKeypoints = 17;

  for (let i = 0; i < numDetections; i++) {
    const score = output[4 * numDetections + i];
    if (score < CONFIDENCE_THRESHOLD) continue;

    // Decode bbox (model space → original image space)
    const cxModel = output[0 * numDetections + i];
    const cyModel = output[1 * numDetections + i];
    const wModel = output[2 * numDetections + i];
    const hModel = output[3 * numDetections + i];

    const cxOrig = (cxModel - pre.padX) / pre.scale;
    const cyOrig = (cyModel - pre.padY) / pre.scale;
    const wOrig = wModel / pre.scale;
    const hOrig = hModel / pre.scale;

    // Decode keypoints
    const keypoints: Keypoint[] = [];
    for (let k = 0; k < numKeypoints; k++) {
      const xCh = kpStartChannel + k * 3;
      const yCh = kpStartChannel + k * 3 + 1;
      const vCh = kpStartChannel + k * 3 + 2;
      const xModel = output[xCh * numDetections + i];
      const yModel = output[yCh * numDetections + i];
      const v = output[vCh * numDetections + i];
      const x = (xModel - pre.padX) / pre.scale;
      const y = (yModel - pre.padY) / pre.scale;
      keypoints.push({ x, y, v });
    }

    people.push({
      box: {
        x: cxOrig - wOrig / 2,
        y: cyOrig - hOrig / 2,
        w: wOrig,
        h: hOrig,
      },
      score,
      keypoints,
    });
  }

  // Non-Maximum Suppression: keep only top non-overlapping detections
  return nms(people, 0.5).filter(
    (p) => p.box.x + p.box.w > 0 && p.box.y + p.box.h > 0 &&
           p.box.x < origW && p.box.y < origH
  );
}

function nms(people: Person[], iouThreshold: number): Person[] {
  const sorted = people.slice().sort((a, b) => b.score - a.score);
  const kept: Person[] = [];
  for (const p of sorted) {
    let suppressed = false;
    for (const k of kept) {
      const x1 = Math.max(p.box.x, k.box.x);
      const y1 = Math.max(p.box.y, k.box.y);
      const x2 = Math.min(p.box.x + p.box.w, k.box.x + k.box.w);
      const y2 = Math.min(p.box.y + p.box.h, k.box.y + k.box.h);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const union = p.box.w * p.box.h + k.box.w * k.box.h - inter;
      if (union > 0 && inter / union > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push(p);
  }
  return kept;
}

// -------- Anchor extraction --------

function extractAnchor(
  person: Person
): {
  anchorX: number;
  anchorY: number;
  anchorScale: number;
  mode: 'eyes-to-shoulders' | 'shoulders-only' | 'eye-to-eye';
} | null {
  const kp = person.keypoints;
  const lEye = kp[KP.LEFT_EYE];
  const rEye = kp[KP.RIGHT_EYE];
  const lSh = kp[KP.LEFT_SHOULDER];
  const rSh = kp[KP.RIGHT_SHOULDER];

  const eyesGood = lEye.v > KEYPOINT_VIS_THRESHOLD && rEye.v > KEYPOINT_VIS_THRESHOLD;
  const shouldersGood = lSh.v > KEYPOINT_VIS_THRESHOLD && rSh.v > KEYPOINT_VIS_THRESHOLD;

  // Best case: both eyes and both shoulders confident → eye anchor + eye-to-shoulder scale
  if (eyesGood && shouldersGood) {
    const eyeMidX = (lEye.x + rEye.x) / 2;
    const eyeMidY = (lEye.y + rEye.y) / 2;
    const shMidX = (lSh.x + rSh.x) / 2;
    const shMidY = (lSh.y + rSh.y) / 2;
    const dx = shMidX - eyeMidX;
    const dy = shMidY - eyeMidY;
    const eyeToShoulder = Math.sqrt(dx * dx + dy * dy);
    return {
      anchorX: eyeMidX,
      anchorY: eyeMidY,
      anchorScale: Math.max(1, eyeToShoulder),
      mode: 'eyes-to-shoulders',
    };
  }

  // Fallback: only shoulders confident → shoulder anchor + shoulder-to-shoulder distance
  if (shouldersGood) {
    const shMidX = (lSh.x + rSh.x) / 2;
    const shMidY = (lSh.y + rSh.y) / 2;
    const dx = rSh.x - lSh.x;
    const dy = rSh.y - lSh.y;
    const shoulderWidth = Math.sqrt(dx * dx + dy * dy);
    return {
      anchorX: shMidX,
      anchorY: shMidY,
      anchorScale: Math.max(1, shoulderWidth),
      mode: 'shoulders-only',
    };
  }

  // Last resort: only eyes confident → eye-to-eye distance as scale
  if (eyesGood) {
    const eyeMidX = (lEye.x + rEye.x) / 2;
    const eyeMidY = (lEye.y + rEye.y) / 2;
    const dx = rEye.x - lEye.x;
    const dy = rEye.y - lEye.y;
    const eyeWidth = Math.sqrt(dx * dx + dy * dy);
    return {
      anchorX: eyeMidX,
      anchorY: eyeMidY,
      anchorScale: Math.max(1, eyeWidth),
      mode: 'eye-to-eye',
    };
  }

  return null;
}

// -------- Detection orchestration --------

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
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      const feeds: Record<string, Tensor> = { [inputName]: inputTensor };
      const outputMap = await session.run(feeds);
      const output = outputMap[outputName];
      const dims = output.dims; // [1, 56, 8400]
      const data = output.data as Float32Array;
      const numDetections = dims[2];

      const people = postprocess(data, numDetections, pre, img.naturalWidth, img.naturalHeight);
      console.log('[CompareShot] Slot', slot.slotIndex, '— person candidates:', people.length);

      if (people.length === 0) {
        results.push(null);
        continue;
      }

      // Pick the largest person (closest)
      let best = people[0];
      let bestArea = best.box.w * best.box.h;
      for (let k = 1; k < people.length; k++) {
        const a = people[k].box.w * people[k].box.h;
        if (a > bestArea) {
          best = people[k];
          bestArea = a;
        }
      }

      const anchor = extractAnchor(best);
      if (!anchor) {
        console.warn('[CompareShot] Slot', slot.slotIndex, '— no usable keypoints');
        results.push(null);
        continue;
      }
      console.log('[CompareShot] Slot', slot.slotIndex, '— mode:', anchor.mode);

      const sharpness = await imageSharpness(img);

      results.push({
        slotIndex: slot.slotIndex,
        anchorX: anchor.anchorX,
        anchorY: anchor.anchorY,
        anchorScale: anchor.anchorScale,
        anchorMode: anchor.mode,
        detectionScore: best.score,
        sharpness,
      });
    } catch (err) {
      console.warn('[CompareShot] Pose detection failed for slot', slot.slotIndex, err);
      results.push(null);
    }
  }
  return results;
}

// -------- Reference selection --------

function pickReference(poses: PoseData[]): number {
  if (poses.length === 0) return 0;
  const sN = normalize(poses.map((f) => f.sharpness));
  const hN = normalize(poses.map((f) => f.anchorScale));
  const dN = normalize(poses.map((f) => f.detectionScore));
  const scores = poses.map((_, i) => 0.55 * hN[i] + 0.3 * sN[i] + 0.15 * dN[i]);
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

// -------- Per-image transform --------

function poseToTransform(
  current: PoseData,
  reference: PoseData,
  currentState: ImageState,
  referenceState: ImageState
): AlignTransform {
  const imageScale = reference.anchorScale / Math.max(1, current.anchorScale);

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

  const refOffsetX = reference.anchorX - referenceState.naturalWidth / 2;
  const refOffsetY = reference.anchorY - referenceState.naturalHeight / 2;
  const refOffsetX_container = refOffsetX * refCover;
  const refOffsetY_container = refOffsetY * refCover;

  const curOffsetX = current.anchorX - currentState.naturalWidth / 2;
  const curOffsetY = current.anchorY - currentState.naturalHeight / 2;
  const curProjectedX = curOffsetX * refCover * imageScale;
  const curProjectedY = curOffsetY * refCover * imageScale;

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

// -------- Public entry --------

export async function poseAlign(
  slots: AlignableSlot[],
  detected: (PoseData | null)[],
  onProgress?: ProgressCallback
): Promise<SmartAlignReport> {
  const withPose: PoseData[] = [];
  for (const f of detected) {
    if (f) withPose.push(f);
  }

  if (withPose.length === 0) {
    return {
      mode: 'face',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({
        slotIndex: s.slotIndex,
        status: 'failed',
        reason: 'No person detected',
      })),
    };
  }

  onProgress?.('Choosing reference…');
  const refLocal = pickReference(withPose);
  const reference = withPose[refLocal];
  const referenceSlotIndex = reference.slotIndex;
  const refState = slots.find((s) => s.slotIndex === referenceSlotIndex)!.state;

  const results: AlignResult[] = [];
  for (let i = 0; i < slots.length; i++) {
    const slot = slots[i];
    if (slot.slotIndex === referenceSlotIndex) {
      results.push({ slotIndex: slot.slotIndex, status: 'reference' });
      continue;
    }
    const pose = detected[i];
    if (!pose) {
      results.push({
        slotIndex: slot.slotIndex,
        status: 'failed',
        reason: 'No person detected — will fall back to features',
      });
      continue;
    }
    onProgress?.(`Aligning ${i + 1}/${slots.length}…`);
    const transform = poseToTransform(pose, reference, slot.state, refState);
    results.push({
      slotIndex: slot.slotIndex,
      status: 'aligned',
      transform,
    });
  }

  return {
    mode: 'face', // We keep this string for backward compatibility with the UI
    referenceSlotIndex,
    results,
  };
}

export type { PoseData };
