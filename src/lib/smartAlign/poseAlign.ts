/**
 * V1.1 — Pose-based Smart Align with HeadScale-primary architecture.
 *
 * Changes from V1:
 *  - HeadScale (from head-only keypoints) is the primary scale anchor
 *  - BodyScale is only used as fallback when HeadReliability is low
 *  - Length-weighted candidate weights via smoothstep
 *  - Candidate Agreement score detects when head measures disagree
 *  - HeadReliability as soft score (not hard product) so it doesn't collapse
 *  - Smooth blend between Head and Body via smoothstep on reliability
 *
 * Anatomical ratios (eye-to-eye = 1.0 reference unit):
 *   eye-mid-to-nose     ≈ 0.65
 *   eye-to-nose-side    ≈ 0.82
 *   ear-to-ear          ≈ 2.20  (only used when ears are confident & plausible)
 */
import type { Tensor } from 'onnxruntime-web';
import { decodeImage } from '@/lib/exportRenderer';
import { ImageState } from '@/lib/types';
import { loadOrt } from './loadOrt';
import { loadYoloPose, KP } from './loadYoloPose';
import {
  AlignResult,
  AlignableSlot,
  ProgressCallback,
  SmartAlignReport,
} from './types';

const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.25;
const KEYPOINT_VIS_THRESHOLD = 0.5;
const EAR_CONFIDENCE_THRESHOLD = 0.6; // ears are noisier, require higher confidence
const MAX_GLOBAL_CROP = 1.15;
const SAFETY_PX = 1;

const MAX_PAN_X_FACTOR = 0.4;
const MAX_PAN_Y_FACTOR = 1.0;

// Anatomical ratios — values relative to eye-to-eye distance
const RATIO_EYE_MID_NOSE = 0.65;
const RATIO_EYE_NOSE_SIDE = 0.82;
const RATIO_EAR_EAR = 2.20;

// Length-weight smoothstep thresholds (in screen pixels after cover scaling)
const LEN_W_LO = 12;
const LEN_W_HI = 50;

// Threshold above which a slot uses HeadScale as-is; below this it blends with Body
const HEAD_RELIABILITY_HEAD_FAVORED = 0.65;
const HEAD_RELIABILITY_BODY_FAVORED = 0.25;

interface Keypoint { x: number; y: number; v: number; }
interface Person { box: { x: number; y: number; w: number; h: number }; score: number; keypoints: Keypoint[]; }

interface ScaleCandidate {
  name: string;
  px: number;
  // value = px / anatomical_ratio. After this normalization, all candidates
  // express the same underlying "person size" in pixels, so they can be
  // compared/medianed directly.
  value: number;
  weight: number;
}

interface PoseData {
  slotIndex: number;
  anchorXNorm: number;
  anchorYNorm: number;
  anchorScale: number;
  anchorMode: 'head-primary' | 'head-body-blend' | 'body-fallback' | 'shoulders-only' | 'eye-to-eye';
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

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function weightedMedian(items: ScaleCandidate[]): number | null {
  const valid = items
    .filter((i) => Number.isFinite(i.value) && i.weight > 0)
    .sort((a, b) => a.value - b.value);
  const total = valid.reduce((sum, i) => sum + i.weight, 0);
  if (valid.length === 0 || total <= 0) return null;
  let acc = 0;
  for (const item of valid) {
    acc += item.weight;
    if (acc >= total / 2) return item.value;
  }
  return valid[valid.length - 1].value;
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

const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
  Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

interface ScaleEstimate {
  scale: number | null;
  reliability: number;
  agreement: number;
  candidates: ScaleCandidate[];
}

/**
 * Estimate scale from head-only keypoints (eyes, nose, optional ears).
 * Each candidate gets a weight based on:
 *  - keypoint confidence (product of the two endpoints)
 *  - length reliability (smoothstep — short distances are noise-prone)
 *  - geometric reliability (yaw penalty for eye-eye and ear-ear when head is turned)
 * Returns weighted-median scale, plus reliability scores for the caller.
 */
function estimateHeadScale(kp: Keypoint[], cover: number): ScaleEstimate {
  const lEye = kp[KP.LEFT_EYE], rEye = kp[KP.RIGHT_EYE];
  const nose = kp[KP.NOSE];
  const lEar = kp[KP.LEFT_EAR], rEar = kp[KP.RIGHT_EAR];

  const candidates: ScaleCandidate[] = [];

  const eyesGood = lEye.v > KEYPOINT_VIS_THRESHOLD && rEye.v > KEYPOINT_VIS_THRESHOLD;
  if (!eyesGood) {
    return { scale: null, reliability: 0, agreement: 0, candidates };
  }

  const eyeMidX = (lEye.x + rEye.x) / 2;
  const eyeMidY = (lEye.y + rEye.y) / 2;
  const eyeEyePx = dist(lEye, rEye);

  // Yaw indicator: how far is the nose from the eye midline (relative to eye-eye)
  // If nose is far off-center, the head is yawed and eye-eye is foreshortened.
  const noseGood = nose.v > KEYPOINT_VIS_THRESHOLD;
  const noseOffsetRel = noseGood ? Math.abs(nose.x - eyeMidX) / Math.max(eyeEyePx, 1) : 0.5;
  const yawReliability = 1 - smoothstep(0.15, 0.45, noseOffsetRel);

  // Eye-to-eye: full yaw penalty applies (foreshortening hits this directly)
  const eyeEyeScreen = eyeEyePx * cover;
  candidates.push({
    name: 'eye-eye',
    px: eyeEyePx,
    value: eyeEyePx / 1.0,
    weight: lEye.v * rEye.v * smoothstep(LEN_W_LO, LEN_W_HI, eyeEyeScreen) * yawReliability,
  });

  if (noseGood) {
    // Eye-to-nose distances suffer less from yaw — at least one side is usually still good.
    const partialYaw = Math.max(0.5, yawReliability);

    const eyeMidNosePx = dist({ x: eyeMidX, y: eyeMidY }, nose);
    candidates.push({
      name: 'eyeMid-nose',
      px: eyeMidNosePx,
      value: eyeMidNosePx / RATIO_EYE_MID_NOSE,
      weight: Math.min(lEye.v, rEye.v) * nose.v *
              smoothstep(LEN_W_LO, LEN_W_HI, eyeMidNosePx * cover) *
              partialYaw,
    });

    const lEyeNosePx = dist(lEye, nose);
    candidates.push({
      name: 'lEye-nose',
      px: lEyeNosePx,
      value: lEyeNosePx / RATIO_EYE_NOSE_SIDE,
      weight: lEye.v * nose.v *
              smoothstep(LEN_W_LO, LEN_W_HI, lEyeNosePx * cover) *
              partialYaw,
    });

    const rEyeNosePx = dist(rEye, nose);
    candidates.push({
      name: 'rEye-nose',
      px: rEyeNosePx,
      value: rEyeNosePx / RATIO_EYE_NOSE_SIDE,
      weight: rEye.v * nose.v *
              smoothstep(LEN_W_LO, LEN_W_HI, rEyeNosePx * cover) *
              partialYaw,
    });
  }

  // Ear-to-ear: only if both ears are clearly visible AND geometry is plausible
  if (lEar.v > EAR_CONFIDENCE_THRESHOLD && rEar.v > EAR_CONFIDENCE_THRESHOLD) {
    const earEarPx = dist(lEar, rEar);
    const plausible = earEarPx > eyeEyePx * 1.4 && earEarPx < eyeEyePx * 3.0;
    if (plausible) {
      candidates.push({
        name: 'ear-ear',
        px: earEarPx,
        value: earEarPx / RATIO_EAR_EAR,
        weight: lEar.v * rEar.v *
                smoothstep(LEN_W_LO * 1.5, LEN_W_HI * 1.5, earEarPx * cover) *
                yawReliability,
      });
    }
  }

  const scale = weightedMedian(candidates);
  const agreement = computeCandidateAgreement(candidates);

  // Head reliability as a soft score (not hard product), so a single mediocre
  // factor doesn't collapse the whole estimate.
  const totalWeight = candidates.reduce((sum, c) => sum + c.weight, 0);
  const usableCount = candidates.filter((c) => c.weight > 0.15).length;
  const weightReliability = clamp01(totalWeight / 2.5);
  const countReliability = clamp01(usableCount / 3);
  const reliability = clamp01(0.40 * weightReliability + 0.25 * countReliability + 0.35 * agreement);

  return { scale, reliability, agreement, candidates };
}

/**
 * Estimate scale from body/torso keypoints (legacy V1 logic).
 * Used as fallback when HeadReliability is low.
 */
function estimateBodyScale(kp: Keypoint[], cover: number): ScaleEstimate {
  const lEye = kp[KP.LEFT_EYE], rEye = kp[KP.RIGHT_EYE];
  const nose = kp[KP.NOSE];
  const lSh = kp[KP.LEFT_SHOULDER], rSh = kp[KP.RIGHT_SHOULDER];
  const lHip = kp[11], rHip = kp[12];

  const candidates: ScaleCandidate[] = [];

  const eyesGood = lEye.v > KEYPOINT_VIS_THRESHOLD && rEye.v > KEYPOINT_VIS_THRESHOLD;
  const shouldersGood = lSh.v > KEYPOINT_VIS_THRESHOLD && rSh.v > KEYPOINT_VIS_THRESHOLD;
  const noseGood = nose.v > KEYPOINT_VIS_THRESHOLD;
  const hipsGood = lHip && rHip && lHip.v > KEYPOINT_VIS_THRESHOLD && rHip.v > KEYPOINT_VIS_THRESHOLD;

  if (!shouldersGood) {
    return { scale: null, reliability: 0, agreement: 0, candidates };
  }

  const shMidX = (lSh.x + rSh.x) / 2;
  const shMidY = (lSh.y + rSh.y) / 2;

  if (eyesGood) {
    const eyeMidX = (lEye.x + rEye.x) / 2;
    const eyeMidY = (lEye.y + rEye.y) / 2;
    const eyeShoulderPx = dist({ x: eyeMidX, y: eyeMidY }, { x: shMidX, y: shMidY });
    candidates.push({
      name: 'eyeMid-shMid',
      px: eyeShoulderPx,
      value: eyeShoulderPx,
      weight: Math.min(lEye.v, rEye.v) * Math.min(lSh.v, rSh.v) *
              smoothstep(LEN_W_LO, LEN_W_HI, eyeShoulderPx * cover),
    });
  }

  const shWidthPx = dist(lSh, rSh);
  candidates.push({
    name: 'sh-sh',
    px: shWidthPx,
    value: shWidthPx / 1.5,
    weight: lSh.v * rSh.v * smoothstep(LEN_W_LO, LEN_W_HI, shWidthPx * cover),
  });

  if (noseGood) {
    const noseShPx = dist(nose, { x: shMidX, y: shMidY });
    candidates.push({
      name: 'nose-shMid',
      px: noseShPx,
      value: noseShPx / 0.85,
      weight: nose.v * Math.min(lSh.v, rSh.v) * smoothstep(LEN_W_LO, LEN_W_HI, noseShPx * cover),
    });
  }

  if (eyesGood && hipsGood) {
    const eyeMidX = (lEye.x + rEye.x) / 2;
    const eyeMidY = (lEye.y + rEye.y) / 2;
    const hipMidX = (lHip.x + rHip.x) / 2;
    const hipMidY = (lHip.y + rHip.y) / 2;
    const eyeHipPx = dist({ x: eyeMidX, y: eyeMidY }, { x: hipMidX, y: hipMidY });
    candidates.push({
      name: 'eyeMid-hipMid',
      px: eyeHipPx,
      value: eyeHipPx / 3.5,
      weight: Math.min(lEye.v, rEye.v) * Math.min(lHip.v, rHip.v) *
              smoothstep(LEN_W_LO, LEN_W_HI * 2, eyeHipPx * cover),
    });
  }

  const scale = weightedMedian(candidates);
  const agreement = computeCandidateAgreement(candidates);
  const totalWeight = candidates.reduce((sum, c) => sum + c.weight, 0);
  const usableCount = candidates.filter((c) => c.weight > 0.15).length;
  const weightReliability = clamp01(totalWeight / 2.5);
  const countReliability = clamp01(usableCount / 2);
  const reliability = clamp01(0.40 * weightReliability + 0.25 * countReliability + 0.35 * agreement);

  return { scale, reliability, agreement, candidates };
}

/**
 * Agreement score: how well do the candidates agree with each other?
 * Returns 1 if they all agree closely, 0 if they disagree wildly.
 * Used to detect when a single bad keypoint corrupts the estimate.
 */
function computeCandidateAgreement(candidates: ScaleCandidate[]): number {
  const usable = candidates.filter((c) => c.weight > 0.15);
  if (usable.length < 2) return 0.3;
  const values = usable.map((c) => c.value);
  const med = median(values);
  const relErrors = values.map((v) => Math.abs(v - med) / Math.max(med, 1));
  const medianRelError = median(relErrors);
  return 1 - smoothstep(0.08, 0.25, medianRelError);
}

interface RawAnchor {
  anchorX: number;
  anchorY: number;
  anchorScale: number;
  mode: 'head-primary' | 'head-body-blend' | 'body-fallback' | 'shoulders-only' | 'eye-to-eye';
}

function extractAnchor(person: Person, slotIndex: number, cover: number): RawAnchor | null {
  const kp = person.keypoints;
  const lEye = kp[KP.LEFT_EYE], rEye = kp[KP.RIGHT_EYE];
  const lSh = kp[KP.LEFT_SHOULDER], rSh = kp[KP.RIGHT_SHOULDER];

  const eyesGood = lEye.v > KEYPOINT_VIS_THRESHOLD && rEye.v > KEYPOINT_VIS_THRESHOLD;
  const shouldersGood = lSh.v > KEYPOINT_VIS_THRESHOLD && rSh.v > KEYPOINT_VIS_THRESHOLD;

  // Position anchor: eye midpoint preferred, fall back to shoulders, then nothing.
  let anchorX: number, anchorY: number;
  if (eyesGood) {
    anchorX = (lEye.x + rEye.x) / 2;
    anchorY = (lEye.y + rEye.y) / 2;
  } else if (shouldersGood) {
    anchorX = (lSh.x + rSh.x) / 2;
    anchorY = (lSh.y + rSh.y) / 2;
  } else {
    return null;
  }

  // Scale: try Head first, then Body as fallback
  const head = estimateHeadScale(kp, cover);
  const body = estimateBodyScale(kp, cover);

  console.log(
    '[CompareShot V1.1] Slot', slotIndex,
    '— HEAD candidates:',
    head.candidates.map((c) =>
      `${c.name}(px=${c.px.toFixed(1)}, val=${c.value.toFixed(1)}, w=${c.weight.toFixed(2)})`
    ).join(', ')
  );
  console.log(
    '[CompareShot V1.1] Slot', slotIndex,
    '— BODY candidates:',
    body.candidates.map((c) =>
      `${c.name}(px=${c.px.toFixed(1)}, val=${c.value.toFixed(1)}, w=${c.weight.toFixed(2)})`
    ).join(', ')
  );
  console.log(
    '[CompareShot V1.1] Slot', slotIndex,
    `— headScale: ${head.scale !== null ? head.scale.toFixed(1) : 'n/a'}`,
    `(reliability: ${head.reliability.toFixed(2)}, agreement: ${head.agreement.toFixed(2)})`,
    `bodyScale: ${body.scale !== null ? body.scale.toFixed(1) : 'n/a'}`,
    `(reliability: ${body.reliability.toFixed(2)}, agreement: ${body.agreement.toFixed(2)})`
  );

  // Choose final scale
  let finalScale: number;
  let mode: RawAnchor['mode'];

  if (head.scale !== null && body.scale !== null) {
    // Both available → smooth blend based on head reliability
    const headWeight = smoothstep(HEAD_RELIABILITY_BODY_FAVORED, HEAD_RELIABILITY_HEAD_FAVORED, head.reliability);
    const bodyWeight = 1 - headWeight;
    finalScale = head.scale * headWeight + body.scale * bodyWeight;

    if (headWeight > 0.85) mode = 'head-primary';
    else if (headWeight < 0.15) mode = 'body-fallback';
    else mode = 'head-body-blend';

    console.log(
      '[CompareShot V1.1] Slot', slotIndex,
      `— chosen: ${mode} (headWeight=${headWeight.toFixed(2)}, finalScale=${finalScale.toFixed(1)})`
    );
  } else if (head.scale !== null) {
    finalScale = head.scale;
    mode = 'head-primary';
    console.log('[CompareShot V1.1] Slot', slotIndex, '— chosen: head-primary (no body)');
  } else if (body.scale !== null) {
    finalScale = body.scale;
    mode = shouldersGood ? 'shoulders-only' : 'body-fallback';
    console.log('[CompareShot V1.1] Slot', slotIndex, '— chosen: body-fallback (no head)');
  } else {
    // Last resort: eye-to-eye if eyes are at least visible
    if (eyesGood) {
      finalScale = dist(lEye, rEye);
      mode = 'eye-to-eye';
      console.log('[CompareShot V1.1] Slot', slotIndex, '— chosen: eye-to-eye fallback');
    } else {
      return null;
    }
  }

  return {
    anchorX,
    anchorY,
    anchorScale: Math.max(1, finalScale),
    mode,
  };
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

      const cover = computeCoverScale(
        img.naturalWidth, img.naturalHeight,
        slot.state._containerW, slot.state._containerH
      );
      const anchor = extractAnchor(best, slot.slotIndex, cover);
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

interface SlotCalc {
  poseIdx: number;
  slotIndex: number;
  state: ImageState;
  pose: PoseData;
  cover: number;
  scaleMatchZoom: number;
  containerW: number;
  containerH: number;
}

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

function solveGlobalCropZoom(calcs: SlotCalc[]): number {
  let { lo, hi } = computeFeasibleYInterval(calcs, 1.0);
  if (lo <= hi) return 1.0;

  ({ lo, hi } = computeFeasibleYInterval(calcs, MAX_GLOBAL_CROP));
  if (lo > hi) return MAX_GLOBAL_CROP;

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
      scaleMatchZoom: 1,
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
      'scaleMatchZoom:', c.scaleMatchZoom.toFixed(3),
      'mode:', c.pose.anchorMode
    );
  }

  const globalCropZoom = solveGlobalCropZoom(calcs);
  console.log('[CompareShot] globalCropZoom:', globalCropZoom.toFixed(4));

  const { lo: globalLoY, hi: globalHiY, perSlot } = computeFeasibleYInterval(calcs, globalCropZoom);
  const naturalEyeYs = perSlot.map((s) => s.baseY);
  const preferredY = median(naturalEyeYs);
  const targetY = clamp(preferredY, globalLoY, globalHiY);
  console.log(
    '[CompareShot] Y-interval feasible:', globalLoY <= globalHiY,
    'lo:', globalLoY.toFixed(1), 'hi:', globalHiY.toFixed(1),
    'preferredY:', preferredY.toFixed(1),
    'targetY:', targetY.toFixed(1)
  );

  const naturalEyeXs = calcs.map((c) => {
    const z = c.scaleMatchZoom * globalCropZoom;
    return (c.pose.anchorXNorm - 0.5) * c.pose.naturalWidth * c.cover * z;
  });
  const targetX = median(naturalEyeXs);

  const results: AlignResult[] = [];
  for (let i = 0; i < slots.length; i++) {
    const slot = slots[i];
    const pose = detected[i];
    if (!pose) {
      results.push({ slotIndex: slot.slotIndex, status: 'failed', reason: 'No person detected — will fall back to features' });
      continue;
    }
    onProgress?.(`Aligning ${i + 1}/${slots.length}…`);
    const calc = calcs.find((x) => x.slotIndex === slot.slotIndex)!;
    const idxInCalcs = calcs.indexOf(calc);

    const finalZoom = calc.scaleMatchZoom * globalCropZoom;
    const drawnW = calc.pose.naturalWidth * calc.cover * finalZoom;
    const drawnH = calc.pose.naturalHeight * calc.cover * finalZoom;
    const reserveX = Math.max(0, (drawnW - calc.containerW) / 2 - SAFETY_PX);
    const reserveY = Math.max(0, (drawnH - calc.containerH) / 2 - SAFETY_PX);

    const baseX = (calc.pose.anchorXNorm - 0.5) * calc.pose.naturalWidth * calc.cover * finalZoom;
    const baseY = perSlot[idxInCalcs].baseY;

    let panX = targetX - baseX;
    let panY = targetY - baseY;

    const personSize = calc.pose.anchorScale * Math.max(calc.pose.naturalWidth, calc.pose.naturalHeight) * calc.cover * finalZoom;
    const maxPanX = personSize * MAX_PAN_X_FACTOR;
    const maxPanY = personSize * MAX_PAN_Y_FACTOR;

    panX = clamp(panX, -maxPanX, maxPanX);
    panY = clamp(panY, -maxPanY, maxPanY);
    panX = clamp(panX, -reserveX, reserveX);
    panY = clamp(panY, -reserveY, reserveY);

    const isReference = slot.slotIndex === referenceSlotIndex;
    console.log(
      '[CompareShot] Transform slot', slot.slotIndex,
      isReference ? '(REFERENCE)' : '',
      'finalZoom:', finalZoom.toFixed(3),
      'panX:', panX.toFixed(1), 'panY:', panY.toFixed(1)
    );

    results.push({
      slotIndex: slot.slotIndex,
      status: isReference ? 'reference' : 'aligned',
      transform: { zoom: finalZoom, panX, panY, rotation: 0 },
    });
  }

  return { mode: 'face', referenceSlotIndex, results };
}

export type { PoseData };
