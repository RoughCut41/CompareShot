/**
 * Lazy-loads the YOLO11n-Pose ONNX model.
 *
 * Model details:
 *   - Input:  [1, 3, 640, 640] (RGB, normalized 0..1, NCHW)
 *   - Output: [1, 56, 8400] where 56 = [4 box (cx,cy,w,h), 1 conf,
 *             17*3 keypoints (x,y,visibility)]
 *   - 17 COCO keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear,
 *     4=right_ear, 5=left_shoulder, 6=right_shoulder, 7=left_elbow,
 *     8=right_elbow, 9=left_wrist, 10=right_wrist, 11=left_hip,
 *     12=right_hip, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
 */
import type { InferenceSession } from 'onnxruntime-web';
import { loadOrt } from './loadOrt';

const MODEL_URL = '/models/yolo11s-pose.onnx';

let sessionPromise: Promise<InferenceSession> | null = null;

export async function loadYoloPose(): Promise<InferenceSession> {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    const ort = await loadOrt();
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return session;
  })().catch((err) => {
    sessionPromise = null;
    throw err;
  });
  return sessionPromise;
}

// COCO keypoint indices for easy reference
export const KP = {
  NOSE: 0,
  LEFT_EYE: 1,
  RIGHT_EYE: 2,
  LEFT_EAR: 3,
  RIGHT_EAR: 4,
  LEFT_SHOULDER: 5,
  RIGHT_SHOULDER: 6,
} as const;
