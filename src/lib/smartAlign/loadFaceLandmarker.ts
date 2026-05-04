/**
 * Lazy-loads MediaPipe FaceLandmarker. The library and the model are fetched
 * from jsDelivr the first time Smart Align is invoked and cached afterwards.
 *
 * MediaPipe's FaceLandmarker provides:
 *   - 478 3D facial landmarks (face-api had 68 2D)
 *   - Iris landmarks (indices 468..477) — sub-pixel precise
 *   - Better detection in challenging lighting than face-api's tinyFaceDetector
 *   - Built on BlazeFace, the same detector Google uses for their AR products
 */
import {
  FaceLandmarker,
  FilesetResolver,
} from '@mediapipe/tasks-vision';

let landmarkerPromise: Promise<FaceLandmarker> | null = null;

const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export async function loadFaceLandmarker(): Promise<FaceLandmarker> {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);
    const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        // CPU is more reliable across browsers; GPU sometimes fails on Safari
        delegate: 'CPU',
      },
      runningMode: 'IMAGE',
      numFaces: 5,
      // Keep blendshapes/transformation matrices off — we only need landmarks
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
    return landmarker;
  })().catch((err) => {
    landmarkerPromise = null;
    throw err;
  });
  return landmarkerPromise;
}
