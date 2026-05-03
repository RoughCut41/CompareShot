/**
 * Lazy-loads face-api.js (the maintained @vladmandic/face-api fork) and its
 * detector + landmark models. The library is installed via npm but split into
 * its own chunk via dynamic import — only fetched when Smart Align is actually
 * invoked.
 */
import type * as FaceApiNS from '@vladmandic/face-api';

type FaceApiModule = typeof FaceApiNS;

let faceApiPromise: Promise<FaceApiModule> | null = null;
let modelsLoaded = false;

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/';

export async function loadFaceApi(): Promise<FaceApiModule> {
  if (!faceApiPromise) {
    faceApiPromise = import('@vladmandic/face-api').catch((err) => {
      faceApiPromise = null;
      throw err;
    });
  }
  const faceapi = await faceApiPromise;

  if (!modelsLoaded) {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    ]);
    modelsLoaded = true;
  }

  return faceapi;
}
