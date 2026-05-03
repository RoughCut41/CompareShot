/**
 * Hybrid Smart Align orchestrator.
 *
 * Strategy:
 *  1. Try face detection on every loaded image.
 *  2. If at least HALF of the loaded images have a detected face, run the
 *     face-based pipeline on the main thread (face-api works fine there).
 *  3. Otherwise, fall back to the feature-matching pipeline running in a
 *     Web Worker so OpenCV.js never blocks the UI thread.
 */
import { ImageState } from '@/lib/types';
import { detectFacesForSlots, faceAlign } from './faceAlign';
import { featureAlignViaWorker } from './workerClient';
import { AlignableSlot, ProgressCallback, SmartAlignReport } from './types';

export type { SmartAlignReport, AlignResult, AlignTransform } from './types';

export interface SmartAlignInput {
  images: (ImageState | null)[];
  onProgress?: ProgressCallback;
}

export async function smartAlign({ images, onProgress }: SmartAlignInput): Promise<SmartAlignReport> {
  const slots: AlignableSlot[] = [];
  images.forEach((state, i) => {
    if (state) slots.push({ slotIndex: i, state });
  });

  if (slots.length < 2) {
    return {
      mode: 'feature',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({ slotIndex: s.slotIndex, status: 'skipped' })),
    };
  }

  // ---- Phase A: Try face detection ----
  let faceDetections: Awaited<ReturnType<typeof detectFacesForSlots>> = [];
  try {
    faceDetections = await detectFacesForSlots(slots, onProgress);
  } catch (err) {
    console.warn('[CompareShot] Face detection failed, falling back to features:', err);
  }

  const facesFound = faceDetections.filter((f) => f !== null).length;
  const useFaceMode = facesFound >= Math.ceil(slots.length / 2);

  if (useFaceMode) {
    const report = await faceAlign(slots, faceDetections, onProgress);
    const aligned = report.results.filter((r) => r.status === 'aligned').length;
    if (aligned > 0 || slots.length === 1) {
      return report;
    }
  }

  // ---- Phase B: Feature matching in Web Worker ----
  return featureAlignViaWorker(slots, onProgress);
}
