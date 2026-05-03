/**
 * Hybrid Smart Align orchestrator.
 *
 * Strategy:
 *  1. Try face detection on every loaded image.
 *  2. If at least HALF of the loaded images have a detected face, run the
 *     face-based pipeline.
 *  3. Otherwise (or if face mode produces too few aligned images), fall back
 *     to the feature-matching pipeline (OpenCV.js + AKAZE).
 *
 * The hybrid choice is automatic — there is no manual toggle in the UI per
 * the project's design decisions.
 */
import { ImageState } from '@/lib/types';
import { detectFacesForSlots, faceAlign } from './faceAlign';
import { featureAlign } from './featureAlign';
import { AlignableSlot, ProgressCallback, SmartAlignReport } from './types';

export type { SmartAlignReport, AlignResult, AlignTransform } from './types';

export interface SmartAlignInput {
  /** Images currently loaded into slots (null entries are filtered out). */
  images: (ImageState | null)[];
  onProgress?: ProgressCallback;
}

export async function smartAlign({ images, onProgress }: SmartAlignInput): Promise<SmartAlignReport> {
  // Build the list of slots that actually have an image
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
  let faceDetectError: Error | null = null;
  try {
    faceDetections = await detectFacesForSlots(slots, onProgress);
  } catch (err) {
    // Face detection failure shouldn't block the whole flow — fall through to
    // feature alignment below.
    faceDetectError = err instanceof Error ? err : new Error(String(err));
    // eslint-disable-next-line no-console
    console.warn('[CompareShot] Face detection failed, falling back to features:', faceDetectError);
  }

  const facesFound = faceDetections.filter((f) => f !== null).length;
  const useFaceMode = facesFound >= Math.ceil(slots.length / 2);

  if (useFaceMode) {
    const report = await faceAlign(slots, faceDetections, onProgress);
    // If face mode somehow only aligned the reference, fall back to features
    const aligned = report.results.filter((r) => r.status === 'aligned').length;
    if (aligned > 0 || slots.length === 1) {
      return report;
    }
    // Otherwise fall through to feature mode
  }

  // ---- Phase B: Feature matching ----
  return featureAlign(slots, onProgress);
}
