/**
 * Hybrid Smart Align orchestrator.
 *
 * Strategy:
 *  1. Try face detection on every loaded image.
 *  2. If at least HALF of the loaded images have a detected face, run the
 *     face-based pipeline. Slots in face mode that have no detected face will
 *     be re-aligned via feature-matching as a fallback (so a single failed
 *     detection doesn't block the whole comparison).
 *  3. Otherwise, run feature matching for everything in a Web Worker so
 *     OpenCV.js never blocks the UI thread.
 */
import { ImageState } from '@/lib/types';
import { detectFacesForSlots, faceAlign } from './faceAlign';
import { featureAlignViaWorker } from './workerClient';
import { AlignableSlot, AlignResult, ProgressCallback, SmartAlignReport } from './types';

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
  console.log('[CompareShot] Face detections:', facesFound, '/', slots.length, '— face mode:', useFaceMode);

  if (useFaceMode) {
    const faceReport = await faceAlign(slots, faceDetections, onProgress);

    // Identify slots that face mode could not align (no face detected)
    const fallbackSlotIndices = new Set<number>();
    for (const r of faceReport.results) {
      if (r.status === 'failed' && r.reason && r.reason.includes('fall back')) {
        fallbackSlotIndices.add(r.slotIndex);
      }
    }

    if (fallbackSlotIndices.size === 0) {
      return faceReport;
    }

    // ---- Feature fallback for slots without a detected face ----
    // We need to feature-align those slots against the SAME reference image
    // that face-align used, so the final positions are consistent with the
    // face-aligned slots. We run featureAlignViaWorker on a sub-list:
    //   [reference slot, ...fallback slots]
    onProgress?.('Aligning faceless slots via features…');
    const refSlot = slots.find((s) => s.slotIndex === faceReport.referenceSlotIndex);
    const fallbackSlots = slots.filter((s) => fallbackSlotIndices.has(s.slotIndex));
    if (refSlot && fallbackSlots.length > 0) {
      const subList: AlignableSlot[] = [refSlot, ...fallbackSlots];
      const featureReport = await featureAlignViaWorker(subList, onProgress);

      // Splice the fallback transforms back into the face report.
      const merged: AlignResult[] = faceReport.results.map((r) => {
        if (!fallbackSlotIndices.has(r.slotIndex)) return r;
        // Find the matching feature result
        const fr = featureReport.results.find((x) => x.slotIndex === r.slotIndex);
        if (fr && fr.status === 'aligned' && fr.transform) {
          return {
            slotIndex: r.slotIndex,
            status: 'aligned',
            transform: fr.transform,
          };
        }
        // Feature align also failed — keep the original failed result
        return r;
      });

      return {
        mode: 'face',
        referenceSlotIndex: faceReport.referenceSlotIndex,
        results: merged,
      };
    }
    return faceReport;
  }

  // ---- Phase B: Feature matching for everything ----
  return featureAlignViaWorker(slots, onProgress);
}
