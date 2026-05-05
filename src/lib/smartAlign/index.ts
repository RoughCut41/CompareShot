/**
 * Hybrid Smart Align orchestrator.
 *
 * Strategy:
 *  1. Run YOLO11n-Pose on all loaded images.
 *  2. If at least HALF the images have a detected person, use pose mode.
 *     Slots without a detected person fall back to feature matching.
 *  3. Otherwise, run feature matching (AKAZE in worker — kept as fallback
 *     until SuperPoint+LightGlue is integrated in phase 3).
 */
import { ImageState } from '@/lib/types';
import { detectPoseForSlots, poseAlign } from './poseAlign';
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

  // ---- Phase A: Try pose detection ----
  let poseDetections: Awaited<ReturnType<typeof detectPoseForSlots>> = [];
  try {
    poseDetections = await detectPoseForSlots(slots, onProgress);
  } catch (err) {
    console.warn('[CompareShot] Pose detection failed entirely:', err);
  }

  const peopleFound = poseDetections.filter((p) => p !== null).length;
  const usePoseMode = peopleFound >= Math.ceil(slots.length / 2);
  console.log('[CompareShot] Persons detected:', peopleFound, '/', slots.length, '— pose mode:', usePoseMode);

  if (usePoseMode) {
    const poseReport = await poseAlign(slots, poseDetections, onProgress);

    // Identify slots that need feature fallback (no person detected)
    const fallbackSlotIndices = new Set<number>();
    for (const r of poseReport.results) {
      if (r.status === 'failed' && r.reason && r.reason.includes('fall back')) {
        fallbackSlotIndices.add(r.slotIndex);
      }
    }

    if (fallbackSlotIndices.size === 0) {
      return poseReport;
    }

    onProgress?.('Aligning personless slots via features…');
    const refSlot = slots.find((s) => s.slotIndex === poseReport.referenceSlotIndex);
    const fallbackSlots = slots.filter((s) => fallbackSlotIndices.has(s.slotIndex));
    if (refSlot && fallbackSlots.length > 0) {
      const subList: AlignableSlot[] = [refSlot, ...fallbackSlots];
      const featureReport = await featureAlignViaWorker(subList, onProgress);

      const merged: AlignResult[] = poseReport.results.map((r) => {
        if (!fallbackSlotIndices.has(r.slotIndex)) return r;
        const fr = featureReport.results.find((x) => x.slotIndex === r.slotIndex);
        if (fr && fr.status === 'aligned' && fr.transform) {
          return {
            slotIndex: r.slotIndex,
            status: 'aligned',
            transform: fr.transform,
          };
        }
        return r;
      });

      return {
        mode: 'face',
        referenceSlotIndex: poseReport.referenceSlotIndex,
        results: merged,
      };
    }
    return poseReport;
  }

  // ---- Phase B: Feature matching for everything ----
  return featureAlignViaWorker(slots, onProgress);
}
