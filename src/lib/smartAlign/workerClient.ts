/**
 * Main-thread client for the Smart Align worker.
 * Handles spawning the worker, downscaling images on the main thread (fast),
 * and translating worker messages back into the SmartAlignReport shape.
 */
import { decodeImage } from '@/lib/exportRenderer';
import {
  AlignableSlot,
  AlignResult,
  ProgressCallback,
  SmartAlignReport,
} from './types';

const ANALYSIS_MAX_DIM = 800;

function downscaleToImageData(img: HTMLImageElement, maxDim: number): { data: ImageData; scale: number } {
  const ratio = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * ratio));
  const h = Math.max(1, Math.round(img.naturalHeight * ratio));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h);
  return { data, scale: ratio };
}

export async function featureAlignViaWorker(
  slots: AlignableSlot[],
  onProgress?: ProgressCallback
): Promise<SmartAlignReport> {
  if (slots.length < 2) {
    return {
      mode: 'feature',
      referenceSlotIndex: slots[0]?.slotIndex ?? 0,
      results: slots.map((s) => ({ slotIndex: s.slotIndex, status: 'skipped' })),
    };
  }

  onProgress?.('Preparing images…');

  // Decode + downscale every image on the main thread (fast — milliseconds per image)
  const payloads = await Promise.all(
    slots.map(async (s) => {
      const img = await decodeImage(s.state.url);
      const { data, scale } = downscaleToImageData(img, ANALYSIS_MAX_DIM);
      return {
        slotIndex: s.slotIndex,
        imageData: data,
        preScale: scale,
        naturalWidth: s.state.naturalWidth,
        naturalHeight: s.state.naturalHeight,
        containerW: s.state._containerW,
        containerH: s.state._containerH,
      };
    })
  );

  onProgress?.('Starting worker…');

  return new Promise<SmartAlignReport>((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL('./worker.ts', import.meta.url));
    } catch (err) {
      reject(err);
      return;
    }

    const timeout = window.setTimeout(() => {
      worker.terminate();
      reject(new Error('Smart Align timed out after 90s'));
    }, 90000);

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        onProgress?.(msg.label);
      } else if (msg.type === 'done') {
        window.clearTimeout(timeout);
        worker.terminate();
        const results: AlignResult[] = msg.results.map(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (r: any) => ({
            slotIndex: r.slotIndex,
            status: r.status,
            reason: r.reason,
            transform: r.transform,
          })
        );
        resolve({
          mode: 'feature',
          referenceSlotIndex: msg.referenceSlotIndex,
          results,
        });
      } else if (msg.type === 'error') {
        window.clearTimeout(timeout);
        worker.terminate();
        reject(new Error(msg.message));
      }
    };

    worker.onerror = (e) => {
      window.clearTimeout(timeout);
      worker.terminate();
      reject(new Error(`Worker error: ${e.message}`));
    };

    // Use transferable ImageData buffers for zero-copy hand-off
    const transferables = payloads.map((p) => p.imageData.data.buffer);
    worker.postMessage({ type: 'run', slots: payloads }, transferables);
  });
}
