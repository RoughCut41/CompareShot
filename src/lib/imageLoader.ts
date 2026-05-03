import { ImageState, DEFAULT_IMAGE_STATE } from './types';

/**
 * Best-effort detection: HEIC files often arrive with mime "" or "image/heic".
 * We sniff the extension as a fallback.
 */
function isHeicFile(file: File): boolean {
  const mime = file.type.toLowerCase();
  if (mime === 'image/heic' || mime === 'image/heif') return true;
  const name = file.name.toLowerCase();
  return name.endsWith('.heic') || name.endsWith('.heif');
}

/**
 * Convert HEIC/HEIF to a JPEG Blob using heic2any (lazy-loaded).
 * Falls back to throwing if conversion fails.
 */
async function convertHeicToJpeg(file: File): Promise<Blob> {
  // Dynamic import — heic2any is heavy and only needed for HEIC files
  const heic2any = (await import('heic2any')).default;
  const result = await heic2any({
    blob: file,
    toType: 'image/jpeg',
    quality: 0.95,
  });
  // heic2any returns Blob | Blob[] depending on whether HEIC has multiple images
  return Array.isArray(result) ? result[0] : result;
}

/**
 * Read an image file, convert HEIC if necessary, and return an ImageState
 * ready to insert into a slot.
 */
export async function loadImageFile(file: File): Promise<ImageState> {
  let blob: Blob = file;

  if (isHeicFile(file)) {
    try {
      blob = await convertHeicToJpeg(file);
    } catch (err) {
      throw new Error(
        `HEIC-Konvertierung fehlgeschlagen: ${err instanceof Error ? err.message : 'unbekannter Fehler'}`
      );
    }
  }

  const url = URL.createObjectURL(blob);

  const { naturalWidth, naturalHeight } = await new Promise<{
    naturalWidth: number;
    naturalHeight: number;
  }>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight });
    };
    img.onerror = () => reject(new Error('Bild kann nicht dekodiert werden'));
    img.src = url;
  });

  return {
    ...DEFAULT_IMAGE_STATE,
    file,
    url,
    naturalWidth,
    naturalHeight,
  };
}

/**
 * Revoke an image's object URL to free memory.
 */
export function disposeImage(state: ImageState | null): void {
  if (state?.url) {
    try {
      URL.revokeObjectURL(state.url);
    } catch {
      /* noop */
    }
  }
}
