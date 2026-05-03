import { EXPORT_HEIGHT, EXPORT_WIDTH, ImageState } from './types';

/**
 * Decode an image from a URL and resolve when it's ready to draw.
 */
export function decodeImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Bild konnte nicht dekodiert werden'));
    img.src = url;
  });
}

/**
 * Compute the "object-cover" base size for a source image inside an EXPORT-sized
 * canvas. This matches what the browser renders for `object-fit: cover` in the UI.
 */
function computeCoverSize(
  naturalW: number,
  naturalH: number,
  destW: number,
  destH: number
): { drawW: number; drawH: number } {
  const sourceAspect = naturalW / naturalH;
  const destAspect = destW / destH;
  if (sourceAspect > destAspect) {
    // source is wider — match height, overflow horizontally
    return { drawW: destH * sourceAspect, drawH: destH };
  }
  // source is taller — match width, overflow vertically
  return { drawW: destW, drawH: destW / sourceAspect };
}

/**
 * Render an ImageState to a fresh canvas at EXPORT_WIDTH x EXPORT_HEIGHT.
 * Uses an already-decoded HTMLImageElement so we can draw synchronously.
 *
 * Coordinate system:
 *   - Canvas origin (0,0) = top-left of the export canvas
 *   - Translate to canvas center, apply pan (scaled), apply rotation, apply flip,
 *     apply zoom, then draw the cover-sized image centered.
 *   - This mirrors the CSS transform: translate(panX, panY) rotate(deg) scale(±zoom).
 */
export async function renderImageToExportCanvas(
  state: ImageState,
  options: { width?: number; height?: number } = {}
): Promise<HTMLCanvasElement> {
  const W = options.width ?? EXPORT_WIDTH;
  const H = options.height ?? EXPORT_HEIGHT;

  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context not available');

  // Black background
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);

  // Decode the image
  const img = await decodeImage(state.url);

  // Compute base cover size (what `object-fit: cover` would produce inside W x H)
  const { drawW, drawH } = computeCoverSize(img.naturalWidth, img.naturalHeight, W, H);

  // Pan was stored in container pixels — scale to export pixels.
  // If _containerW/_containerH are 0 (image never measured), fall back to assuming
  // the container had the export aspect ratio, so the pan factor is 1.
  const containerW = state._containerW > 0 ? state._containerW : W;
  const containerH = state._containerH > 0 ? state._containerH : H;
  const panScaleX = W / containerW;
  const panScaleY = H / containerH;
  const panX = state.panX * panScaleX;
  const panY = state.panY * panScaleY;

  ctx.save();
  // Move origin to canvas center, then apply pan (in export pixels)
  ctx.translate(W / 2 + panX, H / 2 + panY);
  // Apply rotation (degrees -> radians)
  ctx.rotate((state.rotation * Math.PI) / 180);
  // Apply flips and zoom together via scale
  const sx = (state.flipH ? -1 : 1) * state.zoom;
  const sy = (state.flipV ? -1 : 1) * state.zoom;
  ctx.scale(sx, sy);
  // Draw the cover-sized image centered on the (now-transformed) origin
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, -drawW / 2, -drawH / 2, drawW, drawH);
  ctx.restore();

  return canvas;
}

/**
 * Render multiple image states side by side into a single collage canvas.
 * Each tile is EXPORT_WIDTH x EXPORT_HEIGHT, separated by `gap` pixels of black.
 */
export async function renderCollageCanvas(
  states: (ImageState | null)[],
  gap = 0
): Promise<HTMLCanvasElement> {
  const filled = states.filter((s): s is ImageState => s !== null);
  if (filled.length === 0) {
    throw new Error('Keine Bilder zum Exportieren vorhanden.');
  }

  const tileW = EXPORT_WIDTH;
  const tileH = EXPORT_HEIGHT;
  const totalW = tileW * filled.length + gap * (filled.length - 1);
  const totalH = tileH;

  const canvas = document.createElement('canvas');
  canvas.width = totalW;
  canvas.height = totalH;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context not available');

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, totalW, totalH);

  // Render each tile and paste
  for (let i = 0; i < filled.length; i++) {
    const tile = await renderImageToExportCanvas(filled[i]);
    ctx.drawImage(tile, i * (tileW + gap), 0);
  }

  return canvas;
}

/**
 * Convert a canvas to a PNG Blob (lossless).
 */
export function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) reject(new Error('Canvas-Export fehlgeschlagen'));
        else resolve(blob);
      },
      'image/png'
    );
  });
}
