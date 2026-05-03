import { useCallback, useEffect, useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { ImageState, SLOT_LETTERS } from '@/lib/types';
import { ImageUpdater } from '@/hooks/useCompareStore';
import { loadImageFile } from '@/lib/imageLoader';
import { clamp } from '@/lib/utils';
import { ImageOverlayControls } from './ImageOverlayControls';

interface Props {
  state: ImageState | null;
  slotIndex: number;
  onUpdate: (updater: ImageUpdater) => void;
  onSetImage: (state: ImageState) => void;
  onDelete: () => void;
}

const ZOOM_WHEEL_STEP = 0.01;
const ZOOM_MIN = 0.1;
const ZOOM_MAX = 10;

export function ImageContainer({ state, slotIndex, onUpdate, onSetImage, onDelete }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Track container size and write back into image state for export pipeline
  useEffect(() => {
    if (!containerRef.current || !state) return;
    const el = containerRef.current;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const w = Math.round(entry.contentRect.width);
      const h = Math.round(entry.contentRect.height);
      if (w === state._containerW && h === state._containerH) return;
      onUpdate({ _containerW: w, _containerH: h });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [state, onUpdate]);

  // -------- File loading --------
  const handleFiles = useCallback(
    async (files: FileList | File[] | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      console.log('[CompareShot] Loading file:', file.name, file.type, file.size);
      if (!file.type.startsWith('image/') && !/\.(heic|heif)$/i.test(file.name)) {
        setLoadError('Nur Bilddateien werden unterstützt.');
        return;
      }
      setLoading(true);
      setLoadError(null);
      try {
        const newState = await loadImageFile(file);
        console.log('[CompareShot] Image loaded:', newState.naturalWidth, '×', newState.naturalHeight);
        onSetImage(newState);
      } catch (err) {
        console.error('[CompareShot] Image load failed:', err);
        setLoadError(err instanceof Error ? err.message : 'Laden fehlgeschlagen');
      } finally {
        setLoading(false);
      }
    },
    [onSetImage]
  );

  // -------- Drag & drop handlers (apply to BOTH empty and filled containers) --------
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    if (e.currentTarget === e.target) setDragOver(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    void handleFiles(e.dataTransfer.files);
  };

  // -------- Pan / free-rotate via pointer drag --------
  // We allow pan on filled containers; free-rotate replaces pan when toggled on.
  const dragStateRef = useRef<{
    mode: 'pan' | 'rotate';
    startX: number;
    startY: number;
    initialPanX: number;
    initialPanY: number;
    initialAngle: number; // for rotate mode
    centerX: number; // canvas center in viewport coords (for rotate mode)
    centerY: number;
  } | null>(null);

  const onPointerDown = (e: React.PointerEvent) => {
    if (!state) return;
    // Don't start a drag if the user clicked on a control button
    const target = e.target as HTMLElement;
    if (target.closest('[data-controls="true"]')) return;
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    if (!containerRef.current) return;

    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);

    const rect = containerRef.current.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    if (state.freeRotateActive) {
      const initialAngle =
        (Math.atan2(e.clientY - cy, e.clientX - cx) * 180) / Math.PI - state.rotation;
      dragStateRef.current = {
        mode: 'rotate',
        startX: e.clientX,
        startY: e.clientY,
        initialPanX: state.panX,
        initialPanY: state.panY,
        initialAngle,
        centerX: cx,
        centerY: cy,
      };
    } else {
      dragStateRef.current = {
        mode: 'pan',
        startX: e.clientX,
        startY: e.clientY,
        initialPanX: state.panX,
        initialPanY: state.panY,
        initialAngle: 0,
        centerX: cx,
        centerY: cy,
      };
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const ds = dragStateRef.current;
    if (!ds || !state) return;
    if (ds.mode === 'pan') {
      onUpdate({
        panX: ds.initialPanX + (e.clientX - ds.startX),
        panY: ds.initialPanY + (e.clientY - ds.startY),
      });
    } else {
      const angleNow = (Math.atan2(e.clientY - ds.centerY, e.clientX - ds.centerX) * 180) / Math.PI;
      onUpdate({ rotation: angleNow - ds.initialAngle });
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    dragStateRef.current = null;
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
  };

  // -------- Wheel zoom --------
  const onWheel = (e: React.WheelEvent) => {
    if (!state) return;
    // Use a wheel listener attached non-passively (see effect below) so we can preventDefault
    // React synthetic wheel is passive on some setups; we use the non-passive native listener as well.
    e.stopPropagation();
    const delta = e.deltaY < 0 ? ZOOM_WHEEL_STEP : -ZOOM_WHEEL_STEP;
    onUpdate((prev) => ({ zoom: clamp(prev.zoom * (1 + delta * 5), ZOOM_MIN, ZOOM_MAX) }));
  };

  // Native non-passive wheel handler — needed because React's passive default would
  // let the page scroll while the user zooms an image.
  useEffect(() => {
    const el = containerRef.current;
    if (!el || !state) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, [state]);

  const letter = SLOT_LETTERS[slotIndex] ?? `${slotIndex + 1}`;

  // ---------- RENDER ----------
  return (
    <div
      ref={containerRef}
      className="group relative w-full overflow-hidden rounded-lg border border-zinc-800 bg-black no-select"
      style={{ aspectRatio: '960 / 1625' }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onWheel={onWheel}
    >
      {/* Position label */}
      <div className="pointer-events-none absolute left-2 top-2 z-20 rounded bg-black/70 px-2 py-0.5 text-xs font-medium text-zinc-300">
        {letter}
      </div>

      {/* Rule-of-thirds overlay (only when filled) */}
      {state && (
        <div className="pointer-events-none absolute inset-0 z-10">
          <div
            className="absolute top-0 bottom-0"
            style={{ left: '33.3333%', width: 1, background: 'rgba(255,255,255,0.25)' }}
          />
          <div
            className="absolute top-0 bottom-0"
            style={{ left: '66.6666%', width: 1, background: 'rgba(255,255,255,0.25)' }}
          />
        </div>
      )}

      {/* Content: empty state or image */}
      {!state ? (
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex h-full w-full flex-col items-center justify-center gap-2 text-zinc-500 transition-colors hover:bg-zinc-900/40 hover:text-zinc-300"
        >
          <Upload
            size={28}
            className={`transition-colors ${dragOver ? 'text-blue-400' : ''}`}
          />
          <span className="text-xs">
            {loading
              ? 'Lade Bild…'
              : dragOver
                ? 'Drop image'
                : 'Click or drag image here'}
          </span>
          {loadError && <span className="px-3 text-center text-xs text-red-400">{loadError}</span>}
        </button>
      ) : (
        <>
          <img
            src={state.url}
            alt=""
            draggable={false}
            className="absolute inset-0 h-full w-full object-cover"
            style={{
              transform: `translate(${state.panX}px, ${state.panY}px) rotate(${state.rotation}deg) scale(${state.flipH ? -state.zoom : state.zoom}, ${state.flipV ? -state.zoom : state.zoom})`,
              transformOrigin: 'center',
              willChange: 'transform',
            }}
          />
          <ImageOverlayControls state={state} onUpdate={onUpdate} onDelete={onDelete} />
        </>
      )}

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.heic,.heif"
        className="hidden"
        onChange={(e) => {
          const files = e.target.files;
          // Reset the input first so re-uploading the same file works
          e.target.value = '';
          void handleFiles(files);
        }}
      />
    </div>
  );
}
