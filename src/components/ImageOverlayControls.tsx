import {
  FlipHorizontal2,
  FlipVertical2,
  Home,
  Minus,
  Plus,
  RotateCcw,
  RotateCw,
  Compass,
  Trash2,
} from 'lucide-react';
import { ImageState } from '@/lib/types';
import { ImageUpdater } from '@/hooks/useCompareStore';
import { clamp, normalizeAngle } from '@/lib/utils';

interface Props {
  state: ImageState;
  onUpdate: (updater: ImageUpdater) => void;
  onDelete: () => void;
}

const ZOOM_STEP = 0.02;
const ZOOM_MIN = 0.1;
const ZOOM_MAX = 10;

export function ImageOverlayControls({ state, onUpdate, onDelete }: Props) {
  function setZoom(next: number) {
    onUpdate({ zoom: clamp(next, ZOOM_MIN, ZOOM_MAX) });
  }
  function setRotation(deltaDeg: number) {
    onUpdate((prev) => ({ rotation: normalizeAngle(prev.rotation + deltaDeg) }));
  }
  function reset() {
    onUpdate({
      zoom: 1,
      panX: 0,
      panY: 0,
      rotation: 0,
      flipH: false,
      flipV: false,
      freeRotateActive: false,
    });
  }

  // Stop propagation so clicks on controls never trigger image panning behind them.
  const stop = (e: React.MouseEvent | React.PointerEvent) => e.stopPropagation();

  return (
    <>
      {/* Free-rotate angle badge */}
      {state.freeRotateActive && (
        <div className="pointer-events-none absolute top-10 left-1/2 -translate-x-1/2 rounded bg-black/70 px-2 py-0.5 font-mono text-[11px] text-cyan-300 shadow-lg">
          {normalizeAngle(state.rotation).toFixed(1)}°
        </div>
      )}

      {/* Bottom gradient + toolbar */}
      <div
        data-controls="true"
        onPointerDown={stop}
        className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-2 bg-gradient-to-t from-black/85 via-black/50 to-transparent px-2 pb-2 pt-6 opacity-0 transition-opacity group-hover:opacity-100"
      >
        {/* Zoom controls */}
        <div className="flex items-center gap-1 rounded-md bg-black/50 p-0.5 backdrop-blur">
          <button
            data-controls="true"
            onClick={() => setZoom(state.zoom - ZOOM_STEP)}
            className="rounded p-1 text-zinc-300 hover:bg-zinc-700 hover:text-white"
            aria-label="Zoom out"
          >
            <Minus size={14} />
          </button>
          <span className="min-w-[3rem] text-center font-mono text-[11px] text-zinc-200 tabular-nums">
            {Math.round(state.zoom * 100)}%
          </span>
          <button
            data-controls="true"
            onClick={() => setZoom(state.zoom + ZOOM_STEP)}
            className="rounded p-1 text-zinc-300 hover:bg-zinc-700 hover:text-white"
            aria-label="Zoom in"
          >
            <Plus size={14} />
          </button>
        </div>

        {/* Right cluster: rotate / flip / free-rotate / reset / delete */}
        <div className="flex items-center gap-1 rounded-md bg-black/50 p-0.5 backdrop-blur">
          <button
            data-controls="true"
            onClick={() => setRotation(-90)}
            className="rounded p-1 text-zinc-300 hover:bg-zinc-700 hover:text-white"
            aria-label="Rotate left 90 degrees"
          >
            <RotateCcw size={14} />
          </button>
          <button
            data-controls="true"
            onClick={() => setRotation(90)}
            className="rounded p-1 text-zinc-300 hover:bg-zinc-700 hover:text-white"
            aria-label="Rotate right 90 degrees"
          >
            <RotateCw size={14} />
          </button>
          <div className="mx-0.5 h-4 w-px bg-zinc-700" />
          <button
            data-controls="true"
            onClick={() => onUpdate({ flipH: !state.flipH })}
            className={`rounded p-1 hover:bg-zinc-700 hover:text-white ${
              state.flipH ? 'text-cyan-300' : 'text-zinc-300'
            }`}
            aria-label="Flip horizontal"
          >
            <FlipHorizontal2 size={14} />
          </button>
          <button
            data-controls="true"
            onClick={() => onUpdate({ flipV: !state.flipV })}
            className={`rounded p-1 hover:bg-zinc-700 hover:text-white ${
              state.flipV ? 'text-cyan-300' : 'text-zinc-300'
            }`}
            aria-label="Flip vertical"
          >
            <FlipVertical2 size={14} />
          </button>
          <div className="mx-0.5 h-4 w-px bg-zinc-700" />
          <button
            data-controls="true"
            onClick={() => onUpdate({ freeRotateActive: !state.freeRotateActive })}
            className={`rounded p-1 hover:bg-zinc-700 hover:text-white ${
              state.freeRotateActive ? 'text-cyan-300' : 'text-zinc-300'
            }`}
            aria-label="Free rotation"
            title="Free rotate (drag to rotate)"
          >
            <Compass size={14} />
          </button>
          <button
            data-controls="true"
            onClick={reset}
            className="rounded p-1 text-zinc-300 hover:bg-zinc-700 hover:text-white"
            aria-label="Reset view"
          >
            <Home size={14} />
          </button>
          <div className="mx-0.5 h-4 w-px bg-zinc-700" />
          <button
            data-controls="true"
            onClick={onDelete}
            className="rounded p-1 text-zinc-300 hover:bg-red-500/30 hover:text-red-300"
            aria-label="Delete image"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>
    </>
  );
}
