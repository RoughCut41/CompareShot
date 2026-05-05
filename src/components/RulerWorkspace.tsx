import { useEffect, useRef, useState, useCallback, ReactNode } from 'react';
import { uid } from '@/lib/utils';

const RULER_SIZE = 20;
const TICK_MINOR = 50;
const TICK_MAJOR = 100;
const COLOR_BG = '#1c1c1e';
const COLOR_TICK = '#3f3f46';
const COLOR_LABEL = '#71717a';

interface Guideline {
  id: string;
  axis: 'horizontal' | 'vertical';
  /**
   * Normalized position (0..1) relative to the workspace's full content size.
   * For horizontal lines: fraction of workspace.scrollHeight
   * For vertical lines:   fraction of workspace.scrollWidth
   * Storing as a fraction means guidelines stay anchored to the same image
   * region when the workspace resizes (e.g. window resize).
   */
  posFrac: number;
}

function drawHorizontalRuler(canvas: HTMLCanvasElement, scrollLeft: number) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  if (canvas.width !== cssWidth * dpr || canvas.height !== cssHeight * dpr) {
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  ctx.strokeStyle = COLOR_TICK;
  ctx.fillStyle = COLOR_LABEL;
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.textBaseline = 'top';

  const start = Math.floor(scrollLeft / TICK_MINOR) * TICK_MINOR;
  const end = scrollLeft + cssWidth;
  for (let x = start; x <= end; x += TICK_MINOR) {
    const screenX = x - scrollLeft;
    const isMajor = x % TICK_MAJOR === 0;
    ctx.beginPath();
    ctx.moveTo(screenX + 0.5, cssHeight);
    ctx.lineTo(screenX + 0.5, cssHeight - (isMajor ? 10 : 5));
    ctx.stroke();
    if (isMajor && x >= TICK_MAJOR) {
      ctx.fillText(String(x), screenX + 2, 2);
    }
  }
}

function drawVerticalRuler(canvas: HTMLCanvasElement, scrollTop: number) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  if (canvas.width !== cssWidth * dpr || canvas.height !== cssHeight * dpr) {
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  ctx.strokeStyle = COLOR_TICK;
  ctx.fillStyle = COLOR_LABEL;
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.textBaseline = 'top';

  const start = Math.floor(scrollTop / TICK_MINOR) * TICK_MINOR;
  const end = scrollTop + cssHeight;
  for (let y = start; y <= end; y += TICK_MINOR) {
    const screenY = y - scrollTop;
    const isMajor = y % TICK_MAJOR === 0;
    ctx.beginPath();
    ctx.moveTo(cssWidth, screenY + 0.5);
    ctx.lineTo(cssWidth - (isMajor ? 10 : 5), screenY + 0.5);
    ctx.stroke();
    if (isMajor && y >= TICK_MAJOR) {
      ctx.save();
      ctx.translate(2, screenY + 12);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(String(y), 0, 0);
      ctx.restore();
    }
  }
}

interface Props {
  children: ReactNode;
}

export function RulerWorkspace({ children }: Props) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const hRulerRef = useRef<HTMLCanvasElement | null>(null);
  const vRulerRef = useRef<HTMLCanvasElement | null>(null);

  const [guidelines, setGuidelines] = useState<Guideline[]>([]);
  // Track workspace size so guidelines can be rendered in pixels
  const [wsSize, setWsSize] = useState({ w: 0, h: 0 });

  const createDragRef = useRef<{ axis: 'horizontal' | 'vertical'; tempId: string } | null>(null);
  const moveDragRef = useRef<{ id: string } | null>(null);

  const redraw = useCallback(() => {
    const ws = workspaceRef.current;
    if (!ws) return;
    if (hRulerRef.current) drawHorizontalRuler(hRulerRef.current, ws.scrollLeft);
    if (vRulerRef.current) drawVerticalRuler(vRulerRef.current, ws.scrollTop);
  }, []);

  // Track workspace size with ResizeObserver so guidelines update on window/layout resize.
  useEffect(() => {
    const ws = workspaceRef.current;
    if (!ws) return;
    const update = () => {
      // Use scrollWidth/scrollHeight so guidelines anchor to the FULL content size,
      // not the visible viewport. This matches their drag-creation coordinate space.
      setWsSize({ w: ws.scrollWidth, h: ws.scrollHeight });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(ws);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    redraw();
    const onResize = () => redraw();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [redraw]);

  const onWorkspaceScroll = () => redraw();

  // Convert pointer event to a fraction (0..1) of workspace content size
  const eventToFrac = (e: { clientX: number; clientY: number }, axis: 'horizontal' | 'vertical') => {
    const ws = workspaceRef.current;
    if (!ws) return 0;
    const rect = ws.getBoundingClientRect();
    if (axis === 'horizontal') {
      const px = e.clientY - rect.top + ws.scrollTop;
      return ws.scrollHeight > 0 ? px / ws.scrollHeight : 0;
    } else {
      const px = e.clientX - rect.left + ws.scrollLeft;
      return ws.scrollWidth > 0 ? px / ws.scrollWidth : 0;
    }
  };

  // -------- Create guideline (drag from ruler) --------
  const onRulerPointerDown = (axis: 'horizontal' | 'vertical') => (e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    const tempId = uid();
    createDragRef.current = { axis, tempId };
    const posFrac = eventToFrac(e, axis);
    setGuidelines((g) => [...g, { id: tempId, axis, posFrac }]);
  };

  const onAnyPointerMove = (e: React.PointerEvent) => {
    if (createDragRef.current) {
      const { axis, tempId } = createDragRef.current;
      const posFrac = eventToFrac(e, axis);
      setGuidelines((gs) => gs.map((g) => (g.id === tempId ? { ...g, posFrac } : g)));
      return;
    }

    if (moveDragRef.current) {
      const { id } = moveDragRef.current;
      setGuidelines((gs) =>
        gs.map((g) => {
          if (g.id !== id) return g;
          return { ...g, posFrac: eventToFrac(e, g.axis) };
        })
      );
    }
  };

  const onAnyPointerUp = (e: React.PointerEvent) => {
    const ws = workspaceRef.current;
    if (!ws) return;
    const rect = ws.getBoundingClientRect();

    if (createDragRef.current) {
      const { axis, tempId } = createDragRef.current;
      const inside =
        e.clientX >= rect.left &&
        e.clientX <= rect.right &&
        e.clientY >= rect.top &&
        e.clientY <= rect.bottom;
      const onOwnRuler =
        (axis === 'horizontal' && e.clientY < rect.top + 2) ||
        (axis === 'vertical' && e.clientX < rect.left + 2);
      if (!inside || onOwnRuler) {
        setGuidelines((gs) => gs.filter((g) => g.id !== tempId));
      }
      createDragRef.current = null;
    }

    if (moveDragRef.current) {
      const { id } = moveDragRef.current;
      // Delete if dragged back to the very edge (effectively into the ruler)
      setGuidelines((gs) =>
        gs.filter((g) => {
          if (g.id !== id) return true;
          return g.posFrac > 0.001;
        })
      );
      moveDragRef.current = null;
    }
  };

  const onGuidelinePointerDown = (id: string) => (e: React.PointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    moveDragRef.current = { id };
  };

  return (
    <div
      ref={wrapperRef}
      className="relative flex flex-1 flex-col overflow-hidden bg-background"
      onPointerMove={onAnyPointerMove}
      onPointerUp={onAnyPointerUp}
      onPointerCancel={onAnyPointerUp}
    >
      <div className="flex flex-shrink-0">
        <div
          className="border-b border-r border-zinc-800 bg-surface-alt"
          style={{ width: RULER_SIZE, height: RULER_SIZE }}
        />
        <canvas
          ref={hRulerRef}
          onPointerDown={onRulerPointerDown('horizontal')}
          className="block w-full cursor-ns-resize border-b border-zinc-800"
          style={{ height: RULER_SIZE }}
        />
      </div>

      <div className="flex flex-1 overflow-hidden">
        <canvas
          ref={vRulerRef}
          onPointerDown={onRulerPointerDown('vertical')}
          className="block h-full cursor-ew-resize border-r border-zinc-800"
          style={{ width: RULER_SIZE }}
        />
        <div
          ref={workspaceRef}
          onScroll={onWorkspaceScroll}
          className="relative flex-1 overflow-auto scrollbar-thin"
        >
          {children}

          {/* Guidelines — positions converted from fraction to pixels at render time */}
          {guidelines.map((g) =>
            g.axis === 'horizontal' ? (
              <div
                key={g.id}
                onPointerDown={onGuidelinePointerDown(g.id)}
                className="guideline-h absolute left-0 right-0 z-30 cursor-ns-resize"
                style={{ top: g.posFrac * wsSize.h, height: 1 }}
              />
            ) : (
              <div
                key={g.id}
                onPointerDown={onGuidelinePointerDown(g.id)}
                className="guideline-v absolute top-0 bottom-0 z-30 cursor-ew-resize"
                style={{ left: g.posFrac * wsSize.w, width: 1 }}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}
