import { useEffect, useRef, useState } from 'react';
import { Bug, ChevronDown, Download, LayoutGrid, Trash2 } from 'lucide-react';
import JSZip from 'jszip';
import {
  CATEGORIES,
  CategoryData,
  Category,
  Comparison,
  SLOT_LETTERS,
} from '@/lib/types';
import {
  canvasToPngBlob,
  renderCollageCanvas,
  renderImageToExportCanvas,
} from '@/lib/exportRenderer';
import { downloadBlob, sleep } from '@/lib/utils';

interface Props {
  comparison: Comparison | undefined;
  category: Category;
  comparisonIndex: number;
  /** Full app data, needed for "all" exports across categories */
  allData?: CategoryData;
  onDeleteAll: () => void;
  onOpenDebugReport: () => void;
}

type BusyState =
  | null
  | 'collage-current'
  | 'collage-all'
  | 'photos-current'
  | 'photos-all';

export function DownloadActions({
  comparison,
  category,
  comparisonIndex,
  allData,
  onDeleteAll,
  onOpenDebugReport,
}: Props) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState<BusyState>(null);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);

  const [photosOpen, setPhotosOpen] = useState(false);
  const [collageOpen, setCollageOpen] = useState(false);

  const photosRef = useRef<HTMLDivElement | null>(null);
  const collageRef = useRef<HTMLDivElement | null>(null);

  const filledImages = comparison?.images.filter((img) => img !== null) ?? [];
  const hasCurrentImages = filledImages.length > 0;
  const hasAnyImages =
    !!allData &&
    CATEGORIES.some((cat) =>
      allData[cat].some((c) => c.images.some((img) => img !== null))
    );

  // Close dropdowns when clicking outside
  useEffect(() => {
    function onClick(e: MouseEvent) {
      const target = e.target as Node;
      if (photosRef.current && !photosRef.current.contains(target)) setPhotosOpen(false);
      if (collageRef.current && !collageRef.current.contains(target)) setCollageOpen(false);
    }
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, []);

  // ---- CURRENT-comparison handlers (single comparison, direct downloads) ----

  async function handleDownloadCollageCurrent() {
    if (!comparison || !hasCurrentImages) return;
    setBusy('collage-current');
    try {
      const canvas = await renderCollageCanvas(comparison.images);
      const blob = await canvasToPngBlob(canvas);
      downloadBlob(blob, `Collage-${category}-comp${comparisonIndex + 1}.png`);
    } catch (err) {
      console.error('Collage export failed', err);
      alert('Collage-Export fehlgeschlagen: ' + (err instanceof Error ? err.message : 'Unbekannter Fehler'));
    } finally {
      setBusy(null);
      setCollageOpen(false);
    }
  }

  async function handleDownloadPhotosCurrent() {
    if (!comparison || !hasCurrentImages) return;
    setBusy('photos-current');
    try {
      for (let i = 0; i < comparison.images.length; i++) {
        const img = comparison.images[i];
        if (!img) continue;
        const canvas = await renderImageToExportCanvas(img);
        const blob = await canvasToPngBlob(canvas);
        const letter = SLOT_LETTERS[i] ?? `${i + 1}`;
        downloadBlob(blob, `${letter}-${category}-comp${comparisonIndex + 1}.png`);
        await sleep(200);
      }
    } catch (err) {
      console.error('Photo export failed', err);
      alert('Foto-Export fehlgeschlagen: ' + (err instanceof Error ? err.message : 'Unbekannter Fehler'));
    } finally {
      setBusy(null);
      setPhotosOpen(false);
    }
  }

  // ---- ALL-categories handlers (ZIP archives) ----

  /**
   * Count all non-empty image slots across all categories and comparisons.
   * Used to size the progress bar for "all" exports.
   */
  function countAllPhotos(): number {
    if (!allData) return 0;
    let total = 0;
    for (const cat of CATEGORIES) {
      for (const comp of allData[cat]) {
        for (const img of comp.images) {
          if (img !== null) total++;
        }
      }
    }
    return total;
  }

  /**
   * Count all comparisons that contain at least one filled slot — i.e. the
   * number of collages we can produce.
   */
  function countAllCollages(): number {
    if (!allData) return 0;
    let total = 0;
    for (const cat of CATEGORIES) {
      for (const comp of allData[cat]) {
        if (comp.images.some((img) => img !== null)) total++;
      }
    }
    return total;
  }

  async function handleDownloadPhotosAll() {
    if (!allData || !hasAnyImages) return;
    setBusy('photos-all');
    setProgress({ current: 0, total: countAllPhotos() });
    try {
      const zip = new JSZip();
      let done = 0;
      for (const cat of CATEGORIES) {
        const comparisons = allData[cat];
        for (let compIdx = 0; compIdx < comparisons.length; compIdx++) {
          const comp = comparisons[compIdx];
          for (let slotIdx = 0; slotIdx < comp.images.length; slotIdx++) {
            const img = comp.images[slotIdx];
            if (!img) continue;
            const canvas = await renderImageToExportCanvas(img);
            const blob = await canvasToPngBlob(canvas);
            const letter = SLOT_LETTERS[slotIdx] ?? `${slotIdx + 1}`;
            const filename = `${cat}_comp${compIdx + 1}_${letter}.png`;
            zip.file(filename, blob);
            done++;
            setProgress({ current: done, total: countAllPhotos() });
          }
        }
      }
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      downloadBlob(zipBlob, 'CompareShot-AllPhotos.zip');
    } catch (err) {
      console.error('All-photos export failed', err);
      alert('Export fehlgeschlagen: ' + (err instanceof Error ? err.message : 'Unbekannter Fehler'));
    } finally {
      setBusy(null);
      setProgress(null);
      setPhotosOpen(false);
    }
  }

  async function handleDownloadCollagesAll() {
    if (!allData || !hasAnyImages) return;
    setBusy('collage-all');
    setProgress({ current: 0, total: countAllCollages() });
    try {
      const zip = new JSZip();
      let done = 0;
      for (const cat of CATEGORIES) {
        const comparisons = allData[cat];
        for (let compIdx = 0; compIdx < comparisons.length; compIdx++) {
          const comp = comparisons[compIdx];
          // Skip comparisons with no images at all
          if (!comp.images.some((img) => img !== null)) continue;
          const canvas = await renderCollageCanvas(comp.images);
          const blob = await canvasToPngBlob(canvas);
          const filename = `${cat}_comp${compIdx + 1}_collage.png`;
          zip.file(filename, blob);
          done++;
          setProgress({ current: done, total: countAllCollages() });
        }
      }
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      downloadBlob(zipBlob, 'CompareShot-AllCollages.zip');
    } catch (err) {
      console.error('All-collages export failed', err);
      alert('Export fehlgeschlagen: ' + (err instanceof Error ? err.message : 'Unbekannter Fehler'));
    } finally {
      setBusy(null);
      setProgress(null);
      setCollageOpen(false);
    }
  }

  function handleDeleteAll() {
    if (!confirming) {
      setConfirming(true);
      setTimeout(() => setConfirming(false), 4000);
      return;
    }
    onDeleteAll();
    setConfirming(false);
  }

  // ---- Render ----

  const photosBusy = busy === 'photos-current' || busy === 'photos-all';
  const collageBusy = busy === 'collage-current' || busy === 'collage-all';

  // Status text for buttons
  const photosLabel = photosBusy
    ? progress
      ? `Exporting ${progress.current}/${progress.total}…`
      : 'Exporting…'
    : 'Download photos';
  const collageLabel = collageBusy
    ? progress
      ? `Rendering ${progress.current}/${progress.total}…`
      : 'Rendering…'
    : 'Download collage';

  return (
    <div className="flex items-center gap-1.5">
      <button
        onClick={onOpenDebugReport}
        className="flex items-center gap-1.5 rounded-md border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:border-amber-500/40 hover:bg-amber-500/10 hover:text-amber-300"
        title="Debug-Report öffnen"
      >
        <Bug size={12} />
        Debug report
      </button>

      <button
        onClick={handleDeleteAll}
        className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ${
          confirming
            ? 'border-red-500 bg-red-500/20 text-red-300'
            : 'border-zinc-800 text-zinc-400 hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-300'
        }`}
        title={confirming ? 'Erneut klicken zum Bestätigen' : 'Alle Fotos löschen'}
      >
        <Trash2 size={12} />
        {confirming ? 'Confirm' : 'Delete all photos'}
      </button>

      {/* Collage dropdown */}
      <div ref={collageRef} className="relative">
        <button
          onClick={() => setCollageOpen((o) => !o)}
          disabled={busy !== null}
          className="flex items-center gap-1.5 rounded-md border border-green-500/40 bg-green-600/15 px-3 py-1.5 text-xs text-green-300 transition-colors hover:bg-green-600/25 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <LayoutGrid size={12} />
          {collageLabel}
          <ChevronDown size={12} />
        </button>

        {collageOpen && !collageBusy && (
          <div className="absolute right-0 top-full z-50 mt-1 w-56 overflow-hidden rounded-md border border-zinc-700 bg-zinc-900 shadow-xl">
            <button
              onClick={handleDownloadCollageCurrent}
              disabled={!hasCurrentImages}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-xs text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-500"
            >
              <span>Current comparison</span>
              <span className="text-zinc-500">PNG</span>
            </button>
            <div className="h-px bg-zinc-800" />
            <button
              onClick={handleDownloadCollagesAll}
              disabled={!hasAnyImages}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-xs text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-500"
            >
              <span>All categories</span>
              <span className="text-zinc-500">ZIP</span>
            </button>
          </div>
        )}
      </div>

      {/* Photos dropdown */}
      <div ref={photosRef} className="relative">
        <button
          onClick={() => setPhotosOpen((o) => !o)}
          disabled={busy !== null}
          className="flex items-center gap-1.5 rounded-md border border-blue-500/40 bg-blue-600/15 px-3 py-1.5 text-xs text-blue-300 transition-colors hover:bg-blue-600/25 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Download size={12} />
          {photosLabel}
          <ChevronDown size={12} />
        </button>

        {photosOpen && !photosBusy && (
          <div className="absolute right-0 top-full z-50 mt-1 w-56 overflow-hidden rounded-md border border-zinc-700 bg-zinc-900 shadow-xl">
            <button
              onClick={handleDownloadPhotosCurrent}
              disabled={!hasCurrentImages}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-xs text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-500"
            >
              <span>Current comparison</span>
              <span className="text-zinc-500">PNG</span>
            </button>
            <div className="h-px bg-zinc-800" />
            <button
              onClick={handleDownloadPhotosAll}
              disabled={!hasAnyImages}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-xs text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:text-zinc-500"
            >
              <span>All categories</span>
              <span className="text-zinc-500">ZIP</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
