import { useState } from 'react';
import { Bug, Download, LayoutGrid, Trash2 } from 'lucide-react';
import { Category, Comparison, SLOT_LETTERS } from '@/lib/types';
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
  onDeleteAll: () => void;
  onOpenDebugReport: () => void;
}

export function DownloadActions({
  comparison,
  category,
  comparisonIndex,
  onDeleteAll,
  onOpenDebugReport,
}: Props) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState<'collage' | 'photos' | null>(null);

  const filledImages = comparison?.images.filter((img) => img !== null) ?? [];
  const hasImages = filledImages.length > 0;

  async function handleDownloadCollage() {
    if (!comparison || !hasImages) return;
    setBusy('collage');
    try {
      const canvas = await renderCollageCanvas(comparison.images);
      const blob = await canvasToPngBlob(canvas);
      downloadBlob(blob, `Collage-${category}-${comparisonIndex + 1}.png`);
    } catch (err) {
      console.error('Collage export failed', err);
      alert('Collage-Export fehlgeschlagen: ' + (err instanceof Error ? err.message : 'Unbekannter Fehler'));
    } finally {
      setBusy(null);
    }
  }

  async function handleDownloadEach() {
    if (!comparison || !hasImages) return;
    setBusy('photos');
    try {
      for (let i = 0; i < comparison.images.length; i++) {
        const img = comparison.images[i];
        if (!img) continue;
        const canvas = await renderImageToExportCanvas(img);
        const blob = await canvasToPngBlob(canvas);
        const letter = SLOT_LETTERS[i] ?? `${i + 1}`;
        downloadBlob(blob, `${letter}-${category}-${comparisonIndex + 1}.png`);
        await sleep(200);
      }
    } catch (err) {
      console.error('Photo export failed', err);
      alert('Foto-Export fehlgeschlagen: ' + (err instanceof Error ? err.message : 'Unbekannter Fehler'));
    } finally {
      setBusy(null);
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

      <button
        onClick={handleDownloadCollage}
        disabled={!hasImages || busy !== null}
        className="flex items-center gap-1.5 rounded-md border border-green-500/40 bg-green-600/15 px-3 py-1.5 text-xs text-green-300 transition-colors hover:bg-green-600/25 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <LayoutGrid size={12} />
        {busy === 'collage' ? 'Rendering…' : 'Download collage'}
      </button>

      <button
        onClick={handleDownloadEach}
        disabled={!hasImages || busy !== null}
        className="flex items-center gap-1.5 rounded-md border border-blue-500/40 bg-blue-600/15 px-3 py-1.5 text-xs text-blue-300 transition-colors hover:bg-blue-600/25 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Download size={12} />
        {busy === 'photos' ? 'Exporting…' : 'Download photos'}
      </button>
    </div>
  );
}
