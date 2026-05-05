import { useEffect, useRef, useState } from 'react';
import { FolderUp, Loader2, X } from 'lucide-react';
import {
  buildCategoryDataFromScan,
  FolderScanResult,
  prettifyPhoneName,
  scanFolder,
} from '@/lib/folderImport';
import { CategoryData, MAX_SLOTS } from '@/lib/types';

interface Props {
  open: boolean;
  onClose: () => void;
  onImportComplete: (data: CategoryData) => void;
}

type Phase = 'idle' | 'preview' | 'importing' | 'done';

export function FolderImportModal({ open, onClose, onImportComplete }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [scan, setScan] = useState<FolderScanResult | null>(null);
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);

  // Reset when modal closes
  useEffect(() => {
    if (!open) {
      setPhase('idle');
      setScan(null);
      setProgress({ loaded: 0, total: 0 });
      setError(null);
    }
  }, [open]);

  if (!open) return null;

  const handlePickFolder = () => {
    inputRef.current?.click();
  };

  const handleFolderChosen = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setError(null);

    try {
      const result = scanFolder(files);
      if (result.phones.length === 0) {
        setError(
          'Keine "_Phone …" Ordner gefunden. Stell sicher, dass die Struktur stimmt: Root/_Phone X - Modell/Modus/Bilder.'
        );
        return;
      }
      if (result.totalImages === 0) {
        setError('Keine importierbaren Bilder gefunden (.jpg, .jpeg, .png, .heic, .heif).');
        return;
      }
      setScan(result);
      setPhase('preview');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan fehlgeschlagen');
    }

    // Reset input so the same folder can be picked again
    e.target.value = '';
  };

  const handleConfirmImport = async () => {
    if (!scan) return;
    setPhase('importing');
    setProgress({ loaded: 0, total: scan.totalImages });
    try {
      const data = await buildCategoryDataFromScan(scan, (loaded, total) => {
        setProgress({ loaded, total });
      });
      onImportComplete(data);
      setPhase('done');
      window.setTimeout(() => onClose(), 800);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import fehlgeschlagen');
      setPhase('preview');
    }
  };

  const phonesShown = scan?.phones.slice(0, MAX_SLOTS) ?? [];
  const phonesIgnored = scan ? Math.max(0, scan.phones.length - MAX_SLOTS) : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="relative w-full max-w-xl rounded-lg border border-zinc-800 bg-zinc-900 p-6 shadow-2xl">
        <button
          onClick={onClose}
          disabled={phase === 'importing'}
          className="absolute right-3 top-3 rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30"
          aria-label="Close"
        >
          <X size={18} />
        </button>

        <div className="mb-4 flex items-center gap-2">
          <FolderUp size={20} className="text-blue-400" />
          <h2 className="text-lg font-semibold text-white">Import Folder</h2>
        </div>

        {phase === 'idle' && (
          <>
            <p className="mb-4 text-sm text-zinc-400">
              Wähle einen Root-Ordner mit der Struktur:
              <br />
              <span className="font-mono text-xs text-zinc-500">
                Root / _Phone A - … / 1. Picture / Wide / *.jpg
              </span>
            </p>
            <button
              onClick={handlePickFolder}
              className="flex w-full items-center justify-center gap-2 rounded-md border border-blue-500/40 bg-blue-600/15 py-2.5 text-sm text-blue-300 hover:bg-blue-600/25"
            >
              <FolderUp size={14} />
              Ordner auswählen…
            </button>
            {error && (
              <p className="mt-3 rounded border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-300">
                {error}
              </p>
            )}
            <input
              ref={inputRef}
              type="file"
              // @ts-expect-error — webkitdirectory is non-standard but widely supported
              webkitdirectory=""
              directory=""
              multiple
              className="hidden"
              onChange={handleFolderChosen}
            />
          </>
        )}

        {phase === 'preview' && scan && (
          <>
            <p className="mb-3 text-sm text-zinc-300">
              Gefunden: <strong>{scan.phones.length}</strong> Phones,{' '}
              <strong>{scan.totalImages}</strong> Bilder
            </p>

            <div className="mb-3 rounded border border-zinc-800 bg-zinc-950 p-3">
              <p className="mb-2 text-xs font-medium text-zinc-400">Phones (Slot-Reihenfolge):</p>
              <ol className="space-y-1 text-xs text-zinc-200">
                {phonesShown.map((p, i) => (
                  <li key={p}>
                    <span className="mr-2 inline-block w-4 text-zinc-500">{i + 1}.</span>
                    {prettifyPhoneName(p)}
                  </li>
                ))}
                {phonesIgnored > 0 && (
                  <li className="text-amber-400">
                    + {phonesIgnored} weitere Phone(s) werden ignoriert (Maximum: {MAX_SLOTS})
                  </li>
                )}
              </ol>
            </div>

            <div className="mb-4 rounded border border-zinc-800 bg-zinc-950 p-3">
              <p className="mb-2 text-xs font-medium text-zinc-400">Comparisons pro Kategorie:</p>
              <ul className="space-y-0.5 text-xs text-zinc-200">
                {scan.activeCategories.map((cat) => (
                  <li key={cat}>
                    <span className="capitalize">{cat}</span>:{' '}
                    {scan.comparisonsPerCategory[cat]} comparison(s)
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-300">
              ⚠ Alle aktuellen Vergleiche werden überschrieben.
            </div>

            <div className="mt-4 flex gap-2">
              <button
                onClick={onClose}
                className="flex-1 rounded-md border border-zinc-700 py-2 text-sm text-zinc-300 hover:bg-zinc-800"
              >
                Abbrechen
              </button>
              <button
                onClick={handleConfirmImport}
                className="flex-1 rounded-md border border-blue-500/40 bg-blue-600/15 py-2 text-sm text-blue-300 hover:bg-blue-600/25"
              >
                Import starten
              </button>
            </div>

            {error && (
              <p className="mt-3 rounded border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-300">
                {error}
              </p>
            )}
          </>
        )}

        {phase === 'importing' && (
          <>
            <div className="flex items-center gap-2 text-sm text-zinc-200">
              <Loader2 className="animate-spin text-blue-400" size={16} />
              Lade Bilder…
            </div>
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full bg-blue-500 transition-all"
                style={{
                  width: `${progress.total > 0 ? (progress.loaded / progress.total) * 100 : 0}%`,
                }}
              />
            </div>
            <p className="mt-2 text-xs text-zinc-400 tabular-nums">
              {progress.loaded} / {progress.total}
            </p>
          </>
        )}

        {phase === 'done' && (
          <p className="text-center text-sm text-emerald-400">✓ Import abgeschlossen</p>
        )}
      </div>
    </div>
  );
}
