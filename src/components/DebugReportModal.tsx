import { useEffect, useRef, useState } from 'react';
import { Check, Copy, X } from 'lucide-react';

interface Props {
  open: boolean;
  report: string;
  onClose: () => void;
}

export function DebugReportModal({ open, report, onClose }: Props) {
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (open) setCopied(false);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      const ta = textareaRef.current;
      if (ta) {
        ta.focus();
        ta.select();
        try {
          document.execCommand('copy');
          setCopied(true);
          window.setTimeout(() => setCopied(false), 2000);
        } catch {
          /* noop */
        }
      }
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="flex h-[80vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-shrink-0 items-center justify-between border-b border-zinc-800 px-4 py-3">
          <h2 className="text-sm font-semibold text-white">Debug Report</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className={`flex items-center gap-1.5 rounded-md border px-3 py-1 text-xs transition-colors ${
                copied
                  ? 'border-green-500/40 bg-green-600/15 text-green-300'
                  : 'border-blue-500/40 bg-blue-600/15 text-blue-300 hover:bg-blue-600/25'
              }`}
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? 'Copied!' : 'Copy report'}
            </button>
            <button
              onClick={onClose}
              className="rounded-md border border-zinc-700 p-1 text-zinc-400 hover:bg-zinc-800 hover:text-white"
              aria-label="Close"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-hidden p-4">
          <textarea
            ref={textareaRef}
            value={report}
            readOnly
            className="h-full w-full resize-none rounded-md border border-zinc-800 bg-black p-3 font-mono text-[11px] leading-[1.5] text-zinc-300 focus:border-zinc-700 focus:outline-none"
            spellCheck={false}
          />
        </div>

        <div className="flex-shrink-0 border-t border-zinc-800 px-4 py-2 text-[11px] text-zinc-500">
          Tipp: Klick auf "Copy report" und füge den Inhalt in den Chat ein.
        </div>
      </div>
    </div>
  );
}
