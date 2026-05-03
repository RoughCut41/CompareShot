import { Minus, Plus, Sparkles } from 'lucide-react';
import { MAX_SLOTS, MIN_SLOTS } from '@/lib/types';

interface Props {
  slotCount: number;
  onAddSlot: () => void;
  onRemoveSlot: () => void;
  onSmartAlign: () => void;
  smartAlignDisabled?: boolean;
  smartAlignBusy?: boolean;
  /** Live status label shown on the button while aligning, e.g. "Detecting…" */
  smartAlignStatus?: string | null;
}

export function Toolbar({
  slotCount,
  onAddSlot,
  onRemoveSlot,
  onSmartAlign,
  smartAlignDisabled = false,
  smartAlignBusy = false,
  smartAlignStatus = null,
}: Props) {
  const canAdd = slotCount < MAX_SLOTS;
  const canRemove = slotCount > MIN_SLOTS;

  const buttonLabel = smartAlignBusy ? smartAlignStatus ?? 'Detecting…' : 'AI Smart Align';

  return (
    <div className="flex items-center gap-2 border-b border-zinc-800 bg-background px-4 py-2">
      <span className="text-xs text-zinc-500">{slotCount} photos in comparison</span>

      <div className="ml-auto flex items-center gap-1">
        {canRemove && (
          <button
            onClick={onRemoveSlot}
            className="flex items-center gap-1 rounded-md border border-zinc-800 px-2.5 py-1 text-xs text-zinc-400 transition-colors hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-300"
            title="Letzten Slot entfernen"
          >
            <Minus size={12} />
            Remove slot
          </button>
        )}
        {canAdd && (
          <button
            onClick={onAddSlot}
            className="flex items-center gap-1 rounded-md border border-zinc-800 px-2.5 py-1 text-xs text-zinc-400 transition-colors hover:border-blue-500/40 hover:bg-blue-500/10 hover:text-blue-300"
            title="Slot hinzufügen"
          >
            <Plus size={12} />
            Add slot
          </button>
        )}
        <div className="mx-1 h-5 w-px bg-zinc-800" />
        <button
          onClick={onSmartAlign}
          disabled={smartAlignDisabled || smartAlignBusy}
          className="flex min-w-[8.5rem] items-center justify-center gap-1.5 rounded-md border border-blue-500/40 bg-blue-600/15 px-3 py-1 text-xs text-blue-300 transition-colors hover:bg-blue-600/25 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-blue-600/15"
          title="AI-gestützte automatische Ausrichtung"
        >
          {smartAlignBusy ? (
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-blue-400/30 border-t-blue-300" />
          ) : (
            <Sparkles size={12} />
          )}
          <span className="tabular-nums">{buttonLabel}</span>
        </button>
      </div>
    </div>
  );
}
