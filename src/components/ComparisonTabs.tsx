import { Plus, X } from 'lucide-react';
import { Comparison } from '@/lib/types';

interface Props {
  comparisons: Comparison[];
  activeIndex: number;
  onSelect: (idx: number) => void;
  onCreate: () => void;
  onDelete: (idx: number) => void;
}

export function ComparisonTabs({ comparisons, activeIndex, onSelect, onCreate, onDelete }: Props) {
  return (
    <div className="flex items-center gap-2 overflow-x-auto scrollbar-hide border-b border-zinc-800 bg-background px-4 py-2">
      {comparisons.map((c, i) => {
        const isActive = i === activeIndex;
        const canDelete = comparisons.length > 1;
        return (
          <div
            key={c.id}
            className={`group flex flex-shrink-0 items-center gap-1.5 rounded-md border px-3 py-1 text-sm transition-colors ${
              isActive
                ? 'border-zinc-600 bg-zinc-800 text-white'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <button onClick={() => onSelect(i)} className="outline-none">
              {c.name}
            </button>
            {canDelete && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(i);
                }}
                className="rounded p-0.5 opacity-0 transition-opacity hover:bg-zinc-700 hover:text-red-400 group-hover:opacity-100"
                aria-label={`Vergleich ${c.name} löschen`}
              >
                <X size={12} />
              </button>
            )}
          </div>
        );
      })}
      <button
        onClick={onCreate}
        className="flex flex-shrink-0 items-center gap-1 rounded-md border border-dashed border-zinc-700 px-3 py-1 text-sm text-zinc-500 transition-colors hover:border-zinc-600 hover:text-zinc-300"
      >
        <Plus size={14} />
        <span>Create new</span>
      </button>
    </div>
  );
}
