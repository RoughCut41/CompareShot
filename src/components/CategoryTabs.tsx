import {
  Maximize2,
  MonitorSmartphone,
  ZoomIn,
  User,
  Camera,
  Moon,
  Flower2,
  type LucideIcon,
} from 'lucide-react';
import { CATEGORIES, Category } from '@/lib/types';

const ICON_MAP: Record<Category, LucideIcon> = {
  wide: Maximize2,
  ultrawide: MonitorSmartphone,
  zoom: ZoomIn,
  portrait: User,
  front: Camera,
  'low-light': Moon,
  macro: Flower2,
};

const LABEL_MAP: Record<Category, string> = {
  wide: 'Wide',
  ultrawide: 'Ultrawide',
  zoom: 'Zoom',
  portrait: 'Portrait',
  front: 'Front',
  'low-light': 'Low-light',
  macro: 'Macro',
};

interface Props {
  active: Category;
  onSelect: (cat: Category) => void;
}

export function CategoryTabs({ active, onSelect }: Props) {
  return (
    <div className="flex gap-1 overflow-x-auto scrollbar-hide border-b border-zinc-800 bg-background px-4 py-2">
      {CATEGORIES.map((cat) => {
        const Icon = ICON_MAP[cat];
        const isActive = cat === active;
        return (
          <button
            key={cat}
            onClick={() => onSelect(cat)}
            className={`flex flex-shrink-0 items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm transition-colors ${
              isActive
                ? 'border-blue-500/30 bg-blue-600/15 text-blue-400'
                : 'border-transparent text-zinc-500 hover:bg-zinc-800/40 hover:text-zinc-300'
            }`}
          >
            <Icon size={14} />
            <span>{LABEL_MAP[cat]}</span>
          </button>
        );
      })}
    </div>
  );
}
