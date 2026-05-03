import { DownloadActions } from './DownloadActions';
import { Category, Comparison } from '@/lib/types';

interface Props {
  comparison: Comparison | undefined;
  category: Category;
  comparisonIndex: number;
  onDeleteAll: () => void;
}

export function Header({ comparison, category, comparisonIndex, onDeleteAll }: Props) {
  return (
    <header className="flex flex-shrink-0 items-center justify-between border-b border-zinc-800 bg-background px-4 py-3">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-blue-600 font-bold text-white">
          C
        </div>
        <h1 className="text-lg font-semibold text-white">CompareShot</h1>
      </div>
      <DownloadActions
        comparison={comparison}
        category={category}
        comparisonIndex={comparisonIndex}
        onDeleteAll={onDeleteAll}
      />
    </header>
  );
}
