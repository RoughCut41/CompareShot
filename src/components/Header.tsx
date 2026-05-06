import { useState } from 'react';
import { FolderUp } from 'lucide-react';
import { DownloadActions } from './DownloadActions';
import { FolderImportModal } from './FolderImportModal';
import { Category, CategoryData, Comparison } from '@/lib/types';

interface Props {
  comparison: Comparison | undefined;
  category: Category;
  comparisonIndex: number;
  /** Full app data — needed by DownloadActions for "all categories" exports */
  allData: CategoryData;
  onDeleteAll: () => void;
  onOpenDebugReport: () => void;
  onImportFolder: (data: CategoryData) => void;
}

export function Header({
  comparison,
  category,
  comparisonIndex,
  allData,
  onDeleteAll,
  onOpenDebugReport,
  onImportFolder,
}: Props) {
  const [importOpen, setImportOpen] = useState(false);

  return (
    <header className="flex flex-shrink-0 items-center justify-between border-b border-zinc-800 bg-background px-4 py-3">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-blue-600 font-bold text-white">
          C
        </div>
        <h1 className="text-lg font-semibold text-white">CompareShot</h1>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={() => setImportOpen(true)}
          className="flex items-center gap-1.5 rounded-md border border-zinc-800 px-3 py-1 text-xs text-zinc-300 transition-colors hover:border-blue-500/40 hover:bg-blue-500/10 hover:text-blue-300"
          title="Ganzen Ordner importieren"
        >
          <FolderUp size={12} />
          Import Folder
        </button>

        <DownloadActions
          comparison={comparison}
          category={category}
          comparisonIndex={comparisonIndex}
          allData={allData}
          onDeleteAll={onDeleteAll}
          onOpenDebugReport={onOpenDebugReport}
        />
      </div>

      <FolderImportModal
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImportComplete={(data) => {
          onImportFolder(data);
          setImportOpen(false);
        }}
      />
    </header>
  );
}
