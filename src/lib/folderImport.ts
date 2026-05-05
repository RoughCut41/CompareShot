import {
  CATEGORIES,
  Category,
  CategoryData,
  Comparison,
  ImageState,
  MAX_SLOTS,
} from './types';
import { loadImageFile } from './imageLoader';
import { uid } from './utils';

const ALLOWED_EXTENSIONS = /\.(jpe?g|png|heic|heif)$/i;

const CATEGORY_FOLDER_MAP: Record<string, Category> = {
  'wide': 'wide',
  'ultra wide': 'ultrawide',
  'ultrawide': 'ultrawide',
  'zoom': 'zoom',
  'portrait': 'portrait',
  'front camera photo & video': 'front',
  'front camera': 'front',
  'front': 'front',
  'low light': 'low-light',
  'lowlight': 'low-light',
  'low-light': 'low-light',
  'macro': 'macro',
};

function normalizeFolderName(raw: string): string {
  return raw.replace(/^\d+\.\s*/, '').trim().toLowerCase();
}

interface ParsedPath {
  phone: string;
  category: Category | null;
  fileName: string;
  file: File;
}

/**
 * Parse one File's webkitRelativePath into phone + category + filename.
 * The phone folder ("_Phone X - …") is searched at any depth in the path,
 * so the user can pick the parent folder, a wrapper, or the root itself.
 */
function parsePath(file: File): ParsedPath | null {
  const path = (file as File & { webkitRelativePath?: string }).webkitRelativePath ?? '';
  if (!path) return null;

  const parts = path.split('/').filter(Boolean);
  if (parts.length < 3) return null;

  const fileName = parts[parts.length - 1];
  if (!ALLOWED_EXTENSIONS.test(fileName)) return null;

  // Find the phone folder anywhere in the path (don't assume position)
  const phoneIdx = parts.findIndex((p) => p.startsWith('_Phone'));
  if (phoneIdx < 0) return null;

  const phoneFolder = parts[phoneIdx];
  const afterPhone = parts.slice(phoneIdx + 1);
  if (afterPhone.length < 2) return null;

  // Determine category folder
  let categoryFolder: string;
  if (afterPhone.length >= 3 && normalizeFolderName(afterPhone[0]) === 'picture') {
    categoryFolder = afterPhone[1];
  } else {
    categoryFolder = afterPhone[0];
  }

  const normalized = normalizeFolderName(categoryFolder);
  const category = CATEGORY_FOLDER_MAP[normalized] ?? null;
  if (!category) return null;

  return {
    phone: phoneFolder,
    category,
    fileName,
    file,
  };
}

export interface FolderScanResult {
  phones: string[];
  byCategoryByPhone: Record<Category, Record<string, ParsedPath[]>>;
  comparisonsPerCategory: Record<Category, number>;
  totalImages: number;
  activeCategories: Category[];
}

export function scanFolder(files: FileList | File[]): FolderScanResult {
  const fileArray = Array.from(files);

  // ===== DIAGNOSTIC LOGGING =====
  console.log('[FolderImport] Total files received:', fileArray.length);
  console.log('[FolderImport] First 10 paths:');
  for (let i = 0; i < Math.min(10, fileArray.length); i++) {
    const f = fileArray[i] as File & { webkitRelativePath?: string };
    console.log(`  [${i}] name="${f.name}" webkitRelativePath="${f.webkitRelativePath ?? '(empty)'}"`);
  }
  const topLevels = new Set<string>();
  for (const f of fileArray) {
    const path = (f as File & { webkitRelativePath?: string }).webkitRelativePath ?? '';
    const parts = path.split('/').filter(Boolean);
    if (parts.length >= 2) topLevels.add(parts[1]);
  }
  console.log('[FolderImport] Detected top-level folders:', Array.from(topLevels));
  // ===== END DIAGNOSTIC =====

  const parsed: ParsedPath[] = [];
  for (const file of fileArray) {
    const p = parsePath(file);
    if (p) parsed.push(p);
  }

  console.log('[FolderImport] Successfully parsed:', parsed.length, 'of', fileArray.length, 'files');

  const phoneSet = new Set<string>();
  for (const p of parsed) phoneSet.add(p.phone);
  const phones = Array.from(phoneSet).sort();

  const byCategoryByPhone = {} as Record<Category, Record<string, ParsedPath[]>>;
  for (const cat of CATEGORIES) {
    byCategoryByPhone[cat] = {};
    for (const phone of phones) byCategoryByPhone[cat][phone] = [];
  }
  for (const p of parsed) {
    if (p.category) byCategoryByPhone[p.category][p.phone].push(p);
  }
  for (const cat of CATEGORIES) {
    for (const phone of phones) {
      byCategoryByPhone[cat][phone].sort((a, b) =>
        a.fileName.localeCompare(b.fileName, undefined, { numeric: true })
      );
    }
  }

  const comparisonsPerCategory = {} as Record<Category, number>;
  for (const cat of CATEGORIES) {
    let maxCount = 0;
    for (const phone of phones) {
      maxCount = Math.max(maxCount, byCategoryByPhone[cat][phone].length);
    }
    comparisonsPerCategory[cat] = maxCount;
  }

  const totalImages = parsed.length;
  const activeCategories = CATEGORIES.filter((c) => comparisonsPerCategory[c] > 0);

  return {
    phones,
    byCategoryByPhone,
    comparisonsPerCategory,
    totalImages,
    activeCategories,
  };
}

export async function buildCategoryDataFromScan(
  scan: FolderScanResult,
  onProgress?: (loaded: number, total: number) => void
): Promise<CategoryData> {
  const phones = scan.phones.slice(0, MAX_SLOTS);

  const data = {} as CategoryData;
  for (const cat of CATEGORIES) {
    data[cat] = [
      {
        id: uid(),
        name: 'Comparison 1',
        images: phones.map(() => null) as (ImageState | null)[],
      },
    ];
  }

  let loaded = 0;
  const total = scan.totalImages;

  for (const cat of CATEGORIES) {
    const numComparisons = scan.comparisonsPerCategory[cat];
    if (numComparisons === 0) continue;

    const comparisons: Comparison[] = [];
    for (let i = 0; i < numComparisons; i++) {
      comparisons.push({
        id: uid(),
        name: `Comparison ${i + 1}`,
        images: phones.map(() => null) as (ImageState | null)[],
      });
    }

    for (let phoneIdx = 0; phoneIdx < phones.length; phoneIdx++) {
      const phone = phones[phoneIdx];
      const filesForPhone = scan.byCategoryByPhone[cat][phone];

      for (let imgIdx = 0; imgIdx < filesForPhone.length; imgIdx++) {
        const parsedPath = filesForPhone[imgIdx];
        try {
          const imageState = await loadImageFile(parsedPath.file);
          comparisons[imgIdx].images[phoneIdx] = imageState;
        } catch (err) {
          console.warn(
            '[FolderImport] Failed to load',
            parsedPath.fileName,
            'for phone',
            phone,
            err
          );
        }
        loaded++;
        onProgress?.(loaded, total);
      }
    }

    data[cat] = comparisons;
  }

  return data;
}

export function prettifyPhoneName(folderName: string): string {
  const stripped = folderName.replace(/^_/, '');
  const dashIdx = stripped.indexOf(' - ');
  if (dashIdx >= 0) return stripped.slice(dashIdx + 3);
  return stripped;
}
