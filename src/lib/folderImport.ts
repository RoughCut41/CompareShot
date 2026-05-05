/**
 * Folder Import: parses a chosen directory tree (via webkitdirectory) and
 * builds a complete CategoryData structure for the app.
 *
 * Expected tree:
 *   Root/
 *     _Phone A - Galaxy S26 Ultra/
 *       1. Picture/
 *         Wide/        → image files
 *         Ultra Wide/  → image files
 *       2. Portrait/   → image files
 *       8. Zoom/       → image files
 *       9. Macro/      → image files
 *       10. Low Light/ → image files
 *       5. Front Camera Photo & Video/ → image files
 *       (others ignored: Video, Stabilisation, Audio, Slowmotion)
 *     _Phone B - iPhone 17 Pro Max/
 *       (same structure)
 *
 * Phones (folders starting with "_Phone") are sorted alphabetically and
 * become slots 0..N. Categories are filled comparison-by-comparison: the
 * first sorted image of each phone forms Comparison 1, second forms
 * Comparison 2, etc. If a phone has fewer images than another, its slot
 * stays empty in the trailing comparisons.
 */
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

/**
 * Map category-folder names (case-insensitive) to app category keys.
 * Only modes in this map are imported. Everything else is ignored.
 *
 * Note: "Picture" is NOT a category itself — it's a parent folder that
 * contains "Wide" and "Ultra Wide". We resolve those via the parent path.
 */
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

/**
 * Strip "1. ", "10. " etc. prefix and normalize the folder name for matching.
 */
function normalizeFolderName(raw: string): string {
  return raw.replace(/^\d+\.\s*/, '').trim().toLowerCase();
}

interface ParsedPath {
  phone: string;
  /** App category, or null if the folder isn't an importable mode */
  category: Category | null;
  fileName: string;
  file: File;
}

/**
 * Parse one File's webkitRelativePath into phone + category + filename.
 * Returns null if the path doesn't match the expected structure.
 *
 * Path examples:
 *   "Root/_Phone A - Galaxy S26 Ultra/1. Picture/Wide/IMG_001.jpg"
 *   "Root/_Phone A - Galaxy S26 Ultra/2. Portrait/IMG_001.jpg"
 *   "Root/A Roll/something.jpg"  → null (not a phone folder)
 *   "Root/_Phone A/3. Video/clip.mp4"  → null (mode not importable)
 */
function parsePath(file: File): ParsedPath | null {
  // webkitRelativePath is set when the file came from a directory picker
  const path = (file as File & { webkitRelativePath?: string }).webkitRelativePath ?? '';
  if (!path) return null;

  const parts = path.split('/').filter(Boolean);
  // Must be at least: Root, Phone, Category, File (4 parts)
  if (parts.length < 4) return null;

  // First part is the root the user picked — skip it
  // parts[1] should be the phone folder
  const phoneFolder = parts[1];
  if (!phoneFolder.startsWith('_Phone')) return null;

  const fileName = parts[parts.length - 1];
  if (!ALLOWED_EXTENSIONS.test(fileName)) return null;

  // Determine category from folder structure
  // Case A: parts = [Root, Phone, "1. Picture", "Wide", file]   (5 parts)
  // Case B: parts = [Root, Phone, "2. Portrait", file]          (4 parts)
  let categoryFolder: string;
  if (parts.length >= 5 && normalizeFolderName(parts[2]) === 'picture') {
    // "1. Picture" wrapper — the actual category is one level deeper
    categoryFolder = parts[3];
  } else if (parts.length === 4) {
    categoryFolder = parts[2];
  } else {
    // Unexpected nesting — ignore
    return null;
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
  /** Phones found in the folder, sorted alphabetically. */
  phones: string[];
  /** Per-category, per-phone, sorted file list. */
  byCategoryByPhone: Record<Category, Record<string, ParsedPath[]>>;
  /** How many comparisons each category will produce (= max images across phones). */
  comparisonsPerCategory: Record<Category, number>;
  /** Total importable images across all categories. */
  totalImages: number;
  /** Categories that have at least one image. */
  activeCategories: Category[];
}

/**
 * Scan the FileList and produce a structured plan without loading images yet.
 * This is fast (only parses paths) and used for the preview dialog.
 */
export function scanFolder(files: FileList | File[]): FolderScanResult {
  const fileArray = Array.from(files);

  const parsed: ParsedPath[] = [];
  for (const file of fileArray) {
    const p = parsePath(file);
    if (p) parsed.push(p);
  }

  // Collect phones (sorted alphabetically — "_Phone A …" sorts before "_Phone B …")
  const phoneSet = new Set<string>();
  for (const p of parsed) phoneSet.add(p.phone);
  const phones = Array.from(phoneSet).sort();

  // Group by category × phone, sort each group by filename
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

  // Determine number of comparisons per category (= max image count across phones)
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

/**
 * Take a scan result and actually load all images into ImageState objects,
 * building the final CategoryData structure.
 *
 * onProgress is called after each image with (loaded, total).
 */
export async function buildCategoryDataFromScan(
  scan: FolderScanResult,
  onProgress?: (loaded: number, total: number) => void
): Promise<CategoryData> {
  // Truncate phones to MAX_SLOTS — first N phones win
  const phones = scan.phones.slice(0, MAX_SLOTS);

  // Initialize empty CategoryData (one empty Comparison 1 per category)
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

    // Build comparisons for this category
    const comparisons: Comparison[] = [];
    for (let i = 0; i < numComparisons; i++) {
      comparisons.push({
        id: uid(),
        name: `Comparison ${i + 1}`,
        images: phones.map(() => null) as (ImageState | null)[],
      });
    }

    // Fill each comparison: the i-th file from each phone goes into Comparison i+1
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
          // Slot stays null; user can manually retry that one
        }
        loaded++;
        onProgress?.(loaded, total);
      }
    }

    data[cat] = comparisons;
  }

  return data;
}

/**
 * Format a phone folder name for display (strip "_" prefix, drop "Phone X - ").
 * "_Phone A - Galaxy S26 Ultra"  →  "Galaxy S26 Ultra"
 */
export function prettifyPhoneName(folderName: string): string {
  const stripped = folderName.replace(/^_/, '');
  const dashIdx = stripped.indexOf(' - ');
  if (dashIdx >= 0) return stripped.slice(dashIdx + 3);
  return stripped;
}
