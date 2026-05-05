export function scanFolder(files: FileList | File[]): FolderScanResult {
  const fileArray = Array.from(files);

  // ===== DIAGNOSTIC LOGGING =====
  console.log('[FolderImport] Total files received:', fileArray.length);
  console.log('[FolderImport] First 10 paths:');
  for (let i = 0; i < Math.min(10, fileArray.length); i++) {
    const f = fileArray[i] as File & { webkitRelativePath?: string };
    console.log(`  [${i}] name="${f.name}" webkitRelativePath="${f.webkitRelativePath ?? '(empty)'}"`);
  }
  // Show what's at the top level
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
