function parsePath(file: File): ParsedPath | null {
  const path = (file as File & { webkitRelativePath?: string }).webkitRelativePath ?? '';
  if (!path) return null;

  const parts = path.split('/').filter(Boolean);
  // Need at minimum: Phone, Mode, File (3 parts after the phone folder is found)
  if (parts.length < 3) return null;

  const fileName = parts[parts.length - 1];
  if (!ALLOWED_EXTENSIONS.test(fileName)) return null;

  // Find the phone folder anywhere in the path (don't assume position).
  // The user might pick the root folder, a phone parent folder, or a wrapper —
  // we just locate "_Phone X" and work relative to that.
  const phoneIdx = parts.findIndex((p) => p.startsWith('_Phone'));
  if (phoneIdx < 0) return null;

  const phoneFolder = parts[phoneIdx];
  const afterPhone = parts.slice(phoneIdx + 1);
  // afterPhone should contain at least: Mode, File (2 parts)
  // or: "1. Picture", SubMode, File (3 parts)
  if (afterPhone.length < 2) return null;

  // Determine category folder
  let categoryFolder: string;
  if (afterPhone.length >= 3 && normalizeFolderName(afterPhone[0]) === 'picture') {
    // "1. Picture" wrapper — category is one level deeper
    categoryFolder = afterPhone[1];
  } else if (afterPhone.length === 2) {
    // Direct: <Phone>/<Mode>/<File>
    categoryFolder = afterPhone[0];
  } else {
    // Unexpected nesting — try the first folder after phone as a category
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
