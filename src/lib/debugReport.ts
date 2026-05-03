/**
 * Build a plain-text debug report for sharing with support.
 * Includes browser info, app state, last Smart Align report, captured logs.
 */
import { CategoryData, ImageState, Category } from './types';
import { getCapturedLogs } from './logCapture';
import type { SmartAlignReport } from './smartAlign';

export interface DebugReportInput {
  data: CategoryData;
  activeCategory: Category;
  activeIndex: Record<Category, number>;
  lastSmartAlignReport: SmartAlignReport | null;
  lastSmartAlignError: string | null;
  appVersion: string;
}

function fmtImage(img: ImageState | null, slotIdx: number): string {
  if (!img) return `  Slot ${slotIdx}: <empty>`;
  const round = (n: number) => Math.round(n * 100) / 100;
  return [
    `  Slot ${slotIdx}:`,
    `    file:        ${img.file.name} (${(img.file.size / 1024).toFixed(0)} KB, ${img.file.type || '?'})`,
    `    natural:     ${img.naturalWidth}×${img.naturalHeight}`,
    `    container:   ${img._containerW}×${img._containerH}`,
    `    zoom:        ${round(img.zoom)}`,
    `    pan:         (${round(img.panX)}, ${round(img.panY)})`,
    `    rotation:    ${round(img.rotation)}°`,
    `    flipH/V:     ${img.flipH}/${img.flipV}`,
    `    freeRotate:  ${img.freeRotateActive}`,
  ].join('\n');
}

function fmtSmartAlignReport(report: SmartAlignReport | null): string {
  if (!report) return '  <none yet>';
  const lines = [
    `  mode:                ${report.mode}`,
    `  referenceSlotIndex:  ${report.referenceSlotIndex}`,
    `  results:`,
  ];
  for (const r of report.results) {
    const tx = r.transform
      ? ` zoom=${r.transform.zoom.toFixed(3)} pan=(${r.transform.panX.toFixed(1)},${r.transform.panY.toFixed(1)}) rot=${r.transform.rotation.toFixed(2)}°`
      : '';
    lines.push(`    slot ${r.slotIndex}: ${r.status}${r.reason ? ` — ${r.reason}` : ''}${tx}`);
  }
  return lines.join('\n');
}

export function buildDebugReport(input: DebugReportInput): string {
  const now = new Date().toISOString();
  const ua = navigator.userAgent;
  const platform = navigator.platform ?? 'unknown';
  const viewport = `${window.innerWidth}×${window.innerHeight}`;
  const dpr = window.devicePixelRatio || 1;

  const cat = input.activeCategory;
  const compIdx = input.activeIndex[cat];
  const activeComp = input.data[cat]?.[compIdx];
  const allCounts = (Object.keys(input.data) as Category[])
    .map((c) => `${c}=${input.data[c].length}`)
    .join(', ');

  const sections: string[] = [];

  sections.push('=== CompareShot Debug Report ===');
  sections.push(`Generated: ${now}`);
  sections.push(`App version: ${input.appVersion}`);
  sections.push('');

  sections.push('--- Browser ---');
  sections.push(`UA:        ${ua}`);
  sections.push(`Platform:  ${platform}`);
  sections.push(`Viewport:  ${viewport} (DPR ${dpr})`);
  sections.push(`Language:  ${navigator.language}`);
  sections.push(`Online:    ${navigator.onLine}`);
  sections.push('');

  sections.push('--- App state ---');
  sections.push(`Active category:    ${cat}`);
  sections.push(`Active comparison:  ${compIdx}`);
  sections.push(`Comparisons/category: ${allCounts}`);
  if (activeComp) {
    sections.push(`Active "${activeComp.name}" — ${activeComp.images.length} slot(s)`);
    activeComp.images.forEach((img, i) => sections.push(fmtImage(img, i)));
  } else {
    sections.push('No active comparison.');
  }
  sections.push('');

  sections.push('--- Last Smart Align run ---');
  sections.push(fmtSmartAlignReport(input.lastSmartAlignReport));
  if (input.lastSmartAlignError) {
    sections.push(`  ERROR: ${input.lastSmartAlignError}`);
  }
  sections.push('');

  sections.push('--- Console logs (last 200) ---');
  const logs = getCapturedLogs();
  if (logs.length === 0) {
    sections.push('  <none>');
  } else {
    for (const log of logs) {
      const t = new Date(log.timestamp).toISOString().slice(11, 23);
      sections.push(`  [${t}] ${log.level.toUpperCase().padEnd(5)} ${log.message}`);
    }
  }
  sections.push('');

  sections.push('=== End of report ===');
  return sections.join('\n');
}
