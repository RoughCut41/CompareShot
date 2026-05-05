import { useCallback, useEffect, useState } from 'react';
import { useCompareStore } from '@/hooks/useCompareStore';
import { Header } from '@/components/Header';
import { CategoryTabs } from '@/components/CategoryTabs';
import { ComparisonTabs } from '@/components/ComparisonTabs';
import { Toolbar } from '@/components/Toolbar';
import { RulerWorkspace } from '@/components/RulerWorkspace';
import { ImageGrid } from '@/components/ImageGrid';
import { DebugReportModal } from '@/components/DebugReportModal';
import { smartAlign, type SmartAlignReport } from '@/lib/smartAlign';
import { installLogCapture } from '@/lib/logCapture';
import { buildDebugReport } from '@/lib/debugReport';

const APP_VERSION = '0.2.1-phase2';

export default function App() {
  const store = useCompareStore();
  const [aligning, setAligning] = useState(false);
  const [alignStatus, setAlignStatus] = useState<string | null>(null);
  const [lastAlignReport, setLastAlignReport] = useState<SmartAlignReport | null>(null);
  const [lastAlignError, setLastAlignError] = useState<string | null>(null);
  const [debugOpen, setDebugOpen] = useState(false);
  const [debugReport, setDebugReport] = useState('');

  useEffect(() => {
    installLogCapture();
    console.log('[CompareShot] App mounted, version', APP_VERSION);
  }, []);

  const slotCount = store.activeComparison?.images.length ?? 0;
  const filledCount = (store.activeComparison?.images ?? []).filter((i) => i !== null).length;
  const canAlign = filledCount >= 2;

  const handleSmartAlign = useCallback(async () => {
    console.log('[CompareShot] Smart Align button clicked');
    if (!store.activeComparison) {
      console.warn('[CompareShot] No active comparison — aborting');
      return;
    }
    if (!canAlign) {
      console.warn('[CompareShot] Not enough images — aborting');
      return;
    }
    console.log('[CompareShot] Starting Smart Align with', filledCount, 'images');
    setAligning(true);
    setAlignStatus('Loading…');
    setLastAlignError(null);
    try {
      const report = await smartAlign({
        images: store.activeComparison.images,
        onProgress: (label) => {
          console.log('[CompareShot] Progress:', label);
          setAlignStatus(label);
        },
      });
      console.log('[CompareShot] Smart Align complete:', report);
      setLastAlignReport(report);

      // Apply the computed transform to ALL slots that received one — including
      // the reference slot. The "reference" status is now just a label for which
      // slot was chosen as the anchor; mathematically the reference must also
      // receive scaleMatchZoom × globalCropZoom + pan-to-target so heads stay
      // at the same on-screen size as the others.
      for (const result of report.results) {
        if ((result.status === 'aligned' || result.status === 'reference') && result.transform) {
          store.updateImage(result.slotIndex, {
            zoom: result.transform.zoom,
            panX: result.transform.panX,
            panY: result.transform.panY,
            rotation: result.transform.rotation,
            flipH: false,
            flipV: false,
            freeRotateActive: false,
          });
        }
        // 'failed' results are left untouched — user can manually adjust.
      }

      const aligned = report.results.filter((r) => r.status === 'aligned').length;
      const failed = report.results.filter((r) => r.status === 'failed').length;
      const modeLabel = report.mode === 'face' ? 'face' : 'features';
      setAlignStatus(
        failed > 0
          ? `Aligned ${aligned} (${failed} failed) · ${modeLabel}`
          : `Aligned ${aligned} · ${modeLabel}`
      );
      window.setTimeout(() => {
        setAlignStatus(null);
        setAligning(false);
      }, 2200);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[CompareShot] Smart Align failed:', err);
      if (err instanceof Error && err.stack) {
        console.error('[CompareShot] Stack:', err.stack);
      }
      setLastAlignError(msg);
      setAlignStatus('Failed: ' + msg.slice(0, 30));
      window.setTimeout(() => {
        setAlignStatus(null);
        setAligning(false);
      }, 3000);
    }
  }, [store, canAlign, filledCount]);

  const handleOpenDebug = useCallback(() => {
    const report = buildDebugReport({
      data: store.data,
      activeCategory: store.activeCategory,
      activeIndex: store.activeIndex,
      lastSmartAlignReport: lastAlignReport,
      lastSmartAlignError: lastAlignError,
      appVersion: APP_VERSION,
    });
    setDebugReport(report);
    setDebugOpen(true);
  }, [store.data, store.activeCategory, store.activeIndex, lastAlignReport, lastAlignError]);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-background text-zinc-200">
      <Header
        comparison={store.activeComparison}
        category={store.activeCategory}
        comparisonIndex={store.activeIndex[store.activeCategory]}
        onDeleteAll={store.deleteAllPhotos}
        onOpenDebugReport={handleOpenDebug}
        onImportFolder={store.replaceAllData}
      />
      <CategoryTabs active={store.activeCategory} onSelect={store.switchCategory} />
      <ComparisonTabs
        comparisons={store.data[store.activeCategory]}
        activeIndex={store.activeIndex[store.activeCategory]}
        onSelect={store.switchComparison}
        onCreate={store.createComparison}
        onDelete={store.deleteComparison}
      />
      <Toolbar
        slotCount={slotCount}
        onAddSlot={store.addSlot}
        onRemoveSlot={store.removeSlot}
        onSmartAlign={handleSmartAlign}
        smartAlignDisabled={!canAlign}
        smartAlignBusy={aligning}
        smartAlignStatus={alignStatus}
      />
      <RulerWorkspace>
        {store.activeComparison && (
          <ImageGrid
            images={store.activeComparison.images}
            onUpdateImage={store.updateImage}
            onSetImage={store.setSlotImage}
            onDeleteImage={store.deleteImage}
          />
        )}
      </RulerWorkspace>

      <DebugReportModal open={debugOpen} report={debugReport} onClose={() => setDebugOpen(false)} />
    </div>
  );
}
