import { useCallback, useState } from 'react';
import { useCompareStore } from '@/hooks/useCompareStore';
import { Header } from '@/components/Header';
import { CategoryTabs } from '@/components/CategoryTabs';
import { ComparisonTabs } from '@/components/ComparisonTabs';
import { Toolbar } from '@/components/Toolbar';
import { RulerWorkspace } from '@/components/RulerWorkspace';
import { ImageGrid } from '@/components/ImageGrid';
import { smartAlign } from '@/lib/smartAlign';

export default function App() {
  const store = useCompareStore();
  const [aligning, setAligning] = useState(false);
  const [alignStatus, setAlignStatus] = useState<string | null>(null);

  const slotCount = store.activeComparison?.images.length ?? 0;
  const filledCount = (store.activeComparison?.images ?? []).filter((i) => i !== null).length;
  const canAlign = filledCount >= 2;

  const handleSmartAlign = useCallback(async () => {
    if (!store.activeComparison || !canAlign) return;
    setAligning(true);
    setAlignStatus('Loading…');
    try {
      const report = await smartAlign({
        images: store.activeComparison.images,
        onProgress: (label) => setAlignStatus(label),
      });

      // Apply the computed transforms to the slots
      for (const result of report.results) {
        if (result.status === 'aligned' && result.transform) {
          store.updateImage(result.slotIndex, {
            zoom: result.transform.zoom,
            panX: result.transform.panX,
            panY: result.transform.panY,
            rotation: result.transform.rotation,
            // Reset flips — alignment assumes upright images
            flipH: false,
            flipV: false,
            freeRotateActive: false,
          });
        } else if (result.status === 'reference') {
          // Reset reference to identity (no transform applied)
          store.updateImage(result.slotIndex, {
            zoom: 1,
            panX: 0,
            panY: 0,
            rotation: 0,
            flipH: false,
            flipV: false,
            freeRotateActive: false,
          });
        }
      }

      // Brief summary in the button before clearing
      const aligned = report.results.filter((r) => r.status === 'aligned').length;
      const failed = report.results.filter((r) => r.status === 'failed').length;
      const modeLabel = report.mode === 'face' ? 'face' : 'features';
      setAlignStatus(
        failed > 0
          ? `Aligned ${aligned} (${failed} failed) · ${modeLabel}`
          : `Aligned ${aligned} · ${modeLabel}`
      );
      // Auto-clear status after a moment
      window.setTimeout(() => {
        setAlignStatus(null);
        setAligning(false);
      }, 2200);
    } catch (err) {
      console.error('[CompareShot] Smart Align failed:', err);
      setAlignStatus('Failed');
      window.setTimeout(() => {
        setAlignStatus(null);
        setAligning(false);
      }, 2500);
    }
  }, [store, canAlign]);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-background text-zinc-200">
      <Header
        comparison={store.activeComparison}
        category={store.activeCategory}
        comparisonIndex={store.activeIndex[store.activeCategory]}
        onDeleteAll={store.deleteAllPhotos}
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
    </div>
  );
}
