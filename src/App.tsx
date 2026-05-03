import { useCompareStore } from '@/hooks/useCompareStore';
import { Header } from '@/components/Header';
import { CategoryTabs } from '@/components/CategoryTabs';
import { ComparisonTabs } from '@/components/ComparisonTabs';
import { Toolbar } from '@/components/Toolbar';
import { RulerWorkspace } from '@/components/RulerWorkspace';
import { ImageGrid } from '@/components/ImageGrid';

export default function App() {
  const store = useCompareStore();

  const slotCount = store.activeComparison?.images.length ?? 0;
  const hasAnyImage = (store.activeComparison?.images ?? []).some((i) => i !== null);

  function handleSmartAlign() {
    // Phase 2 — to be implemented after Phase 1 testing
    alert(
      'Smart Align kommt in Phase 2. Phase 1 ist erstmal das vollständige Grundgerüst — ' +
        'sobald du das getestet und deployed hast, bauen wir die Auto-Ausrichtung dazu.'
    );
  }

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
        smartAlignDisabled={!hasAnyImage}
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
