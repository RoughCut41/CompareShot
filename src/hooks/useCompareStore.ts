import { useCallback, useMemo, useState } from 'react';
import {
  ActiveIndexMap,
  CATEGORIES,
  Category,
  CategoryData,
  Comparison,
  ImageState,
  MAX_SLOTS,
  MIN_SLOTS,
} from '@/lib/types';
import { disposeImage } from '@/lib/imageLoader';
import { uid } from '@/lib/utils';

function makeEmptyComparison(name: string, slotCount = MIN_SLOTS): Comparison {
  return {
    id: uid(),
    name,
    images: Array.from({ length: slotCount }, () => null),
  };
}

function makeInitialData(): CategoryData {
  const data = {} as CategoryData;
  for (const cat of CATEGORIES) {
    data[cat] = [makeEmptyComparison('Comparison 1')];
  }
  return data;
}

function makeInitialActiveIndex(): ActiveIndexMap {
  const map = {} as ActiveIndexMap;
  for (const cat of CATEGORIES) map[cat] = 0;
  return map;
}

export type ImageUpdater = Partial<ImageState> | ((prev: ImageState) => Partial<ImageState>);

export function useCompareStore() {
  const [data, setData] = useState<CategoryData>(makeInitialData);
  const [activeIndex, setActiveIndex] = useState<ActiveIndexMap>(makeInitialActiveIndex);
  const [activeCategory, setActiveCategory] = useState<Category>('wide');

  const activeComparison = useMemo<Comparison | undefined>(
    () => data[activeCategory]?.[activeIndex[activeCategory]],
    [data, activeCategory, activeIndex]
  );

  /** Patch the active comparison with a partial or updater. */
  const patchActiveComparison = useCallback(
    (patcher: (c: Comparison) => Comparison) => {
      setData((prev) => {
        const list = prev[activeCategory];
        const idx = activeIndex[activeCategory];
        const target = list[idx];
        if (!target) return prev;
        const next = patcher(target);
        if (next === target) return prev;
        const newList = list.slice();
        newList[idx] = next;
        return { ...prev, [activeCategory]: newList };
      });
    },
    [activeCategory, activeIndex]
  );

  /** Update a single image slot in the active comparison. */
  const updateImage = useCallback(
    (slotIndex: number, updater: ImageUpdater) => {
      patchActiveComparison((c) => {
        const cur = c.images[slotIndex];
        if (!cur) return c;
        const patch = typeof updater === 'function' ? updater(cur) : updater;
        const newImages = c.images.slice();
        newImages[slotIndex] = { ...cur, ...patch };
        return { ...c, images: newImages };
      });
    },
    [patchActiveComparison]
  );

  /** Set a slot's image to a new ImageState (or null to clear). */
  const setSlotImage = useCallback(
    (slotIndex: number, state: ImageState | null) => {
      patchActiveComparison((c) => {
        const prev = c.images[slotIndex];
        // Dispose the previous image's object URL if replacing
        if (prev && prev !== state) disposeImage(prev);
        const newImages = c.images.slice();
        newImages[slotIndex] = state;
        return { ...c, images: newImages };
      });
    },
    [patchActiveComparison]
  );

  /** Delete (clear) a slot's image. */
  const deleteImage = useCallback(
    (slotIndex: number) => {
      setSlotImage(slotIndex, null);
    },
    [setSlotImage]
  );

  /** Add a new empty slot to the active comparison (max MAX_SLOTS). */
  const addSlot = useCallback(() => {
    patchActiveComparison((c) => {
      if (c.images.length >= MAX_SLOTS) return c;
      return { ...c, images: [...c.images, null] };
    });
  }, [patchActiveComparison]);

  /** Remove the last slot from the active comparison (min MIN_SLOTS). */
  const removeSlot = useCallback(() => {
    patchActiveComparison((c) => {
      if (c.images.length <= MIN_SLOTS) return c;
      const removed = c.images[c.images.length - 1];
      if (removed) disposeImage(removed);
      return { ...c, images: c.images.slice(0, -1) };
    });
  }, [patchActiveComparison]);

  /** Switch to another category tab. */
  const switchCategory = useCallback((cat: Category) => {
    setActiveCategory(cat);
  }, []);

  /** Switch to another comparison tab within the active category. */
  const switchComparison = useCallback(
    (idx: number) => {
      setActiveIndex((prev) => ({ ...prev, [activeCategory]: idx }));
    },
    [activeCategory]
  );

  /** Create a new comparison in the active category. */
  const createComparison = useCallback(() => {
    let newIndex = 0;
    setData((prev) => {
      const list = prev[activeCategory];
      const newComp = makeEmptyComparison(`Comparison ${list.length + 1}`);
      newIndex = list.length;
      return { ...prev, [activeCategory]: [...list, newComp] };
    });
    // Activate the newly created comparison
    setActiveIndex((prev) => ({
      ...prev,
      [activeCategory]: newIndex,
    }));
  }, [activeCategory]);

  /** Delete a comparison by index in the active category. */
  const deleteComparison = useCallback(
    (idx: number) => {
      setData((prev) => {
        const list = prev[activeCategory];
        if (list.length <= 1) return prev; // keep at least one
        // Dispose images in the comparison being deleted
        list[idx]?.images.forEach((img) => disposeImage(img));
        const newList = list.slice();
        newList.splice(idx, 1);
        return { ...prev, [activeCategory]: newList };
      });
      // Adjust active index if needed
      setActiveIndex((prev) => {
        const cur = prev[activeCategory];
        const newCur = cur >= idx && cur > 0 ? cur - 1 : cur;
        return { ...prev, [activeCategory]: newCur };
      });
    },
    [activeCategory]
  );

  /** Reset all images in all categories. */
  const deleteAllPhotos = useCallback(() => {
    setData((prev) => {
      // Dispose every image
      for (const cat of CATEGORIES) {
        for (const c of prev[cat]) {
          c.images.forEach((img) => disposeImage(img));
        }
      }
      return makeInitialData();
    });
    setActiveIndex(makeInitialActiveIndex());
  }, []);

  return {
    data,
    activeCategory,
    activeIndex,
    activeComparison,
    switchCategory,
    switchComparison,
    createComparison,
    deleteComparison,
    setSlotImage,
    updateImage,
    deleteImage,
    addSlot,
    removeSlot,
    deleteAllPhotos,
  };
}

export type CompareStore = ReturnType<typeof useCompareStore>;
