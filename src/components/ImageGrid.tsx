import { useEffect, useState } from 'react';
import { ImageContainer } from './ImageContainer';
import { ImageState } from '@/lib/types';
import { ImageUpdater } from '@/hooks/useCompareStore';

interface Props {
  images: (ImageState | null)[];
  onUpdateImage: (slotIndex: number, updater: ImageUpdater) => void;
  onSetImage: (slotIndex: number, state: ImageState) => void;
  onDeleteImage: (slotIndex: number) => void;
}

export function ImageGrid({ images, onUpdateImage, onSetImage, onDeleteImage }: Props) {
  const [slotW, setSlotW] = useState(360);

  useEffect(() => {
    function compute() {
      // Available width = workspace width minus rulers (20px) minus padding/gap allowance
      const workspaceW = window.innerWidth - 100;
      const gap = 8 * (images.length - 1);
      const perSlot = Math.floor((workspaceW - gap) / images.length);
      // Cap individual slot width at 450 px (matches base44 doc)
      setSlotW(Math.max(160, Math.min(450, perSlot)));
    }
    compute();
    window.addEventListener('resize', compute);
    return () => window.removeEventListener('resize', compute);
  }, [images.length]);

  return (
    <div className="flex justify-center gap-2 p-4">
      {images.map((imgState, i) => (
        <div key={i} style={{ width: slotW, maxHeight: 'calc(100vh - 220px)' }}>
          <ImageContainer
            state={imgState}
            slotIndex={i}
            onUpdate={(updater) => onUpdateImage(i, updater)}
            onSetImage={(state) => onSetImage(i, state)}
            onDelete={() => onDeleteImage(i)}
          />
        </div>
      ))}
    </div>
  );
}
