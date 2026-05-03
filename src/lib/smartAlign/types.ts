import { ImageState } from '@/lib/types';

/**
 * The transform we compute and apply to a non-reference image so that
 * it lines up with the reference. Stored in CONTAINER pixels (same coordinate
 * system as ImageState.panX / panY).
 */
export interface AlignTransform {
  zoom: number; // multiplier on top of the natural cover-fit zoom
  panX: number; // container px
  panY: number; // container px
  rotation: number; // degrees
}

/**
 * Per-image result of a Smart Align run.
 */
export interface AlignResult {
  slotIndex: number;
  status: 'reference' | 'aligned' | 'failed' | 'skipped';
  reason?: string;
  transform?: AlignTransform;
}

/**
 * Final result returned by smartAlign.
 */
export interface SmartAlignReport {
  mode: 'face' | 'feature';
  referenceSlotIndex: number;
  results: AlignResult[];
}

/**
 * Slot input passed into smartAlign — pairs the slot index with its image state.
 */
export interface AlignableSlot {
  slotIndex: number;
  state: ImageState;
}

/**
 * Progress callback used to update the toolbar button label.
 */
export type ProgressCallback = (label: string) => void;
