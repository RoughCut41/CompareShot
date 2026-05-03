export type Category =
  | 'wide'
  | 'ultrawide'
  | 'zoom'
  | 'portrait'
  | 'front'
  | 'low-light'
  | 'macro';

export const CATEGORIES: Category[] = [
  'wide',
  'ultrawide',
  'zoom',
  'portrait',
  'front',
  'low-light',
  'macro',
];

/**
 * State of a single image in a slot.
 * Pan values (panX, panY) are stored in CONTAINER PIXELS (screen coords).
 * The export pipeline scales them to 960x1625 using _containerW/_containerH.
 */
export interface ImageState {
  file: File;
  url: string; // object URL (always JPEG/PNG, HEIC is converted)
  zoom: number; // 0.1 .. 10
  panX: number; // px (in container space)
  panY: number; // px (in container space)
  rotation: number; // degrees
  flipH: boolean;
  flipV: boolean;
  freeRotateActive: boolean;
  _containerW: number; // last measured container width (px)
  _containerH: number; // last measured container height (px)
  // Original image natural dimensions, useful for export math
  naturalWidth: number;
  naturalHeight: number;
}

export interface Comparison {
  id: string;
  name: string;
  images: (ImageState | null)[]; // length 2..5
}

export type CategoryData = Record<Category, Comparison[]>;
export type ActiveIndexMap = Record<Category, number>;

export const MIN_SLOTS = 2;
export const MAX_SLOTS = 5;

export const EXPORT_WIDTH = 960;
export const EXPORT_HEIGHT = 1625;
export const EXPORT_ASPECT = EXPORT_WIDTH / EXPORT_HEIGHT;

export const SLOT_LETTERS = ['A', 'B', 'C', 'D', 'E'] as const;

export const DEFAULT_IMAGE_STATE: Omit<
  ImageState,
  'file' | 'url' | 'naturalWidth' | 'naturalHeight'
> = {
  zoom: 1,
  panX: 0,
  panY: 0,
  rotation: 0,
  flipH: false,
  flipV: false,
  freeRotateActive: false,
  _containerW: 0,
  _containerH: 0,
};
