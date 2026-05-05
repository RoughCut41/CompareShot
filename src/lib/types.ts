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

export interface ImageState {
  file: File;
  url: string;
  zoom: number;
  panX: number;
  panY: number;
  rotation: number;
  flipH: boolean;
  flipV: boolean;
  freeRotateActive: boolean;
  _containerW: number;
  _containerH: number;
  naturalWidth: number;
  naturalHeight: number;
}

export interface Comparison {
  id: string;
  name: string;
  images: (ImageState | null)[];
}

export type CategoryData = Record<Category, Comparison[]>;
export type ActiveIndexMap = Record<Category, number>;

export const MIN_SLOTS = 2;
export const MAX_SLOTS = 8;
export const EXPORT_WIDTH = 960;
export const EXPORT_HEIGHT = 1625;
export const EXPORT_ASPECT = EXPORT_WIDTH / EXPORT_HEIGHT;

export const SLOT_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] as const;

export const DEFAULT_IMAGE_STATE: Omit
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
