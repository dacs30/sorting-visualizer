import { sortColors } from "./colors";

/**
 * Generate an array of `size` unique integer heights evenly spread across
 * [min, max], then randomly shuffle them.
 */
export function generateArray(size: number, min = 8, max = 96): number[] {
  return Array.from({ length: size }, (_, i) =>
    Math.round(min + (i / (size - 1)) * (max - min)),
  ).sort(() => Math.random() - 0.5);
}

/**
 * Determine the display color for a bar at position `pos` given the
 * current step's highlighting sets.
 */
export function barColor(
  pos: number,
  comparing: Set<number>,
  swapping: Set<number>,
  sorted: Set<number>,
  pivot: number | undefined,
): string {
  if (sorted.has(pos))    return sortColors.sorted;
  if (swapping.has(pos))  return sortColors.swapping;
  if (pivot === pos)      return sortColors.pivot;
  if (comparing.has(pos)) return sortColors.comparing;
  return sortColors.unsorted;
}
