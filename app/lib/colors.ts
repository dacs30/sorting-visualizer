/** Shared color palette used across all visualizers. */
export const palette = {
  purple:  "#a090c8",
  blue:    "#3b90cc",
  orange:  "#c86030",
  green:   "#3a9a50",
  gold:    "#b89020",
  primary: "#7a6248",
} as const;

/** Semantic color aliases for sorting visualizations. */
export const sortColors = {
  unsorted:  palette.purple,
  comparing: palette.blue,
  swapping:  palette.orange,
  sorted:    palette.green,
  pivot:     palette.gold,
} as const;

/** Semantic color aliases for algorithm diagrams. */
export const diagramColors = {
  bar:     palette.purple,
  sorted:  palette.green,
  compare: palette.blue,
  swap:    palette.orange,
  pivot:   palette.gold,
  left:    "#4a80c4",
  right:   "#8060b0",
} as const;
