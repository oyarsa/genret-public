import type { FacetName } from "./types";

/** Primary facet colors — deep hues with enough contrast against light surfaces. */
export const FACET_COLORS: Record<FacetName, string> = {
  problems: "#2b5cd6",
  methods: "#0f9d6b",
  results: "#cf6b14",
  contributions: "#8b3fcf",
};

/** Dimmed versions for backgrounds and inactive states. */
export const FACET_COLORS_DIM: Record<FacetName, string> = {
  problems: "#2b5cd630",
  methods: "#0f9d6b30",
  results: "#cf6b1430",
  contributions: "#8b3fcf30",
};

/** Glow colors for SVG filters and box-shadows. */
export const FACET_GLOW: Record<FacetName, string> = {
  problems: "#2b5cd666",
  methods: "#0f9d6b66",
  results: "#cf6b1466",
  contributions: "#8b3fcf66",
};

/** Get the CSS color for a facet edge. Falls back to gray for cross-facet edges. */
export function edgeColor(facet: string): string {
  return FACET_COLORS[facet as FacetName] ?? "#6b7280";
}

/** Determine dominant facet from facet scores. */
export function dominantFacet(scores: Record<string, number>): FacetName {
  let best: FacetName = "problems";
  let bestScore = -Infinity;
  for (const [name, score] of Object.entries(scores)) {
    if (score > bestScore && name in FACET_COLORS) {
      bestScore = score;
      best = name as FacetName;
    }
  }
  return best;
}

/** Get node fill color based on dominant facet score. */
export function nodeColor(scores: Record<string, number>): string {
  return FACET_COLORS[dominantFacet(scores)];
}

/** Get node glow color based on dominant facet. */
export function nodeGlow(scores: Record<string, number>): string {
  return FACET_GLOW[dominantFacet(scores)];
}
