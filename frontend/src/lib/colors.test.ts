import { describe, expect, it } from "vitest";
import { dominantFacet, edgeColor, FACET_COLORS, nodeColor } from "./colors";

describe("dominantFacet", () => {
  it("picks the facet with the highest score", () => {
    expect(
      dominantFacet({ problems: 0.1, methods: 0.9, results: 0.2, contributions: 0 }),
    ).toBe("methods");
  });

  it("ignores unknown facet names", () => {
    expect(dominantFacet({ bogus: 99, results: 0.5 })).toBe("results");
  });

  it("falls back to problems for empty scores", () => {
    expect(dominantFacet({})).toBe("problems");
  });
});

describe("edgeColor", () => {
  it("returns the facet color for known facets", () => {
    expect(edgeColor("methods")).toBe(FACET_COLORS.methods);
  });

  it("falls back to gray for cross-facet edges", () => {
    expect(edgeColor("problems_x_contributions")).toBe("#6b7280");
  });
});

describe("nodeColor", () => {
  it("colors by the dominant facet", () => {
    expect(nodeColor({ problems: 1, methods: 0 })).toBe(FACET_COLORS.problems);
  });
});
