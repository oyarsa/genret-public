import { describe, expect, it } from "vitest";
import { fanPositions, focusTransform, nodeRadius } from "./graph";

const parent = { x: 500, y: 300 };
const viewport = { width: 800, height: 600 };

describe("fanPositions", () => {
  it("places every node at the given radius from the parent", () => {
    const slots = fanPositions(parent, { x: 100, y: 300 }, 5, 150);
    expect(slots).toHaveLength(5);
    for (const s of slots) {
      expect(Math.hypot(s.x - parent.x, s.y - parent.y)).toBeCloseTo(150, 5);
    }
  });

  it("fans away from the graph centroid", () => {
    // Centroid to the left of the parent, so the fan opens to the right
    const slots = fanPositions(parent, { x: 100, y: 300 }, 3, 150);
    for (const s of slots) {
      expect(s.x).toBeGreaterThan(parent.x);
    }
    // Mirrored centroid mirrors the fan
    const mirrored = fanPositions(parent, { x: 900, y: 300 }, 3, 150);
    for (const s of mirrored) {
      expect(s.x).toBeLessThan(parent.x);
    }
  });

  it("centres a single node on the outward direction", () => {
    const [only] = fanPositions(parent, { x: 100, y: 300 }, 1, 150);
    expect(only).toEqual({ x: 650, y: 300 });
  });

  it("separates adjacent nodes", () => {
    const slots = fanPositions(parent, { x: 100, y: 300 }, 5, 150);
    for (let i = 1; i < slots.length; i++) {
      const a = slots[i - 1]!;
      const b = slots[i]!;
      expect(Math.hypot(b.x - a.x, b.y - a.y)).toBeGreaterThan(40);
    }
  });

  it("is deterministic", () => {
    const a = fanPositions(parent, { x: 100, y: 300 }, 4);
    const b = fanPositions(parent, { x: 100, y: 300 }, 4);
    expect(a).toEqual(b);
  });

  it("handles a parent at the centroid without producing NaN", () => {
    const slots = fanPositions(parent, parent, 3, 150);
    for (const s of slots) {
      expect(Number.isFinite(s.x)).toBe(true);
      expect(Number.isFinite(s.y)).toBe(true);
    }
  });
});

describe("focusTransform", () => {
  const group = [
    { x: 1000, y: 800, r: 10 },
    { x: 1100, y: 900, r: 10 },
  ];

  it("pans at the current scale when the group fits", () => {
    const t = focusTransform(group, viewport, 1)!;
    expect(t.k).toBe(1);
    // The group's centre lands in the middle of the viewport
    expect(t.x + 1050 * t.k).toBeCloseTo(viewport.width / 2, 5);
    expect(t.y + 850 * t.k).toBeCloseTo(viewport.height / 2, 5);
  });

  it("never zooms in, even when the group is tiny", () => {
    const t = focusTransform([{ x: 10, y: 10, r: 5 }], viewport, 0.5)!;
    expect(t.k).toBe(0.5);
  });

  it("zooms out only when the group cannot fit", () => {
    const wide = [
      { x: 0, y: 0, r: 10 },
      { x: 4000, y: 0, r: 10 },
    ];
    const t = focusTransform(wide, viewport, 1)!;
    expect(t.k).toBeLessThan(1);
    // Still centred on the group
    expect(t.x + 2000 * t.k).toBeCloseTo(viewport.width / 2, 5);
  });

  it("returns null with nothing to focus or no viewport", () => {
    expect(focusTransform([], viewport, 1)).toBeNull();
    expect(focusTransform(group, { width: 0, height: 0 }, 1)).toBeNull();
  });
});

describe("nodeRadius", () => {
  it("grows with the query score", () => {
    expect(nodeRadius(0)).toBeLessThan(nodeRadius(1));
  });
});
