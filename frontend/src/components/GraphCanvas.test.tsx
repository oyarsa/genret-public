import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import GraphCanvas from "./GraphCanvas";
import { FACET_NAMES } from "@/lib/types";
import { paperNode, provenanceEdge } from "@/test/fixtures";

function renderCanvas(overrides: Partial<Parameters<typeof GraphCanvas>[0]> = {}) {
  const a = paperNode({ corpusid: 1001, title: "Paper Alpha" });
  const b = paperNode({ corpusid: 1002, title: "Paper Beta" });
  const props = {
    nodes: new Map([
      [a.corpusid, a],
      [b.corpusid, b],
    ]),
    edges: [provenanceEdge({ source: 1001, target: 1002 })],
    activeFacets: new Set<string>(FACET_NAMES),
    selectedNodeId: null,
    maxEdgesPerNode: 5,
    onNodeClick: vi.fn(),
    ...overrides,
  };
  const view = render(<GraphCanvas {...props} />);
  return { props, view };
}

describe("GraphCanvas", () => {
  it("renders one accessible circle per paper", () => {
    const { view } = renderCanvas();
    const circles = view.container.querySelectorAll("circle.node-circle");
    expect(circles).toHaveLength(2);
    const labels = [...circles].map((c) => c.getAttribute("aria-label"));
    expect(labels).toEqual(
      expect.arrayContaining(["Paper Alpha", "Paper Beta"]),
    );
    for (const circle of circles) {
      expect(circle.getAttribute("role")).toBe("button");
    }
  });

  it("draws edges between connected papers", () => {
    const { view } = renderCanvas();
    expect(view.container.querySelectorAll("line").length).toBeGreaterThan(0);
  });

  it("reports node clicks", () => {
    const { props, view } = renderCanvas();
    const alpha = view.container.querySelector(
      'circle[aria-label="Paper Alpha"]',
    );
    expect(alpha).not.toBeNull();
    alpha!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(props.onNodeClick).toHaveBeenCalledWith(1001);
  });

  it("marks the selected node with the stronger glow", () => {
    const { view } = renderCanvas({ selectedNodeId: 1002 });
    const beta = view.container.querySelector('circle[aria-label="Paper Beta"]');
    expect(beta!.getAttribute("filter")).toBe("url(#selected-glow)");
    const alpha = view.container.querySelector(
      'circle[aria-label="Paper Alpha"]',
    );
    expect(alpha!.getAttribute("filter")).toBe("url(#node-glow)");
  });

  it("hides edges whose facet is filtered out", () => {
    const { view } = renderCanvas({
      activeFacets: new Set(["problems"]),
    });
    // The only edge is a methods edge, so nothing is visible
    expect(view.container.querySelectorAll("line")).toHaveLength(0);
  });
});
