import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useGraph } from "./useGraph";
import { expandFromPaper, expandNode } from "@/api/client";
import { paperNode, provenanceEdge } from "@/test/fixtures";

vi.mock("@/api/client");

const expandNodeMock = vi.mocked(expandNode);
const expandFromPaperMock = vi.mocked(expandFromPaper);

beforeEach(() => {
  expandNodeMock.mockReset();
  expandFromPaperMock.mockReset();
});

function seedSearch(
  result: ReturnType<typeof renderHook<ReturnType<typeof useGraph>, unknown>>["result"],
  queryId: string | null = "q-1",
  seedCorpusId: number | null = null,
) {
  act(() => {
    result.current.setSearchResults(
      [paperNode({ corpusid: 1001 }), paperNode({ corpusid: 1002 })],
      [provenanceEdge()],
      queryId,
      seedCorpusId,
    );
  });
}

describe("useGraph", () => {
  it("stores search results and resets selection", () => {
    const { result } = renderHook(() => useGraph());
    act(() => result.current.setSelectedNodeId(1001));

    seedSearch(result);

    expect(result.current.state.nodes.size).toBe(2);
    expect(result.current.state.queryId).toBe("q-1");
    expect(result.current.selectedNodeId).toBeNull();
    expect(result.current.state.activeFacetFilters.size).toBe(4);
  });

  it("expands via /api/expand in text-search mode", async () => {
    expandNodeMock.mockResolvedValue({
      new_papers: [paperNode({ corpusid: 1003 })],
      new_edges: [provenanceEdge({ source: 1001, target: 1003 })],
    });
    const { result } = renderHook(() => useGraph());
    seedSearch(result);

    await act(() => result.current.handleExpand(1001, "methods"));

    // The papers already on screen are sent so the server can exclude them
    expect(expandNodeMock).toHaveBeenCalledWith(1001, "q-1", "methods", [1001, 1002]);
    expect(expandFromPaperMock).not.toHaveBeenCalled();
    expect(result.current.state.nodes.has(1003)).toBe(true);
    expect(result.current.state.edges).toHaveLength(2);
  });

  it("expands via /api/expand-from in recommend mode", async () => {
    expandFromPaperMock.mockResolvedValue({ new_papers: [], new_edges: [] });
    const { result } = renderHook(() => useGraph());
    seedSearch(result, null, 2001);

    await act(() => result.current.handleExpand(1001));

    expect(expandFromPaperMock).toHaveBeenCalledWith(1001, undefined, [1001, 1002]);
    expect(expandNodeMock).not.toHaveBeenCalled();
  });

  it("notices when the neighbourhood is exhausted", async () => {
    expandNodeMock.mockResolvedValue({ new_papers: [], new_edges: [] });
    const { result } = renderHook(() => useGraph());
    seedSearch(result);

    await act(() => result.current.handleExpand(1001, "methods"));

    expect(result.current.expandNotice).toBe(
      "No further methods neighbours for this paper.",
    );
    expect(result.current.expandError).toBeNull();
  });

  it("does nothing without a query id or seed", async () => {
    const { result } = renderHook(() => useGraph());
    seedSearch(result, null, null);

    await act(() => result.current.handleExpand(1001));

    expect(expandNodeMock).not.toHaveBeenCalled();
    expect(expandFromPaperMock).not.toHaveBeenCalled();
  });

  it("records provenance for newly discovered nodes only", async () => {
    const first = provenanceEdge({ source: 1001, target: 1003, facet: "methods" });
    const second = provenanceEdge({ source: 1002, target: 1003, facet: "results" });
    expandNodeMock.mockResolvedValue({
      new_papers: [paperNode({ corpusid: 1003 })],
      new_edges: [first, second],
    });
    const { result } = renderHook(() => useGraph());
    seedSearch(result);

    await act(() => result.current.handleExpand(1001));

    // The first edge that discovered 1003 wins
    expect(result.current.getProvenanceTrail(1003)).toEqual([first]);
  });

  it("surfaces a friendly message when the cached query has expired", async () => {
    expandNodeMock.mockRejectedValue(new Error("API error: 404 Not Found"));
    const { result } = renderHook(() => useGraph());
    seedSearch(result);

    await act(() => result.current.handleExpand(1001));

    expect(result.current.expandError).toMatch(/expired/);
    expect(result.current.isExpanding).toBe(false);
  });

  it("surfaces other expand failures verbatim", async () => {
    expandNodeMock.mockRejectedValue(new Error("API error: 500 Internal"));
    const { result } = renderHook(() => useGraph());
    seedSearch(result);

    await act(() => result.current.handleExpand(1001));

    expect(result.current.expandError).toContain("API error: 500");
  });

  it("clears the expand error on new results", async () => {
    expandNodeMock.mockRejectedValue(new Error("API error: 404 Not Found"));
    const { result } = renderHook(() => useGraph());
    seedSearch(result);
    await act(() => result.current.handleExpand(1001));
    expect(result.current.expandError).not.toBeNull();

    seedSearch(result);
    expect(result.current.expandError).toBeNull();
  });

  it("toggles facet filters", () => {
    const { result } = renderHook(() => useGraph());
    seedSearch(result);

    act(() => result.current.toggleFacetFilter("methods"));
    expect(result.current.state.activeFacetFilters.has("methods")).toBe(false);

    act(() => result.current.toggleFacetFilter("methods"));
    expect(result.current.state.activeFacetFilters.has("methods")).toBe(true);
  });
});
