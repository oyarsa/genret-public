import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePaperLookup } from "./usePaperLookup";
import {
  getEdges,
  recommendByCorpusId,
  searchPapersByTitle,
  type RecommendCallbacks,
} from "@/api/client";
import {
  provenanceEdge,
  recommendResult,
  searchPaperResult,
} from "@/test/fixtures";

vi.mock("@/api/client");

const searchByTitleMock = vi.mocked(searchPapersByTitle);
const recommendMock = vi.mocked(recommendByCorpusId);
const getEdgesMock = vi.mocked(getEdges);

beforeEach(() => {
  searchByTitleMock.mockReset();
  recommendMock.mockReset();
  getEdgesMock.mockReset();
  getEdgesMock.mockResolvedValue({ edges: [] });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("suggestions", () => {
  it("debounces title lookups", async () => {
    vi.useFakeTimers();
    searchByTitleMock.mockResolvedValue({ papers: [searchPaperResult()] });
    const { result } = renderHook(() => usePaperLookup(vi.fn()));

    act(() => {
      result.current.onQueryChange("att");
      result.current.onQueryChange("atten");
      result.current.onQueryChange("attention");
    });
    expect(searchByTitleMock).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(250);
      await vi.runAllTimersAsync();
    });

    // Only the last keystroke survives the debounce
    expect(searchByTitleMock).toHaveBeenCalledTimes(1);
    expect(searchByTitleMock).toHaveBeenCalledWith("attention", 10, true);
    expect(result.current.suggestions).toHaveLength(1);
  });

  it("clears suggestions for blank queries", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => usePaperLookup(vi.fn()));

    act(() => result.current.onQueryChange("   "));
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(searchByTitleMock).not.toHaveBeenCalled();
    expect(result.current.suggestions).toEqual([]);
  });
});

describe("selectPaper", () => {
  function captureCallbacks(): {
    callbacks: () => RecommendCallbacks;
    close: ReturnType<typeof vi.fn>;
  } {
    const close = vi.fn();
    recommendMock.mockImplementation((_cid, cbs) => {
      captured = cbs;
      return close;
    });
    let captured: RecommendCallbacks | null = null;
    return {
      callbacks: () => {
        if (!captured) throw new Error("recommendByCorpusId not called");
        return captured;
      },
      close,
    };
  }

  it("streams status updates and adapts the result", async () => {
    const { callbacks } = captureCallbacks();
    const inter = provenanceEdge({ source: 1002, target: 1003 });
    getEdgesMock.mockResolvedValue({ edges: [inter] });
    const onResults = vi.fn();
    const { result } = renderHook(() => usePaperLookup(onResults));

    act(() => result.current.selectPaper(searchPaperResult({ corpusid: 2001 })));
    expect(result.current.isRecommending).toBe(true);

    act(() => callbacks().onStatus?.({ stage: "extracting", message: "Extracting..." }));
    expect(result.current.status).toBe("Extracting...");

    await act(async () => callbacks().onResult(recommendResult()));
    await waitFor(() => expect(result.current.isRecommending).toBe(false));

    expect(onResults).toHaveBeenCalledTimes(1);
    const [papers, edges, queryId, seed] = onResults.mock.calls[0]!;
    // Query paper first, then the recommendations
    expect(papers.map((p: { corpusid: number }) => p.corpusid)).toEqual([
      2001, 1002, 1003,
    ]);
    // Recommend mode: no cached query, seed drives expand instead
    expect(queryId).toBeNull();
    expect(seed).toBe(2001);
    // Query-anchored SSE edges (rescaled into 0.4-0.6) plus inter edges
    expect(edges).toHaveLength(3);
    const sseEdges = edges.filter((e: { source: number }) => e.source === 2001);
    for (const e of sseEdges) {
      expect(e.score).toBeGreaterThanOrEqual(0.4);
      expect(e.score).toBeLessThanOrEqual(0.6);
    }
    expect(result.current.status).toBeNull();
  });

  it("surfaces SSE errors and stops recommending", () => {
    const { callbacks } = captureCallbacks();
    const { result } = renderHook(() => usePaperLookup(vi.fn()));

    act(() => result.current.selectPaper(searchPaperResult()));
    act(() => callbacks().onError?.({ message: "not available" }));

    expect(result.current.error).toBe("not available");
    expect(result.current.isRecommending).toBe(false);
    expect(result.current.status).toBeNull();
  });

  it("recommendById works without a suggestion object (permalinks)", () => {
    captureCallbacks();
    const { result } = renderHook(() => usePaperLookup(vi.fn()));

    act(() => result.current.recommendById(4242));

    expect(recommendMock).toHaveBeenCalledWith(4242, expect.anything());
    expect(result.current.isRecommending).toBe(true);
  });

  it("carries matched facets through to the adapted nodes", async () => {
    const { callbacks } = captureCallbacks();
    const onResults = vi.fn();
    const { result } = renderHook(() => usePaperLookup(onResults));

    act(() => result.current.selectPaper(searchPaperResult()));
    await act(async () => callbacks().onResult(recommendResult()));
    await waitFor(() => expect(onResults).toHaveBeenCalled());

    const [papers] = onResults.mock.calls[0]!;
    const rec = papers.find((p: { corpusid: number }) => p.corpusid === 1002);
    expect(rec.matched_facets).toEqual(["problems", "methods"]);
  });

  it("cancels an in-flight recommend when a new paper is selected", () => {
    const { close } = captureCallbacks();
    const { result } = renderHook(() => usePaperLookup(vi.fn()));

    act(() => result.current.selectPaper(searchPaperResult({ corpusid: 1 })));
    act(() => result.current.selectPaper(searchPaperResult({ corpusid: 2 })));

    expect(close).toHaveBeenCalledTimes(1);
    expect(recommendMock).toHaveBeenCalledTimes(2);
  });

  it("closes the stream on unmount", () => {
    const { close } = captureCallbacks();
    const { result, unmount } = renderHook(() => usePaperLookup(vi.fn()));

    act(() => result.current.selectPaper(searchPaperResult()));
    unmount();

    expect(close).toHaveBeenCalled();
  });
});
