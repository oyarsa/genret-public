import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSearch } from "./useSearch";
import { searchPapers } from "@/api/client";
import { paperNode } from "@/test/fixtures";

vi.mock("@/api/client");

const searchPapersMock = vi.mocked(searchPapers);

beforeEach(() => {
  searchPapersMock.mockReset();
});

describe("useSearch", () => {
  it("delivers results to the callback", async () => {
    const response = {
      papers: [paperNode()],
      edges: [],
      query_id: "q-1",
    };
    searchPapersMock.mockResolvedValue(response);
    const onResults = vi.fn();
    const { result } = renderHook(() => useSearch(onResults));

    await act(() => result.current.search("attention"));

    expect(onResults).toHaveBeenCalledWith(response.papers, [], "q-1");
    expect(result.current.isSearching).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("ignores blank queries", async () => {
    const onResults = vi.fn();
    const { result } = renderHook(() => useSearch(onResults));

    await act(() => result.current.search("   "));

    expect(searchPapersMock).not.toHaveBeenCalled();
    expect(onResults).not.toHaveBeenCalled();
  });

  it("captures API failures as error state", async () => {
    searchPapersMock.mockRejectedValue(new Error("API error: 500"));
    const onResults = vi.fn();
    const { result } = renderHook(() => useSearch(onResults));

    await act(() => result.current.search("attention"));

    expect(result.current.error).toBe("API error: 500");
    expect(onResults).not.toHaveBeenCalled();
    expect(result.current.isSearching).toBe(false);
  });
});
