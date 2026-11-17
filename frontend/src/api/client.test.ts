import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  expandNode,
  explainConnection,
  getPaperDetail,
  recommendByCorpusId,
  searchPapers,
  searchPapersByTitle,
} from "./client";
import { FakeEventSource, paperNode, recommendResult } from "@/test/fixtures";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("EventSource", FakeEventSource);
  FakeEventSource.reset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

describe("searchPapers", () => {
  it("posts the query and parses the response", async () => {
    const body = { papers: [paperNode()], edges: [], query_id: "q-1" };
    fetchMock.mockResolvedValue(jsonResponse(body));

    const result = await searchPapers("attention", 5);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/search",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ query: "attention", top_k: 5 }),
      }),
    );
    expect(result.query_id).toBe("q-1");
  });

  it("throws on a non-OK status", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: "bad" }, 400));
    await expect(searchPapers("")).rejects.toThrow("API error: 400");
  });

  it("throws when the payload fails schema validation", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ nonsense: true }));
    await expect(searchPapers("attention")).rejects.toThrow();
  });
});

describe("expandNode", () => {
  it("sends null facet and an empty known set by default", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ new_papers: [], new_edges: [] }));
    await expandNode(1001, "q-1");
    const body = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
    expect(body).toEqual({
      corpusid: 1001,
      query_id: "q-1",
      facet: null,
      top_k: 5,
      known_corpusids: [],
    });
  });

  it("forwards the papers already on screen", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ new_papers: [], new_edges: [] }));
    await expandNode(1001, "q-1", "methods", [1001, 1002]);
    const body = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
    expect(body.known_corpusids).toEqual([1001, 1002]);
    expect(body.facet).toBe("methods");
  });
});

describe("getPaperDetail", () => {
  it("fetches by corpus id", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        corpusid: 7,
        title: "T",
        authors: [],
        year: null,
        venue: null,
        citation_count: null,
        url: null,
        facets: {},
        facet_scores: null,
      }),
    );
    const detail = await getPaperDetail(7);
    expect(fetchMock).toHaveBeenCalledWith("/api/paper/7");
    expect(detail.corpusid).toBe(7);
  });
});

describe("searchPapersByTitle", () => {
  it("defaults to local-only search", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ papers: [] }));
    await searchPapersByTitle("attention");
    const body = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
    expect(body.local_only).toBe(true);
  });
});

describe("explainConnection", () => {
  it("sends a query context", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ explanation: "Because reasons." }));
    const text = await explainConnection(1001, { query: "attention" });
    const body = JSON.parse(
      (fetchMock.mock.calls[0]![1] as RequestInit).body as string,
    );
    expect(body).toEqual({ corpusid: 1001, query: "attention", seed_corpusid: null });
    expect(text).toBe("Because reasons.");
  });

  it("sends a seed-paper context", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ explanation: "Seeded." }));
    await explainConnection(1001, { seedCorpusid: 2001 });
    const body = JSON.parse(
      (fetchMock.mock.calls[0]![1] as RequestInit).body as string,
    );
    expect(body).toEqual({ corpusid: 1001, query: null, seed_corpusid: 2001 });
  });
});

describe("recommendByCorpusId", () => {
  it("opens an EventSource on the recommend endpoint", () => {
    recommendByCorpusId(42, { onResult: vi.fn() });
    expect(FakeEventSource.last.url).toBe("/api/recommend/42");
  });

  it("delivers status then result, closing the source", () => {
    const onStatus = vi.fn();
    const onResult = vi.fn();
    recommendByCorpusId(42, { onStatus, onResult });
    const source = FakeEventSource.last;

    source.emit("status", { stage: "checking", message: "Looking up..." });
    expect(onStatus).toHaveBeenCalledWith({
      stage: "checking",
      message: "Looking up...",
    });

    source.emit("result", recommendResult());
    expect(onResult).toHaveBeenCalledTimes(1);
    expect(source.closed).toBe(true);
  });

  it("parses server-sent error events", () => {
    const onError = vi.fn();
    recommendByCorpusId(42, { onResult: vi.fn(), onError });
    const source = FakeEventSource.last;

    source.emit("error", { message: "paper not available" });
    expect(onError).toHaveBeenCalledWith({ message: "paper not available" });
    expect(source.closed).toBe(true);
  });

  it("reports transport failures as connection errors", () => {
    const onError = vi.fn();
    recommendByCorpusId(42, { onResult: vi.fn(), onError });
    const source = FakeEventSource.last;

    // A bare Event (no data) is what the browser fires on disconnect
    source.emit("error");
    expect(onError).toHaveBeenCalledWith({ message: "SSE connection error" });
    expect(source.closed).toBe(true);
  });

  it("ignores malformed result payloads but still closes", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const onResult = vi.fn();
    recommendByCorpusId(42, { onResult });
    const source = FakeEventSource.last;

    source.emit("result", { garbage: true });
    expect(onResult).not.toHaveBeenCalled();
    expect(source.closed).toBe(true);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("returns a cancel function that closes the source", () => {
    const cancel = recommendByCorpusId(42, { onResult: vi.fn() });
    cancel();
    expect(FakeEventSource.last.closed).toBe(true);
  });
});
