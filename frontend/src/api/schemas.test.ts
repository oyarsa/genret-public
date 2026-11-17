import { describe, expect, it } from "vitest";
import {
  PaperDetailSchema,
  PaperNodeSchema,
  RecommendResultSchema,
  SearchResponseSchema,
  SSEErrorSchema,
  SSEStatusSchema,
} from "./schemas";
import { paperNode, recommendResult } from "@/test/fixtures";

describe("PaperNodeSchema", () => {
  it("accepts a valid paper node", () => {
    expect(PaperNodeSchema.parse(paperNode())).toMatchObject({ corpusid: 1001 });
  });

  it("accepts null year and venue", () => {
    const node = paperNode({ year: null, venue: null });
    expect(PaperNodeSchema.parse(node).year).toBeNull();
  });

  it("rejects missing facet_scores", () => {
    const rest: Record<string, unknown> = { ...paperNode() };
    delete rest.facet_scores;
    expect(() => PaperNodeSchema.parse(rest)).toThrow();
  });

  it("rejects incomplete facet_scores", () => {
    const node = paperNode({
      facet_scores: { problems: 0.1 } as never,
    });
    expect(() => PaperNodeSchema.parse(node)).toThrow();
  });
});

describe("SearchResponseSchema", () => {
  it("accepts a full search response", () => {
    const resp = {
      papers: [paperNode()],
      edges: [{ source: 1, target: 2, facet: "methods", score: 0.5 }],
      query_id: "abc-123",
    };
    expect(SearchResponseSchema.parse(resp).query_id).toBe("abc-123");
  });

  it("rejects a response without query_id", () => {
    expect(() =>
      SearchResponseSchema.parse({ papers: [], edges: [] }),
    ).toThrow();
  });
});

describe("PaperDetailSchema", () => {
  it("accepts a detail with null citation count and url", () => {
    const detail = {
      corpusid: 1,
      title: "T",
      authors: [],
      year: null,
      venue: null,
      citation_count: null,
      url: null,
      facets: {},
    };
    expect(PaperDetailSchema.parse(detail).url).toBeNull();
  });
});

describe("RecommendResultSchema", () => {
  it("accepts a full recommend result", () => {
    const parsed = RecommendResultSchema.parse(recommendResult());
    expect(parsed.query_paper.source).toBe("in_corpus");
    expect(parsed.papers).toHaveLength(2);
  });

  it("rejects a result without query_paper", () => {
    const rest: Record<string, unknown> = { ...recommendResult() };
    delete rest.query_paper;
    expect(() => RecommendResultSchema.parse(rest)).toThrow();
  });
});

describe("SSE schemas", () => {
  it("parses status and error payloads", () => {
    expect(
      SSEStatusSchema.parse({ stage: "checking", message: "Looking up..." }).stage,
    ).toBe("checking");
    expect(SSEErrorSchema.parse({ message: "boom" }).message).toBe("boom");
  });

  it("rejects a status without a stage", () => {
    expect(() => SSEStatusSchema.parse({ message: "hi" })).toThrow();
  });
});
