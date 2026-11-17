// Shared builders for frontend tests.
import type {
  FacetScores,
  PaperDetail,
  PaperNode,
  ProvenanceEdge,
  RecommendResult,
  SearchPaperResult,
} from "@/lib/types";

export function facetScores(overrides: Partial<FacetScores> = {}): FacetScores {
  return { problems: 0.1, methods: 0.2, results: 0.3, contributions: 0.4, ...overrides };
}

export function paperNode(overrides: Partial<PaperNode> = {}): PaperNode {
  return {
    corpusid: 1001,
    title: "Graph Paper 1001 on Attention Mechanisms",
    authors: ["Ada Author", "Ben Builder"],
    year: 2024,
    venue: "TestConf",
    facets: {
      problems: ["problems item 1 of paper 1001"],
      methods: ["methods item 1 of paper 1001"],
    },
    query_score: 0.5,
    facet_scores: facetScores(),
    matched_facets: [],
    ...overrides,
  };
}

export function paperDetail(overrides: Partial<PaperDetail> = {}): PaperDetail {
  return {
    corpusid: 1001,
    title: "Graph Paper 1001 on Attention Mechanisms",
    authors: ["Ada Author"],
    year: 2024,
    venue: "TestConf",
    citation_count: 42,
    url: "https://example.org/1001",
    facets: {},
    ...overrides,
  };
}

export function provenanceEdge(
  overrides: Partial<ProvenanceEdge> = {},
): ProvenanceEdge {
  return { source: 1001, target: 1002, facet: "methods", score: 0.8, ...overrides };
}

export function searchPaperResult(
  overrides: Partial<SearchPaperResult> = {},
): SearchPaperResult {
  return {
    corpusid: 1001,
    title: "Graph Paper 1001 on Attention Mechanisms",
    authors: ["Ada Author"],
    year: 2024,
    venue: "TestConf",
    source: "in_graph",
    ...overrides,
  };
}

export function recommendResult(
  overrides: Partial<RecommendResult> = {},
): RecommendResult {
  return {
    papers: [
      {
        corpusid: 1002,
        title: "Recommended Paper 1002",
        authors: ["Cara Coder"],
        year: 2023,
        venue: "TestConf",
        facets: { problems: ["p"] },
        score: 0.9,
        matched_facets: ["problems", "methods"],
      },
      {
        corpusid: 1003,
        title: "Recommended Paper 1003",
        authors: [],
        year: null,
        venue: null,
        facets: {},
        score: 0.4,
        matched_facets: [],
      },
    ],
    edges: [
      { target: 1002, facet: "methods", score: 0.9 },
      { target: 1003, facet: "problems", score: 0.45 },
    ],
    query_paper: {
      corpusid: 2001,
      title: "Corpus Paper 2001 on Neural Retrieval",
      authors: ["Ada Author"],
      source: "in_corpus",
      facets: { problems: ["q"] },
    },
    ...overrides,
  };
}

type SSEListener = (e: MessageEvent) => void;

/**
 * Minimal EventSource stand-in: jsdom has no EventSource, and the client
 * only uses addEventListener + close. Tests drive it by emitting events.
 */
export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  readonly url: string;
  closed = false;
  private listeners = new Map<string, SSEListener[]>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  static reset(): void {
    FakeEventSource.instances = [];
  }

  static get last(): FakeEventSource {
    const inst = FakeEventSource.instances.at(-1);
    if (!inst) throw new Error("No FakeEventSource instantiated");
    return inst;
  }

  addEventListener(type: string, listener: SSEListener): void {
    const existing = this.listeners.get(type) ?? [];
    this.listeners.set(type, [...existing, listener]);
  }

  close(): void {
    this.closed = true;
  }

  /** Emit a server-sent event with a JSON payload. */
  emit(type: string, data?: unknown): void {
    const event =
      data === undefined
        ? new Event(type)
        : new MessageEvent(type, { data: JSON.stringify(data) });
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event as MessageEvent);
    }
  }
}
