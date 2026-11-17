import {
  SearchResponseSchema,
  ExpandResponseSchema,
  EdgesResponseSchema,
  ExplainResponseSchema,
  PaperDetailSchema,
  SearchPapersResponseSchema,
  RecommendResultSchema,
  SSEStatusSchema,
  SSEErrorSchema,
} from "./schemas";
import type {
  SearchResponse,
  ExpandResponse,
  EdgesResponse,
  PaperDetail,
  SearchPapersResponse,
  RecommendResult,
  SSEStatus,
  SSEError,
} from "@/lib/types";

async function post<T>(
  url: string,
  body: unknown,
  schema: { parse: (data: unknown) => T },
): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  const data: unknown = await res.json();
  return schema.parse(data);
}

async function get<T>(
  url: string,
  schema: { parse: (data: unknown) => T },
): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  const data: unknown = await res.json();
  return schema.parse(data);
}

export async function searchPapers(
  query: string,
  topK: number = 8,
): Promise<SearchResponse> {
  return post("/api/search", { query, top_k: topK }, SearchResponseSchema);
}

export async function expandNode(
  corpusid: number,
  queryId: string,
  facet?: string,
  knownCorpusids: number[] = [],
  topK: number = 5,
): Promise<ExpandResponse> {
  return post(
    "/api/expand",
    {
      corpusid,
      query_id: queryId,
      facet: facet ?? null,
      top_k: topK,
      known_corpusids: knownCorpusids,
    },
    ExpandResponseSchema,
  );
}

export async function expandFromPaper(
  corpusid: number,
  facet?: string,
  knownCorpusids: number[] = [],
  topK: number = 5,
): Promise<ExpandResponse> {
  return post(
    "/api/expand-from",
    {
      corpusid,
      facet: facet ?? null,
      top_k: topK,
      known_corpusids: knownCorpusids,
    },
    ExpandResponseSchema,
  );
}

export async function getEdges(
  corpusids: number[],
): Promise<EdgesResponse> {
  return post("/api/edges", { corpusids }, EdgesResponseSchema);
}

export async function getPaperDetail(
  corpusid: number,
): Promise<PaperDetail> {
  return get(`/api/paper/${corpusid}`, PaperDetailSchema);
}

export async function searchPapersByTitle(
  query: string,
  limit: number = 10,
  localOnly: boolean = true,
): Promise<SearchPapersResponse> {
  return post(
    "/api/search-papers",
    { query, limit, local_only: localOnly },
    SearchPapersResponseSchema,
  );
}

export type ExplainContext =
  | { query: string }
  | { seedCorpusid: number };

export async function explainConnection(
  corpusid: number,
  context: ExplainContext,
): Promise<string> {
  const body = {
    corpusid,
    query: "query" in context ? context.query : null,
    seed_corpusid: "seedCorpusid" in context ? context.seedCorpusid : null,
  };
  const result = await post("/api/explain", body, ExplainResponseSchema);
  return result.explanation;
}

export interface RecommendCallbacks {
  onStatus?: (status: SSEStatus) => void;
  onResult: (result: RecommendResult) => void;
  onError?: (error: SSEError) => void;
}

export function recommendByCorpusId(
  corpusid: number,
  callbacks: RecommendCallbacks,
): () => void {
  const source = new EventSource(`/api/recommend/${corpusid}`);

  source.addEventListener("status", (e) => {
    try {
      const parsed = SSEStatusSchema.parse(JSON.parse(e.data));
      callbacks.onStatus?.(parsed);
    } catch (err) {
      console.warn("Failed to parse status event", err);
    }
  });

  source.addEventListener("result", (e) => {
    try {
      const parsed = RecommendResultSchema.parse(JSON.parse(e.data));
      callbacks.onResult(parsed);
    } catch (err) {
      console.warn("Failed to parse result event", err);
    } finally {
      source.close();
    }
  });

  source.addEventListener("error", (e) => {
    if (e instanceof MessageEvent && typeof e.data === "string") {
      try {
        const parsed = SSEErrorSchema.parse(JSON.parse(e.data));
        callbacks.onError?.(parsed);
      } catch {
        callbacks.onError?.({ message: "SSE stream error" });
      }
    } else {
      callbacks.onError?.({ message: "SSE connection error" });
    }
    source.close();
  });

  return () => source.close();
}
