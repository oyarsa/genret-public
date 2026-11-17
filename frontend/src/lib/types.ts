import { z } from "zod";
import {
  FacetScoresSchema,
  PaperNodeSchema,
  ProvenanceEdgeSchema,
  SearchResponseSchema,
  ExpandResponseSchema,
  EdgesResponseSchema,
  PaperDetailSchema,
  SearchPaperResultSchema,
  SearchPapersResponseSchema,
  RecommendResultSchema,
  SSEStatusSchema,
  SSEErrorSchema,
} from "@/api/schemas";

export type FacetScores = z.infer<typeof FacetScoresSchema>;
export type PaperNode = z.infer<typeof PaperNodeSchema>;
export type ProvenanceEdge = z.infer<typeof ProvenanceEdgeSchema>;
export type SearchResponse = z.infer<typeof SearchResponseSchema>;
export type ExpandResponse = z.infer<typeof ExpandResponseSchema>;
export type EdgesResponse = z.infer<typeof EdgesResponseSchema>;
export type PaperDetail = z.infer<typeof PaperDetailSchema>;
export type SearchPaperResult = z.infer<typeof SearchPaperResultSchema>;
export type SearchPapersResponse = z.infer<typeof SearchPapersResponseSchema>;
export type RecommendResult = z.infer<typeof RecommendResultSchema>;
export type SSEStatus = z.infer<typeof SSEStatusSchema>;
export type SSEError = z.infer<typeof SSEErrorSchema>;

export const FACET_NAMES = [
  "problems",
  "methods",
  "results",
  "contributions",
] as const;
export type FacetName = (typeof FACET_NAMES)[number];

export interface GraphState {
  nodes: Map<number, PaperNode>;
  edges: ProvenanceEdge[];
  provenanceTree: Map<number, ProvenanceEdge>;
  /**
   * Session token returned by /api/search; used by /api/expand to look
   * up the cached query vectors. Null when the graph came from the
   * paper-anchored recommend flow (which has no cached query).
   */
  queryId: string | null;
  /**
   * Corpus id of the seed paper, set when the graph came from the
   * paper-anchored recommend flow. Null in text-search mode. Drives
   * both the expand dispatch (use /api/expand-from when set) and UI
   * affordances that need to identify the seed node.
   */
  seedCorpusId: number | null;
  activeFacetFilters: Set<string>;
}
