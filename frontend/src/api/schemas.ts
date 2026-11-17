import { z } from "zod";

export const FacetScoresSchema = z.object({
  problems: z.number(),
  methods: z.number(),
  results: z.number(),
  contributions: z.number(),
});

export const PaperNodeSchema = z.object({
  corpusid: z.number(),
  title: z.string(),
  authors: z.array(z.string()),
  year: z.number().nullable(),
  venue: z.string().nullable(),
  facets: z.record(z.string(), z.array(z.string())),
  query_score: z.number(),
  facet_scores: FacetScoresSchema,
  matched_facets: z.array(z.string()).default([]),
});

export const ProvenanceEdgeSchema = z.object({
  source: z.number(),
  target: z.number(),
  facet: z.string(),
  score: z.number(),
});

export const SearchResponseSchema = z.object({
  papers: z.array(PaperNodeSchema),
  edges: z.array(ProvenanceEdgeSchema),
  query_id: z.string(),
});

export const ExpandResponseSchema = z.object({
  new_papers: z.array(PaperNodeSchema),
  new_edges: z.array(ProvenanceEdgeSchema),
});

export const EdgesResponseSchema = z.object({
  edges: z.array(ProvenanceEdgeSchema),
});

export const PaperDetailSchema = z.object({
  corpusid: z.number(),
  title: z.string(),
  authors: z.array(z.string()),
  year: z.number().nullable(),
  venue: z.string().nullable(),
  citation_count: z.number().nullable(),
  url: z.string().nullable(),
  facets: z.record(z.string(), z.array(z.string())),
});

export const SearchPaperResultSchema = z.object({
  corpusid: z.number(),
  title: z.string(),
  authors: z.array(z.string()),
  year: z.number().nullable(),
  venue: z.string().nullable(),
  source: z.string(),
});

export const SearchPapersResponseSchema = z.object({
  papers: z.array(SearchPaperResultSchema),
});

export const RecommendPaperSchema = z.object({
  corpusid: z.number(),
  title: z.string(),
  authors: z.array(z.string()),
  year: z.number().nullable(),
  venue: z.string().nullable(),
  facets: z.record(z.string(), z.array(z.string())),
  score: z.number(),
  matched_facets: z.array(z.string()).default([]),
});

export const RecommendEdgeSchema = z.object({
  target: z.number(),
  facet: z.string(),
  score: z.number(),
});

export const QueryPaperSchema = z.object({
  corpusid: z.number(),
  title: z.string(),
  authors: z.array(z.string()),
  source: z.string(),
  facets: z.record(z.string(), z.array(z.string())),
});

export const RecommendResultSchema = z.object({
  papers: z.array(RecommendPaperSchema),
  edges: z.array(RecommendEdgeSchema),
  query_paper: QueryPaperSchema,
});

export const ExplainResponseSchema = z.object({
  explanation: z.string(),
});

export const SSEStatusSchema = z.object({
  stage: z.string(),
  message: z.string(),
});

export const SSEErrorSchema = z.object({
  message: z.string(),
});
