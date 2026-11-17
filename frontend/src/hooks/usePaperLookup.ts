import { useCallback, useEffect, useRef, useState } from "react";
import {
  getEdges,
  recommendByCorpusId,
  searchPapersByTitle,
} from "@/api/client";
import type {
  PaperNode,
  ProvenanceEdge,
  RecommendResult,
  SearchPaperResult,
} from "@/lib/types";

const DEBOUNCE_MS = 250;

function emptyFacetScores() {
  return { problems: 0, methods: 0, results: 0, contributions: 0 };
}

function recommendToSearchShape(result: RecommendResult): {
  papers: PaperNode[];
  edges: ProvenanceEdge[];
  queryId: string | null;
  seedCorpusId: number;
} {
  // Match the query-node radius to the top retrieved (with a tiny
  // bump so the "top result" dashed ring marks only the query, not
  // the highest-scoring neighbour). Layout will still centre the
  // query naturally from its edge count.
  const maxScore = result.papers.reduce(
    (m, p) => (p.score > m ? p.score : m),
    0,
  );
  const queryNode: PaperNode = {
    corpusid: result.query_paper.corpusid,
    title: result.query_paper.title,
    authors: result.query_paper.authors,
    year: null,
    venue: null,
    facets: result.query_paper.facets,
    query_score: maxScore + 1e-6,
    facet_scores: emptyFacetScores(),
    matched_facets: [],
  };

  const recommendedNodes: PaperNode[] = result.papers.map((p) => ({
    corpusid: p.corpusid,
    title: p.title,
    authors: p.authors,
    year: p.year,
    venue: p.venue,
    facets: p.facets,
    query_score: p.score,
    facet_scores: emptyFacetScores(),
    matched_facets: p.matched_facets,
  }));

  // Build query-anchored edges from the SSE result. /api/edges only
  // returns query-anchored edges when the query is in the retrieval
  // graph (in_graph), so for in_corpus / out_of_corpus queries this is
  // the only source. Rescale the SSE scores into the facet-edge range
  // so edge opacity is comparable to the inter-paper edges that come
  // from /api/edges. The rescale is a visualization transform only.
  const sseScores = result.edges.map((e) => e.score);
  const sseMax = sseScores.length > 0 ? Math.max(...sseScores) : 1;
  const TARGET_MAX = 0.6;
  const TARGET_MIN = 0.4;
  const queryEdges: ProvenanceEdge[] = result.edges.map((e) => ({
    source: result.query_paper.corpusid,
    target: e.target,
    facet: e.facet,
    score: sseMax > 0
      ? TARGET_MIN + (e.score / sseMax) * (TARGET_MAX - TARGET_MIN)
      : TARGET_MIN,
  }));

  // queryId is null in recommend mode: no /api/search query vectors
  // were cached, so /api/expand can't be called. seedCorpusId is the
  // anchor that drives expand dispatch and seed-node UI affordances.
  return {
    papers: [queryNode, ...recommendedNodes],
    edges: queryEdges,
    queryId: null,
    seedCorpusId: result.query_paper.corpusid,
  };
}

export function usePaperLookup(
  onResults: (
    papers: PaperNode[],
    edges: ProvenanceEdge[],
    queryId: string | null,
    seedCorpusId?: number | null,
  ) => void,
) {
  const [suggestions, setSuggestions] = useState<SearchPaperResult[]>([]);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [isRecommending, setIsRecommending] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  const fetchSuggestions = useCallback(async (q: string) => {
    if (!q.trim()) {
      setSuggestions([]);
      return;
    }
    setIsSuggesting(true);
    try {
      // local_only=true: substring match against the explorer's loaded
      // metadata (in-graph + in-corpus papers) without hitting the
      // Semantic Scholar API. Cheaper and lower-latency for autocomplete
      // on each keystroke. Pass local_only=false if the UI ever needs
      // to suggest papers that are not yet in the corpus.
      const res = await searchPapersByTitle(q, 10, true);
      setSuggestions(res.papers);
    } catch (e) {
      console.warn("Paper lookup failed", e);
      setSuggestions([]);
    } finally {
      setIsSuggesting(false);
    }
  }, []);

  const onQueryChange = useCallback(
    (q: string) => {
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current);
      }
      debounceRef.current = window.setTimeout(() => {
        fetchSuggestions(q);
      }, DEBOUNCE_MS);
    },
    [fetchSuggestions],
  );

  const recommendById = useCallback(
    (corpusid: number) => {
      // Cancel any in-flight recommend
      if (abortRef.current) {
        abortRef.current();
        abortRef.current = null;
      }
      setError(null);
      setStatus("Starting…");
      setIsRecommending(true);
      const close = recommendByCorpusId(corpusid, {
        onStatus: (s) => setStatus(s.message),
        onResult: (result) => {
          void (async () => {
            const adapted = recommendToSearchShape(result);
            // Pull inter-paper edges (result <-> result) to give the graph
            // the same density as the text-search mode.
            const cids = adapted.papers.map((p) => p.corpusid);
            try {
              const inter = await getEdges(cids);
              adapted.edges = [...adapted.edges, ...inter.edges];
            } catch (err) {
              console.warn("Failed to fetch inter-paper edges", err);
            }
            onResults(
              adapted.papers,
              adapted.edges,
              adapted.queryId,
              adapted.seedCorpusId,
            );
            setStatus(null);
            setIsRecommending(false);
            abortRef.current = null;
          })();
        },
        onError: (e) => {
          setError(e.message);
          setStatus(null);
          setIsRecommending(false);
          abortRef.current = null;
        },
      });
      abortRef.current = close;
    },
    [onResults],
  );

  const selectPaper = useCallback(
    (paper: SearchPaperResult) => recommendById(paper.corpusid),
    [recommendById],
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current);
      }
      if (abortRef.current) {
        abortRef.current();
      }
    };
  }, []);

  const clearSuggestions = useCallback(() => setSuggestions([]), []);

  return {
    suggestions,
    isSuggesting,
    isRecommending,
    status,
    error,
    onQueryChange,
    selectPaper,
    recommendById,
    clearSuggestions,
  };
}
