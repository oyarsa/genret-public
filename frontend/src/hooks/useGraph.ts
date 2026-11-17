import { useCallback, useState } from "react";
import type {
  GraphState,
  PaperNode,
  ProvenanceEdge,
  FacetName,
} from "@/lib/types";
import { FACET_NAMES } from "@/lib/types";
import { expandFromPaper, expandNode } from "@/api/client";

const INITIAL_STATE: GraphState = {
  nodes: new Map(),
  edges: [],
  provenanceTree: new Map(),
  queryId: null,
  seedCorpusId: null,
  activeFacetFilters: new Set(FACET_NAMES as unknown as string[]),
};

export function useGraph() {
  const [state, setState] = useState<GraphState>(INITIAL_STATE);
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [isExpanding, setIsExpanding] = useState(false);
  const [expandError, setExpandError] = useState<string | null>(null);
  const [expandNotice, setExpandNotice] = useState<string | null>(null);

  const setSearchResults = useCallback(
    (
      papers: PaperNode[],
      edges: ProvenanceEdge[],
      queryId: string | null,
      seedCorpusId: number | null = null,
    ) => {
      const nodes = new Map<number, PaperNode>();
      for (const p of papers) {
        nodes.set(p.corpusid, p);
      }
      setState({
        nodes,
        edges,
        provenanceTree: new Map(),
        queryId,
        seedCorpusId,
        activeFacetFilters: new Set(FACET_NAMES as unknown as string[]),
      });
      setSelectedNodeId(null);
      setExpandError(null);
      setExpandNotice(null);
    },
    [],
  );

  const handleExpand = useCallback(
    async (corpusid: number, facet?: string) => {
      const isRecommendMode = state.seedCorpusId !== null;
      if (!isRecommendMode && !state.queryId) return;
      if (isExpanding) return;
      setIsExpanding(true);
      setExpandError(null);
      setExpandNotice(null);
      // Papers already on screen: the server drops them from the results
      // (so every expansion adds something new) and uses them to connect
      // the new papers to the existing graph.
      const known = Array.from(state.nodes.keys());
      try {
        const result = isRecommendMode
          ? await expandFromPaper(corpusid, facet, known)
          : await expandNode(corpusid, state.queryId!, facet, known);
        if (result.new_papers.length === 0) {
          setExpandNotice(
            facet
              ? `No further ${facet} neighbours for this paper.`
              : "No further related papers for this paper.",
          );
        }
        setState((prev) => {
          const nodes = new Map(prev.nodes);
          const newEdges = [...prev.edges];
          const provTree = new Map(prev.provenanceTree);

          for (const paper of result.new_papers) {
            if (!nodes.has(paper.corpusid)) {
              nodes.set(paper.corpusid, paper);
            }
          }
          for (const edge of result.new_edges) {
            newEdges.push(edge);
            // Record provenance: first edge that discovered this node
            if (!provTree.has(edge.target)) {
              provTree.set(edge.target, edge);
            }
          }

          return { ...prev, nodes, edges: newEdges, provenanceTree: provTree };
        });
      } catch (e) {
        // The query-vector cache holds 100 entries server-wide, so a
        // cached query can expire under other users' traffic; a fresh
        // search re-caches it.
        const message = e instanceof Error ? e.message : "Expand failed";
        setExpandError(
          message.includes("404")
            ? "This search session has expired — run the search again to keep expanding."
            : `Expanding failed: ${message}`,
        );
      } finally {
        setIsExpanding(false);
      }
    },
    [state.queryId, state.seedCorpusId, state.nodes, isExpanding],
  );

  const toggleFacetFilter = useCallback((facet: FacetName) => {
    setState((prev) => {
      const filters = new Set(prev.activeFacetFilters);
      if (filters.has(facet)) {
        filters.delete(facet);
      } else {
        filters.add(facet);
      }
      return { ...prev, activeFacetFilters: filters };
    });
  }, []);

  const getProvenanceTrail = useCallback(
    (corpusid: number): ProvenanceEdge[] => {
      const trail: ProvenanceEdge[] = [];
      let current = corpusid;
      const visited = new Set<number>();

      while (state.provenanceTree.has(current) && !visited.has(current)) {
        visited.add(current);
        const edge = state.provenanceTree.get(current)!;
        trail.unshift(edge);
        current = edge.source;
      }
      return trail;
    },
    [state.provenanceTree],
  );

  return {
    state,
    selectedNodeId,
    setSelectedNodeId,
    setSearchResults,
    handleExpand,
    toggleFacetFilter,
    getProvenanceTrail,
    isExpanding,
    expandError,
    expandNotice,
  };
}
