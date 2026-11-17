import { useCallback, useEffect, useMemo, useState } from "react";
import SearchBar, { type SearchMode } from "./SearchBar";
import GraphCanvas from "./GraphCanvas";
import NodeDetail from "./NodeDetail";
import FacetFilter from "./FacetFilter";
import { useGraph } from "@/hooks/useGraph";
import { useSearch } from "@/hooks/useSearch";
import { usePaperLookup } from "@/hooks/usePaperLookup";
import {
  explainConnection,
  getPaperDetail,
  type ExplainContext,
} from "@/api/client";
import type { PaperDetail, SearchPaperResult } from "@/lib/types";

function getInitialQuery(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get("q") ?? "";
}

function getInitialPaper(): number | null {
  const raw = new URLSearchParams(window.location.search).get("paper");
  const cid = raw ? Number(raw) : NaN;
  return Number.isInteger(cid) && cid > 0 ? cid : null;
}

function setUrlParam(key: "q" | "paper", value: string | null) {
  const url = new URL(window.location.href);
  url.searchParams.delete("q");
  url.searchParams.delete("paper");
  if (value) {
    url.searchParams.set(key, value);
  }
  window.history.replaceState(null, "", url);
}

export default function App() {
  const {
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
  } = useGraph();

  const [maxEdgesPerNode, setMaxEdgesPerNode] = useState(5);
  const [mode, setMode] = useState<SearchMode>(
    getInitialPaper() !== null ? "paper" : "text",
  );
  const { search, isSearching, error } = useSearch(setSearchResults);
  const {
    suggestions,
    isSuggesting,
    isRecommending,
    status: recommendStatus,
    error: recommendError,
    onQueryChange,
    selectPaper,
    recommendById,
    clearSuggestions,
  } = usePaperLookup(setSearchResults);

  // The active query text, for URL sync and explanation context
  const [lastQuery, setLastQuery] = useState(getInitialQuery());

  // Sync query to URL on search
  const handleSearch = useCallback(
    (query: string) => {
      setUrlParam("q", query.trim() || null);
      setLastQuery(query.trim());
      search(query);
    },
    [search],
  );

  // Sync the seed paper to the URL on recommend, so views are shareable
  const handlePaperSelect = useCallback(
    (paper: SearchPaperResult) => {
      setUrlParam("paper", String(paper.corpusid));
      selectPaper(paper);
    },
    [selectPaper],
  );

  // Auto-search / auto-recommend from URL on mount
  useEffect(() => {
    const paper = getInitialPaper();
    if (paper !== null) {
      recommendById(paper);
      return;
    }
    const q = getInitialQuery();
    if (q) {
      search(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch full metadata (paper link, citation count) for the selected node
  const [selectedDetail, setSelectedDetail] = useState<PaperDetail | null>(null);
  useEffect(() => {
    setSelectedDetail(null);
    if (selectedNodeId === null) return;
    let stale = false;
    getPaperDetail(selectedNodeId)
      .then((detail) => {
        if (!stale) setSelectedDetail(detail);
      })
      .catch(() => {
        // Non-essential enrichment; the panel works without it
      });
    return () => {
      stale = true;
    };
  }, [selectedNodeId]);

  // On-demand rich explanation for the selected node's connection
  const [richExplanation, setRichExplanation] = useState<string | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  useEffect(() => {
    setRichExplanation(null);
    setIsExplaining(false);
  }, [selectedNodeId]);

  const isSeedSelected =
    selectedNodeId !== null && selectedNodeId === state.seedCorpusId;
  const explainContext: ExplainContext | null = useMemo(
    () =>
      state.seedCorpusId !== null
        ? { seedCorpusid: state.seedCorpusId }
        : lastQuery
          ? { query: lastQuery }
          : null,
    [state.seedCorpusId, lastQuery],
  );

  // Spell out what an explanation would be relative to
  const explainContextLabel = useMemo(() => {
    if (state.seedCorpusId !== null) {
      const seed = state.nodes.get(state.seedCorpusId);
      const title = seed?.title ?? "the seed paper";
      return title.length > 44 ? `“${title.slice(0, 41)}…”` : `“${title}”`;
    }
    if (!lastQuery) return null;
    return `your search “${lastQuery}”`;
  }, [state.seedCorpusId, state.nodes, lastQuery]);

  const handleExplain = useCallback(() => {
    if (selectedNodeId === null || explainContext === null) return;
    setIsExplaining(true);
    explainConnection(selectedNodeId, explainContext)
      .then(setRichExplanation)
      .catch(() => setRichExplanation("Explanation unavailable right now."))
      .finally(() => setIsExplaining(false));
  }, [selectedNodeId, explainContext]);

  // Escape key dismisses detail panel
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectedNodeId !== null) {
        setSelectedNodeId(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedNodeId, setSelectedNodeId]);

  const selectedPaper =
    selectedNodeId !== null ? state.nodes.get(selectedNodeId) ?? null : null;

  const trail =
    selectedNodeId !== null ? getProvenanceTrail(selectedNodeId) : [];

  // In recommend mode the seed paper itself is typically in_corpus
  // (not in_graph), so expanding from it would 404 server-side. Use
  // the structured seedCorpusId from GraphState to disable the action
  // for that one node.


  return (
    <div className="flex h-screen flex-col" style={{ background: "var(--color-surface-0)" }}>
      {/* Header */}
      <header
        className="relative z-10 px-5 py-3"
        style={{
          background: "var(--color-surface-1)",
          borderBottom: "1px solid var(--color-surface-3)",
        }}
      >
        <div className="mx-auto flex max-w-7xl items-center gap-5">
          <h1
            className="whitespace-nowrap text-lg tracking-tight"
            style={{
              fontFamily: "var(--font-display)",
              fontWeight: 500,
              color: "var(--color-text-primary)",
              fontOpticalSizing: "auto",
            }}
          >
            Graft
          </h1>
          <div className="flex-1">
            <SearchBar
              mode={mode}
              onModeChange={setMode}
              onTextSearch={handleSearch}
              isSearching={isSearching}
              isRecommending={isRecommending}
              status={recommendStatus}
              initialQuery={getInitialQuery()}
              suggestions={suggestions}
              isSuggesting={isSuggesting}
              onQueryChange={onQueryChange}
              onPaperSelect={handlePaperSelect}
              onClearSuggestions={clearSuggestions}
            />
          </div>
        </div>
        {(error || recommendError || expandError || expandNotice) && (
          <div
            className="mx-auto mt-2 max-w-7xl text-sm"
            style={{
              color:
                error || recommendError || expandError
                  ? "#ef6b6b"
                  : "var(--color-text-muted)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.75rem",
            }}
          >
            {error ?? recommendError ?? expandError ?? expandNotice}
          </div>
        )}
      </header>

      {/* Body */}
      <div className="flex min-h-0 flex-1">
        {/* Graph area */}
        <div className="graph-bg relative flex-1">
          {state.nodes.size === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-3">
              <div
                className="empty-pulse text-center"
                style={{
                  fontFamily: "var(--font-display)",
                  fontSize: "1.25rem",
                  fontWeight: 300,
                  color: "var(--color-text-muted)",
                  letterSpacing: "-0.01em",
                }}
              >
                Search for papers to explore the graph
              </div>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.65rem",
                  color: "var(--color-text-muted)",
                  opacity: 0.5,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                }}
              >
                click node to inspect &middot; expand from sidebar
              </div>
            </div>
          ) : (
            <>
              <GraphCanvas
                nodes={state.nodes}
                edges={state.edges}
                activeFacets={state.activeFacetFilters}
                selectedNodeId={selectedNodeId}
                maxEdgesPerNode={maxEdgesPerNode}
                onNodeClick={setSelectedNodeId}
              />
              {/* Facet filter + edge density overlay */}
              <div className="absolute bottom-4 left-4 flex flex-col gap-2">
                <FacetFilter
                  activeFacets={state.activeFacetFilters}
                  onToggle={toggleFacetFilter}
                />
                <div
                  className="flex items-center gap-2 rounded px-2 py-1"
                  style={{
                    background: "var(--color-surface-2)",
                    fontSize: "0.65rem",
                    color: "var(--color-text-secondary)",
                    fontFamily: "var(--font-mono)",
                    letterSpacing: "0.04em",
                  }}
                >
                  <span>edges</span>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    value={maxEdgesPerNode}
                    onChange={(e) => setMaxEdgesPerNode(Number(e.target.value))}
                    style={{ width: 80, accentColor: "var(--color-text-muted)" }}
                  />
                  <span style={{ minWidth: "1.2em", textAlign: "right" }}>
                    {maxEdgesPerNode}
                  </span>
                </div>
              </div>
              {/* Stats overlay */}
              <div
                className="stats-overlay absolute top-4 left-4"
                style={{
                  fontSize: "0.65rem",
                  color: "var(--color-text-muted)",
                  letterSpacing: "0.04em",
                }}
              >
                {state.nodes.size} papers &middot; {state.edges.length} edges
              </div>
            </>
          )}
        </div>

        {/* Detail panel */}
        {selectedPaper && (
          <div
            className="detail-panel relative w-[420px]"
            style={{
              background: "var(--color-surface-1)",
              borderLeft: "1px solid var(--color-surface-3)",
            }}
          >
            <NodeDetail
              paper={selectedPaper}
              detail={selectedDetail}
              trail={trail}
              allNodes={state.nodes}
              onExpand={handleExpand}
              isExpanding={isExpanding}
              onClose={() => setSelectedNodeId(null)}
              expandDisabled={selectedPaper.corpusid === state.seedCorpusId}
              onExplain={
                explainContext !== null && !isSeedSelected
                  ? handleExplain
                  : undefined
              }
              explainContextLabel={isSeedSelected ? null : explainContextLabel}
              richExplanation={richExplanation}
              isExplaining={isExplaining}
            />
          </div>
        )}
      </div>
    </div>
  );
}
