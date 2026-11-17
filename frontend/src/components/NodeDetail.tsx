import { useState } from "react";
import type {
  FacetName,
  PaperDetail,
  PaperNode,
  ProvenanceEdge,
} from "@/lib/types";
import { FACET_NAMES } from "@/lib/types";
import { FACET_COLORS } from "@/lib/colors";
import FacetBadge from "./FacetBadge";
import ProvenanceTrail from "./ProvenanceTrail";

type ExpandFacet = "all" | FacetName;

const CROSS_FACET = "problems_x_contributions";

interface NodeDetailProps {
  paper: PaperNode;
  /**
   * Full metadata fetched on selection (paper URL, citation count).
   * Null while loading or when unavailable; the panel works without it.
   */
  detail: PaperDetail | null;
  trail: ProvenanceEdge[];
  allNodes: Map<number, PaperNode>;
  onExpand: (corpusid: number, facet?: string) => void;
  isExpanding: boolean;
  onClose: () => void;
  /**
   * Disable the "Find related papers" action. Used for the recommend
   * flow's seed paper, which would 404 server-side because it has no
   * row in the retrieval graph.
   */
  expandDisabled?: boolean;
  /**
   * Request an LLM explanation of how this paper relates to the current
   * context. Undefined when there is no context (or for the seed paper),
   * which hides the action.
   */
  onExplain?: () => void;
  /** What the explanation compares against, e.g. a query or paper title. */
  explainContextLabel?: string | null;
  richExplanation?: string | null;
  isExplaining?: boolean;
}

export default function NodeDetail({
  paper,
  detail,
  trail,
  allNodes,
  onExpand,
  isExpanding,
  onClose,
  expandDisabled = false,
  onExplain,
  explainContextLabel = null,
  richExplanation = null,
  isExplaining = false,
}: NodeDetailProps) {
  const [expandVia, setExpandVia] = useState<ExpandFacet>("all");
  // Kept across selections so facets can be compared between papers.
  const [preferredFacet, setPreferredFacet] = useState<FacetName | null>(null);

  const handleFind = () => {
    onExpand(paper.corpusid, expandVia === "all" ? undefined : expandVia);
  };
  const findDisabled = expandDisabled || isExpanding;

  const populated = FACET_NAMES.filter(
    (f) => (paper.facets[f]?.length ?? 0) > 0,
  );
  const activeFacet =
    preferredFacet && populated.includes(preferredFacet)
      ? preferredFacet
      : (populated[0] ?? null);
  const matched = new Set(paper.matched_facets);
  const items = activeFacet ? (paper.facets[activeFacet] ?? []) : [];

  return (
    <div className="detail-scroll flex h-full flex-col overflow-y-auto p-5">
      {/* Close button */}
      <button
        onClick={onClose}
        className="close-btn absolute top-3 right-3 flex h-7 w-7 items-center justify-center rounded-md"
        style={{
          background: "var(--color-surface-3)",
          color: "var(--color-text-muted)",
          fontFamily: "var(--font-mono)",
          fontSize: "0.85rem",
          lineHeight: 1,
          border: "none",
          cursor: "pointer",
        }}
        aria-label="Close detail panel"
      >
        &times;
      </button>

      {/* Title */}
      <h2
        className="mb-1 leading-snug"
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "1.15rem",
          fontWeight: 500,
          color: "var(--color-text-primary)",
        }}
      >
        {paper.title}
      </h2>

      {/* Authors & metadata */}
      <p
        className="mb-2"
        style={{
          fontFamily: "var(--font-body)",
          fontSize: "0.8rem",
          color: "var(--color-text-secondary)",
          lineHeight: 1.5,
        }}
      >
        {paper.authors.slice(0, 3).join(", ")}
        {paper.authors.length > 3 && " et al."}
        {paper.year && ` (${paper.year})`}
        {paper.venue && <span style={{ opacity: 0.6 }}> &mdash; {paper.venue}</span>}
      </p>

      {/* Paper link + citation count (from the detail fetch) */}
      {detail && (detail.url || detail.citation_count !== null) && (
        <p
          className="mb-3 flex items-center gap-3"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.68rem",
            letterSpacing: "0.03em",
          }}
        >
          {detail.url && (
            <a
              href={detail.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "var(--color-accent, #5b8def)" }}
            >
              View on Semantic Scholar ↗
            </a>
          )}
          {detail.citation_count !== null && (
            <span style={{ color: "var(--color-text-muted)" }}>
              {detail.citation_count.toLocaleString()} citations
            </span>
          )}
        </p>
      )}

      {/* How this paper was reached */}
      <ProvenanceTrail trail={trail} nodes={allNodes} />

      {/* Facet score badges — hide near-zero scores to keep the panel
          uncluttered for nodes with no query-vs-paper signal (e.g. the
          recommend-mode query node itself). */}
      <div className="mt-4 mb-3 flex flex-wrap gap-1.5">
        {FACET_NAMES.filter((f) => paper.facet_scores[f] > 1e-3).map((f) => (
          <FacetBadge key={f} facet={f} score={paper.facet_scores[f]} />
        ))}
      </div>

      {/* Query score — only meaningful for retrieval results, not for the
          query paper itself. */}
      {FACET_NAMES.some((f) => paper.facet_scores[f] > 1e-3) && (
        <div
          className="mb-3"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.65rem",
            color: "var(--color-text-muted)",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          Query score {paper.query_score.toFixed(4)}
        </div>
      )}

      {/* LLM explanation of the link to the query or seed paper. The
          context is spelled out: "this connection" alone left users
          guessing what it was connecting. */}
      {(onExplain || richExplanation) && explainContextLabel && (
        <div className="mb-4">
          <div
            className="mb-1.5"
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.6rem",
              color: "var(--color-text-muted)",
              letterSpacing: "0.04em",
              lineHeight: 1.4,
            }}
          >
            Compared with {explainContextLabel}
          </div>
          {richExplanation ? (
            <div
              className="rounded-md p-3"
              style={{
                background: "var(--color-surface-2)",
                fontFamily: "var(--font-body)",
                fontSize: "0.78rem",
                lineHeight: 1.55,
                color: "var(--color-text-primary)",
              }}
            >
              {richExplanation}
            </div>
          ) : (
            onExplain && (
              <button
                onClick={onExplain}
                disabled={isExplaining}
                className="rounded-md px-2.5 py-1 disabled:opacity-50"
                style={{
                  background: "transparent",
                  border: "1px solid var(--color-surface-4)",
                  color: "var(--color-text-secondary)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.65rem",
                  letterSpacing: "0.03em",
                  cursor: isExplaining ? "wait" : "pointer",
                }}
              >
                {isExplaining ? "Explaining…" : "Why are these related?"}
              </button>
            )
          )}
        </div>
      )}

      {/* Find related papers — adds new nodes to the graph anchored at
          this paper. The "via" selector restricts the signal. */}
      <div className="mb-1 flex items-stretch gap-1.5">
        <button
          onClick={handleFind}
          disabled={findDisabled}
          title={
            expandDisabled
              ? "This is the seed paper; pick a neighbour to expand from."
              : undefined
          }
          className="expand-btn flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 disabled:opacity-40"
          style={{
            background: "var(--color-accent, #5b8def)",
            color: "#fff",
            fontFamily: "var(--font-mono)",
            fontSize: "0.75rem",
            fontWeight: 600,
            letterSpacing: "0.03em",
            border: "none",
            cursor: findDisabled
              ? expandDisabled
                ? "not-allowed"
                : "wait"
              : "pointer",
          }}
        >
          <span aria-hidden="true" style={{ fontWeight: 700, fontSize: "0.9rem", lineHeight: 1 }}>+</span>
          {isExpanding ? "Finding…" : "Find related papers"}
        </button>
        <select
          aria-label="Refine which facet drives the search"
          value={expandVia}
          onChange={(e) => setExpandVia(e.target.value as ExpandFacet)}
          disabled={findDisabled}
          style={{
            padding: "0 8px",
            borderRadius: 6,
            background: "var(--color-surface-2)",
            border: "1px solid var(--color-surface-4)",
            color: "var(--color-text-primary)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.7rem",
            cursor: findDisabled ? "not-allowed" : "pointer",
          }}
        >
          <option value="all">via all facets</option>
          {FACET_NAMES.map((f) => (
            <option key={f} value={f}>
              via {f}
            </option>
          ))}
        </select>
      </div>
      {expandDisabled && (
        <div
          className="mb-2"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.6rem",
            color: "var(--color-text-muted)",
            letterSpacing: "0.04em",
            lineHeight: 1.4,
            paddingLeft: "0.2rem",
          }}
        >
          Seed paper of this view — click any neighbour to expand from
          there instead.
        </div>
      )}

      {/* Facets in one tabbed panel. A dot marks the facets that
          contributed this paper to the results. */}
      {activeFacet && (
        <div className="mt-4">
          <div
            role="tablist"
            aria-label="Facets"
            className="flex gap-1"
            style={{ borderBottom: "1px solid var(--color-surface-3)" }}
          >
            {populated.map((facet) => {
              const isActive = facet === activeFacet;
              const color = FACET_COLORS[facet];
              return (
                <button
                  key={facet}
                  role="tab"
                  aria-selected={isActive}
                  onClick={() => setPreferredFacet(facet)}
                  title={
                    matched.has(facet)
                      ? `${facet} — matched the query`
                      : facet
                  }
                  className="flex items-center gap-1 px-2 py-1.5"
                  style={{
                    background: "transparent",
                    border: "none",
                    borderBottom: `2px solid ${isActive ? color : "transparent"}`,
                    marginBottom: -1,
                    color: isActive ? color : "var(--color-text-muted)",
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.6rem",
                    fontWeight: isActive ? 600 : 400,
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                    cursor: "pointer",
                  }}
                >
                  {matched.has(facet) && (
                    <span
                      aria-hidden="true"
                      style={{
                        display: "inline-block",
                        width: 4,
                        height: 4,
                        borderRadius: "50%",
                        background: color,
                      }}
                    />
                  )}
                  {facet}
                </button>
              );
            })}
          </div>

          <ul
            className="list-none space-y-2 pt-3"
            style={{ borderLeft: "none" }}
          >
            {items.map((item, i) => (
              <li
                key={i}
                className="flex gap-2"
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: "0.78rem",
                  color: "var(--color-text-secondary)",
                  lineHeight: 1.5,
                }}
              >
                <span
                  aria-hidden="true"
                  style={{
                    color: FACET_COLORS[activeFacet],
                    opacity: 0.7,
                    flexShrink: 0,
                  }}
                >
                  •
                </span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          {matched.has(CROSS_FACET) && (
            <div
              className="pt-3"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.6rem",
                color: "var(--color-text-muted)",
                letterSpacing: "0.04em",
                lineHeight: 1.4,
              }}
            >
              Also matched via problems → contributions
            </div>
          )}
        </div>
      )}
    </div>
  );
}
