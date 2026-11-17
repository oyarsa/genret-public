import type { ProvenanceEdge, PaperNode } from "@/lib/types";
import { FACET_COLORS } from "@/lib/colors";
import type { FacetName } from "@/lib/types";
import FacetBadge from "./FacetBadge";

interface ProvenanceTrailProps {
  trail: ProvenanceEdge[];
  nodes: Map<number, PaperNode>;
}

export default function ProvenanceTrail({
  trail,
  nodes,
}: ProvenanceTrailProps) {
  if (trail.length === 0) return null;

  return (
    <div className="mt-4">
      <h4
        className="mb-2"
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.6rem",
          fontWeight: 500,
          color: "var(--color-text-muted)",
          letterSpacing: "0.06em",
          textTransform: "uppercase",
        }}
      >
        Provenance trail
      </h4>
      <div className="space-y-0">
        {trail.map((edge, i) => {
          const sourceNode = nodes.get(edge.source);
          const targetNode = nodes.get(edge.target);
          const facetColor =
            FACET_COLORS[edge.facet as FacetName] ?? "var(--color-text-muted)";
          const isLast = i === trail.length - 1;

          return (
            <div key={i} className="trail-step flex items-start gap-2 pb-3 pl-1">
              {/* Dot */}
              <div
                className="mt-1 h-3.5 w-3.5 shrink-0 rounded-full"
                style={{
                  background: `${facetColor}30`,
                  border: `1.5px solid ${facetColor}`,
                }}
              />

              {/* Content */}
              <div className="min-w-0 flex-1">
                <div
                  className="mb-0.5 truncate"
                  title={sourceNode?.title}
                  style={{
                    fontFamily: "var(--font-body)",
                    fontSize: "0.75rem",
                    color: "var(--color-text-secondary)",
                  }}
                >
                  {sourceNode?.title ?? `#${edge.source}`}
                </div>
                <div className="flex items-center gap-1.5">
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.55rem",
                      color: "var(--color-text-muted)",
                    }}
                  >
                    &darr;
                  </span>
                  <FacetBadge facet={edge.facet} score={edge.score} />
                </div>
                {isLast && (
                  <div
                    className="mt-1.5 truncate"
                    title={targetNode?.title}
                    style={{
                      fontFamily: "var(--font-body)",
                      fontSize: "0.75rem",
                      color: "var(--color-text-primary)",
                      fontWeight: 500,
                    }}
                  >
                    {targetNode?.title ?? `#${edge.target}`}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
