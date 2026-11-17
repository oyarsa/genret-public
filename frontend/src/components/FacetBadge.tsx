import { FACET_COLORS, FACET_COLORS_DIM } from "@/lib/colors";
import type { FacetName } from "@/lib/types";

interface FacetBadgeProps {
  facet: string;
  score?: number;
}

export default function FacetBadge({ facet, score }: FacetBadgeProps) {
  const color = FACET_COLORS[facet as FacetName] ?? "#555a70";
  const bg = FACET_COLORS_DIM[facet as FacetName] ?? "#555a7020";

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded px-2 py-0.5"
      style={{
        background: bg,
        color,
        fontFamily: "var(--font-mono)",
        fontSize: "0.65rem",
        fontWeight: 500,
        letterSpacing: "0.02em",
      }}
    >
      {facet}
      {score !== undefined && (
        <span style={{ opacity: 0.7 }}>{score.toFixed(2)}</span>
      )}
    </span>
  );
}
