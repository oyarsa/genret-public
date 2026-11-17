import { FACET_COLORS } from "@/lib/colors";
import { FACET_NAMES, type FacetName } from "@/lib/types";

interface FacetFilterProps {
  activeFacets: Set<string>;
  onToggle: (facet: FacetName) => void;
}

export default function FacetFilter({
  activeFacets,
  onToggle,
}: FacetFilterProps) {
  return (
    <div
      className="flex flex-col gap-1.5 rounded-lg px-3 py-2.5"
      style={{
        background: "var(--color-surface-1)",
        border: "1px solid var(--color-surface-3)",
        minWidth: "11rem",
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.6rem",
          color: "var(--color-text-muted)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}
      >
        Edges by facet
      </div>
      <div className="flex flex-col gap-1">
        {FACET_NAMES.map((facet) => {
          const active = activeFacets.has(facet);
          const color = FACET_COLORS[facet];
          return (
            <button
              key={facet}
              type="button"
              onClick={() => onToggle(facet)}
              className="flex items-center gap-2 rounded text-left transition-opacity"
              style={{
                background: "transparent",
                border: "none",
                padding: "2px 4px",
                fontFamily: "var(--font-mono)",
                fontSize: "0.7rem",
                color: active
                  ? "var(--color-text-primary)"
                  : "var(--color-text-muted)",
                opacity: active ? 1 : 0.55,
                cursor: "pointer",
                letterSpacing: "0.02em",
              }}
            >
              <span
                aria-hidden="true"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 14,
                  height: 14,
                  borderRadius: 3,
                  border: `1.5px solid ${active ? color : "var(--color-surface-4)"}`,
                  background: active ? color : "transparent",
                  color: "#fff",
                  fontSize: "0.65rem",
                  lineHeight: 1,
                }}
              >
                {active ? "✓" : ""}
              </span>
              <span style={{ flex: 1 }}>{facet}</span>
              <span
                aria-hidden="true"
                style={{
                  display: "inline-block",
                  width: 22,
                  height: 2,
                  background: color,
                  opacity: active ? 1 : 0.35,
                  borderRadius: 1,
                }}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}
