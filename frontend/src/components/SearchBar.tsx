import { useEffect, useRef, useState } from "react";
import type { SearchPaperResult } from "@/lib/types";

export type SearchMode = "text" | "paper";

interface SearchBarProps {
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
  onTextSearch: (query: string) => void;
  isSearching: boolean;
  isRecommending: boolean;
  status: string | null;
  initialQuery?: string;
  suggestions: SearchPaperResult[];
  isSuggesting: boolean;
  onQueryChange: (query: string) => void;
  onPaperSelect: (paper: SearchPaperResult) => void;
  onClearSuggestions: () => void;
}

const PLACEHOLDER: Record<SearchMode, string> = {
  text: "e.g. attention mechanisms for machine translation",
  paper: "type a paper title to search the corpus",
};

const MODE_LABEL: Record<SearchMode, string> = {
  text: "Text",
  paper: "Paper",
};

export default function SearchBar({
  mode,
  onModeChange,
  onTextSearch,
  isSearching,
  isRecommending,
  status,
  initialQuery,
  suggestions,
  isSuggesting,
  onQueryChange,
  onPaperSelect,
  onClearSuggestions,
}: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery ?? "");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  // Selecting a suggestion sets the input to the paper title, which would
  // re-trigger this lookup effect and pop the dropdown back open over the
  // results; skip the one lookup caused by the selection itself.
  const skipLookupRef = useRef(false);

  useEffect(() => {
    if (skipLookupRef.current) {
      skipLookupRef.current = false;
      return;
    }
    if (mode === "paper" && query.trim().length > 0) {
      onQueryChange(query);
      setDropdownOpen(true);
    } else {
      setDropdownOpen(false);
      onClearSuggestions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, mode]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setDropdownOpen(false);
      }
    };
    window.addEventListener("mousedown", handleClickOutside);
    return () => window.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "text") {
      onTextSearch(query);
    } else if (mode === "paper") {
      const first = suggestions[0];
      if (first) {
        onPaperSelect(first);
        setDropdownOpen(false);
      }
    }
  };

  const handleSelect = (paper: SearchPaperResult) => {
    skipLookupRef.current = true;
    setQuery(paper.title);
    setDropdownOpen(false);
    onClearSuggestions();
    onPaperSelect(paper);
  };

  const busy = isSearching || isRecommending;
  const submitDisabled =
    busy ||
    !query.trim() ||
    (mode === "paper" && suggestions.length === 0);

  return (
    <div ref={containerRef} className="relative flex flex-col gap-1">
      <form onSubmit={handleSubmit} className="flex gap-2">
        {/* Mode toggle */}
        <div
          className="flex rounded-md p-0.5"
          style={{
            background: "var(--color-surface-2)",
            border: "1px solid var(--color-surface-4)",
          }}
        >
          {(["text", "paper"] as SearchMode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => {
                onModeChange(m);
                setQuery("");
                setDropdownOpen(false);
                onClearSuggestions();
              }}
              disabled={busy}
              className="rounded px-2 py-1 transition-all"
              style={{
                background:
                  mode === m ? "var(--color-accent)" : "transparent",
                color:
                  mode === m ? "#fff" : "var(--color-text-secondary)",
                fontFamily: "var(--font-mono)",
                fontSize: "0.7rem",
                fontWeight: 500,
                letterSpacing: "0.04em",
                textTransform: "uppercase",
                cursor: busy ? "not-allowed" : "pointer",
              }}
            >
              {MODE_LABEL[m]}
            </button>
          ))}
        </div>

        {/* Input */}
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => {
            if (mode === "paper" && suggestions.length > 0) {
              setDropdownOpen(true);
            }
          }}
          placeholder={PLACEHOLDER[mode]}
          disabled={busy}
          className="search-input flex-1 rounded-md border px-4 py-2 outline-none transition-shadow"
          style={{
            background: "var(--color-surface-2)",
            border: "1px solid var(--color-surface-4)",
            color: "var(--color-text-primary)",
            fontFamily: "var(--font-body)",
            fontSize: "0.875rem",
          }}
        />

        {/* Submit */}
        <button
          type="submit"
          disabled={submitDisabled}
          className="expand-btn rounded-md px-5 py-2 transition-all disabled:opacity-40"
          style={{
            background: "var(--color-accent)",
            color: "#fff",
            fontFamily: "var(--font-mono)",
            fontSize: "0.75rem",
            fontWeight: 500,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          {busy ? (
            <span className="inline-flex items-center gap-1.5">
              <span className="search-spinner" />
              {mode === "paper" ? "Recommending…" : "Searching…"}
            </span>
          ) : mode === "paper" ? (
            "Recommend"
          ) : (
            "Search"
          )}
        </button>
      </form>

      {mode === "paper" && status && (
        <div
          style={{
            color: "var(--color-text-secondary)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.7rem",
            letterSpacing: "0.04em",
            paddingLeft: "0.25rem",
          }}
        >
          {status}
        </div>
      )}

      {/* Autocomplete dropdown */}
      {mode === "paper" &&
        dropdownOpen &&
        (suggestions.length > 0 || isSuggesting) && (
          <div
            className="absolute left-0 right-0 z-20 mt-1 max-h-[18rem] overflow-y-auto rounded-md"
            style={{
              top: "100%",
              background: "var(--color-surface-1)",
              border: "1px solid var(--color-surface-3)",
              boxShadow: "0 4px 20px rgba(15, 23, 42, 0.18)",
            }}
          >
            {isSuggesting && suggestions.length === 0 ? (
              <div
                className="px-4 py-2"
                style={{
                  color: "var(--color-text-muted)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.75rem",
                }}
              >
                Searching titles…
              </div>
            ) : (
              suggestions.map((paper) => (
                <button
                  key={`${paper.corpusid}-${paper.source}`}
                  type="button"
                  onClick={() => handleSelect(paper)}
                  className="flex w-full flex-col gap-0.5 px-4 py-2 text-left transition-colors"
                  style={{
                    background: "transparent",
                    borderBottom: "1px solid var(--color-surface-3)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background =
                      "var(--color-surface-2)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "transparent";
                  }}
                >
                  <div
                    style={{
                      color: "var(--color-text-primary)",
                      fontFamily: "var(--font-body)",
                      fontSize: "0.85rem",
                      lineHeight: 1.3,
                    }}
                  >
                    {paper.title}
                  </div>
                  <div
                    style={{
                      color: "var(--color-text-muted)",
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.65rem",
                      letterSpacing: "0.04em",
                    }}
                  >
                    {paper.authors.slice(0, 3).join(", ")}
                    {paper.authors.length > 3 ? " et al." : ""}
                    {paper.year ? ` · ${paper.year}` : ""}
                    {paper.venue ? ` · ${paper.venue}` : ""}
                    {" · "}
                    <span style={{ textTransform: "uppercase" }}>
                      {paper.source.replace("_", " ")}
                    </span>
                  </div>
                </button>
              ))
            )}
          </div>
        )}
    </div>
  );
}
