import { useCallback, useState } from "react";
import { searchPapers } from "@/api/client";
import type { SearchResponse } from "@/lib/types";

export function useSearch(
  onResults: (
    papers: SearchResponse["papers"],
    edges: SearchResponse["edges"],
    queryId: string,
  ) => void,
) {
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(
    async (query: string) => {
      if (!query.trim()) return;
      setIsSearching(true);
      setError(null);
      try {
        const result = await searchPapers(query);
        onResults(result.papers, result.edges, result.query_id);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Search failed");
      } finally {
        setIsSearching(false);
      }
    },
    [onResults],
  );

  return { search, isSearching, error };
}
