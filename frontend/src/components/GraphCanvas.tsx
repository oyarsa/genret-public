import { useEffect, useRef } from "react";
import type { PaperNode, ProvenanceEdge } from "@/lib/types";
import {
  fanPositions,
  renderGraph,
  updateSelection,
  updateEdges,
  type SimLink,
  type SimNode,
} from "@/lib/graph";
import type { Simulation } from "d3";

interface GraphCanvasProps {
  nodes: Map<number, PaperNode>;
  edges: ProvenanceEdge[];
  activeFacets: Set<string>;
  selectedNodeId: number | null;
  maxEdgesPerNode: number;
  onNodeClick: (id: number) => void;
}

export default function GraphCanvas({
  nodes,
  edges,
  activeFacets,
  selectedNodeId,
  maxEdgesPerNode,
  onNodeClick,
}: GraphCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<Simulation<SimNode, SimLink> | null>(null);
  const prevNodesRef = useRef<SimNode[]>([]);
  const simLinksRef = useRef<SimLink[]>([]);
  // The dashed ring marks the top hit of the search or recommend that
  // built this graph. Expansion scores are on a different scale, so it
  // must not be recomputed when nodes are added.
  const topResultIdRef = useRef<number | null>(null);

  // Rebuild the graph when nodes or edges change
  useEffect(() => {
    if (!svgRef.current || nodes.size === 0) return;

    // An expansion adds to the current graph; a new search or recommend
    // replaces it. Only the former reuses positions -- a replaced graph
    // gets a fresh layout and a fresh framing.
    const prevNodes = prevNodesRef.current;
    const isExpansion =
      prevNodes.length > 0 &&
      nodes.size > prevNodes.length &&
      prevNodes.every((n) => nodes.has(n.id));

    const oldPositions = new Map<number, { x: number; y: number }>();
    if (isExpansion) {
      for (const n of prevNodes) {
        if (n.x != null && n.y != null) {
          oldPositions.set(n.id, { x: n.x, y: n.y });
        }
      }
    }

    const papers = Array.from(nodes.values());
    const incremental = oldPositions.size > 0;

    if (!incremental) {
      const topScore = Math.max(...papers.map((p) => p.query_score));
      topResultIdRef.current =
        papers.find((p) => p.query_score === topScore)?.corpusid ?? null;
    }

    // Which placed node each new node hangs off. Provenance edges point
    // from the expanded node to its new neighbours, so prefer that
    // direction; inter-edges are a fallback.
    const parentOf = new Map<number, number>();
    for (const e of edges) {
      if (oldPositions.has(e.source) && !oldPositions.has(e.target)) {
        if (!parentOf.has(e.target)) parentOf.set(e.target, e.source);
      } else if (oldPositions.has(e.target) && !oldPositions.has(e.source)) {
        if (!parentOf.has(e.source)) parentOf.set(e.source, e.target);
      }
    }

    // Centroid of the placed nodes: new nodes fan away from it, into
    // empty space rather than over the existing graph.
    const placed = [...oldPositions.values()];
    const centroid = placed.length
      ? {
          x: placed.reduce((s, p) => s + p.x, 0) / placed.length,
          y: placed.reduce((s, p) => s + p.y, 0) / placed.length,
        }
      : { x: 0, y: 0 };

    // Group the new nodes by parent and lay each group out in a fan.
    const byParent = new Map<number, number[]>();
    for (const p of papers) {
      if (oldPositions.has(p.corpusid)) continue;
      const parent = parentOf.get(p.corpusid);
      if (parent === undefined) continue;
      byParent.set(parent, [...(byParent.get(parent) ?? []), p.corpusid]);
    }
    const spawnPositions = new Map<number, { x: number; y: number }>();
    for (const [parent, children] of byParent) {
      const anchor = oldPositions.get(parent)!;
      const slots = fanPositions(anchor, centroid, children.length);
      children.forEach((cid, i) => spawnPositions.set(cid, slots[i]!));
    }

    const simNodes: SimNode[] = papers.map((p) => {
      const pos = oldPositions.get(p.corpusid);
      const isTopResult = p.corpusid === topResultIdRef.current;
      if (pos) {
        // Pin what is already on screen: an expansion must not rearrange
        // the graph the user is reading.
        return { id: p.corpusid, paper: p, isTopResult, ...pos, fx: pos.x, fy: pos.y };
      }
      const spawn = spawnPositions.get(p.corpusid);
      return {
        id: p.corpusid,
        paper: p,
        isTopResult,
        ...(spawn ?? {}),
        ...(spawn ? { anchorX: spawn.x, anchorY: spawn.y } : {}),
      };
    });

    const nodeIds = new Set(nodes.keys());
    const simLinks: SimLink[] = edges
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        edge: e,
      }));

    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    prevNodesRef.current = simNodes;
    simLinksRef.current = simLinks;

    // Recentre on the expansion: the node that was expanded plus the
    // papers it produced.
    const focusIds = incremental
      ? [...spawnPositions.keys(), ...byParent.keys()]
      : [];

    const sim = renderGraph(
      svgRef.current,
      simNodes,
      simLinks,
      activeFacets,
      maxEdgesPerNode,
      onNodeClick,
      incremental,
      focusIds,
    );

    simulationRef.current = sim;

    updateSelection(svgRef.current, selectedNodeId);

    return () => {
      sim.stop();
    };
    // activeFacets, maxEdgesPerNode, selectedNodeId handled by effects below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges, onNodeClick]);

  // Update selection styling without rebuilding the graph
  useEffect(() => {
    if (svgRef.current) {
      updateSelection(svgRef.current, selectedNodeId);
    }
  }, [selectedNodeId]);

  // Update visible edges without rebuilding the graph
  useEffect(() => {
    if (svgRef.current && simLinksRef.current.length > 0) {
      updateEdges(svgRef.current, simLinksRef.current, activeFacets, maxEdgesPerNode);
    }
  }, [activeFacets, maxEdgesPerNode]);

  return <svg ref={svgRef} className="h-full w-full" />;
}
