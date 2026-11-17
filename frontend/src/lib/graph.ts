import * as d3 from "d3";
import type { PaperNode, ProvenanceEdge } from "./types";
import { edgeColor, nodeColor, nodeGlow } from "./colors";

export interface SimNode extends d3.SimulationNodeDatum {
  id: number;
  paper: PaperNode;
  isTopResult: boolean;
  /**
   * Position a newly added node is held near, so expansions grow around
   * the node they came from instead of being dragged across the canvas
   * by link forces. Unset for nodes that are already placed.
   */
  anchorX?: number;
  anchorY?: number;
}

export interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  edge: ProvenanceEdge;
}

/**
 * Deterministic positions for newly added nodes: an arc around their
 * parent, pointing away from the rest of the graph.
 *
 * @param parent Position of the node that was expanded.
 * @param awayFrom Point to fan away from (the graph's centroid).
 * @param count Number of new nodes to place.
 * @param radius Distance from the parent.
 */
export function fanPositions(
  parent: { x: number; y: number },
  awayFrom: { x: number; y: number },
  count: number,
  radius = 150,
): { x: number; y: number }[] {
  const dx = parent.x - awayFrom.x;
  const dy = parent.y - awayFrom.y;
  const base = Math.hypot(dx, dy) < 1e-6 ? 0 : Math.atan2(dy, dx);
  const spread = Math.min(2.6, count * 0.5);

  return Array.from({ length: count }, (_, i) => {
    const offset = count === 1 ? 0 : ((i / (count - 1)) * 2 - 1) * (spread / 2);
    const angle = base + offset;
    return {
      x: parent.x + Math.cos(angle) * radius,
      y: parent.y + Math.sin(angle) * radius,
    };
  });
}

/**
 * Transform that centres a group of nodes in the viewport.
 *
 * Pans at the current scale whenever the group fits, and only zooms out
 * when it cannot: a rescale is far more disorienting than a translation,
 * which preserves every relative position on screen. Never zooms in.
 *
 * @returns The target transform, or null when there is nothing to focus.
 */
export function focusTransform(
  targets: { x: number; y: number; r: number }[],
  viewport: { width: number; height: number },
  currentK: number,
  padding = 70,
): { x: number; y: number; k: number } | null {
  if (targets.length === 0 || viewport.width === 0 || viewport.height === 0) {
    return null;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const t of targets) {
    minX = Math.min(minX, t.x - t.r);
    minY = Math.min(minY, t.y - t.r);
    maxX = Math.max(maxX, t.x + t.r);
    maxY = Math.max(maxY, t.y + t.r);
  }

  const boxW = maxX - minX + padding * 2;
  const boxH = maxY - minY + padding * 2;
  const k = Math.min(currentK, viewport.width / boxW, viewport.height / boxH);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  return {
    x: viewport.width / 2 - cx * k,
    y: viewport.height / 2 - cy * k,
    k,
  };
}

export function createSimulation(
  nodes: SimNode[],
  links: SimLink[],
  width: number,
  height: number,
  incremental = false,
): d3.Simulation<SimNode, SimLink> {
  const sim = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink<SimNode, SimLink>(links)
        .id((d) => d.id)
        .distance(300)
        .strength((d) => d.edge.score * 0.25),
    )
    // distanceMax keeps disconnected result clusters from shoving each
    // other off the canvas: without it two components with no edges
    // between them drift apart until the auto-fit shrinks everything.
    .force("charge", d3.forceManyBody().strength(-800).distanceMax(500))
    .force("collision", d3.forceCollide().radius(65));

  if (incremental) {
    // Existing nodes are pinned, so only the new ones move. Hold them
    // near their fan slot; without this the link force drags them across
    // the graph towards the neighbours they share with the pinned nodes.
    sim
      .force(
        "anchorX",
        d3
          .forceX<SimNode>((d) => d.anchorX ?? 0)
          .strength((d) => (d.anchorX == null ? 0 : 0.4)),
      )
      .force(
        "anchorY",
        d3
          .forceY<SimNode>((d) => d.anchorY ?? 0)
          .strength((d) => (d.anchorY == null ? 0 : 0.4)),
      );
  } else {
    // forceCenter only recentres the mean; the weak positional forces
    // are what actually gather separate components into one readable
    // cluster instead of leaving them scattered.
    sim
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("gatherX", d3.forceX(width / 2).strength(0.06))
      .force("gatherY", d3.forceY(height / 2).strength(0.06));
  }

  return sim;
}

export function nodeRadius(queryScore: number): number {
  return 6 + queryScore * 35;
}

function addSvgDefs(defs: d3.Selection<SVGDefsElement, unknown, null, undefined>): void {
  // Glow filter for nodes
  const glow = defs.append("filter").attr("id", "node-glow");
  glow
    .append("feGaussianBlur")
    .attr("stdDeviation", "4")
    .attr("result", "blur");
  glow
    .append("feComposite")
    .attr("in", "SourceGraphic")
    .attr("in2", "blur")
    .attr("operator", "over");

  // Stronger glow for selected node
  const selectedGlow = defs.append("filter").attr("id", "selected-glow");
  selectedGlow
    .append("feGaussianBlur")
    .attr("stdDeviation", "8")
    .attr("result", "blur");
  selectedGlow
    .append("feComposite")
    .attr("in", "SourceGraphic")
    .attr("in2", "blur")
    .attr("operator", "over");
}

/**
 * Keep only the top-N edges per node (by score), across all facet types.
 * An edge is included if at least one of its endpoints still has room.
 */
function pruneEdges(links: SimLink[], maxPerNode: number): SimLink[] {
  const sorted = [...links].sort((a, b) => b.edge.score - a.edge.score);
  const counts = new Map<number, number>();
  const result: SimLink[] = [];

  for (const link of sorted) {
    const src =
      typeof link.source === "object"
        ? (link.source as SimNode).id
        : (link.source as number);
    const tgt =
      typeof link.target === "object"
        ? (link.target as SimNode).id
        : (link.target as number);

    const srcCount = counts.get(src) ?? 0;
    const tgtCount = counts.get(tgt) ?? 0;

    if (srcCount < maxPerNode || tgtCount < maxPerNode) {
      result.push(link);
      counts.set(src, srcCount + 1);
      counts.set(tgt, tgtCount + 1);
    }
  }
  return result;
}

/**
 * Update selection styling without rebuilding the graph.
 */
export function updateSelection(svg: SVGSVGElement, selectedId: number | null): void {
  svg.dataset.selectedId = String(selectedId ?? -1);
  const d3svg = d3.select(svg);

  d3svg
    .selectAll<SVGCircleElement, SimNode>(".node-circle")
    .attr("stroke", (d) =>
      d.id === selectedId ? "#1a1e2e" : "rgba(15, 23, 42, 0.18)",
    )
    .attr("stroke-width", (d) => (d.id === selectedId ? 2 : 1))
    .attr("filter", (d) =>
      d.id === selectedId ? "url(#selected-glow)" : "url(#node-glow)",
    );

  d3svg
    .selectAll<SVGCircleElement, SimNode>(".node-halo")
    .attr("opacity", (d) => (d.id === selectedId ? 0.5 : 0.15));
}

/**
 * Update visible edges without rebuilding the graph or restarting the simulation.
 */
export function updateEdges(
  svg: SVGSVGElement,
  allLinks: SimLink[],
  activeFacets: Set<string>,
  maxEdgesPerNode: number,
): void {
  const facetFiltered = allLinks.filter(
    (l) => activeFacets.size === 0 || activeFacets.has(l.edge.facet),
  );
  const visibleLinks = pruneEdges(facetFiltered, maxEdgesPerNode);

  const g = d3.select(svg).select<SVGGElement>("g");
  const linkGroup = g.select<SVGGElement>(".edge-group");

  linkGroup
    .selectAll<SVGLineElement, SimLink>("line")
    .data(visibleLinks, (d) => `${(d.source as SimNode).id ?? d.source}-${(d.target as SimNode).id ?? d.target}-${d.edge.facet}`)
    .join(
      (enter) =>
        enter
          .append("line")
          .attr("stroke", (d) => edgeColor(d.edge.facet))
          .attr("stroke-opacity", (d) => 0.15 + d.edge.score * 0.35)
          .attr("stroke-width", (d) => 0.5 + d.edge.score * 2.5)
          .attr("x1", (d) => (d.source as SimNode).x ?? 0)
          .attr("y1", (d) => (d.source as SimNode).y ?? 0)
          .attr("x2", (d) => (d.target as SimNode).x ?? 0)
          .attr("y2", (d) => (d.target as SimNode).y ?? 0),
      (update) => update,
      (exit) => exit.remove(),
    );
}

export function renderGraph(
  svg: SVGSVGElement,
  nodes: SimNode[],
  links: SimLink[],
  activeFacets: Set<string>,
  maxEdgesPerNode: number,
  onNodeClick: (id: number) => void,
  incremental = false,
  focusIds: number[] = [],
): d3.Simulation<SimNode, SimLink> {
  const width = svg.clientWidth;
  const height = svg.clientHeight;

  const d3svg = d3.select(svg);
  d3svg.selectAll("*").remove();

  // Clean up stale tooltip from previous render
  const container = svg.parentElement!;
  container.querySelector(".graph-tooltip")?.remove();

  // SVG definitions (filters)
  const defs = d3svg.append("defs");
  addSvgDefs(defs);

  const g = d3svg.append("g");

  // Zoom
  const zoom = d3
    .zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 8])
    .on("zoom", (e) => {
      g.attr("transform", e.transform as string);
    });
  d3svg.call(zoom);

  // The <g> is recreated on every render while d3 keeps the zoom state on
  // the <svg>, so re-apply it here. Without this the view snaps back to
  // identity on every expansion and the whole graph appears to jump.
  g.attr("transform", d3.zoomTransform(svg).toString());

  // Auto-fit the simulated layout to the canvas with a margin so the
  // graph fills the available space instead of clustering near the
  // centre. The fit runs after the simulation has had time to settle.
  const fitToView = (): void => {
    const nodeData = nodes;
    if (nodeData.length === 0) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const n of nodeData) {
      const r = nodeRadius(n.paper.query_score);
      if (n.x == null || n.y == null) continue;
      minX = Math.min(minX, n.x - r);
      minY = Math.min(minY, n.y - r);
      maxX = Math.max(maxX, n.x + r);
      maxY = Math.max(maxY, n.y + r);
    }
    if (!Number.isFinite(minX)) return;
    // Generous padding: node labels extend well beyond the circles, and
    // the facet legend sits over the bottom-left corner.
    const padding = 90;
    const bbW = maxX - minX + padding * 2;
    const bbH = maxY - minY + padding * 2;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const k = Math.min(rect.width / bbW, rect.height / bbH, 8);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const tx = rect.width / 2 - cx * k;
    const ty = rect.height / 2 - cy * k;
    d3svg
      .transition()
      .duration(400)
      .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(k));
  };

  /**
   * Pan so the expanded node and the papers it just produced sit in the
   * middle of the canvas. The layout itself does not change: this is a
   * translation at the current zoom (falling back to a zoom-out only if
   * the group cannot fit), so every relative position is preserved.
   */
  const focusSet = new Set(focusIds);
  const focusOnExpansion = (): void => {
    const rect = svg.getBoundingClientRect();
    const targets = nodes
      .filter((n) => focusSet.has(n.id) && n.x != null && n.y != null)
      .map((n) => ({
        x: n.x!,
        y: n.y!,
        r: nodeRadius(n.paper.query_score),
      }));

    const current = d3.zoomTransform(svg);
    const target = focusTransform(
      targets,
      { width: rect.width, height: rect.height },
      current.k,
    );
    if (target === null) return;

    // Skip imperceptible corrections so a central expansion does not
    // nudge the view for no reason.
    const settled =
      Math.abs(target.x - current.x) < 40 &&
      Math.abs(target.y - current.y) < 40 &&
      Math.abs(target.k - current.k) < 0.01;
    if (settled) return;

    d3svg
      .transition()
      .duration(500)
      .call(
        zoom.transform,
        d3.zoomIdentity.translate(target.x, target.y).scale(target.k),
      );
  };

  // Fit once the simulation has cooled. A timeout drives the initial fit
  // so users see the auto-framed layout quickly. Expansions deliberately
  // do NOT re-frame: rescaling the view makes the whole graph appear to
  // move even though the placed nodes are pinned. Re-frame only if the
  // new nodes would otherwise land off-screen.
  if (incremental) {
    // The fan slots are known immediately, so recentre without waiting
    // for the simulation to cool; the "end" pass then corrects for how
    // the new nodes actually settled.
    window.setTimeout(focusOnExpansion, 350);
  } else {
    window.setTimeout(fitToView, 1200);
  }

  // Filter links by active facets, then prune to top-N per node
  const facetFiltered = links.filter(
    (l) => activeFacets.size === 0 || activeFacets.has(l.edge.facet),
  );
  const visibleLinks = pruneEdges(facetFiltered, maxEdgesPerNode);

  // Draw edges — with variable opacity based on score
  g.append("g")
    .attr("class", "edge-group")
    .selectAll<SVGLineElement, SimLink>("line")
    .data(visibleLinks)
    .join("line")
    .attr("stroke", (d) => edgeColor(d.edge.facet))
    .attr("stroke-opacity", (d) => 0.15 + d.edge.score * 0.35)
    .attr("stroke-width", (d) => 0.5 + d.edge.score * 2.5);

  // Draw node glow halos (behind the nodes)
  const halo = g
    .append("g")
    .selectAll<SVGCircleElement, SimNode>("circle")
    .data(nodes)
    .join("circle")
    .attr("class", "node-halo")
    .attr("r", (d) => nodeRadius(d.paper.query_score) + 6)
    .attr("fill", (d) => nodeGlow(d.paper.facet_scores))
    .attr("opacity", 0.15);

  // Draw accent ring around top result
  const topResultNodes = nodes.filter((d) => d.isTopResult);
  const topRing = g
    .append("g")
    .selectAll<SVGCircleElement, SimNode>("circle")
    .data(topResultNodes)
    .join("circle")
    .attr("r", (d) => nodeRadius(d.paper.query_score) + 4)
    .attr("fill", "none")
    .attr("stroke", "#1a1e2e")
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "4 3")
    .attr("opacity", 0.6);

  // Draw nodes
  const node = g
    .append("g")
    .selectAll<SVGCircleElement, SimNode>("circle")
    .data(nodes)
    .join("circle")
    .attr("class", "node-circle")
    .attr("r", (d) => nodeRadius(d.paper.query_score))
    .attr("fill", (d) => nodeColor(d.paper.facet_scores))
    .attr("fill-opacity", 0.85)
    .attr("stroke", "rgba(15, 23, 42, 0.18)")
    .attr("stroke-width", 1)
    .attr("filter", "url(#node-glow)")
    .attr("cursor", "pointer")
    // Accessible name so screen readers and test selectors can tell
    // nodes apart; without it they are anonymous graphics symbols.
    .attr("role", "button")
    .attr("aria-label", (d) => d.paper.title)
    .on("click", (_e, d) => onNodeClick(d.id));

  // HTML tooltip (lives outside the SVG for proper text rendering)
  const tooltip = document.createElement("div");
  tooltip.className = "graph-tooltip";
  Object.assign(tooltip.style, {
      position: "absolute",
      pointerEvents: "none",
      opacity: "0",
      transition: "opacity 150ms",
      background: "var(--color-surface-2)",
      border: "1px solid var(--color-surface-3)",
      borderRadius: "6px",
      padding: "8px 10px",
      maxWidth: "320px",
      fontFamily: "var(--font-body)",
      fontSize: "0.75rem",
      lineHeight: "1.4",
      color: "var(--color-text-primary)",
      zIndex: "10",
    });
  container.style.position = "relative";
  container.appendChild(tooltip);

  function showTooltip(event: MouseEvent, d: SimNode): void {
    const p = d.paper;
    const authors = p.authors.length > 3
      ? p.authors.slice(0, 3).join(", ") + ` +${p.authors.length - 3}`
      : p.authors.join(", ");

    tooltip.innerHTML =
      `<div style="font-weight:600;margin-bottom:2px">${p.title}</div>` +
      (p.year ? `<div style="color:var(--color-text-secondary);font-size:0.7rem;margin-bottom:2px">${p.year}</div>` : "") +
      `<div style="color:var(--color-text-muted);font-size:0.65rem">${authors}</div>`;

    const rect = container.getBoundingClientRect();
    tooltip.style.left = `${event.clientX - rect.left + 12}px`;
    tooltip.style.top = `${event.clientY - rect.top - 10}px`;
    tooltip.style.opacity = "1";
  }

  function hideTooltip(): void {
    tooltip.style.opacity = "0";
  }

  // Hover effects
  node
    .on("mouseenter", function (event, d) {
      d3.select(this)
        .transition()
        .duration(150)
        .attr("fill-opacity", 1)
        .attr("stroke", "#1a1e2e")
        .attr("stroke-width", 1.5);
      halo
        .filter((h) => h.id === d.id)
        .transition()
        .duration(150)
        .attr("opacity", 0.4);
      showTooltip(event, d);
    })
    .on("mousemove", function (event) {
      const rect = container.getBoundingClientRect();
      tooltip.style.left = `${event.clientX - rect.left + 12}px`;
      tooltip.style.top = `${event.clientY - rect.top - 10}px`;
    })
    .on("mouseleave", function (_e, d) {
      const sel = Number(svg.dataset.selectedId ?? -1);
      const isSelected = d.id === sel;
      d3.select(this)
        .transition()
        .duration(200)
        .attr("fill-opacity", 0.85)
        .attr("stroke", isSelected ? "#1a1e2e" : "rgba(15, 23, 42, 0.18)")
        .attr("stroke-width", isSelected ? 2 : 1);
      halo
        .filter((h) => h.id === d.id)
        .transition()
        .duration(200)
        .attr("opacity", isSelected ? 0.5 : 0.15);
      hideTooltip();
    });

  // Labels for all nodes — mono font, subtle
  const labels = g
    .append("g")
    .selectAll<SVGTextElement, SimNode>("text")
    .data(nodes)
    .join("text")
    .attr("class", "node-label")
    .attr("font-family", '"DM Mono", monospace')
    .attr("font-size", 9)
    .attr("font-weight", 400)
    .attr("fill", "#555a70")
    .attr("text-anchor", "middle")
    .attr("dy", (d) => -nodeRadius(d.paper.query_score) - 8)
    .text((d) =>
      d.paper.title.length > 45
        ? d.paper.title.slice(0, 42) + "\u2026"
        : d.paper.title,
    );

  // Drop shadow behind labels for readability
  labels
    .clone(true)
    .lower()
    .attr("stroke", "#fafbfd")
    .attr("stroke-width", 3)
    .attr("stroke-linejoin", "round");

  // Drag
  const drag = d3
    .drag<SVGCircleElement, SimNode>()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x as number;
      d.fy = event.y as number;
    })
    .on("end", (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      // Stay where dropped: the layout is sticky, so releasing the pin
      // here would let the node drift back on the next tick.
      d.fx = d.x;
      d.fy = d.y;
    });

  node.call(drag);

  const simulation = createSimulation(
    nodes,
    visibleLinks,
    width,
    height,
    incremental,
  );

  // On an expansion the placed nodes are pinned, so a short warm run is
  // enough for the new nodes to settle into their fan.
  if (incremental) {
    simulation.alpha(0.5);
  }

  simulation.on("tick", () => {
    g.select(".edge-group")
      .selectAll<SVGLineElement, SimLink>("line")
      .attr("x1", (d) => (d.source as SimNode).x!)
      .attr("y1", (d) => (d.source as SimNode).y!)
      .attr("x2", (d) => (d.target as SimNode).x!)
      .attr("y2", (d) => (d.target as SimNode).y!);

    halo.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
    topRing.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
    node.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);

    g.selectAll<SVGTextElement, SimNode>(".node-label")
      .attr("x", (d) => d.x!)
      .attr("y", (d) => d.y!);
  });

  simulation.on("end", () => {
    if (incremental) {
      focusOnExpansion();
    } else {
      fitToView();
    }
  });

  return simulation;
}
