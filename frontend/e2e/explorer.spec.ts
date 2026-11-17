import { expect, test, type Page } from "@playwright/test";

// Fixture dataset (tests/explorer_harness.py): 20 graph papers titled
// "Graph Paper <cid> on Attention Mechanisms" (cids 1001-1020), 5
// in-corpus papers "Corpus Paper <cid> on Neural Retrieval" (2001-2005,
// 2001 with pre-cached facets). The fake encoder embeds "paper N"
// mentions near that paper's vectors, so queries quoting facet items
// rank deterministically.

async function searchFor(page: Page, query: string) {
  await page.getByRole("textbox").fill(query);
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(page.locator("text=/\\d+ papers/")).toBeVisible();
}

/**
 * Search so that a specific fixture paper is in the results. Quoting one of
 * its facet items makes the identity-encoding fake encoder rank it first;
 * a generic query returns an arbitrary 8 of the 20 graph papers.
 */
async function searchForPaper(page: Page, cid: number) {
  await searchFor(page, `problems item 1 of paper ${cid}`);
  await expect(
    page.getByRole("button", { name: `Graph Paper ${cid} on Attention Mechanisms` }),
  ).toBeVisible();
}

async function edgeCount(page: Page): Promise<number> {
  const stats = await page.locator("text=/\\d+ papers/").innerText();
  const match = stats.match(/(\d+) edges/);
  return match ? Number(match[1]) : 0;
}

function nodeCount(page: Page): Promise<number> {
  return page.locator("circle.node-circle").count();
}

test("empty state prompts for a search", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByText("Search for papers to explore the graph"),
  ).toBeVisible();
});

test("text search renders a readable graph", async ({ page }) => {
  await page.goto("/");
  await searchFor(page, "attention mechanisms");

  // Small by default (~2 per facet) so the graph stays legible
  await expect(page.getByText("8 papers", { exact: false })).toBeVisible();
  expect(await nodeCount(page)).toBe(8);
  await expect(
    page.getByRole("button", { name: /Graph Paper \d+ on Attention Mechanisms/ }).first(),
  ).toBeVisible();
  // Facet filter overlay appears with the graph
  await expect(page.getByText("Edges by facet")).toBeVisible();
});

test("query quoting a paper's facet item ranks it as top result", async ({
  page,
}) => {
  await page.goto("/");
  await searchFor(page, "methods item 1 of paper 1008");

  // The top result carries the dashed accent ring; there is exactly one
  await expect(page.locator("circle[stroke-dasharray]")).toHaveCount(1);
});

test("node click opens the detail panel; Escape closes it", async ({
  page,
}) => {
  await page.goto("/");
  await searchForPaper(page, 1005);

  await page
    .getByRole("button", { name: "Graph Paper 1005 on Attention Mechanisms" })
    .dispatchEvent("click");

  await expect(
    page.getByRole("heading", { name: "Graph Paper 1005 on Attention Mechanisms" }),
  ).toBeVisible();
  // exact: true because the "Compared with your search" line echoes the
  // query, which is itself this facet item
  await expect(
    page.getByText("problems item 1 of paper 1005", { exact: true }),
  ).toBeVisible();
  // Detail enrichment: paper link + citation count from /api/paper
  await expect(
    page.getByRole("link", { name: /View on Semantic Scholar/ }),
  ).toHaveAttribute("href", "https://example.org/1005");
  await expect(page.getByText("7 citations")).toBeVisible();
  // Facets are one tabbed panel; contributing ones are marked
  await expect(page.getByRole("tab", { name: "problems" })).toHaveAttribute(
    "title",
    "problems — matched the query",
  );
  await page.getByRole("tab", { name: "methods" }).click();
  await expect(
    page.getByText("methods item 1 of paper 1005", { exact: true }),
  ).toBeVisible();

  // On-demand rich explanation, with its comparison spelled out (the e2e
  // server cans the LLM call)
  await expect(page.getByText(/Compared with your search/)).toBeVisible();
  await page.getByRole("button", { name: "Why are these related?" }).click();
  await expect(
    page.getByText("These papers share a canned e2e connection."),
  ).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("heading", { name: "Graph Paper 1005 on Attention Mechanisms" }),
  ).not.toBeVisible();
});

test("find related papers adds new nodes and edges", async ({ page }) => {
  await page.goto("/");
  await searchForPaper(page, 1003);
  const nodesBefore = await nodeCount(page);
  const edgesBefore = await edgeCount(page);

  await page
    .getByRole("button", { name: "Graph Paper 1003 on Attention Mechanisms" })
    .dispatchEvent("click");
  await page.getByRole("button", { name: /Find related papers/ }).click();

  // Papers already on screen are excluded server-side, so expansion must
  // grow the node count, not just re-add what is already there
  await expect
    .poll(async () => nodeCount(page), { timeout: 10_000 })
    .toBeGreaterThan(nodesBefore);
  expect(await edgeCount(page)).toBeGreaterThan(edgesBefore);
});

test("facet filter hides that facet's edges", async ({ page }) => {
  await page.goto("/");
  await searchFor(page, "attention mechanisms");
  // Methods edges use the methods facet color. Total line count is NOT a
  // valid signal: per-node pruning back-fills with other facets' edges.
  const methodsLines = page.locator('svg line[stroke="#0f9d6b"]');
  expect(await methodsLines.count()).toBeGreaterThan(0);

  await page.getByRole("button", { name: "methods", exact: true }).click();

  await expect.poll(async () => methodsLines.count()).toBe(0);
});

test("url query auto-searches on load", async ({ page }) => {
  await page.goto("/?q=attention%20mechanisms");
  await expect(page.getByText("8 papers", { exact: false })).toBeVisible();
});

test("paper permalink auto-recommends on load", async ({ page }) => {
  await page.goto("/?paper=2001");
  await expect(page.locator("text=/\\d+ papers/")).toBeVisible({
    timeout: 15_000,
  });
  await expect(
    page.getByRole("button", {
      name: "Corpus Paper 2001 on Neural Retrieval",
      exact: true,
    }),
  ).toBeVisible();
});

test("paper mode recommends from an in-corpus paper", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Paper", exact: true }).click();
  await page.getByRole("textbox").fill("Corpus Paper 2001");

  // Match the suggestion, not the graph node that later shares this
  // aria-label: only the suggestion carries the author/source line.
  const suggestion = page.getByRole("button", {
    name: /Corpus Paper 2001 on Neural Retrieval.*in corpus/i,
  });
  await expect(suggestion).toBeVisible();
  await suggestion.click();

  // Selecting closes the dropdown for good; it must not reopen over the
  // results when the input text changes to the selected title
  await expect(suggestion).not.toBeVisible();

  // Seed paper + its recommendations. exact: true matches only the graph
  // node; the autocomplete suggestion has trailing author text.
  await expect(page.locator("text=/\\d+ papers/")).toBeVisible({
    timeout: 15_000,
  });
  // The recommend view is shareable
  expect(new URL(page.url()).searchParams.get("paper")).toBe("2001");
  const seedNode = page.getByRole("button", {
    name: "Corpus Paper 2001 on Neural Retrieval",
    exact: true,
  });
  await expect(seedNode).toBeVisible();

  // The seed paper's detail panel disables expansion
  await seedNode.dispatchEvent("click");
  await expect(page.getByText(/Seed paper of this view/)).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Find related papers/ }),
  ).toBeDisabled();
});
