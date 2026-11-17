import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import {
  expandNode,
  explainConnection,
  getPaperDetail,
  recommendByCorpusId,
  searchPapers,
} from "@/api/client";
import { paperDetail, paperNode, provenanceEdge } from "@/test/fixtures";

vi.mock("@/api/client");

const searchPapersMock = vi.mocked(searchPapers);
const expandNodeMock = vi.mocked(expandNode);
const getPaperDetailMock = vi.mocked(getPaperDetail);
const recommendMock = vi.mocked(recommendByCorpusId);
const explainMock = vi.mocked(explainConnection);

const SEARCH_RESPONSE = {
  papers: [
    paperNode({ corpusid: 1001, title: "Paper Alpha" }),
    paperNode({ corpusid: 1002, title: "Paper Beta" }),
  ],
  edges: [provenanceEdge({ source: 1001, target: 1002 })],
  query_id: "q-1",
};

beforeEach(() => {
  searchPapersMock.mockReset();
  expandNodeMock.mockReset();
  getPaperDetailMock.mockReset();
  getPaperDetailMock.mockResolvedValue(paperDetail());
  recommendMock.mockReset();
  recommendMock.mockReturnValue(() => {});
  explainMock.mockReset();
  window.history.replaceState(null, "", "/");
});

async function searchAndWait(user: ReturnType<typeof userEvent.setup>) {
  searchPapersMock.mockResolvedValue(SEARCH_RESPONSE);
  await user.type(screen.getByRole("textbox"), "attention");
  await user.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() =>
    expect(screen.getByText(/2 papers/)).toBeInTheDocument(),
  );
}

describe("App", () => {
  it("shows the empty state before any search", () => {
    render(<App />);
    expect(
      screen.getByText("Search for papers to explore the graph"),
    ).toBeInTheDocument();
  });

  it("searches, renders the graph, and syncs the URL", async () => {
    const user = userEvent.setup();
    render(<App />);

    await searchAndWait(user);

    expect(searchPapersMock).toHaveBeenCalledWith("attention");
    expect(window.location.search).toBe("?q=attention");
    expect(
      document.querySelector('circle[aria-label="Paper Alpha"]'),
    ).not.toBeNull();
  });

  it("auto-searches from the URL on load", async () => {
    window.history.replaceState(null, "", "/?q=preloaded");
    searchPapersMock.mockResolvedValue(SEARCH_RESPONSE);

    render(<App />);

    await waitFor(() => expect(searchPapersMock).toHaveBeenCalledWith("preloaded"));
  });

  it("opens the detail panel on node click and dismisses with Escape", async () => {
    const user = userEvent.setup();
    render(<App />);
    await searchAndWait(user);

    document
      .querySelector('circle[aria-label="Paper Alpha"]')!
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(
      await screen.findByRole("button", { name: "Close detail panel" }),
    ).toBeInTheDocument();

    // The detail fetch enriches the panel with the paper link
    expect(getPaperDetailMock).toHaveBeenCalledWith(1001);
    expect(
      await screen.findByRole("link", { name: /View on Semantic Scholar/ }),
    ).toHaveAttribute("href", "https://example.org/1001");

    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Close detail panel" }),
      ).not.toBeInTheDocument(),
    );
  });

  it("explains the connection to the current query on demand", async () => {
    const user = userEvent.setup();
    explainMock.mockResolvedValue("They are deeply connected.");
    render(<App />);
    await searchAndWait(user);

    document
      .querySelector('circle[aria-label="Paper Alpha"]')!
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await user.click(
      await screen.findByRole("button", { name: "Why are these related?" }),
    );

    expect(explainMock).toHaveBeenCalledWith(1001, { query: "attention" });
    expect(
      await screen.findByText("They are deeply connected."),
    ).toBeInTheDocument();
  });

  it("auto-recommends from a ?paper= permalink", async () => {
    window.history.replaceState(null, "", "/?paper=2001");
    render(<App />);

    await waitFor(() =>
      expect(recommendMock).toHaveBeenCalledWith(2001, expect.anything()),
    );
    // Permalink loads land in paper mode with the recommend in flight
    expect(
      screen.getByRole("button", { name: /Recommending/ }),
    ).toBeInTheDocument();
  });

  it("surfaces search errors", async () => {
    const user = userEvent.setup();
    searchPapersMock.mockRejectedValue(new Error("API error: 500"));
    render(<App />);

    await user.type(screen.getByRole("textbox"), "attention");
    await user.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("API error: 500")).toBeInTheDocument();
  });
});
