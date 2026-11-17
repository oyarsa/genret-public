import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import NodeDetail from "./NodeDetail";
import {
  facetScores,
  paperDetail,
  paperNode,
  provenanceEdge,
} from "@/test/fixtures";

const FOUR_FACETS = {
  problems: ["problem one", "problem two"],
  methods: ["method one"],
  results: ["result one"],
  contributions: ["contribution one"],
};

function renderDetail(overrides: Partial<Parameters<typeof NodeDetail>[0]> = {}) {
  const props = {
    paper: paperNode({ facets: FOUR_FACETS }),
    detail: null,
    trail: [],
    allNodes: new Map([[1001, paperNode()]]),
    onExpand: vi.fn(),
    isExpanding: false,
    onClose: vi.fn(),
    ...overrides,
  };
  const view = render(<NodeDetail {...props} />);
  return { props, view };
}

describe("rendering", () => {
  it("shows title, authors, and metadata", () => {
    renderDetail();
    expect(
      screen.getByRole("heading", { name: /Graph Paper 1001/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Ada Author, Ben Builder/)).toBeInTheDocument();
  });

  it("hides near-zero facet score badges", () => {
    renderDetail({
      paper: paperNode({
        facets: {},
        facet_scores: facetScores({
          problems: 0,
          methods: 0.7,
          results: 0,
          contributions: 0,
        }),
      }),
    });
    expect(screen.getByText("methods")).toBeInTheDocument();
    expect(screen.queryByText("problems")).not.toBeInTheDocument();
  });

  it("links to the paper with its citation count when detail is loaded", () => {
    renderDetail({ detail: paperDetail() });
    const link = screen.getByRole("link", { name: /View on Semantic Scholar/ });
    expect(link).toHaveAttribute("href", "https://example.org/1001");
    expect(screen.getByText("42 citations")).toBeInTheDocument();
  });

  it("puts the provenance trail above the facet tabs", () => {
    const source = paperNode({ corpusid: 1001, title: "Source Paper" });
    const target = paperNode({ corpusid: 1002, title: "Target", facets: FOUR_FACETS });
    const { view } = renderDetail({
      paper: target,
      trail: [provenanceEdge({ source: 1001, target: 1002 })],
      allNodes: new Map([
        [1001, source],
        [1002, target],
      ]),
    });

    const trail = screen.getByText("Provenance trail");
    const tabs = screen.getByRole("tablist");
    expect(
      trail.compareDocumentPosition(tabs) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(view.container.textContent).toContain("Source Paper");
  });
});

describe("facet tabs", () => {
  it("shows one tab per populated facet, with the first active", () => {
    renderDetail();
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual([
      "problems",
      "methods",
      "results",
      "contributions",
    ]);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    // Only the active facet's items are shown
    expect(screen.getByText("problem one")).toBeInTheDocument();
    expect(screen.queryByText("method one")).not.toBeInTheDocument();
  });

  it("switches items when another tab is picked", async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(screen.getByRole("tab", { name: "methods" }));

    expect(screen.getByText("method one")).toBeInTheDocument();
    expect(screen.queryByText("problem one")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "methods" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("omits facets with no items", () => {
    renderDetail({
      paper: paperNode({ facets: { problems: ["only this"] } }),
    });
    expect(screen.getAllByRole("tab")).toHaveLength(1);
  });

  it("marks each item with a bullet", () => {
    renderDetail();
    const list = screen.getByRole("list");
    expect(within(list).getAllByText("•")).toHaveLength(2);
  });

  it("renders no tab panel when the paper has no facets", () => {
    renderDetail({ paper: paperNode({ facets: {} }) });
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });
});

describe("matched facets", () => {
  it("marks the facets that contributed the paper", () => {
    renderDetail({
      paper: paperNode({
        facets: FOUR_FACETS,
        matched_facets: ["methods", "results"],
      }),
    });
    expect(screen.getByRole("tab", { name: "methods" })).toHaveAttribute(
      "title",
      "methods — matched the query",
    );
    expect(screen.getByRole("tab", { name: "problems" })).toHaveAttribute(
      "title",
      "problems",
    );
  });

  it("notes the cross-facet signal, which has no tab of its own", () => {
    renderDetail({
      paper: paperNode({
        facets: FOUR_FACETS,
        matched_facets: ["problems_x_contributions"],
      }),
    });
    expect(
      screen.getByText(/Also matched via problems → contributions/),
    ).toBeInTheDocument();
  });
});

describe("explanations", () => {
  it("names what the explanation compares against", () => {
    renderDetail({
      onExplain: vi.fn(),
      explainContextLabel: 'your search “attention”',
    });
    expect(
      screen.getByText(/Compared with your search “attention”/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Why are these related?" }),
    ).toBeInTheDocument();
  });

  it("requests and then shows the explanation", async () => {
    const user = userEvent.setup();
    const onExplain = vi.fn();
    const { view } = renderDetail({
      onExplain,
      explainContextLabel: "your search “x”",
    });

    await user.click(screen.getByRole("button", { name: "Why are these related?" }));
    expect(onExplain).toHaveBeenCalled();

    view.rerender(
      <NodeDetail
        paper={paperNode({ facets: FOUR_FACETS })}
        detail={null}
        trail={[]}
        allNodes={new Map()}
        onExpand={vi.fn()}
        isExpanding={false}
        onClose={vi.fn()}
        onExplain={onExplain}
        explainContextLabel="your search “x”"
        richExplanation="Both study attention from different angles."
      />,
    );
    expect(
      screen.getByText("Both study attention from different angles."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Why are these related?" }),
    ).not.toBeInTheDocument();
  });

  it("hides the explanation block without a context", () => {
    renderDetail({ onExplain: undefined, explainContextLabel: null });
    expect(
      screen.queryByRole("button", { name: "Why are these related?" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Compared with/)).not.toBeInTheDocument();
  });
});

describe("actions", () => {
  it("closes via the close button", async () => {
    const user = userEvent.setup();
    const { props } = renderDetail();

    await user.click(screen.getByRole("button", { name: "Close detail panel" }));

    expect(props.onClose).toHaveBeenCalled();
  });

  it("expands via all facets by default", async () => {
    const user = userEvent.setup();
    const { props } = renderDetail();

    await user.click(screen.getByRole("button", { name: /Find related papers/ }));

    expect(props.onExpand).toHaveBeenCalledWith(1001, undefined);
  });

  it("expands via a chosen facet", async () => {
    const user = userEvent.setup();
    const { props } = renderDetail();

    await user.selectOptions(
      screen.getByRole("combobox", { name: /facet/i }),
      "methods",
    );
    await user.click(screen.getByRole("button", { name: /Find related papers/ }));

    expect(props.onExpand).toHaveBeenCalledWith(1001, "methods");
  });

  it("disables expansion for the recommend seed paper", () => {
    renderDetail({ expandDisabled: true });
    expect(
      screen.getByRole("button", { name: /Find related papers/ }),
    ).toBeDisabled();
    expect(screen.getByText(/Seed paper of this view/)).toBeInTheDocument();
  });
});
