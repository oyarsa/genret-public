import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import SearchBar, { type SearchMode } from "./SearchBar";
import { searchPaperResult } from "@/test/fixtures";

function renderBar(overrides: Partial<Parameters<typeof SearchBar>[0]> = {}) {
  const props = {
    mode: "text" as SearchMode,
    onModeChange: vi.fn(),
    onTextSearch: vi.fn(),
    isSearching: false,
    isRecommending: false,
    status: null,
    suggestions: [],
    isSuggesting: false,
    onQueryChange: vi.fn(),
    onPaperSelect: vi.fn(),
    onClearSuggestions: vi.fn(),
    ...overrides,
  };
  const view = render(<SearchBar {...props} />);
  return { props, view };
}

describe("text mode", () => {
  it("submits the typed query", async () => {
    const user = userEvent.setup();
    const { props } = renderBar();

    await user.type(screen.getByRole("textbox"), "attention mechanisms");
    await user.click(screen.getByRole("button", { name: "Search" }));

    expect(props.onTextSearch).toHaveBeenCalledWith("attention mechanisms");
  });

  it("disables submit for empty queries", () => {
    renderBar();
    expect(screen.getByRole("button", { name: "Search" })).toBeDisabled();
  });

  it("shows a busy label while searching", () => {
    renderBar({ isSearching: true });
    expect(screen.getByText("Searching…")).toBeInTheDocument();
  });
});

describe("mode toggle", () => {
  it("switches modes and clears state", async () => {
    const user = userEvent.setup();
    const { props } = renderBar();

    await user.click(screen.getByRole("button", { name: "Paper" }));

    expect(props.onModeChange).toHaveBeenCalledWith("paper");
    expect(props.onClearSuggestions).toHaveBeenCalled();
  });
});

describe("paper mode", () => {
  it("forwards keystrokes for suggestion lookup", async () => {
    const user = userEvent.setup();
    const { props } = renderBar({ mode: "paper" });

    await user.type(screen.getByRole("textbox"), "atten");

    expect(props.onQueryChange).toHaveBeenCalledWith("atten");
  });

  it("renders suggestions with source tags and selects on click", async () => {
    const user = userEvent.setup();
    const suggestion = searchPaperResult({ source: "in_corpus" });
    const { props } = renderBar({ mode: "paper", suggestions: [suggestion] });

    await user.type(screen.getByRole("textbox"), "graph paper");
    expect(screen.getByText("in corpus")).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: new RegExp(suggestion.title) }),
    );

    expect(props.onPaperSelect).toHaveBeenCalledWith(suggestion);
  });

  it("does not reopen the suggestion lookup after selecting", async () => {
    const user = userEvent.setup();
    const suggestion = searchPaperResult();
    const { props } = renderBar({ mode: "paper", suggestions: [suggestion] });

    await user.type(screen.getByRole("textbox"), "graph");
    await user.click(
      screen.getByRole("button", { name: new RegExp(suggestion.title) }),
    );

    // Selecting sets the input to the title; that change must not trigger
    // a fresh lookup (which used to pop the dropdown back open)
    expect(props.onQueryChange).not.toHaveBeenCalledWith(suggestion.title);
    expect(props.onClearSuggestions).toHaveBeenCalled();
  });

  it("submits the first suggestion on enter", async () => {
    const user = userEvent.setup();
    const first = searchPaperResult({ corpusid: 1 });
    const second = searchPaperResult({ corpusid: 2, title: "Another Paper" });
    const { props } = renderBar({
      mode: "paper",
      suggestions: [first, second],
    });

    await user.type(screen.getByRole("textbox"), "graph paper{Enter}");

    expect(props.onPaperSelect).toHaveBeenCalledWith(first);
  });

  it("disables recommend without suggestions", async () => {
    const user = userEvent.setup();
    renderBar({ mode: "paper" });

    await user.type(screen.getByRole("textbox"), "nothing matches this");

    expect(screen.getByRole("button", { name: "Recommend" })).toBeDisabled();
  });

  it("shows streaming status messages", () => {
    renderBar({ mode: "paper", status: "Extracting facets..." });
    expect(screen.getByText("Extracting facets...")).toBeInTheDocument();
  });
});
