import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import FacetFilter from "./FacetFilter";
import { FACET_NAMES } from "@/lib/types";

describe("FacetFilter", () => {
  it("renders a toggle per facet", () => {
    render(
      <FacetFilter activeFacets={new Set(FACET_NAMES)} onToggle={vi.fn()} />,
    );
    for (const facet of FACET_NAMES) {
      expect(screen.getByRole("button", { name: facet })).toBeInTheDocument();
    }
  });

  it("reports toggles for the clicked facet", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(
      <FacetFilter activeFacets={new Set(FACET_NAMES)} onToggle={onToggle} />,
    );

    await user.click(screen.getByRole("button", { name: "methods" }));

    expect(onToggle).toHaveBeenCalledWith("methods");
  });
});
