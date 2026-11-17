// Vitest setup: jest-dom matchers (toBeInTheDocument, toBeDisabled, ...)
// registered on vitest's expect, with their type augmentation.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// With vitest globals disabled, testing-library cannot auto-register its
// cleanup hook; without this, renders leak across tests in the same file.
afterEach(() => {
  cleanup();
});
