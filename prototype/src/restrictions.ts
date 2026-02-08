import type { RestrictionZone } from "./types";

export const DEFAULT_RESTRICTION_ZONES: RestrictionZone[] = [
  {
    id: "mental-health",
    label: "Mental Health",
    x0: 0.0,
    y0: 0.82,
    x1: 0.15,
    y1: 1.0,
  },
  {
    id: "political",
    label: "Political Content",
    x0: 0.85,
    y0: 0.0,
    x1: 1.0,
    y1: 0.18,
  },
];

/**
 * Check if a point (x, y) falls within any restriction zone.
 */
export function isRestricted(
  x: number,
  y: number,
  zones: RestrictionZone[],
): boolean {
  for (const z of zones) {
    if (x >= z.x0 && x <= z.x1 && y >= z.y0 && y <= z.y1) {
      return true;
    }
  }
  return false;
}
