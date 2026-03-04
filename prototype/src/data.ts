import type { Advertiser, ImpressionCluster } from "./types";
export { DEFAULT_RESTRICTION_ZONES } from "./restrictions";

export const DEFAULT_ADVERTISERS: Advertiser[] = [
  { id: "nike", name: "Nike", center: [0.6, 0.3], bid: 5.0, sigma: 0.3, color: "#FF6B35" },
  { id: "wholefoods", name: "Whole Foods", center: [0.3, 0.7], bid: 3.0, sigma: 0.25, color: "#4CAF50" },
  { id: "peloton", name: "Peloton", center: [0.5, 0.5], bid: 4.0, sigma: 0.2, color: "#2196F3" },
  { id: "gnc", name: "GNC", center: [0.7, 0.7], bid: 2.5, sigma: 0.35, color: "#9C27B0" },
  { id: "fitbit", name: "Fitbit", center: [0.4, 0.3], bid: 3.5, sigma: 0.25, color: "#FF9800" },
];

// Keyword mode: all advertisers have σ ≈ 0, collapsed to category cluster centers
// Fitness cluster: [0.55, 0.35], Nutrition cluster: [0.35, 0.7], Gear cluster: [0.65, 0.65]
export const KEYWORD_ADVERTISERS: Advertiser[] = [
  { id: "nike", name: "Nike", center: [0.55, 0.35], bid: 5.0, sigma: 0.01, color: "#FF6B35" },
  { id: "wholefoods", name: "Whole Foods", center: [0.35, 0.7], bid: 3.0, sigma: 0.01, color: "#4CAF50" },
  { id: "peloton", name: "Peloton", center: [0.55, 0.35], bid: 4.0, sigma: 0.01, color: "#2196F3" },
  { id: "gnc", name: "GNC", center: [0.35, 0.7], bid: 2.5, sigma: 0.01, color: "#9C27B0" },
  { id: "fitbit", name: "Fitbit", center: [0.65, 0.65], bid: 3.5, sigma: 0.01, color: "#FF9800" },
];

export const DEFAULT_CLUSTERS: ImpressionCluster[] = [
  { center: [0.5, 0.4], weight: 0.4, sigma: 0.15 },  // general fitness intent
  { center: [0.3, 0.6], weight: 0.3, sigma: 0.1 },   // nutrition-focused
  { center: [0.7, 0.5], weight: 0.3, sigma: 0.12 },  // purchase-ready fitness
];
