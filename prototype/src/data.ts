import type { Advertiser, ImpressionCluster } from "./types";

export const DEFAULT_ADVERTISERS: Advertiser[] = [
  { id: "nike", name: "Nike", center: [0.6, 0.3], bid: 5.0, sigma: 0.3, color: "#FF6B35" },
  { id: "wholefoods", name: "Whole Foods", center: [0.3, 0.7], bid: 3.0, sigma: 0.25, color: "#4CAF50" },
  { id: "peloton", name: "Peloton", center: [0.5, 0.5], bid: 4.0, sigma: 0.2, color: "#2196F3" },
  { id: "gnc", name: "GNC", center: [0.7, 0.7], bid: 2.5, sigma: 0.35, color: "#9C27B0" },
  { id: "fitbit", name: "Fitbit", center: [0.4, 0.3], bid: 3.5, sigma: 0.25, color: "#FF9800" },
];

export const DEFAULT_CLUSTERS: ImpressionCluster[] = [
  { center: [0.5, 0.4], weight: 0.4, sigma: 0.15 },  // general fitness intent
  { center: [0.3, 0.6], weight: 0.3, sigma: 0.1 },   // nutrition-focused
  { center: [0.7, 0.5], weight: 0.3, sigma: 0.12 },  // purchase-ready fitness
];
