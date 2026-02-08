export interface Advertiser {
  id: string;
  name: string;
  center: [number, number]; // (x, y) in [0, 1]
  bid: number;
  sigma: number;
  color: string;
  sigmaX?: number;
  sigmaY?: number;
}

export interface ImpressionCluster {
  center: [number, number];
  weight: number;
  sigma: number;
}

export interface AdvertiserMetrics {
  name: string;
  impressions: number;
  territoryArea: number;
  spend: number;
  cpi: number;
}

export interface AuctionMetrics {
  perAdvertiser: AdvertiserMetrics[];
  platformRevenue: number;
  socialWelfare: number;
}

export interface RestrictionZone {
  id: string;
  label: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type TargetingPhase = "initial" | "refining" | "volume" | "locked";

export interface CandidateDirection {
  label: string;
  gloss: string;
  examples: string[];
  position: [number, number];
  distance: "nearby" | "distant";
  estimatedCPM: number;
}

export interface TargetingState {
  phase: TargetingPhase;
  locus: [number, number];
  breadcrumbs: [number, number][];
  reach: number; // sigma for the targeting circle
  refinementCount: number;
}
