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
