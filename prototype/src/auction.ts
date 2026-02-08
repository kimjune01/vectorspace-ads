import type { Advertiser, ImpressionCluster, AuctionMetrics } from "./types";

/**
 * Compute the power diagram allocation and metrics on a grid.
 *
 * For each pixel, the winner is:
 *   argmax_i [log(b_i) - ||x - c_i||^2 / sigma_i^2]
 *
 * For anisotropic mode:
 *   argmax_i [log(b_i) - (x - cx)^2/sx^2 - (y - cy)^2/sy^2]
 */
export function computeAuction(
  advertisers: Advertiser[],
  clusters: ImpressionCluster[],
  resolution: number,
  anisotropic: boolean,
): {
  allocation: Uint8Array;
  density: Float32Array;
  metrics: AuctionMetrics;
} {
  const n = advertisers.length;
  const totalPixels = resolution * resolution;

  const allocation = new Uint8Array(totalPixels);
  const density = new Float32Array(totalPixels);

  // Precompute advertiser params
  const cx = advertisers.map((a) => a.center[0]);
  const cy = advertisers.map((a) => a.center[1]);
  const logBids = advertisers.map((a) => Math.log(a.bid));
  const sxSq = advertisers.map((a) => {
    const sx = anisotropic && a.sigmaX != null ? a.sigmaX : a.sigma;
    return sx * sx;
  });
  const sySq = advertisers.map((a) => {
    const sy = anisotropic && a.sigmaY != null ? a.sigmaY : a.sigma;
    return sy * sy;
  });

  // Compute density normalization
  let densitySum = 0;
  for (let py = 0; py < resolution; py++) {
    const y = (py + 0.5) / resolution;
    for (let px = 0; px < resolution; px++) {
      const x = (px + 0.5) / resolution;
      const idx = py * resolution + px;

      // Impression density
      let d = 0;
      for (const cl of clusters) {
        const ddx = x - cl.center[0];
        const ddy = y - cl.center[1];
        d += cl.weight * Math.exp(-(ddx * ddx + ddy * ddy) / (2 * cl.sigma * cl.sigma));
      }
      density[idx] = d;
      densitySum += d;
    }
  }

  // Normalize density
  if (densitySum > 0) {
    for (let i = 0; i < totalPixels; i++) {
      density[i] /= densitySum;
    }
  }

  // Compute allocation and metrics
  const impressions = new Float64Array(n);
  const areas = new Float64Array(n);
  const pixelArea = 1 / totalPixels;

  for (let py = 0; py < resolution; py++) {
    const y = (py + 0.5) / resolution;
    for (let px = 0; px < resolution; px++) {
      const x = (px + 0.5) / resolution;
      const idx = py * resolution + px;

      // Find winner
      let bestScore = -Infinity;
      let bestIdx = 0;

      for (let i = 0; i < n; i++) {
        const dx = x - cx[i];
        const dy = y - cy[i];
        const score = logBids[i] - (dx * dx) / sxSq[i] - (dy * dy) / sySq[i];
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }

      allocation[idx] = bestIdx;
      impressions[bestIdx] += density[idx];
      areas[bestIdx] += pixelArea;
    }
  }

  // Build metrics
  const perAdvertiser = advertisers.map((adv, i) => ({
    name: adv.name,
    impressions: impressions[i],
    territoryArea: areas[i],
    spend: adv.bid * impressions[i],
    cpi: impressions[i] > 0 ? adv.bid : 0,
  }));

  const platformRevenue = perAdvertiser.reduce((sum, m) => sum + m.spend, 0);
  const socialWelfare = perAdvertiser.reduce(
    (sum, m, i) => sum + m.impressions * advertisers[i].bid,
    0,
  );

  return {
    allocation,
    density,
    metrics: { perAdvertiser, platformRevenue, socialWelfare },
  };
}

/**
 * Convert a hex color string to [r, g, b] (0-255).
 */
export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}
