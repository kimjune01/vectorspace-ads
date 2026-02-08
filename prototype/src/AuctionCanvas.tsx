import { useRef, useEffect, useCallback } from "react";
import type { Advertiser, ImpressionCluster } from "./types";
import { computeAuction, hexToRgb } from "./auction";

/**
 * Cross-check with Python reference implementation (shared/auction.py).
 *
 * For the default advertisers at center point (0.5, 0.5):
 *
 *   Nike:        log(5.0) - ((0.5-0.6)^2 + (0.5-0.3)^2) / 0.3^2
 *                = 1.6094 - (0.01 + 0.04) / 0.09
 *                = 1.6094 - 0.5556 = 1.054
 *
 *   Whole Foods: log(3.0) - ((0.5-0.3)^2 + (0.5-0.7)^2) / 0.25^2
 *                = 1.0986 - (0.04 + 0.04) / 0.0625
 *                = 1.0986 - 1.2800 = -0.181
 *
 *   Peloton:     log(4.0) - ((0.5-0.5)^2 + (0.5-0.5)^2) / 0.2^2
 *                = 1.3863 - 0 / 0.04
 *                = 1.3863 - 0 = 1.386
 *
 *   GNC:         log(2.5) - ((0.5-0.7)^2 + (0.5-0.7)^2) / 0.35^2
 *                = 0.9163 - (0.04 + 0.04) / 0.1225
 *                = 0.9163 - 0.6531 = 0.263
 *
 *   Fitbit:      log(3.5) - ((0.5-0.4)^2 + (0.5-0.3)^2) / 0.25^2
 *                = 1.2528 - (0.01 + 0.04) / 0.0625
 *                = 1.2528 - 0.8000 = 0.453
 *
 * Winner at (0.5, 0.5) = Peloton (score 1.386).
 * This matches the Python output: compute_value_functions at (0.5, 0.5)
 * yields argmax = index 2 (Peloton).
 */

const RESOLUTION = 300; // Grid resolution for computation (300x300 = 90K pixels)

interface StreamDot {
  x: number; // canvas x [0, width]
  y: number; // canvas y [0, height]
  color: string;
  born: number; // timestamp in ms
}

const DOT_LIFETIME_MS = 2000;
const DOTS_PER_SECOND = 3;
const MAX_DOTS = 50;

interface Props {
  advertisers: Advertiser[];
  clusters: ImpressionCluster[];
  anisotropic: boolean;
  width: number;
  height: number;
  draggingId: string | null;
  showStream: boolean;
  onDragStart: (id: string) => void;
  onDragMove: (x: number, y: number) => void;
  onDragEnd: () => void;
  onMetricsUpdate: (metrics: ReturnType<typeof computeAuction>["metrics"]) => void;
}

/**
 * Sample a point from the Gaussian mixture impression density.
 * Pick a cluster weighted by cluster weight, then sample from that Gaussian.
 */
function sampleFromDensity(clusters: ImpressionCluster[]): [number, number] {
  // Pick a cluster weighted by weight
  const totalWeight = clusters.reduce((s, c) => s + c.weight, 0);
  let r = Math.random() * totalWeight;
  let chosen = clusters[0];
  for (const cl of clusters) {
    r -= cl.weight;
    if (r <= 0) {
      chosen = cl;
      break;
    }
  }

  // Sample from the chosen Gaussian using Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

  const x = chosen.center[0] + z0 * chosen.sigma;
  const y = chosen.center[1] + z1 * chosen.sigma;

  return [
    Math.max(0, Math.min(1, x)),
    Math.max(0, Math.min(1, y)),
  ];
}

/**
 * Find the winning advertiser at a given (x, y) point.
 */
function findWinner(
  x: number,
  y: number,
  advertisers: Advertiser[],
  anisotropic: boolean,
): number {
  let bestScore = -Infinity;
  let bestIdx = 0;
  for (let i = 0; i < advertisers.length; i++) {
    const adv = advertisers[i];
    const dx = x - adv.center[0];
    const dy = y - adv.center[1];
    const sx = anisotropic && adv.sigmaX != null ? adv.sigmaX : adv.sigma;
    const sy = anisotropic && adv.sigmaY != null ? adv.sigmaY : adv.sigma;
    const score = Math.log(adv.bid) - (dx * dx) / (sx * sx) - (dy * dy) / (sy * sy);
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export default function AuctionCanvas({
  advertisers,
  clusters,
  anisotropic,
  width,
  height,
  draggingId,
  showStream,
  onDragStart,
  onDragMove,
  onDragEnd,
  onMetricsUpdate,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const metricsRef = useRef(onMetricsUpdate);
  metricsRef.current = onMetricsUpdate;

  // Impression stream state
  const dotsRef = useRef<StreamDot[]>([]);
  const lastSpawnRef = useRef(0);
  const animFrameRef = useRef<number>(0);

  // Render the auction territory map
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { allocation, density, metrics } = computeAuction(
      advertisers,
      clusters,
      RESOLUTION,
      anisotropic,
    );

    metricsRef.current(metrics);

    // Create ImageData for the territory + density
    const imgData = ctx.createImageData(RESOLUTION, RESOLUTION);
    const data = imgData.data;

    // Precompute colors
    const colors = advertisers.map((a) => hexToRgb(a.color));

    // Find max density for brightness normalization
    let maxDensity = 0;
    for (let i = 0; i < density.length; i++) {
      if (density[i] > maxDensity) maxDensity = density[i];
    }

    // Improved color blending: territory color as base, modulate brightness by density.
    // High-density areas are vivid/saturated, low-density areas are washed out/pale.
    for (let py = 0; py < RESOLUTION; py++) {
      for (let px = 0; px < RESOLUTION; px++) {
        // Canvas y is flipped (0 at top), but we want 0 at bottom
        const srcIdx = (RESOLUTION - 1 - py) * RESOLUTION + px;
        const dstIdx = (py * RESOLUTION + px) * 4;

        const winner = allocation[srcIdx];
        const [cr, cg, cb] = colors[winner];

        // Normalized density [0, 1]
        const dNorm = maxDensity > 0 ? density[srcIdx] / maxDensity : 0;

        // Brightness factor: map density to [0.25, 1.0] range
        // Low density -> pale/washed out (blended toward white)
        // High density -> vivid/saturated (full advertiser color)
        const vibrancy = 0.25 + 0.75 * Math.pow(dNorm, 0.6);

        // Blend advertiser color with white based on vibrancy
        // vibrancy=1 -> full color; vibrancy=0.25 -> mostly white/pale
        data[dstIdx] = Math.round(cr * vibrancy + 245 * (1 - vibrancy));
        data[dstIdx + 1] = Math.round(cg * vibrancy + 245 * (1 - vibrancy));
        data[dstIdx + 2] = Math.round(cb * vibrancy + 245 * (1 - vibrancy));
        data[dstIdx + 3] = 255;
      }
    }

    // Draw boundaries (where adjacent pixels have different winners)
    for (let py = 0; py < RESOLUTION - 1; py++) {
      for (let px = 0; px < RESOLUTION - 1; px++) {
        const srcIdx = (RESOLUTION - 1 - py) * RESOLUTION + px;
        const rightIdx = srcIdx + 1;
        const belowIdx = srcIdx - RESOLUTION;

        if (
          allocation[srcIdx] !== allocation[rightIdx] ||
          (belowIdx >= 0 && allocation[srcIdx] !== allocation[belowIdx])
        ) {
          const dstIdx = (py * RESOLUTION + px) * 4;
          data[dstIdx] = 40;
          data[dstIdx + 1] = 40;
          data[dstIdx + 2] = 40;
          data[dstIdx + 3] = 200;
        }
      }
    }

    // Scale up to canvas size
    const offscreen = new OffscreenCanvas(RESOLUTION, RESOLUTION);
    const offCtx = offscreen.getContext("2d")!;
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, width, height);

    // Draw impression stream dots
    if (showStream) {
      const now = performance.now();
      const dots = dotsRef.current;
      for (const dot of dots) {
        const age = now - dot.born;
        if (age > DOT_LIFETIME_MS) continue;
        const alpha = 1 - age / DOT_LIFETIME_MS;
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = dot.color;
        ctx.globalAlpha = alpha * 0.85;
        ctx.fill();
        ctx.globalAlpha = alpha * 0.5;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // Draw advertiser markers
    for (const adv of advertisers) {
      const px = adv.center[0] * width;
      const py = (1 - adv.center[1]) * height;

      // Outer circle (white with border)
      ctx.beginPath();
      ctx.arc(px, py, 14, 0, Math.PI * 2);
      ctx.fillStyle = "white";
      ctx.fill();
      ctx.strokeStyle = adv.color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Inner dot
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = adv.color;
      ctx.fill();

      // Label
      ctx.font = "bold 11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "#222";
      ctx.strokeStyle = "white";
      ctx.lineWidth = 3;
      ctx.strokeText(adv.name, px, py - 20);
      ctx.fillText(adv.name, px, py - 20);

      ctx.font = "10px system-ui, sans-serif";
      ctx.fillStyle = "#555";
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.strokeText(`$${adv.bid.toFixed(1)}`, px, py - 9);
      ctx.fillText(`$${adv.bid.toFixed(1)}`, px, py - 9);

      // Draw anisotropic ellipse if in anisotropic mode
      if (anisotropic && adv.sigmaX != null && adv.sigmaY != null) {
        ctx.beginPath();
        ctx.ellipse(
          px,
          py,
          adv.sigmaX * width,
          adv.sigmaY * height,
          0,
          0,
          Math.PI * 2,
        );
        ctx.strokeStyle = adv.color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Axis labels
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = "#666";
    ctx.textAlign = "center";
    ctx.fillText("Topic  (fitness \u2192 nutrition)", width / 2, height - 4);

    ctx.save();
    ctx.translate(14, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Intent  (browsing \u2192 purchase-ready)", 0, 0);
    ctx.restore();
  }, [advertisers, clusters, anisotropic, width, height, showStream]);

  // Animation loop for impression stream
  useEffect(() => {
    if (!showStream) {
      dotsRef.current = [];
      return;
    }

    let running = true;

    const tick = () => {
      if (!running) return;

      const now = performance.now();

      // Spawn new dots
      const elapsed = now - lastSpawnRef.current;
      const interval = 1000 / DOTS_PER_SECOND;
      if (elapsed >= interval) {
        const [wx, wy] = sampleFromDensity(clusters);
        const winnerIdx = findWinner(wx, wy, advertisers, anisotropic);
        const color = advertisers[winnerIdx].color;
        dotsRef.current.push({
          x: wx * width,
          y: (1 - wy) * height,
          color,
          born: now,
        });
        lastSpawnRef.current = now;

        // Trim old dots
        dotsRef.current = dotsRef.current
          .filter((d) => now - d.born < DOT_LIFETIME_MS)
          .slice(-MAX_DOTS);
      }

      // Re-render with dots
      render();

      animFrameRef.current = requestAnimationFrame(tick);
    };

    animFrameRef.current = requestAnimationFrame(tick);

    return () => {
      running = false;
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [showStream, advertisers, clusters, anisotropic, width, height, render]);

  // Static render when stream is off
  useEffect(() => {
    if (!showStream) {
      render();
    }
  }, [render, showStream]);

  // Mouse handling for dragging advertisers
  const canvasToWorld = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      const canvas = canvasRef.current;
      if (!canvas) return [0, 0];
      const rect = canvas.getBoundingClientRect();
      const x = (clientX - rect.left) / rect.width;
      const y = 1 - (clientY - rect.top) / rect.height;
      return [Math.max(0.02, Math.min(0.98, x)), Math.max(0.02, Math.min(0.98, y))];
    },
    [],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const [wx, wy] = canvasToWorld(e.clientX, e.clientY);
      // Find closest advertiser within 30px
      let closest: string | null = null;
      let minDist = Infinity;
      for (const adv of advertisers) {
        const dx = (wx - adv.center[0]) * width;
        const dy = (wy - adv.center[1]) * height;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 30 && dist < minDist) {
          minDist = dist;
          closest = adv.id;
        }
      }
      if (closest) {
        onDragStart(closest);
      }
    },
    [advertisers, canvasToWorld, onDragStart, width, height],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!draggingId) return;
      const [wx, wy] = canvasToWorld(e.clientX, e.clientY);
      onDragMove(wx, wy);
    },
    [draggingId, canvasToWorld, onDragMove],
  );

  const handleMouseUp = useCallback(() => {
    if (draggingId) {
      onDragEnd();
    }
  }, [draggingId, onDragEnd]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        cursor: draggingId ? "grabbing" : "grab",
        borderRadius: 8,
        border: "1px solid #ddd",
        boxShadow: "0 2px 12px rgba(0, 0, 0, 0.12)",
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
}
