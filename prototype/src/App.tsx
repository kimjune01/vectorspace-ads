import { useState, useCallback } from "react";
import type { Advertiser, AuctionMetrics } from "./types";
import { DEFAULT_ADVERTISERS, DEFAULT_CLUSTERS } from "./data";
import AuctionCanvas from "./AuctionCanvas";
import ControlPanel from "./ControlPanel";

const CANVAS_SIZE = 600;

function makeInitialAdvertisers(): Advertiser[] {
  return DEFAULT_ADVERTISERS.map((a) => ({
    ...a,
    sigmaX: a.sigma,
    sigmaY: a.sigma,
  }));
}

function App() {
  const [advertisers, setAdvertisers] = useState<Advertiser[]>(makeInitialAdvertisers);
  const [anisotropic, setAnisotropic] = useState(false);
  const [showStream, setShowStream] = useState(false);
  const [metrics, setMetrics] = useState<AuctionMetrics | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);

  const handleDragStart = useCallback((id: string) => {
    setDraggingId(id);
  }, []);

  const handleDragMove = useCallback(
    (x: number, y: number) => {
      if (!draggingId) return;
      setAdvertisers((prev) =>
        prev.map((a) => (a.id === draggingId ? { ...a, center: [x, y] as [number, number] } : a)),
      );
    },
    [draggingId],
  );

  const handleDragEnd = useCallback(() => {
    setDraggingId(null);
  }, []);

  const handleAdvertiserUpdate = useCallback((id: string, updates: Partial<Advertiser>) => {
    setAdvertisers((prev) => prev.map((a) => (a.id === id ? { ...a, ...updates } : a)));
  }, []);

  const handleAnisotropicToggle = useCallback(() => {
    setAnisotropic((prev) => !prev);
  }, []);

  const handleStreamToggle = useCallback(() => {
    setShowStream((prev) => !prev);
  }, []);

  const handleReset = useCallback(() => {
    setAdvertisers(makeInitialAdvertisers());
    setAnisotropic(false);
    setShowStream(false);
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f5f5f5",
        padding: 24,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Title */}
        <div style={{ marginBottom: 20 }}>
          <h1 style={{ margin: 0, fontSize: 24, color: "#1a1a2e" }}>
            Embedding Ad Auction Explorer
          </h1>
          <p style={{ margin: "4px 0 0", color: "#666", fontSize: 14 }}>
            Visualizing how power diagrams allocate advertising territory in continuous embedding
            space
          </p>
        </div>

        {/* Main layout */}
        <div style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
          <AuctionCanvas
            advertisers={advertisers}
            clusters={DEFAULT_CLUSTERS}
            anisotropic={anisotropic}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            draggingId={draggingId}
            showStream={showStream}
            onDragStart={handleDragStart}
            onDragMove={handleDragMove}
            onDragEnd={handleDragEnd}
            onMetricsUpdate={setMetrics}
          />
          <ControlPanel
            advertisers={advertisers}
            metrics={metrics}
            anisotropic={anisotropic}
            showStream={showStream}
            onAnisotropicToggle={handleAnisotropicToggle}
            onStreamToggle={handleStreamToggle}
            onReset={handleReset}
            onAdvertiserUpdate={handleAdvertiserUpdate}
          />
        </div>

        {/* Footer explanation */}
        <div
          style={{
            marginTop: 20,
            padding: 16,
            background: "white",
            borderRadius: 8,
            border: "1px solid #e0e0e0",
            fontSize: 13,
            color: "#555",
            lineHeight: 1.5,
          }}
        >
          <strong>How it works:</strong> Each advertiser has a position, bid, and reach (sigma). The
          winner at each point is{" "}
          <code>argmax_i [log(bid_i) - ||x - center_i||^2 / sigma_i^2]</code>, forming a{" "}
          <strong>power diagram</strong> -- a bid-weighted Voronoi tessellation. Color vibrancy
          indicates impression traffic density.
        </div>
      </div>
    </div>
  );
}

export default App;
