import { useState, useCallback, useMemo } from "react";
import type { Advertiser, AuctionMetrics, TargetingState } from "./types";
import { DEFAULT_ADVERTISERS, DEFAULT_CLUSTERS, DEFAULT_RESTRICTION_ZONES } from "./data";
import AuctionCanvas from "./AuctionCanvas";
import ControlPanel from "./ControlPanel";
import TargetingWizard from "./TargetingWizard";

const CANVAS_SIZE = 600;

function makeInitialAdvertisers(): Advertiser[] {
  return DEFAULT_ADVERTISERS.map((a) => ({
    ...a,
    sigmaX: a.sigma,
    sigmaY: a.sigma,
  }));
}

type AppMode = "explorer" | "targeting";

function App() {
  const [mode, setMode] = useState<AppMode>("targeting");
  const [advertisers, setAdvertisers] = useState<Advertiser[]>(makeInitialAdvertisers);
  const [anisotropic, setAnisotropic] = useState(false);
  const [showStream, setShowStream] = useState(false);
  const [showRestrictions, setShowRestrictions] = useState(false);
  const [metrics, setMetrics] = useState<AuctionMetrics | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [targetingAdvertiserId, setTargetingAdvertiserId] = useState<string | null>("nike");
  const [targetingState, setTargetingState] = useState<TargetingState | null>({
    phase: "refining",
    locus: [0.6, 0.3],
    breadcrumbs: [[0.4, 0.6], [0.6, 0.3]],
    reach: 0.3,
    refinementCount: 1,
  });
  const [ghostPreview, setGhostPreview] = useState<[number, number] | null>(null);

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

  const handleRestrictionsToggle = useCallback(() => {
    setShowRestrictions((prev) => !prev);
  }, []);

  const handleReset = useCallback(() => {
    setAdvertisers(makeInitialAdvertisers());
    setAnisotropic(false);
    setShowStream(false);
    setTargetingAdvertiserId(null);
    setTargetingState(null);
    setGhostPreview(null);
  }, []);

  const handleRefineTargeting = useCallback((id: string) => {
    const adv = advertisers.find((a) => a.id === id);
    if (!adv) return;
    setMode("targeting");
    setTargetingAdvertiserId(id);
    setTargetingState({
      phase: "refining",
      locus: adv.center,
      breadcrumbs: [adv.center],
      reach: adv.sigma,
      refinementCount: 0,
    });
  }, [advertisers]);

  const handleSwitchToExplorer = useCallback(() => {
    setMode("explorer");
    setGhostPreview(null);
  }, []);

  const handleSwitchToTargeting = useCallback(() => {
    setMode("targeting");
    // If no advertiser selected, default to Nike
    if (!targetingAdvertiserId) {
      const nike = advertisers.find((a) => a.id === "nike")!;
      setTargetingAdvertiserId("nike");
      setTargetingState({
        phase: "refining",
        locus: nike.center,
        breadcrumbs: [nike.center],
        reach: nike.sigma,
        refinementCount: 0,
      });
    }
  }, [targetingAdvertiserId, advertisers]);

  const targetingAdvertiser = useMemo(
    () => advertisers.find((a) => a.id === targetingAdvertiserId) ?? null,
    [advertisers, targetingAdvertiserId],
  );

  // Reach circle for the canvas overlay
  const reachCircle = useMemo(() => {
    if (!targetingState || !targetingAdvertiser) return null;
    if (targetingState.phase === "initial") return null;
    return {
      center: targetingState.locus,
      radius: targetingState.reach,
    };
  }, [targetingState, targetingAdvertiser]);

  const showWizard = mode === "targeting" && targetingAdvertiserId !== null && targetingState !== null && targetingAdvertiser !== null;

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
        <div style={{ marginBottom: 0 }}>
          <h1 style={{ margin: 0, fontSize: 24, color: "#1a1a2e" }}>
            Embedding Ad Auction Explorer
          </h1>
          <p style={{ margin: "4px 0 0", color: "#666", fontSize: 14 }}>
            Visualizing how power diagrams allocate advertising territory in continuous embedding
            space
          </p>
        </div>

        {/* Tab bar */}
        <div style={{ display: "flex", gap: 0, marginTop: 16, marginBottom: 16 }}>
          {([
            { key: "explorer" as AppMode, label: "Auction Explorer", desc: "Drag advertisers, adjust bids, watch territories shift" },
            { key: "targeting" as AppMode, label: "Targeting Wizard", desc: "Navigate embedding space through language and examples" },
          ]).map((tab) => (
            <button
              key={tab.key}
              onClick={() => tab.key === "explorer" ? handleSwitchToExplorer() : handleSwitchToTargeting()}
              style={{
                flex: 1,
                padding: "12px 16px",
                background: mode === tab.key ? "#1a1a2e" : "white",
                color: mode === tab.key ? "white" : "#555",
                border: mode === tab.key ? "1px solid #1a1a2e" : "1px solid #ddd",
                borderRadius: tab.key === "explorer" ? "8px 0 0 8px" : "0 8px 8px 0",
                cursor: "pointer",
                textAlign: "left",
                transition: "all 0.15s",
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 600 }}>{tab.label}</div>
              <div style={{ fontSize: 11, marginTop: 2, opacity: 0.7 }}>{tab.desc}</div>
            </button>
          ))}
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
            restrictionZones={DEFAULT_RESTRICTION_ZONES}
            showRestrictions={showRestrictions}
            ghostPreview={showWizard ? ghostPreview : null}
            breadcrumbs={showWizard ? targetingState?.breadcrumbs : undefined}
            reachCircle={showWizard ? reachCircle : null}
            activeAdvertiserId={showWizard ? targetingAdvertiserId : null}
          />
          {showWizard ? (
            <TargetingWizard
              advertiser={targetingAdvertiser}
              advertisers={advertisers}
              clusters={DEFAULT_CLUSTERS}
              anisotropic={anisotropic}
              targetingState={targetingState}
              onStateChange={setTargetingState}
              onAdvertiserUpdate={handleAdvertiserUpdate}
              onGhostPreview={setGhostPreview}
              onSwitchAdvertiser={handleRefineTargeting}
            />
          ) : (
            <ControlPanel
              advertisers={advertisers}
              metrics={metrics}
              anisotropic={anisotropic}
              showStream={showStream}
              showRestrictions={showRestrictions}
              onAnisotropicToggle={handleAnisotropicToggle}
              onStreamToggle={handleStreamToggle}
              onRestrictionsToggle={handleRestrictionsToggle}
              onReset={handleReset}
              onAdvertiserUpdate={handleAdvertiserUpdate}
              onRefineTargeting={handleRefineTargeting}
            />
          )}
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
