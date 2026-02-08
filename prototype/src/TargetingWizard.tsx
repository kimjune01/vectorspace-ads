import { useState, useCallback, useMemo } from "react";
import type { Advertiser, ImpressionCluster, CandidateDirection, TargetingState } from "./types";
import { getNearestAnchor, getCandidates, lookupPhrase } from "./semanticField";
import { estimateCPM } from "./auction";
import DirectionCard from "./DirectionCard";

interface Props {
  advertiser: Advertiser;
  advertisers: Advertiser[];
  clusters: ImpressionCluster[];
  anisotropic: boolean;
  targetingState: TargetingState;
  onStateChange: (state: TargetingState) => void;
  onAdvertiserUpdate: (id: string, updates: Partial<Advertiser>) => void;
  onGhostPreview: (pos: [number, number] | null) => void;
  onClose: () => void;
}

const REACH_PRESETS = [
  { label: "Sniper", value: 0.08, desc: "~3% of impressions" },
  { label: "Balanced", value: 0.18, desc: "~12% of impressions" },
  { label: "Blanket", value: 0.35, desc: "~35% of impressions" },
];

export default function TargetingWizard({
  advertiser,
  advertisers,
  clusters,
  anisotropic,
  targetingState,
  onStateChange,
  onAdvertiserUpdate,
  onGhostPreview,
  onClose,
}: Props) {
  const [inputText, setInputText] = useState("");
  const [freeText, setFreeText] = useState("");

  const { phase, locus, breadcrumbs, reach, refinementCount } = targetingState;

  // Current anchor info
  const currentAnchor = useMemo(() => getNearestAnchor(locus[0], locus[1]), [locus]);

  // Get candidate directions with CPM estimates
  const candidates = useMemo((): CandidateDirection[] => {
    const { nearby, distant } = getCandidates(locus[0], locus[1]);
    const toCandidate = (anchor: ReturnType<typeof getNearestAnchor>, distance: "nearby" | "distant"): CandidateDirection => ({
      label: anchor.label,
      gloss: anchor.gloss,
      examples: anchor.examples,
      position: anchor.position,
      distance,
      estimatedCPM: estimateCPM(anchor.position, advertisers, clusters, anisotropic),
    });
    return [
      ...nearby.map((a) => toCandidate(a, "nearby")),
      ...distant.map((a) => toCandidate(a, "distant")),
    ];
  }, [locus, advertisers, clusters, anisotropic]);

  const handleInitialSubmit = useCallback(() => {
    if (!inputText.trim()) return;
    const pos = lookupPhrase(inputText);
    onAdvertiserUpdate(advertiser.id, { center: pos });
    onStateChange({
      ...targetingState,
      phase: "refining",
      locus: pos,
      breadcrumbs: [pos],
      refinementCount: 0,
    });
  }, [inputText, advertiser.id, onAdvertiserUpdate, onStateChange, targetingState]);

  const handleCandidateClick = useCallback((candidate: CandidateDirection) => {
    const newPos = candidate.position;
    onAdvertiserUpdate(advertiser.id, { center: newPos });
    onGhostPreview(null);
    onStateChange({
      ...targetingState,
      locus: newPos,
      breadcrumbs: [...breadcrumbs, newPos],
      refinementCount: refinementCount + 1,
    });
  }, [advertiser.id, onAdvertiserUpdate, onGhostPreview, onStateChange, targetingState, breadcrumbs, refinementCount]);

  const handleFreeTextSubmit = useCallback(() => {
    if (!freeText.trim()) return;
    const pos = lookupPhrase(freeText);
    onAdvertiserUpdate(advertiser.id, { center: pos });
    onGhostPreview(null);
    setFreeText("");
    onStateChange({
      ...targetingState,
      locus: pos,
      breadcrumbs: [...breadcrumbs, pos],
      refinementCount: refinementCount + 1,
    });
  }, [freeText, advertiser.id, onAdvertiserUpdate, onGhostPreview, onStateChange, targetingState, breadcrumbs, refinementCount]);

  const handleLockIn = useCallback(() => {
    onStateChange({ ...targetingState, phase: "volume" });
  }, [onStateChange, targetingState]);

  const handleReachChange = useCallback((newReach: number) => {
    onAdvertiserUpdate(advertiser.id, { sigma: newReach });
    onStateChange({ ...targetingState, reach: newReach });
  }, [advertiser.id, onAdvertiserUpdate, onStateChange, targetingState]);

  const handleFinalize = useCallback(() => {
    onStateChange({ ...targetingState, phase: "locked" });
  }, [onStateChange, targetingState]);

  const handleRetarget = useCallback(() => {
    onStateChange({ ...targetingState, phase: "refining" });
  }, [onStateChange, targetingState]);

  const currentCPM = useMemo(
    () => estimateCPM(locus, advertisers, clusters, anisotropic),
    [locus, advertisers, clusters, anisotropic],
  );

  // Estimate reach percentage based on sigma
  const reachPercent = useMemo(() => {
    // Approximate: fraction of [0,1]^2 within sigma radius
    const area = Math.PI * reach * reach;
    return Math.min(100, area * 100);
  }, [reach]);

  return (
    <div style={{ width: 320, display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Header */}
      <div style={{
        padding: "10px 14px",
        background: "#1a1a2e",
        borderRadius: 8,
        color: "white",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div>
          <div style={{ fontSize: 11, color: "#aaa", textTransform: "uppercase", letterSpacing: 0.5 }}>
            Targeting Wizard
          </div>
          <div style={{ fontSize: 14, fontWeight: 600, marginTop: 2 }}>
            {advertiser.name}
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            background: "rgba(255,255,255,0.15)",
            border: "none",
            color: "#aaa",
            cursor: "pointer",
            borderRadius: 4,
            padding: "4px 8px",
            fontSize: 12,
          }}
        >
          Close
        </button>
      </div>

      {/* Phase 1: Initial Guess */}
      {phase === "initial" && (
        <div style={{
          padding: "14px",
          background: "white",
          borderRadius: 8,
          border: "1px solid #e0e0e0",
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
            Describe your target audience
          </div>
          <div style={{ fontSize: 12, color: "#666", marginBottom: 10 }}>
            Start with a natural-language description. The system will place you in the semantic space.
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleInitialSubmit()}
              placeholder='e.g., "high-intent fitness shoppers"'
              style={{
                flex: 1,
                padding: "8px 10px",
                border: "1.5px solid #ddd",
                borderRadius: 6,
                fontSize: 13,
                outline: "none",
              }}
            />
            <button
              onClick={handleInitialSubmit}
              style={{
                padding: "8px 14px",
                background: "#1a1a2e",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 13,
                fontWeight: 500,
              }}
            >
              Go
            </button>
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: "#999" }}>
            Try: &ldquo;fitness nutrition&rdquo;, &ldquo;running enthusiasts&rdquo;, &ldquo;supplement shoppers&rdquo;
          </div>
        </div>
      )}

      {/* Phase 2: Iterative Refinement */}
      {phase === "refining" && (
        <>
          {/* Current position */}
          <div style={{
            padding: "10px 14px",
            background: "white",
            borderRadius: 8,
            border: `2px solid ${advertiser.color}`,
          }}>
            <div style={{ fontSize: 11, color: "#888", marginBottom: 2 }}>
              You&apos;re targeting:
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#1a1a2e", marginBottom: 6 }}>
              {currentAnchor.label}
            </div>
            <div style={{ fontSize: 11, color: "#666", marginBottom: 6 }}>
              {currentAnchor.gloss}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 6 }}>
              {currentAnchor.examples.map((ex, i) => (
                <span
                  key={i}
                  style={{
                    fontSize: 10,
                    color: "#555",
                    background: "#f0f0f0",
                    padding: "2px 6px",
                    borderRadius: 4,
                    fontStyle: "italic",
                  }}
                >
                  &ldquo;{ex}&rdquo;
                </span>
              ))}
            </div>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#2e7d32" }}>
              Est. CPM: ${currentCPM.toFixed(2)}
            </div>
          </div>

          {/* Nearby suggestions */}
          {candidates.filter((c) => c.distance === "nearby").length > 0 && (
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#666", marginBottom: 4, padding: "0 2px" }}>
                Nearby adjustments
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {candidates
                  .filter((c) => c.distance === "nearby")
                  .map((c) => (
                    <DirectionCard
                      key={c.label}
                      candidate={c}
                      onClick={() => handleCandidateClick(c)}
                      onHover={(pos) => onGhostPreview(pos)}
                    />
                  ))}
              </div>
            </div>
          )}

          {/* Distant suggestions */}
          {candidates.filter((c) => c.distance === "distant").length > 0 && (
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#666", marginBottom: 4, padding: "0 2px" }}>
                Explore new territory
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {candidates
                  .filter((c) => c.distance === "distant")
                  .map((c) => (
                    <DirectionCard
                      key={c.label}
                      candidate={c}
                      onClick={() => handleCandidateClick(c)}
                      onHover={(pos) => onGhostPreview(pos)}
                    />
                  ))}
              </div>
            </div>
          )}

          {/* Free-text input */}
          <div style={{
            padding: "10px 12px",
            background: "#fafafa",
            borderRadius: 8,
            border: "1px solid #e0e0e0",
          }}>
            <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>
              Or describe what you&apos;re looking for...
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="text"
                value={freeText}
                onChange={(e) => setFreeText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleFreeTextSubmit()}
                placeholder='e.g., "more premium, less casual"'
                style={{
                  flex: 1,
                  padding: "6px 8px",
                  border: "1px solid #ddd",
                  borderRadius: 6,
                  fontSize: 12,
                  outline: "none",
                }}
              />
              <button
                onClick={handleFreeTextSubmit}
                style={{
                  padding: "6px 10px",
                  background: "#eee",
                  border: "1px solid #ddd",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 12,
                }}
              >
                Move
              </button>
            </div>
          </div>

          {/* Lock In button */}
          <button
            onClick={handleLockIn}
            style={{
              padding: refinementCount >= 2 ? "12px 16px" : "8px 16px",
              background: refinementCount >= 2 ? "#1a1a2e" : "white",
              color: refinementCount >= 2 ? "white" : "#555",
              border: refinementCount >= 2 ? "none" : "1px solid #ccc",
              borderRadius: 8,
              cursor: "pointer",
              fontSize: refinementCount >= 2 ? 14 : 13,
              fontWeight: refinementCount >= 2 ? 600 : 500,
              transition: "all 0.2s",
            }}
          >
            Lock In Position
            {refinementCount > 0 && (
              <span style={{ fontSize: 11, marginLeft: 6, opacity: 0.7 }}>
                ({refinementCount} step{refinementCount > 1 ? "s" : ""} taken)
              </span>
            )}
          </button>
        </>
      )}

      {/* Phase 3: Volume / Reach */}
      {phase === "volume" && (
        <div style={{
          padding: "14px",
          background: "white",
          borderRadius: 8,
          border: "1px solid #e0e0e0",
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
            Set Targeting Breadth
          </div>
          <div style={{ fontSize: 12, color: "#666", marginBottom: 12 }}>
            How wide should your targeting radius be?
          </div>

          {/* Presets */}
          <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
            {REACH_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => handleReachChange(p.value)}
                style={{
                  flex: 1,
                  padding: "8px 4px",
                  background: Math.abs(reach - p.value) < 0.01 ? "#e3f2fd" : "#f5f5f5",
                  border: Math.abs(reach - p.value) < 0.01 ? "1.5px solid #90caf9" : "1px solid #ddd",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontSize: 12,
                  fontWeight: Math.abs(reach - p.value) < 0.01 ? 600 : 400,
                }}
              >
                <div>{p.label}</div>
                <div style={{ fontSize: 10, color: "#888", marginTop: 2 }}>{p.desc}</div>
              </button>
            ))}
          </div>

          {/* Slider */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#666", marginBottom: 4 }}>
              <span>Targeting Breadth</span>
              <span style={{ fontWeight: 600 }}>Reaching ~{reachPercent.toFixed(1)}% of impressions</span>
            </div>
            <input
              type="range"
              min="0.03"
              max="0.5"
              step="0.01"
              value={reach}
              onChange={(e) => handleReachChange(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: advertiser.color }}
            />
          </div>

          {/* Confirm */}
          <button
            onClick={handleFinalize}
            style={{
              marginTop: 12,
              width: "100%",
              padding: "10px",
              background: "#1a1a2e",
              color: "white",
              border: "none",
              borderRadius: 8,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 600,
            }}
          >
            Confirm Targeting
          </button>
        </div>
      )}

      {/* Phase 4: Locked */}
      {phase === "locked" && (
        <>
          <div style={{
            padding: "14px",
            background: "white",
            borderRadius: 8,
            border: `2px solid ${advertiser.color}`,
          }}>
            <div style={{ fontSize: 11, color: "#888", marginBottom: 2 }}>Locked target</div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#1a1a2e" }}>
              {currentAnchor.label}
            </div>
            <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
              {REACH_PRESETS.find((p) => Math.abs(reach - p.value) < 0.02)?.label ?? "Custom"} reach ({reachPercent.toFixed(1)}% of impressions)
            </div>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#2e7d32", marginTop: 4 }}>
              Est. CPM: ${currentCPM.toFixed(2)}
            </div>
          </div>

          {/* Bid slider */}
          <div style={{
            padding: "10px 14px",
            background: "white",
            borderRadius: 8,
            border: "1px solid #e0e0e0",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#666" }}>
              <span>Bid</span>
              <span style={{ fontWeight: 600 }}>${advertiser.bid.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="15"
              step="0.1"
              value={advertiser.bid}
              onChange={(e) => onAdvertiserUpdate(advertiser.id, { bid: parseFloat(e.target.value) })}
              style={{ width: "100%", accentColor: advertiser.color }}
            />
          </div>

          <button
            onClick={handleRetarget}
            style={{
              padding: "8px 16px",
              background: "white",
              border: "1px solid #ccc",
              borderRadius: 8,
              cursor: "pointer",
              fontSize: 13,
              fontWeight: 500,
              color: "#555",
            }}
          >
            Re-target
          </button>
        </>
      )}
    </div>
  );
}
