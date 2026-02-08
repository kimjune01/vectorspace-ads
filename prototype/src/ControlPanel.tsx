import { useState } from "react";
import type { Advertiser, AuctionMetrics } from "./types";

interface Props {
  advertisers: Advertiser[];
  metrics: AuctionMetrics | null;
  anisotropic: boolean;
  showStream: boolean;
  showRestrictions: boolean;
  onAnisotropicToggle: () => void;
  onStreamToggle: () => void;
  onRestrictionsToggle: () => void;
  onReset: () => void;
  onAdvertiserUpdate: (id: string, updates: Partial<Advertiser>) => void;
  onRefineTargeting: (id: string) => void;
}

function Toggle({
  checked,
  label,
  activeColor,
  onClick,
}: {
  checked: boolean;
  label: string;
  activeColor: string;
  onClick: () => void;
}) {
  return (
    <div
      style={{
        padding: "8px 16px",
        background: checked ? "#e3f2fd" : "#f5f5f5",
        borderRadius: 8,
        display: "flex",
        alignItems: "center",
        gap: 8,
        cursor: "pointer",
        border: checked ? "1px solid #90caf9" : "1px solid #e0e0e0",
      }}
      onClick={onClick}
    >
      <div
        style={{
          width: 36,
          height: 20,
          borderRadius: 10,
          background: checked ? activeColor : "#bbb",
          position: "relative",
          transition: "background 0.2s",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            width: 16,
            height: 16,
            borderRadius: 8,
            background: "white",
            position: "absolute",
            top: 2,
            left: checked ? 18 : 2,
            transition: "left 0.2s",
          }}
        />
      </div>
      <span style={{ fontSize: 13, fontWeight: 500 }}>{label}</span>
    </div>
  );
}

export default function ControlPanel({
  advertisers,
  metrics,
  anisotropic,
  showStream,
  showRestrictions,
  onAnisotropicToggle,
  onStreamToggle,
  onRestrictionsToggle,
  onReset,
  onAdvertiserUpdate,
  onRefineTargeting,
}: Props) {
  const [showFormula, setShowFormula] = useState(false);

  return (
    <div style={{ width: 320, display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Header */}
      <div style={{ padding: "12px 16px", background: "#1a1a2e", borderRadius: 8, color: "white" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <h2 style={{ margin: 0, fontSize: 16 }}>Power Diagram Ad Auction</h2>
            <p style={{ margin: "4px 0 0", fontSize: 12, color: "#aaa" }}>
              Drag advertisers. Adjust bids. Watch territories shift.
            </p>
          </div>
          <div
            title="Score formula: argmax_i [log(bid_i) - ||x - c_i||^2 / sigma_i^2]"
            style={{
              width: 22,
              height: 22,
              borderRadius: 11,
              background: "rgba(255,255,255,0.15)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: 700,
              color: "#aaa",
              flexShrink: 0,
            }}
            onClick={() => setShowFormula((p) => !p)}
          >
            i
          </div>
        </div>
        {showFormula && (
          <div
            style={{
              marginTop: 8,
              padding: "8px 10px",
              background: "rgba(255,255,255,0.08)",
              borderRadius: 6,
              fontSize: 11,
              color: "#ccc",
              fontFamily: "monospace",
              lineHeight: 1.5,
            }}
          >
            winner(x) = argmax_i [ log(b_i) - ||x - c_i||^2 / sigma_i^2 ]
            <br />
            Higher bids expand territory; smaller sigma concentrates it.
          </div>
        )}
      </div>

      {/* Toggles */}
      <Toggle
        checked={anisotropic}
        label={`Anisotropic Mode ${anisotropic ? "ON" : "OFF"}`}
        activeColor="#2196F3"
        onClick={onAnisotropicToggle}
      />
      <Toggle
        checked={showStream}
        label={`Impression Stream ${showStream ? "ON" : "OFF"}`}
        activeColor="#4CAF50"
        onClick={onStreamToggle}
      />
      <Toggle
        checked={showRestrictions}
        label={`Restriction Zones ${showRestrictions ? "ON" : "OFF"}`}
        activeColor="#F44336"
        onClick={onRestrictionsToggle}
      />

      {/* Advertiser Controls */}
      {advertisers.map((adv, i) => {
        const m = metrics?.perAdvertiser[i];
        return (
          <div
            key={adv.id}
            style={{
              padding: "10px 14px",
              background: "white",
              borderRadius: 8,
              border: `2px solid ${adv.color}`,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 6,
                  background: adv.color,
                }}
              />
              <span style={{ fontWeight: 600, fontSize: 13 }}>{adv.name}</span>
              {m && (
                <span style={{ marginLeft: "auto", fontSize: 11, color: "#888" }}>
                  {(m.impressions * 100).toFixed(1)}% impr
                </span>
              )}
            </div>

            {/* Bid slider */}
            <div style={{ marginBottom: 4 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#666" }}>
                <span>Bid</span>
                <span style={{ fontWeight: 600 }}>${adv.bid.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="15"
                step="0.1"
                value={adv.bid}
                onChange={(e) => onAdvertiserUpdate(adv.id, { bid: parseFloat(e.target.value) })}
                style={{ width: "100%", accentColor: adv.color }}
              />
            </div>

            {/* Sigma slider */}
            {!anisotropic && (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#666" }}>
                  <span>Reach (sigma)</span>
                  <span>{adv.sigma.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.6"
                  step="0.01"
                  value={adv.sigma}
                  onChange={(e) => onAdvertiserUpdate(adv.id, { sigma: parseFloat(e.target.value) })}
                  style={{ width: "100%", accentColor: adv.color }}
                />
              </div>
            )}

            {/* Anisotropic sliders */}
            {anisotropic && (
              <>
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#666" }}>
                    <span>Reach X (sigma_x)</span>
                    <span>{(adv.sigmaX ?? adv.sigma).toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.6"
                    step="0.01"
                    value={adv.sigmaX ?? adv.sigma}
                    onChange={(e) => onAdvertiserUpdate(adv.id, { sigmaX: parseFloat(e.target.value) })}
                    style={{ width: "100%", accentColor: adv.color }}
                  />
                </div>
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#666" }}>
                    <span>Reach Y (sigma_y)</span>
                    <span>{(adv.sigmaY ?? adv.sigma).toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.6"
                    step="0.01"
                    value={adv.sigmaY ?? adv.sigma}
                    onChange={(e) => onAdvertiserUpdate(adv.id, { sigmaY: parseFloat(e.target.value) })}
                    style={{ width: "100%", accentColor: adv.color }}
                  />
                </div>
              </>
            )}

            {/* Metrics */}
            {m && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 4,
                  marginTop: 6,
                  fontSize: 10,
                  color: "#888",
                }}
              >
                <span>Area: {(m.territoryArea * 100).toFixed(1)}%</span>
                <span>Spend: ${m.spend.toFixed(3)}</span>
              </div>
            )}

            {/* Refine Targeting button */}
            <button
              onClick={() => onRefineTargeting(adv.id)}
              style={{
                marginTop: 6,
                width: "100%",
                padding: "5px 8px",
                background: "#f8f9fa",
                border: `1px solid ${adv.color}40`,
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 11,
                fontWeight: 500,
                color: adv.color,
                transition: "background 0.15s",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = `${adv.color}15`; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "#f8f9fa"; }}
            >
              Refine Targeting
            </button>
          </div>
        );
      })}

      {/* Platform Metrics */}
      {metrics && (
        <div
          style={{
            padding: "10px 14px",
            background: "#f8f9fa",
            borderRadius: 8,
            border: "1px solid #dee2e6",
          }}
        >
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>Platform Metrics</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 12 }}>
            <span style={{ color: "#666" }}>Revenue:</span>
            <span style={{ fontWeight: 600 }}>${metrics.platformRevenue.toFixed(3)}</span>
            <span style={{ color: "#666" }}>Welfare:</span>
            <span style={{ fontWeight: 600 }}>${metrics.socialWelfare.toFixed(3)}</span>
          </div>
        </div>
      )}

      {/* Reset button */}
      <button
        onClick={onReset}
        style={{
          padding: "8px 16px",
          background: "white",
          borderRadius: 8,
          border: "1px solid #ccc",
          cursor: "pointer",
          fontSize: 13,
          fontWeight: 500,
          color: "#555",
          transition: "background 0.15s",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = "#f0f0f0"; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = "white"; }}
      >
        Reset to Defaults
      </button>

      {/* Info */}
      <div style={{ fontSize: 11, color: "#999", padding: "0 4px" }}>
        <p style={{ margin: 0 }}>
          Each pixel shows the winning advertiser. Boundaries mark equal bid-adjusted distances.
          Color vibrancy indicates impression density.
        </p>
      </div>
    </div>
  );
}
