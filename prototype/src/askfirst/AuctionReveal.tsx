import { useMemo } from "react";
import type { ScoredAdvertiser } from "./plumberEmbeddings";
import { secondPricePayment } from "./plumberEmbeddings";

interface AuctionRevealProps {
  scores: ScoredAdvertiser[];
  onClose: () => void;
}

export default function AuctionReveal({ scores, onClose }: AuctionRevealProps) {
  const payment = useMemo(() => secondPricePayment(scores), [scores]);
  const topN = scores.slice(0, 4);
  const winner = scores[0];
  if (!winner) return null;

  const maxScore = Math.max(...topN.map((s) => s.score));
  const minScore = Math.min(...topN.map((s) => s.score));
  const scoreRange = maxScore - minScore || 1;

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.4)",
          zIndex: 100,
          animation: "fadeIn 0.2s ease-out",
        }}
      />

      {/* Bottom sheet */}
      <div
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          maxHeight: "80vh",
          background: "white",
          borderRadius: "16px 16px 0 0",
          zIndex: 101,
          animation: "slideUp 0.3s ease-out",
          overflowY: "auto",
          padding: "20px 20px 32px",
          maxWidth: 520,
          margin: "0 auto",
        }}
      >
        {/* Handle & close */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ width: 40, height: 4, background: "#ddd", borderRadius: 2 }} />
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              fontSize: 20,
              cursor: "pointer",
              color: "#888",
              padding: "0 4px",
            }}
          >
            ✕
          </button>
        </div>

        <div style={{ fontSize: 12, color: "#888", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
          Auction Result
        </div>

        {/* Winner card */}
        <div
          style={{
            background: "#fdf6ee",
            border: `2px solid ${winner.advertiser.color}`,
            borderRadius: 12,
            padding: 16,
            marginBottom: 20,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: winner.advertiser.color,
                flexShrink: 0,
              }}
            />
            <div style={{ fontSize: 18, fontWeight: 700, color: "#1a1a2e" }}>
              {winner.advertiser.name}
            </div>
          </div>
          <div style={{ fontSize: 14, color: "#555", marginBottom: 12, fontFamily: "Lora, Georgia, serif" }}>
            {winner.advertiser.description}
          </div>

          {/* Relevance bar */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#888", marginBottom: 4 }}>
              <span>Relevance</span>
              <span>{(winner.similarity * 100).toFixed(0)}%</span>
            </div>
            <div style={{ height: 6, background: "#e8e8e8", borderRadius: 3 }}>
              <div
                style={{
                  height: "100%",
                  width: `${winner.similarity * 100}%`,
                  background: winner.advertiser.color,
                  borderRadius: 3,
                  transition: "width 0.5s",
                }}
              />
            </div>
          </div>

          {/* Score breakdown */}
          <div
            style={{
              background: "#f8f6f3",
              borderRadius: 8,
              padding: 10,
              fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
              fontSize: 12,
              lineHeight: 1.8,
              color: "#444",
            }}
          >
            <div>log(bid)     = {Math.log(winner.advertiser.bid).toFixed(3)}</div>
            <div>1 - cosine   = {(1 - winner.similarity).toFixed(3)}</div>
            <div>penalty/σ²   = {((1 - winner.similarity) ** 2 / winner.advertiser.sigma ** 2).toFixed(3)}</div>
            <div style={{ borderTop: "1px solid #ddd", paddingTop: 4, marginTop: 4, fontWeight: 600 }}>
              score        = {winner.score.toFixed(3)}
            </div>
          </div>
        </div>

        {/* Payment */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, padding: "0 4px" }}>
          <span style={{ fontSize: 13, color: "#888" }}>Second-price payment:</span>
          <span style={{ fontSize: 16, fontWeight: 700, color: "#1a1a2e" }}>${payment.toFixed(2)}</span>
        </div>

        {/* All competitors */}
        <div style={{ fontSize: 12, color: "#888", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>
          Top Competitors
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {topN.map((s, i) => (
            <div key={s.advertiser.id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 18, textAlign: "right", fontSize: 12, color: "#aaa", fontWeight: 600 }}>
                #{i + 1}
              </div>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: s.advertiser.color,
                  flexShrink: 0,
                }}
              />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                  <span style={{ fontSize: 13, fontWeight: i === 0 ? 700 : 400, color: "#333" }}>
                    {s.advertiser.name}
                  </span>
                  <span style={{ fontSize: 11, color: "#aaa" }}>
                    {s.score.toFixed(2)}
                  </span>
                </div>
                <div style={{ height: 4, background: "#eee", borderRadius: 2 }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${((s.score - minScore) / scoreRange) * 100}%`,
                      background: s.advertiser.color,
                      borderRadius: 2,
                      opacity: i === 0 ? 1 : 0.6,
                      transition: "width 0.4s",
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
