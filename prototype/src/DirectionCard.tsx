import type { CandidateDirection } from "./types";

interface Props {
  candidate: CandidateDirection;
  onClick: () => void;
  onHover: (position: [number, number] | null) => void;
}

export default function DirectionCard({ candidate, onClick, onHover }: Props) {
  const isNearby = candidate.distance === "nearby";

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => onHover(candidate.position)}
      onMouseLeave={() => onHover(null)}
      style={{
        padding: "10px 12px",
        background: "white",
        borderRadius: 8,
        border: `1.5px solid ${isNearby ? "#90caf9" : "#ce93d8"}`,
        cursor: "pointer",
        transition: "box-shadow 0.15s, transform 0.15s",
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
      }}
      onMouseOver={(e) => {
        e.currentTarget.style.boxShadow = "0 3px 12px rgba(0,0,0,0.15)";
        e.currentTarget.style.transform = "translateY(-1px)";
      }}
      onMouseOut={(e) => {
        e.currentTarget.style.boxShadow = "0 1px 3px rgba(0,0,0,0.08)";
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              fontSize: 9,
              fontWeight: 600,
              padding: "1px 6px",
              borderRadius: 4,
              background: isNearby ? "#e3f2fd" : "#f3e5f5",
              color: isNearby ? "#1565c0" : "#7b1fa2",
              textTransform: "uppercase",
              letterSpacing: 0.5,
            }}
          >
            {isNearby ? "Fine-tune" : "Explore"}
          </span>
          <span style={{ fontWeight: 600, fontSize: 13, color: "#1a1a2e" }}>
            {candidate.label}
          </span>
        </div>
        <span style={{ fontWeight: 700, fontSize: 13, color: "#2e7d32" }}>
          ${candidate.estimatedCPM.toFixed(2)}
        </span>
      </div>

      {/* Gloss */}
      <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>
        {candidate.gloss}
      </div>

      {/* Example queries */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {candidate.examples.slice(0, 3).map((ex, i) => (
          <span
            key={i}
            style={{
              fontSize: 10,
              color: "#555",
              background: "#f5f5f5",
              padding: "2px 6px",
              borderRadius: 4,
              fontStyle: "italic",
            }}
          >
            &ldquo;{ex}&rdquo;
          </span>
        ))}
      </div>
    </div>
  );
}
