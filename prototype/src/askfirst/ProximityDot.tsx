interface ProximityDotProps {
  proximity: number; // 0-1
  onTap: () => void;
}

export default function ProximityDot({ proximity, onTap }: ProximityDotProps) {
  // Thresholds tuned to realistic cosine range (sparse message vs dense advertiser vectors)
  // After 1 message: ~0.35-0.45, after 2: ~0.50-0.60, after 3+: ~0.60+
  const t = Math.max(0, Math.min(1, (proximity - 0.15) / 0.45));
  const intensity = t * t; // quadratic ease-in

  const opacity = 0.05 + intensity * 0.95;
  const glowSize = intensity * 12;
  const scale = 1 + intensity * 0.15;
  const shouldPulse = proximity > 0.45;

  return (
    <button
      onClick={onTap}
      aria-label="View nearby advertisers"
      style={{
        position: "absolute",
        bottom: 4,
        right: 4,
        width: 44,
        height: 44,
        background: "none",
        border: "none",
        padding: 0,
        cursor: intensity > 0.1 ? "pointer" : "default",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 2,
        zIndex: 10,
      }}
    >
      {/* Visible dot */}
      <div
        style={{
          width: 18,
          height: 18,
          borderRadius: "50%",
          background: "#D4883A",
          opacity,
          transform: `scale(${scale})`,
          boxShadow: glowSize > 0 ? `0 0 ${glowSize}px ${glowSize * 0.5}px #F5B041` : "none",
          transition: "opacity 0.4s, transform 0.4s, box-shadow 0.4s",
          // Pulse via filter so it doesn't conflict with inline transform
          animation: shouldPulse ? "dotPulse 2s ease-in-out infinite" : "none",
        }}
      />
      {/* Hint label */}
      {intensity > 0.25 && (
        <div
          style={{
            fontSize: 9,
            fontWeight: 600,
            color: "#D4883A",
            opacity: Math.min(1, (intensity - 0.25) * 4),
            transition: "opacity 0.4s",
            letterSpacing: 0.5,
          }}
        >
          TAP
        </div>
      )}
    </button>
  );
}
