import { colors, fonts } from '../theme';
import { EMBEDDING_MAP_POINTS } from '../data';

interface Props {
  showQuery: boolean;
  showAdvertisers: boolean;
  showWinner: boolean;
}

export function EmbeddingMap({ showQuery, showAdvertisers, showWinner }: Props) {
  const w = 440;
  const h = 380;
  const pad = 50;

  function toX(x: number) { return pad + x * (w - 2 * pad); }
  function toY(y: number) { return pad + (1 - y) * (h - 2 * pad); }

  const queryPt = EMBEDDING_MAP_POINTS.find(p => p.id === 'query')!;
  const drChen = EMBEDDING_MAP_POINTS.find(p => p.id === 'drchen')!;
  const metro = EMBEDDING_MAP_POINTS.find(p => p.id === 'metro')!;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Embedding map showing user query position relative to advertisers" style={{ width: '100%', maxWidth: w }}>
      {/* Grid dots for background */}
      {Array.from({ length: 8 }, (_, i) =>
        Array.from({ length: 7 }, (_, j) => (
          <circle
            key={`${i}-${j}`}
            cx={pad + i * (w - 2 * pad) / 7}
            cy={pad + j * (h - 2 * pad) / 6}
            r={1}
            fill="#222"
          />
        ))
      )}

      {/* Concept labels (always shown as dim context) */}
      {EMBEDDING_MAP_POINTS
        .filter(p => p.type === 'concept')
        .map(p => (
          <text
            key={p.id}
            x={toX(p.x)}
            y={toY(p.y)}
            fill="#444"
            fontSize={9}
            fontFamily={fonts.mono}
            textAnchor="middle"
          >
            {p.label}
          </text>
        ))}

      {/* Query pin */}
      {showQuery && (
        <g style={{ transition: 'opacity 0.5s' }}>
          <circle
            cx={toX(queryPt.x)}
            cy={toY(queryPt.y)}
            r={20}
            fill="none"
            stroke={colors.glow}
            strokeWidth={1.5}
            opacity={0.3}
          >
            <animate attributeName="r" from="8" to="24" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" from="0.5" to="0" dur="2s" repeatCount="indefinite" />
          </circle>
          <circle cx={toX(queryPt.x)} cy={toY(queryPt.y)} r={6} fill={colors.glow} />
          <text
            x={toX(queryPt.x) - 14}
            y={toY(queryPt.y) + 4}
            fill={colors.glow}
            fontSize={10}
            fontFamily={fonts.mono}
            fontWeight={600}
            textAnchor="end"
          >
            Your query
          </text>
        </g>
      )}

      {/* Advertiser flags */}
      {showAdvertisers && (
        <>
          <AdvertiserFlag
            x={toX(drChen.x)}
            y={toY(drChen.y)}
            label="Dr. Chen"
            bid="$2"
            color={colors.embedGreen}
            highlight={showWinner}
          />
          <AdvertiserFlag
            x={toX(metro.x)}
            y={toY(metro.y)}
            label="Metro Ortho"
            bid="$8"
            color={colors.googleRed}
            highlight={false}
          />
          {/* Distance line from query to Dr. Chen */}
          {showWinner && (
            <line
              x1={toX(queryPt.x)}
              y1={toY(queryPt.y)}
              x2={toX(drChen.x)}
              y2={toY(drChen.y)}
              stroke={colors.embedGreen}
              strokeWidth={1.5}
              strokeDasharray="4 4"
              opacity={0.6}
            />
          )}
          {/* Distance line from query to Metro (faded) */}
          {showWinner && (
            <line
              x1={toX(queryPt.x)}
              y1={toY(queryPt.y)}
              x2={toX(metro.x)}
              y2={toY(metro.y)}
              stroke={colors.googleRed}
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.2}
            />
          )}
        </>
      )}
    </svg>
  );
}

function AdvertiserFlag({
  x, y, label, bid, color, highlight,
}: {
  x: number; y: number; label: string; bid: string; color: string; highlight: boolean;
}) {
  return (
    <g>
      {/* Flag pole */}
      <line x1={x} y1={y} x2={x} y2={y - 28} stroke={color} strokeWidth={highlight ? 2 : 1} />
      {/* Flag */}
      <rect
        x={x}
        y={y - 28}
        width={56}
        height={18}
        rx={3}
        fill={highlight ? color : '#222'}
        stroke={color}
        strokeWidth={highlight ? 2 : 1}
      />
      <text
        x={x + 28}
        y={y - 16}
        fill={highlight ? '#fff' : color}
        fontSize={8}
        fontFamily={fonts.mono}
        fontWeight={600}
        textAnchor="middle"
      >
        {label} {bid}
      </text>
      {/* Dot at base */}
      <circle cx={x} cy={y} r={3} fill={color} />
    </g>
  );
}
