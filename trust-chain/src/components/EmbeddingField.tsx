import { colors, fonts } from '../theme';
import { EMBEDDING_MAP_POINTS, FIELD_ADVERTISERS } from '../data';

interface Props {
  stepId: string;
}

type SubState =
  | 'gate-opens'
  | 'protocol-gap'
  | 'sigma-intro'
  | 'sigma-incentives'
  | 'keywords-tiny-circles'
  | 'hotelling'
  | 'relocation-fee'
  | 'everyone-wins';

const STEP_TO_SUBSTATE: Record<string, SubState> = {
  'gate-opens': 'gate-opens',
  'protocol-gap': 'protocol-gap',
  'sigma-intro': 'sigma-intro',
  'sigma-incentives': 'sigma-incentives',
  'keywords-tiny-circles': 'keywords-tiny-circles',
  'hotelling': 'hotelling',
  'relocation-fee': 'relocation-fee',
  'everyone-wins': 'everyone-wins',
};

export function EmbeddingField({ stepId }: Props) {
  const subState = STEP_TO_SUBSTATE[stepId] ?? 'gate-opens';
  const w = 440;
  const h = 380;
  const pad = 50;

  function toX(x: number) { return pad + x * (w - 2 * pad); }
  function toY(y: number) { return pad + (1 - y) * (h - 2 * pad); }

  const queryPt = EMBEDDING_MAP_POINTS.find(p => p.id === 'query')!;
  const drChen = FIELD_ADVERTISERS.find(a => a.id === 'drchen')!;
  const metro = FIELD_ADVERTISERS.find(a => a.id === 'metro')!;

  // Determine visual parameters based on sub-state
  const isGreenField = subState !== 'hotelling';
  const showProtocolOverlay = subState === 'protocol-gap';
  const showSigmaCircles = ['sigma-intro', 'sigma-incentives', 'keywords-tiny-circles', 'hotelling', 'relocation-fee', 'everyone-wins'].includes(subState);
  const showKeywordDots = ['keywords-tiny-circles', 'hotelling', 'relocation-fee', 'everyone-wins'].includes(subState);
  const showSigmaSlider = subState === 'keywords-tiny-circles';
  const isHotelling = subState === 'hotelling';
  const showRelocFees = subState === 'relocation-fee';
  const isEquilibrium = subState === 'everyone-wins';
  const showInflateAnim = subState === 'sigma-incentives';

  // Positions shift during hotelling
  const drChenPos = isHotelling
    ? { x: 0.45, y: 0.48 }
    : { x: drChen.x, y: drChen.y };
  const metroPos = isHotelling
    ? { x: 0.42, y: 0.42 }
    : { x: metro.x, y: metro.y };

  // Background tint
  const bgGradient = isGreenField
    ? 'rgba(76, 175, 80, 0.03)'
    : 'rgba(255, 68, 68, 0.03)';

  return (
    <div style={{ width: '100%', maxWidth: w, position: 'relative' }}>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', background: bgGradient, borderRadius: 8 }}>
        {/* Grid dots */}
        {Array.from({ length: 8 }, (_, i) =>
          Array.from({ length: 7 }, (_, j) => (
            <circle
              key={`${i}-${j}`}
              cx={pad + i * (w - 2 * pad) / 7}
              cy={pad + j * (h - 2 * pad) / 6}
              r={1}
              fill={isGreenField ? '#2a3a2a' : '#3a2a2a'}
            />
          ))
        )}

        {/* Concept labels — stagger in */}
        {EMBEDDING_MAP_POINTS
          .filter(p => p.type === 'concept')
          .map((p, i) => (
            <g key={p.id} style={{
              opacity: 0,
              animation: `fieldFadeIn 0.4s ease ${0.1 + i * 0.08}s forwards`,
            }}>
              <text
                x={toX(p.x)}
                y={toY(p.y)}
                fill="#555"
                fontSize={9}
                fontFamily={fonts.mono}
                textAnchor="middle"
              >
                {p.label}
              </text>
            </g>
          ))}

        {/* Query pin — drops in with delay */}
        <g style={{ opacity: 0, animation: 'fieldFadeIn 0.5s ease 0.3s forwards' }}>
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
            x={toX(queryPt.x)}
            y={toY(queryPt.y) + 18}
            fill={colors.glow}
            fontSize={10}
            fontFamily={fonts.mono}
            fontWeight={600}
            textAnchor="middle"
          >
            Your query
          </text>
        </g>

        {/* Sigma circles for Dr. Chen and Metro */}
        {showSigmaCircles && (
          <g>
            {/* Dr. Chen circle */}
            <circle
              cx={toX(drChenPos.x)}
              cy={toY(drChenPos.y)}
              r={showInflateAnim ? 30 : (isHotelling ? 40 : 30)}
              fill={`${drChen.color}11`}
              stroke={drChen.color}
              strokeWidth={2}
              opacity={isEquilibrium ? 1 : 0.8}
              style={{ transition: 'cx 0.8s, cy 0.8s, r 0.5s' }}
            >
              {showInflateAnim && (
                <>
                  <animate
                    attributeName="r"
                    values="30;90;90;30"
                    dur="3s"
                    repeatCount="indefinite"
                    keyTimes="0;0.3;0.7;1"
                  />
                </>
              )}
            </circle>
            {/* Dr. Chen dot */}
            <circle
              cx={toX(drChenPos.x)}
              cy={toY(drChenPos.y)}
              r={4}
              fill={drChen.color}
              style={{ transition: 'cx 0.8s, cy 0.8s' }}
            />
            {/* Dr. Chen label */}
            <text
              x={toX(drChenPos.x)}
              y={toY(drChenPos.y) - (showInflateAnim ? 40 : 38)}
              fill={drChen.color}
              fontSize={9}
              fontFamily={fonts.mono}
              fontWeight={600}
              textAnchor="middle"
              style={{ transition: 'x 0.8s, y 0.8s' }}
            >
              Dr. Chen · $2
            </text>
            {isEquilibrium && (
              <text
                x={toX(drChenPos.x)}
                y={toY(drChenPos.y) - 26}
                fill={drChen.color}
                fontSize={8}
                fontFamily={fonts.mono}
                textAnchor="middle"
                opacity={0.7}
              >
                σ = small · specialist
              </text>
            )}

            {/* Inflate warning labels */}
            {showInflateAnim && (
              <g>
                <text x={toX(0.55)} y={toY(0.35)} fill={colors.googleRed} fontSize={8} fontFamily={fonts.mono} textAnchor="middle" opacity={0.8}>
                  bad match
                </text>
                <text x={toX(0.90)} y={toY(0.45)} fill={colors.googleRed} fontSize={8} fontFamily={fonts.mono} textAnchor="middle" opacity={0.8}>
                  wasted spend
                </text>
              </g>
            )}

            {/* Metro circle */}
            <circle
              cx={toX(metroPos.x)}
              cy={toY(metroPos.y)}
              r={isHotelling ? 45 : 70}
              fill={`${metro.color}08`}
              stroke={metro.color}
              strokeWidth={1}
              opacity={0.5}
              style={{ transition: 'cx 0.8s, cy 0.8s, r 0.5s' }}
            />
            <circle
              cx={toX(metroPos.x)}
              cy={toY(metroPos.y)}
              r={4}
              fill={metro.color}
              style={{ transition: 'cx 0.8s, cy 0.8s' }}
            />
            <text
              x={toX(metroPos.x)}
              y={toY(metroPos.y) + 82}
              fill={metro.color}
              fontSize={9}
              fontFamily={fonts.mono}
              textAnchor="middle"
              opacity={0.7}
              style={{ transition: 'x 0.8s, y 0.8s' }}
            >
              Metro Ortho · $8
            </text>
            {isEquilibrium && (
              <text
                x={toX(metroPos.x)}
                y={toY(metroPos.y) + 94}
                fill={metro.color}
                fontSize={8}
                fontFamily={fonts.mono}
                textAnchor="middle"
                opacity={0.5}
              >
                σ = large · generalist
              </text>
            )}
          </g>
        )}

        {/* Keyword dots (tiny circles) */}
        {showKeywordDots && (
          <g>
            {[
              { x: 0.30, y: 0.50, label: 'knee pain' },
              { x: 0.50, y: 0.40, label: 'running injury' },
              { x: 0.40, y: 0.65, label: 'PT near me' },
              { x: 0.55, y: 0.55, label: 'knee brace' },
            ].map((kw, i) => (
              <g key={i}>
                <circle
                  cx={toX(isHotelling ? 0.43 + i * 0.02 : kw.x)}
                  cy={toY(isHotelling ? 0.45 + i * 0.02 : kw.y)}
                  r={3}
                  fill={colors.googleOrange}
                  opacity={0.6}
                  style={{ transition: 'cx 0.8s, cy 0.8s' }}
                />
                {!isHotelling && (
                  <text
                    x={toX(kw.x) + 6}
                    y={toY(kw.y) + 3}
                    fill={colors.googleOrange}
                    fontSize={7}
                    fontFamily={fonts.mono}
                    opacity={0.5}
                  >
                    {kw.label}
                  </text>
                )}
              </g>
            ))}
          </g>
        )}

        {/* Relocation fee arrows */}
        {showRelocFees && (
          <g>
            {/* Arrow from center back to Dr. Chen's position */}
            <line
              x1={toX(0.45)}
              y1={toY(0.48)}
              x2={toX(drChen.x)}
              y2={toY(drChen.y)}
              stroke={colors.embedGreen}
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.5}
              markerEnd="url(#arrowGreen)"
            />
            <text
              x={toX(0.62)}
              y={toY(0.56)}
              fill={colors.embedGreen}
              fontSize={8}
              fontFamily={fonts.mono}
              textAnchor="middle"
            >
              $0 (stays)
            </text>

            {/* Cost indicator for hypothetical move */}
            <line
              x1={toX(drChen.x)}
              y1={toY(drChen.y)}
              x2={toX(0.42)}
              y2={toY(0.42)}
              stroke={colors.googleRed}
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.3}
            />
            <text
              x={toX(0.58)}
              y={toY(0.42)}
              fill={colors.googleRed}
              fontSize={8}
              fontFamily={fonts.mono}
              textAnchor="middle"
              opacity={0.6}
            >
              $$$ (expensive)
            </text>
          </g>
        )}

        {/* Winner line (everyone-wins) */}
        {isEquilibrium && (
          <line
            x1={toX(queryPt.x)}
            y1={toY(queryPt.y)}
            x2={toX(drChen.x)}
            y2={toY(drChen.y)}
            stroke={colors.embedGreen}
            strokeWidth={2}
            strokeDasharray="4 4"
            opacity={0.6}
          />
        )}

        {/* Arrow defs */}
        <defs>
          <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={colors.embedGreen} />
          </marker>
        </defs>
      </svg>

      {/* Sigma slider (keywords-tiny-circles) */}
      {showSigmaSlider && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          marginTop: 8,
          fontFamily: fonts.mono,
          fontSize: '0.7rem',
          color: '#888',
        }}>
          <span style={{ color: colors.googleOrange }}>σ ≈ 0</span>
          <div style={{
            width: 120,
            height: 4,
            background: 'linear-gradient(90deg, #FF8800, #4CAF50, #2196F3)',
            borderRadius: 2,
          }} />
          <span style={{ color: colors.embedBlue }}>σ large</span>
        </div>
      )}

      {/* Sigma slider legend */}
      {showSigmaSlider && (
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 4,
          fontFamily: fonts.mono,
          fontSize: '0.6rem',
          color: '#666',
          padding: '0 20px',
        }}>
          <span>keyword</span>
          <span>niche specialist</span>
          <span>broad practice</span>
        </div>
      )}

      <style>{`
        @keyframes fieldFadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>

      {/* Protocol overlay (protocol-gap) */}
      {showProtocolOverlay && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(10, 10, 26, 0.95)',
          border: `2px solid ${colors.embedBlue}`,
          borderRadius: 10,
          padding: 20,
          maxWidth: 260,
          width: '90%',
        }}>
          <div style={{
            fontSize: '0.7rem',
            color: '#888',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            marginBottom: 12,
            fontFamily: fonts.mono,
          }}>
            OpenRTB Bid Request
          </div>
          {[
            { label: 'keywords', value: '"knee, pain, running"', ok: true },
            { label: 'category', value: '"IAB7-1"', ok: true },
            { label: 'geo', value: '"US-CA"', ok: true },
            { label: 'embedding', value: '???', ok: false },
          ].map(f => (
            <div key={f.label} style={{
              display: 'flex',
              justifyContent: 'space-between',
              padding: '6px 0',
              borderBottom: '1px solid #222',
            }}>
              <span style={{
                fontFamily: fonts.mono,
                fontSize: '0.8rem',
                color: f.ok ? '#888' : colors.googleOrange,
              }}>
                {f.label}:
              </span>
              <span style={{
                fontFamily: fonts.mono,
                fontSize: '0.8rem',
                color: f.ok ? '#666' : colors.googleOrange,
                fontStyle: f.ok ? 'normal' : 'italic',
              }}>
                {f.value}
              </span>
            </div>
          ))}
          <div style={{
            marginTop: 10,
            fontSize: '0.7rem',
            color: colors.embedBlue,
            textAlign: 'center',
            fontFamily: fonts.mono,
          }}>
            One new field: embedding: [0.70, 0.68, ...]
          </div>
        </div>
      )}
    </div>
  );
}
