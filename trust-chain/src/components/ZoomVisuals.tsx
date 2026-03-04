import { colors, fonts } from '../theme';

// ─── Surveillance comparison ───────────────────────────────

export function SurveillanceCompare() {
  const w = 440;
  const h = 320;

  // Keyword tracking: tangled web of nodes
  const trackingNodes = [
    { x: 60, y: 40, label: 'User', color: '#bbb' },
    { x: 180, y: 30, label: 'Cookie', color: colors.googleRed },
    { x: 300, y: 50, label: 'Tracker', color: colors.googleRed },
    { x: 120, y: 90, label: 'Fingerprint', color: colors.googleRed },
    { x: 240, y: 100, label: 'Data broker', color: colors.googleRed },
    { x: 370, y: 80, label: 'Profile', color: colors.googleOrange },
    { x: 370, y: 130, label: 'Ad', color: '#888' },
  ];
  const trackingEdges = [
    [0,1],[0,2],[0,3],[1,2],[1,4],[2,4],[3,4],[4,5],[5,6],[2,5],
  ];

  return (
    <div style={{ width: '100%', maxWidth: 440 }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
        {/* Top: keyword tracking web */}
        <text x={20} y={16} fill={colors.googleRed} fontSize="10" fontFamily={fonts.mono} style={{ textTransform: 'uppercase' as const, letterSpacing: '0.1em' }}>
          keyword ads
        </text>

        {/* Edges */}
        {trackingEdges.map(([a, b], i) => (
          <line
            key={`e${i}`}
            x1={trackingNodes[a].x} y1={trackingNodes[a].y}
            x2={trackingNodes[b].x} y2={trackingNodes[b].y}
            stroke={`${colors.googleRed}40`}
            strokeWidth={1}
            strokeDasharray="3 2"
            style={{ opacity: 0, animation: `zoomFadeIn 0.3s ease ${0.05 * i}s forwards` }}
          />
        ))}

        {/* Nodes */}
        {trackingNodes.map((n, i) => (
          <g key={`n${i}`} style={{ opacity: 0, animation: `zoomFadeIn 0.3s ease ${0.05 * i + 0.1}s forwards` }}>
            <circle cx={n.x} cy={n.y} r={6} fill={`${n.color}33`} stroke={n.color} strokeWidth={1} />
            <text x={n.x} y={n.y + 18} textAnchor="middle" fill="#666" fontSize="7" fontFamily={fonts.mono}>
              {n.label}
            </text>
          </g>
        ))}

        {/* Divider */}
        <line x1={20} y1={170} x2={420} y2={170} stroke="#222" strokeWidth={1}
          style={{ opacity: 0, animation: 'zoomFadeIn 0.3s ease 0.5s forwards' }}
        />

        {/* Bottom: embedding — clean direct line */}
        <text x={20} y={196} fill={colors.embedGreen} fontSize="10" fontFamily={fonts.mono} style={{ textTransform: 'uppercase' as const, letterSpacing: '0.1em' }}>
          embedding ads
        </text>

        {/* Simple flow: User → Vector → Enclave → Match */}
        {['User', 'Vector', 'Enclave', 'Match'].map((label, i) => {
          const x = 60 + i * 110;
          const y = 240;
          const c = i === 0 ? '#bbb' : colors.embedGreen;
          return (
            <g key={label} style={{ opacity: 0, animation: `zoomFadeIn 0.3s ease ${0.6 + i * 0.1}s forwards` }}>
              <circle cx={x} cy={y} r={8} fill={`${c}22`} stroke={c} strokeWidth={1.5} />
              <text x={x} y={y + 22} textAnchor="middle" fill="#888" fontSize="8" fontFamily={fonts.mono}>
                {label}
              </text>
              {i < 3 && (
                <line x1={x + 12} y1={y} x2={x + 98} y2={y} stroke={`${colors.embedGreen}55`} strokeWidth={1.5} />
              )}
            </g>
          );
        })}

        {/* No tracking label */}
        <text x={w / 2} y={290} textAnchor="middle" fill="#555" fontSize="9" fontFamily={fonts.mono}
          style={{ opacity: 0, animation: 'zoomFadeIn 0.3s ease 1s forwards' }}
        >
          no cookies · no profile · no third party
        </text>
      </svg>

      <style>{`
        @keyframes zoomFadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

// ─── Independence / absorption visual ──────────────────────

const CHATBOT_COMPANIES = [
  { name: 'OpenAI', color: '#10a37f', burn: '−$5B/year' },
  { name: 'Anthropic', color: '#d4a574', burn: '−$3B/year' },
  { name: 'Mistral', color: '#ff7000', burn: '−$600M/year' },
  { name: 'Perplexity', color: '#20b2aa', burn: '−$65M/year' },
  { name: 'Cohere', color: '#39a0ed', burn: '−$200M/year' },
  { name: 'xAI', color: '#aaaaaa', burn: '−$2B/year' },
];

const ABSORBERS = [
  { name: 'Facebook', color: '#1877F2' },
  { name: 'Google', color: '#FF4444' },
  { name: 'Government', color: '#888888' },
];

export function AbsorptionVisual() {
  return (
    <div style={{
      width: '100%',
      maxWidth: 440,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 20,
    }}>
      {/* VC funding draining */}
      <div style={{
        fontSize: '0.7rem',
        fontFamily: fonts.mono,
        color: '#666',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        opacity: 0,
        animation: 'zoomFadeIn 0.4s ease forwards',
      }}>
        VC Funding
      </div>

      {/* Companies — two rows of three */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 10,
        width: '100%',
      }}>
        {CHATBOT_COMPANIES.map((co, i) => (
          <div key={co.name} style={{
            border: `1px solid ${co.color}66`,
            borderRadius: 8,
            padding: '10px 8px',
            textAlign: 'center',
            background: `${co.color}08`,
            opacity: 0,
            animation: `zoomFadeIn 0.4s ease ${0.1 + i * 0.08}s forwards`,
          }}>
            <div style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              color: co.color,
              marginBottom: 4,
            }}>
              {co.name}
            </div>
            <div style={{
              fontSize: '0.55rem',
              fontFamily: fonts.mono,
              color: '#666',
            }}>
              {co.burn}
            </div>
          </div>
        ))}
      </div>

      {/* Drain indicator */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        opacity: 0,
        animation: 'zoomFadeIn 0.4s ease 0.5s forwards',
      }}>
        <div style={{
          height: 1,
          width: 60,
          background: 'linear-gradient(90deg, transparent, #FF444488)',
        }} />
        <span style={{
          fontSize: '0.65rem',
          fontFamily: fonts.mono,
          color: colors.googleRed,
        }}>
          funding runs out
        </span>
        <div style={{
          height: 1,
          width: 60,
          background: 'linear-gradient(90deg, #FF444488, transparent)',
        }} />
      </div>

      {/* Arrows down */}
      <div style={{
        color: '#444',
        fontSize: '1.2rem',
        opacity: 0,
        animation: 'zoomFadeIn 0.3s ease 0.7s forwards',
      }}>
        ↓
      </div>

      {/* Absorbers */}
      <div style={{
        display: 'flex',
        gap: 12,
        width: '100%',
        justifyContent: 'center',
      }}>
        {ABSORBERS.map((ab, i) => (
          <div key={ab.name} style={{
            flex: 1,
            maxWidth: 120,
            border: `2px solid ${ab.color}44`,
            borderRadius: 10,
            padding: '14px 8px',
            textAlign: 'center',
            background: `${ab.color}0a`,
            opacity: 0,
            animation: `zoomFadeIn 0.4s ease ${0.8 + i * 0.1}s forwards`,
          }}>
            <div style={{
              fontSize: '0.85rem',
              fontWeight: 700,
              color: ab.color,
            }}>
              {ab.name}
            </div>
            <div style={{
              fontSize: '0.55rem',
              fontFamily: fonts.mono,
              color: '#555',
              marginTop: 4,
            }}>
              {ab.name === 'Government' ? 'subsidized' : 'acquired'}
            </div>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes zoomFadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

// ─── Populated field — many specialists ────────────────────

const SPECIALISTS = [
  { x: 0.12, y: 0.85, label: 'ACL coach', color: '#4CAF50' },
  { x: 0.25, y: 0.72, label: 'Tagalog lawyer', color: '#2196F3' },
  { x: 0.08, y: 0.55, label: 'Century HVAC', color: '#FF9800' },
  { x: 0.38, y: 0.90, label: 'ADHD tutor', color: '#9C27B0' },
  { x: 0.80, y: 0.62, label: 'Dr. Chen', color: '#4CAF50' },
  { x: 0.65, y: 0.35, label: 'Luthier', color: '#E91E63' },
  { x: 0.50, y: 0.78, label: 'Midwife', color: '#00BCD4' },
  { x: 0.72, y: 0.15, label: 'Trail guide', color: '#8BC34A' },
  { x: 0.15, y: 0.25, label: 'Bookkeeper', color: '#FF5722' },
  { x: 0.90, y: 0.45, label: 'Dog rehab', color: '#FFEB3B' },
  { x: 0.42, y: 0.48, label: 'Brewer', color: '#795548' },
  { x: 0.58, y: 0.60, label: 'Solar tech', color: '#FFC107' },
  { x: 0.30, y: 0.40, label: 'Herbalist', color: '#66BB6A' },
  { x: 0.85, y: 0.82, label: 'Violin teacher', color: '#AB47BC' },
  { x: 0.18, y: 0.68, label: 'Doula', color: '#EC407A' },
  { x: 0.75, y: 0.88, label: 'Farrier', color: '#8D6E63' },
  { x: 0.48, y: 0.20, label: 'Arborist', color: '#43A047' },
  { x: 0.62, y: 0.50, label: 'Welder', color: '#FF7043' },
  { x: 0.35, y: 0.15, label: 'Sign lang.', color: '#29B6F6' },
  { x: 0.55, y: 0.92, label: 'Beekeeper', color: '#FFD54F' },
];

export function PopulatedField({ stepId }: { stepId: string }) {
  const showJobs = stepId === 'zoom-jobs';
  const isClosing = stepId === 'closing';
  const w = 440;
  const h = 380;
  const pad = 30;

  function toX(x: number) { return pad + x * (w - 2 * pad); }
  function toY(y: number) { return pad + (1 - y) * (h - 2 * pad); }

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: 440 }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
        <defs>
          <pattern id="popGrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#ffffff06" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width={w} height={h} fill="url(#popGrid)" rx="8" />

        {SPECIALISTS.map((sp, i) => {
          const cx = toX(sp.x);
          const cy = toY(sp.y);
          const jobs = JOBS_PER_SPECIALIST[sp.label] ?? [];
          const orbitR = 20;
          const isDrChen = sp.label === 'Dr. Chen';
          return (
            <g key={i} style={{
              ...(isClosing
                ? isDrChen
                  ? { opacity: 0.3, filter: 'saturate(0)', animation: 'chenReveal 0.6s ease 1.8s forwards' }
                  : { opacity: 0.3, filter: 'saturate(0)', transition: 'opacity 0.5s ease, filter 0.5s ease' }
                : { opacity: 0, animation: `popDotAppear 0.4s ease ${0.05 * i}s forwards` }),
            }}>
              <circle cx={cx} cy={cy} r={16} fill={`${sp.color}15`} />
              <circle cx={cx} cy={cy} r={4} fill={sp.color} />
              <text x={cx} y={cy - 10} textAnchor="middle" fill="#aaa" fontSize="8" fontFamily={fonts.mono}>
                {sp.label}
              </text>
              {/* Job satellites — always in DOM, opacity transitions */}
              {jobs.map((job, j) => {
                const angle = (j / jobs.length) * Math.PI * 2 - Math.PI / 2;
                const jx = cx + Math.cos(angle) * orbitR;
                const jy = cy + Math.sin(angle) * orbitR;
                return (
                  <g key={j} style={{
                    opacity: showJobs && !isClosing ? 1 : 0,
                    transition: `opacity 0.4s ease ${0.03 * i + j * 0.05}s`,
                  }}>
                    <line x1={cx} y1={cy} x2={jx} y2={jy} stroke={`${sp.color}30`} strokeWidth={0.5} />
                    <circle cx={jx} cy={jy} r={2} fill={`${sp.color}88`} />
                    <text x={jx} y={jy - 5} textAnchor="middle" fill="#666" fontSize="5.5" fontFamily={fonts.mono}>
                      {job}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}

        <text x={w / 2} y={h - 6} textAnchor="middle" fill="#333" fontSize="9" fontFamily={fonts.mono}
          style={{ opacity: isClosing ? 0 : 1, transition: 'opacity 0.4s ease' }}
        >
          {showJobs ? 'each dot = a business = 2–3 jobs' : 'semantic space'}
        </text>
      </svg>

      {/* Chat bubble overlaid on closing */}
      {isClosing && (
        <div style={{
          position: 'absolute',
          top: '45%',
          left: '50%',
          transform: 'translate(-50%, -50%) scale(0)',
          background: '#2a2a4a',
          borderRadius: '18px 18px 4px 18px',
          padding: '14px 20px',
          maxWidth: 280,
          fontFamily: "'Inter', sans-serif",
          fontSize: '0.95rem',
          color: '#e0e0e0',
          lineHeight: 1.5,
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          animation: 'closingBubbleIn 0.5s ease 0.5s forwards',
          marginLeft: -30,
          transformOrigin: 'bottom right',
          zIndex: 2,
        }}>
          my knee hurts when I run downhill but not uphill
        </div>
      )}

      <style>{`
        @keyframes popDotAppear {
          from { opacity: 0; transform: scale(0.5); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes closingBubbleIn {
          from { transform: translate(-50%, -50%) scale(0); opacity: 0; }
          to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
        @keyframes chenReveal {
          to { opacity: 1; filter: saturate(1); }
        }
      `}</style>
    </div>
  );
}

const JOBS_PER_SPECIALIST: Record<string, string[]> = {
  'Dr. Chen': ['receptionist', 'PT aide', 'billing'],
  'Tagalog lawyer': ['paralegal', 'translator'],
  'Century HVAC': ['apprentice', 'dispatcher'],
  'ADHD tutor': ['tutor 2', 'scheduler'],
  'Luthier': ['apprentice', 'shipping'],
  'Midwife': ['doula', 'admin'],
  'Bookkeeper': ['clerk', 'payroll'],
  'Solar tech': ['installer', 'surveyor'],
  'Brewer': ['bartender', 'delivery'],
  'Dog rehab': ['vet tech', 'groomer'],
};

// ─── Dot field — dots appearing beside conversations ─────

const DOT_CONVERSATIONS = [
  { query: 'my knee hurts going downhill', specialist: 'Dr. Chen Sports', color: '#4CAF50' },
  { query: 'need a Tagalog-speaking lawyer', specialist: 'Reyes Law Office', color: '#2196F3' },
  { query: 'my 1920s radiator is leaking', specialist: 'Century HVAC', color: '#FF9800' },
  { query: 'calculus tutor for my kid with ADHD', specialist: 'Focus Tutoring', color: '#9C27B0' },
  { query: 'violin teacher for adult beginner', specialist: 'Midtown Strings', color: '#AB47BC' },
  { query: 'dog tore her ACL, need rehab', specialist: 'PawMotion Rehab', color: '#FFEB3B' },
  { query: 'bees in my wall, want them saved', specialist: 'Urban Beekeeper Co', color: '#FFD54F' },
];

export function DotField() {
  return (
    <div style={{
      width: '100%',
      maxWidth: 440,
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
    }}>
      {DOT_CONVERSATIONS.map((conv, i) => (
        <div
          key={i}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '8px 12px',
            borderRadius: 8,
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid #1a1a2e',
            opacity: 0,
            animation: `dotRowIn 0.4s ease ${i * 0.15}s forwards`,
          }}
        >
          {/* The dot — appears colored, then fades to gray */}
          <div style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            flexShrink: 0,
            opacity: 0,
            animation: `dotLife${i} 1.3s ease ${i * 0.15 + 0.3}s forwards`,
          }} />
          {/* Query + specialist */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: '0.7rem',
              color: '#888',
              fontFamily: fonts.mono,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}>
              {conv.query}
            </div>
            <div style={{
              fontSize: '0.65rem',
              fontFamily: fonts.mono,
              fontWeight: 600,
              opacity: 0,
              animation: `dotText${i} 1.2s ease ${i * 0.15 + 0.5}s forwards`,
            }}>
              → {conv.specialist}
            </div>
          </div>
        </div>
      ))}

      <style>{`
        @keyframes dotRowIn {
          from { opacity: 0; transform: translateX(-8px); }
          to { opacity: 1; transform: translateX(0); }
        }
        ${DOT_CONVERSATIONS.map((conv, i) => `
          @keyframes dotLife${i} {
            0% { opacity: 0; transform: scale(0.3); background-color: ${conv.color}; box-shadow: 0 0 8px ${conv.color}88; }
            54% { opacity: 1; transform: scale(1); background-color: ${conv.color}; box-shadow: 0 0 8px ${conv.color}88; }
            100% { opacity: 1; transform: scale(1); background-color: #555; box-shadow: none; }
          }
          @keyframes dotText${i} {
            0% { opacity: 0; color: ${conv.color}; }
            58% { opacity: 1; color: ${conv.color}; }
            100% { opacity: 1; color: #555; }
          }
        `).join('')}
      `}</style>
    </div>
  );
}
