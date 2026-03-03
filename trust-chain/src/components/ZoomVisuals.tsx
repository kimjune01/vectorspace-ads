import { colors, fonts } from '../theme';

// ─── Surveillance comparison ───────────────────────────────

export function SurveillanceCompare() {
  return (
    <div style={{
      width: '100%',
      maxWidth: 440,
      display: 'flex',
      flexDirection: 'column',
      gap: 24,
    }}>
      {/* Keyword side */}
      <div style={{
        border: '1px solid #442222',
        borderRadius: 10,
        padding: '16px 20px',
        background: 'rgba(255, 68, 68, 0.04)',
        opacity: 0,
        animation: 'zoomFadeIn 0.5s ease forwards',
      }}>
        <div style={{
          fontSize: '0.7rem',
          fontFamily: fonts.mono,
          color: colors.googleRed,
          marginBottom: 12,
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Keyword Ads
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[
            { icon: '◉', label: 'Cookies', desc: 'track across sites' },
            { icon: '◉', label: 'Fingerprinting', desc: 'identify device' },
            { icon: '◉', label: 'Data brokers', desc: 'buy offline behavior' },
            { icon: '◉', label: 'Profile', desc: 'infer intent from history' },
          ].map((item, i) => (
            <div key={i} style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              opacity: 0,
              animation: `zoomFadeIn 0.3s ease ${0.1 + i * 0.1}s forwards`,
            }}>
              <span style={{ color: colors.googleRed, fontSize: '0.6rem' }}>{item.icon}</span>
              <span style={{ color: '#bbb', fontSize: '0.8rem', fontWeight: 500 }}>{item.label}</span>
              <span style={{ color: '#555', fontSize: '0.7rem', fontFamily: fonts.mono }}>→ {item.desc}</span>
            </div>
          ))}
        </div>
        <div style={{
          marginTop: 10,
          padding: '6px 10px',
          background: 'rgba(255, 68, 68, 0.08)',
          borderRadius: 6,
          fontSize: '0.7rem',
          color: '#888',
          fontFamily: fonts.mono,
        }}>
          user → cookie → data broker → ad network → guess
        </div>
      </div>

      {/* Arrow */}
      <div style={{
        textAlign: 'center',
        color: '#444',
        fontSize: '1.2rem',
        opacity: 0,
        animation: 'zoomFadeIn 0.3s ease 0.5s forwards',
      }}>
        ↓
      </div>

      {/* Embedding side */}
      <div style={{
        border: `1px solid ${colors.embedGreen}44`,
        borderRadius: 10,
        padding: '16px 20px',
        background: `rgba(76, 175, 80, 0.04)`,
        opacity: 0,
        animation: 'zoomFadeIn 0.5s ease 0.6s forwards',
      }}>
        <div style={{
          fontSize: '0.7rem',
          fontFamily: fonts.mono,
          color: colors.embedGreen,
          marginBottom: 12,
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Embedding Ads
        </div>
        <div style={{
          padding: '6px 10px',
          background: `rgba(76, 175, 80, 0.08)`,
          borderRadius: 6,
          fontSize: '0.7rem',
          color: '#bbb',
          fontFamily: fonts.mono,
        }}>
          user says it → vector → sealed enclave → match
        </div>
        <div style={{
          marginTop: 10,
          fontSize: '0.75rem',
          color: '#666',
        }}>
          No cookies. No profile. No third party.
        </div>
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

// ─── Independence / absorption visual ──────────────────────

const CHATBOT_COMPANIES = [
  { name: 'OpenAI', color: '#10a37f' },
  { name: 'Anthropic', color: '#d4a574' },
  { name: 'Perplexity', color: '#20b2aa' },
];

const ABSORBERS = [
  { name: 'Microsoft', color: '#00a4ef' },
  { name: 'Google', color: '#FF4444' },
  { name: 'State', color: '#888' },
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

      {/* Companies */}
      <div style={{
        display: 'flex',
        gap: 12,
        width: '100%',
        justifyContent: 'center',
      }}>
        {CHATBOT_COMPANIES.map((co, i) => (
          <div key={co.name} style={{
            flex: 1,
            maxWidth: 120,
            border: `1px solid ${co.color}66`,
            borderRadius: 8,
            padding: '12px 8px',
            textAlign: 'center',
            background: `${co.color}08`,
            opacity: 0,
            animation: `zoomFadeIn 0.4s ease ${0.1 + i * 0.1}s forwards`,
          }}>
            <div style={{
              fontSize: '0.8rem',
              fontWeight: 600,
              color: co.color,
              marginBottom: 6,
            }}>
              {co.name}
            </div>
            <div style={{
              fontSize: '0.6rem',
              fontFamily: fonts.mono,
              color: '#666',
            }}>
              −$B/year
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
              {ab.name === 'State' ? 'subsidized' : 'acquired'}
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

export function PopulatedField() {
  const w = 440;
  const h = 380;
  const pad = 30;

  function toX(x: number) { return pad + x * (w - 2 * pad); }
  function toY(y: number) { return pad + (1 - y) * (h - 2 * pad); }

  return (
    <div style={{ width: '100%', maxWidth: 440 }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }}>
        {/* Grid background */}
        <defs>
          <pattern id="popGrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#ffffff06" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width={w} height={h} fill="url(#popGrid)" rx="8" />

        {/* Each specialist as a small glowing dot with label */}
        {SPECIALISTS.map((sp, i) => {
          const cx = toX(sp.x);
          const cy = toY(sp.y);
          return (
            <g key={i} style={{
              opacity: 0,
              animation: `popDotAppear 0.4s ease ${0.05 * i}s forwards`,
            }}>
              {/* Glow */}
              <circle cx={cx} cy={cy} r={16} fill={`${sp.color}15`} />
              {/* Dot */}
              <circle cx={cx} cy={cy} r={4} fill={sp.color} />
              {/* Label */}
              <text
                x={cx}
                y={cy - 10}
                textAnchor="middle"
                fill="#aaa"
                fontSize="8"
                fontFamily={fonts.mono}
              >
                {sp.label}
              </text>
            </g>
          );
        })}

        {/* Axis labels */}
        <text x={w / 2} y={h - 6} textAnchor="middle" fill="#333" fontSize="9" fontFamily={fonts.mono}>
          semantic space
        </text>
      </svg>

      <style>{`
        @keyframes popDotAppear {
          from { opacity: 0; transform: scale(0.5); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}
