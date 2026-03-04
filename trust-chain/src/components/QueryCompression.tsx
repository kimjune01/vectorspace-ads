import { colors, fonts } from '../theme';

/**
 * Animated visual: a full chat query compresses into 3 flat keywords.
 * Words that won't survive get struck through one by one.
 * The critical losses (downhill, not, uphill) flash brighter to
 * emphasize that the most important detail is what gets deleted.
 */

interface WordInfo {
  text: string;
  discarded: boolean;
  /** Critical discards — the specificity that matters most */
  critical: boolean;
  /** Staggered animation delay (seconds) */
  delay: number;
}

const WORDS: WordInfo[] = [
  { text: 'my',       discarded: true,  critical: false, delay: 0.6 },
  { text: 'knee',     discarded: false, critical: false, delay: 0 },
  { text: 'hurts',    discarded: false, critical: false, delay: 0 },
  { text: 'when',     discarded: true,  critical: false, delay: 0.9 },
  { text: 'I',        discarded: true,  critical: false, delay: 0.9 },
  { text: 'run',      discarded: false, critical: false, delay: 0 },
  { text: 'downhill', discarded: true,  critical: true,  delay: 1.3 },
  { text: 'but',      discarded: true,  critical: false, delay: 1.7 },
  { text: 'not',      discarded: true,  critical: true,  delay: 2.0 },
  { text: 'uphill',   discarded: true,  critical: true,  delay: 2.3 },
];

const KEYWORDS = ['knee', 'pain', 'running'];

export function QueryCompression() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 0,
      width: '100%',
      maxWidth: 400,
    }}>
      {/* Chat bubble with animated word strikes */}
      <div style={{
        background: colors.chat.userBubble,
        borderRadius: '18px 18px 4px 18px',
        padding: '16px 20px',
        maxWidth: 360,
        lineHeight: 1.8,
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0 7px',
        position: 'relative',
      }}>
        {WORDS.map((w, i) => (
          <span
            key={i}
            style={{
              fontFamily: fonts.body,
              fontSize: '1.1rem',
              position: 'relative',
              display: 'inline-block',
              color: w.discarded ? '#ccc' : '#fff',
              fontWeight: w.discarded ? 400 : 600,
              ...(w.discarded ? {
                animation: w.critical
                  ? `wordCriticalFade 0.8s ease ${w.delay}s forwards`
                  : `wordFade 0.6s ease ${w.delay}s forwards`,
              } : {
                animation: `wordSurvive 0.4s ease 2.8s forwards`,
              }),
            }}
          >
            {w.text}
          </span>
        ))}
      </div>

      {/* Compression arrow */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        margin: '8px 0',
        opacity: 0,
        animation: 'compArrow 0.4s ease 2.6s forwards',
      }}>
        <div style={{
          width: 2,
          height: 24,
          background: `linear-gradient(to bottom, ${colors.chat.userBubble}, ${colors.googleOrange})`,
        }} />
        <div style={{
          width: 0,
          height: 0,
          borderLeft: '6px solid transparent',
          borderRight: '6px solid transparent',
          borderTop: `8px solid ${colors.googleOrange}`,
        }} />
      </div>

      {/* Keyword bin */}
      <div style={{
        border: `2px solid ${colors.googleOrange}`,
        borderRadius: 8,
        padding: '10px 16px',
        background: 'rgba(255, 136, 0, 0.06)',
        display: 'flex',
        gap: 8,
        opacity: 0,
        animation: 'binAppear 0.4s ease 3.0s forwards',
      }}>
        {KEYWORDS.map((kw, i) => (
          <span
            key={kw}
            style={{
              fontFamily: fonts.mono,
              fontSize: '0.9rem',
              fontWeight: 600,
              color: colors.googleOrange,
              padding: '4px 14px',
              background: 'rgba(255, 136, 0, 0.12)',
              borderRadius: 16,
              border: `1px solid ${colors.googleOrange}44`,
              opacity: 0,
              animation: `chipAppear 0.3s ease ${3.1 + i * 0.15}s forwards`,
            }}
          >
            {kw}
          </span>
        ))}
      </div>

      <style>{`
        @keyframes wordFade {
          0% { color: #ccc; text-decoration: none; }
          50% { color: #ef4444; text-decoration: line-through; }
          100% { color: #884444; text-decoration: line-through; }
        }
        @keyframes wordCriticalFade {
          0% { color: #ccc; text-decoration: none; }
          25% { color: #ff6666; text-decoration: none; transform: scale(1.08); }
          60% { color: #ef4444; text-decoration: line-through; transform: scale(1); }
          100% { color: #aa4444; text-decoration: line-through; }
        }
        @keyframes wordSurvive {
          0% { color: #fff; }
          100% { color: ${colors.googleOrange}; font-weight: 700; }
        }
        @keyframes compArrow {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes binAppear {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes chipAppear {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
