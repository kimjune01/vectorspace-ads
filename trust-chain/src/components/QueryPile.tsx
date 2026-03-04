import { useState, useEffect } from 'react';
import { colors, fonts } from '../theme';

/**
 * Cycles through multiple diverse queries, each playing the same
 * word-by-word compression animation as QueryCompression.
 * All queries compress into the same keyword bin, emphasizing
 * that keywords can't differentiate them.
 */

interface WordInfo {
  text: string;
  discarded: boolean;
  critical: boolean;
  delay: number;
}

interface QueryDef {
  words: WordInfo[];
}

const QUERIES: QueryDef[] = [
  {
    // Downhill vs uphill contrast — eccentric loading
    words: [
      { text: 'knee',     discarded: false, critical: false, delay: 0 },
      { text: 'pain',     discarded: false, critical: false, delay: 0 },
      { text: 'running',  discarded: false, critical: false, delay: 0 },
      { text: 'downhill', discarded: true,  critical: true,  delay: 0.6 },
      { text: 'but',      discarded: true,  critical: false, delay: 1.0 },
      { text: 'not',      discarded: true,  critical: true,  delay: 1.3 },
      { text: 'uphill',   discarded: true,  critical: true,  delay: 1.6 },
    ],
  },
  {
    // Warm-up pain that resolves — temporal pattern
    words: [
      { text: 'knee',      discarded: false, critical: false, delay: 0 },
      { text: 'pain',      discarded: false, critical: false, delay: 0 },
      { text: 'only',      discarded: true,  critical: true,  delay: 0.6 },
      { text: 'the',       discarded: true,  critical: false, delay: 0.6 },
      { text: 'first',     discarded: true,  critical: true,  delay: 0.9 },
      { text: 'mile',      discarded: true,  critical: true,  delay: 0.9 },
      { text: 'of',        discarded: true,  critical: false, delay: 1.2 },
      { text: 'running',   discarded: false, critical: false, delay: 0 },
      { text: 'then',      discarded: true,  critical: true,  delay: 1.4 },
      { text: 'goes',      discarded: true,  critical: true,  delay: 1.7 },
      { text: 'away',      discarded: true,  critical: true,  delay: 1.7 },
    ],
  },
  {
    // Overuse / training load increase
    words: [
      { text: 'knee',      discarded: false, critical: false, delay: 0 },
      { text: 'pain',      discarded: false, critical: false, delay: 0 },
      { text: 'running',   discarded: false, critical: false, delay: 0 },
      { text: 'that',      discarded: true,  critical: false, delay: 0.6 },
      { text: 'started',   discarded: true,  critical: true,  delay: 0.9 },
      { text: 'when',      discarded: true,  critical: false, delay: 1.1 },
      { text: 'I',         discarded: true,  critical: false, delay: 1.1 },
      { text: 'increased', discarded: true,  critical: true,  delay: 1.4 },
      { text: 'mileage',   discarded: true,  critical: true,  delay: 1.7 },
    ],
  },
  {
    // Surface-dependent — concrete vs trail
    words: [
      { text: 'knee',      discarded: false, critical: false, delay: 0 },
      { text: 'pain',      discarded: false, critical: false, delay: 0 },
      { text: 'running',   discarded: false, critical: false, delay: 0 },
      { text: 'on',        discarded: true,  critical: false, delay: 0.6 },
      { text: 'concrete',  discarded: true,  critical: true,  delay: 0.9 },
      { text: 'but',       discarded: true,  critical: false, delay: 1.2 },
      { text: 'not',       discarded: true,  critical: true,  delay: 1.4 },
      { text: 'on',        discarded: true,  critical: false, delay: 1.4 },
      { text: 'trails',    discarded: true,  critical: true,  delay: 1.7 },
    ],
  },
  {
    // Radiating pain — different workup entirely
    words: [
      { text: 'knee',      discarded: false, critical: false, delay: 0 },
      { text: 'pain',      discarded: false, critical: false, delay: 0 },
      { text: 'from',      discarded: true,  critical: false, delay: 0.6 },
      { text: 'running',   discarded: false, critical: false, delay: 0 },
      { text: 'that',      discarded: true,  critical: false, delay: 0.9 },
      { text: 'moves',     discarded: true,  critical: true,  delay: 1.2 },
      { text: 'to',        discarded: true,  critical: false, delay: 1.2 },
      { text: 'the',       discarded: true,  critical: false, delay: 1.2 },
      { text: 'hip',       discarded: true,  critical: true,  delay: 1.5 },
      { text: 'afterwards', discarded: true, critical: true,  delay: 1.8 },
    ],
  },
];

const KEYWORDS = ['knee', 'pain', 'running'];
const CYCLE_DURATION = 3360; // ms per query before swapping

export function QueryPile() {
  const [queryIndex, setQueryIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setQueryIndex((prev) => (prev + 1) % QUERIES.length);
    }, CYCLE_DURATION);
    return () => clearInterval(timer);
  }, []);

  const current = QUERIES[queryIndex];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 0,
      width: '100%',
      maxWidth: 400,
    }}>
      {/* Cycling chat bubble with animated word strikes */}
      <div
        key={queryIndex}
        style={{
          background: colors.chat.userBubble,
          borderRadius: '18px 18px 4px 18px',
          padding: '16px 20px',
          maxWidth: 360,
          lineHeight: 1.8,
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0 7px',
          position: 'relative',
          animation: 'qpBubbleIn 0.3s ease',
        }}
      >
        {current.words.map((w, i) => (
          <span
            key={`${queryIndex}-${i}`}
            style={{
              fontFamily: fonts.body,
              fontSize: '1.1rem',
              position: 'relative',
              display: 'inline-block',
              color: w.discarded ? '#ccc' : colors.googleOrange,
              fontWeight: w.discarded ? 400 : 700,
              ...(w.discarded ? {
                animation: w.critical
                  ? `qpCriticalFade 0.8s ease ${w.delay}s forwards`
                  : `qpWordFade 0.6s ease ${w.delay}s forwards`,
              } : {}),
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
        margin: '4px 0',
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

      {/* Keyword bin — always visible */}
      <div style={{
        border: `2px solid ${colors.googleOrange}`,
        borderRadius: 8,
        padding: '10px 16px',
        background: 'rgba(255, 136, 0, 0.06)',
        display: 'flex',
        gap: 8,
      }}>
        {KEYWORDS.map((kw) => (
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
            }}
          >
            {kw}
          </span>
        ))}
      </div>

      <style>{`
        @keyframes qpBubbleIn {
          from { opacity: 0; transform: translateY(-6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes qpWordFade {
          0% { color: #ccc; text-decoration: none; }
          50% { color: #ef4444; text-decoration: line-through; }
          100% { color: #884444; text-decoration: line-through; }
        }
        @keyframes qpCriticalFade {
          0% { color: #ccc; text-decoration: none; }
          25% { color: #ff6666; text-decoration: none; transform: scale(1.08); }
          60% { color: #ef4444; text-decoration: line-through; transform: scale(1); }
          100% { color: #aa4444; text-decoration: line-through; }
        }
      `}</style>
    </div>
  );
}
