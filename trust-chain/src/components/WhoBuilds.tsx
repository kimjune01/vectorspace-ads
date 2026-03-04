import { colors, fonts } from '../theme';
import { GOOGLE_BIDDERS } from '../data';
import { BidPaddle } from './BidPaddle';

interface Props {
  stepId: string;
}

type SubState = 'chatbots-stuck' | 'incumbents-wrong' | 'trust-tee';

const STEP_TO_SUBSTATE: Record<string, SubState> = {
  'chatbots-stuck': 'chatbots-stuck',
  'incumbents-wrong': 'incumbents-wrong',
  'trust-tee': 'trust-tee',
};

export function WhoBuilds({ stepId }: Props) {
  const subState = STEP_TO_SUBSTATE[stepId] ?? 'chatbots-stuck';

  let content;
  if (subState === 'chatbots-stuck') content = <ChatbotsStuck />;
  else if (subState === 'incumbents-wrong') content = <IncumbentsWrong />;
  else content = <TrustTee />;

  return (
    <div key={subState} style={{ animation: 'whoBuildsFadeIn 0.4s ease', width: '100%' }}>
      {content}
      <style>{`
        @keyframes whoBuildsFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

function ChatbotsStuck() {
  const chatbots = [
    { name: 'ChatGPT', color: '#10A37F' },
    { name: 'Claude', color: '#D4A574' },
    { name: 'Perplexity', color: '#20B2AA' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20, width: '100%' }}>
      {chatbots.map((bot, i) => (
        <div key={bot.name} style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          width: '100%',
          opacity: 0,
          animation: `fadeSlideIn 0.4s ease ${i * 0.15}s forwards`,
        }}>
          <div style={{
            width: 44,
            height: 44,
            borderRadius: 10,
            background: `${bot.color}22`,
            border: `1px solid ${bot.color}44`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}>
            <span style={{
              fontFamily: fonts.mono,
              fontSize: '0.7rem',
              color: bot.color,
              fontWeight: 600,
            }}>
              {bot.name.slice(0, 2).toUpperCase()}
            </span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            flex: 1,
          }}>
            <span style={{
              fontFamily: fonts.mono,
              fontSize: '0.7rem',
              color: colors.embedBlue,
              opacity: 0.8,
            }}>
              [0.71, 0.68, ...]
            </span>
            <span style={{ color: '#555', fontSize: '0.8rem' }}>→</span>
            <span style={{
              fontFamily: fonts.mono,
              fontSize: '0.7rem',
              color: '#555',
              opacity: 0.3,
              textDecoration: 'line-through',
            }}>
              ∅
            </span>
          </div>
        </div>
      ))}

      <div style={{
        fontFamily: fonts.mono,
        fontSize: '0.75rem',
        color: '#555',
        textAlign: 'center',
        marginTop: 8,
        fontStyle: 'italic',
        opacity: 0,
        animation: 'fadeSlideIn 0.4s ease 0.5s forwards',
      }}>
        Rich intent vectors — thrown away
      </div>

      <style>{`
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

function IncumbentsWrong() {
  // Rich embedding bar colors — vibrant semantic dimensions
  const embeddingColors = [
    '#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
    '#00BCD4', '#FF5722', '#8BC34A', '#3F51B5', '#FFEB3B',
    '#009688', '#F44336', '#03A9F4', '#CDDC39', '#673AB7',
    '#FFC107', '#00E5FF', '#76FF03', '#FF4081', '#7C4DFF',
  ];

  // Dull keyword chips
  const keywords = ['knee', 'pain', 'running'];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 0,
      width: '100%',
      maxWidth: 380,
    }}>
      {/* Top: rich embedding — wide colorful bar */}
      <div style={{
        opacity: 0,
        animation: 'incFadeIn 0.5s ease 0s forwards',
        width: '100%',
        textAlign: 'center',
      }}>
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.65rem',
          color: '#888',
          marginBottom: 6,
        }}>
          embedding — 384 dimensions
        </div>
        <div style={{
          display: 'flex',
          width: '100%',
          height: 32,
          borderRadius: 6,
          overflow: 'hidden',
          boxShadow: '0 0 20px rgba(33, 150, 243, 0.15)',
        }}>
          {embeddingColors.map((c, i) => (
            <div key={i} style={{
              flex: 1,
              background: c,
              opacity: 0.5 + Math.random() * 0.5,
            }} />
          ))}
        </div>
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.55rem',
          color: '#666',
          marginTop: 4,
        }}>
          "my knee hurts when I run downhill but not uphill"
        </div>
      </div>

      {/* Funnel: narrowing trapezoid */}
      <div style={{
        width: '100%',
        height: 48,
        display: 'flex',
        justifyContent: 'center',
        opacity: 0,
        animation: 'incFadeIn 0.4s ease 0.3s forwards',
      }}>
        <svg width="100%" height="48" viewBox="0 0 380 48" preserveAspectRatio="none" aria-hidden="true">
          <path
            d="M 0 0 L 380 0 L 220 48 L 160 48 Z"
            fill="none"
            stroke="#333"
            strokeWidth="1"
            strokeDasharray="4 3"
          />
          <path
            d="M 190 24 L 190 48"
            stroke="#555"
            strokeWidth="1"
          />
        </svg>
      </div>

      {/* Bottom: dull keyword chips — narrow */}
      <div style={{
        opacity: 0,
        animation: 'incFadeIn 0.5s ease 0.5s forwards',
        textAlign: 'center',
      }}>
        <div style={{
          display: 'flex',
          gap: 8,
          justifyContent: 'center',
        }}>
          {keywords.map(kw => (
            <div key={kw} style={{
              fontFamily: fonts.mono,
              fontSize: '0.75rem',
              color: '#666',
              padding: '4px 12px',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid #333',
              borderRadius: 4,
            }}>
              {kw}
            </div>
          ))}
        </div>
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.65rem',
          color: '#555',
          marginTop: 6,
        }}>
          keywords — 3 bins
        </div>
      </div>

      {/* Bid auction table — desaturated to show it's the broken outcome */}
      <div style={{
        marginTop: 16,
        width: '100%',
        background: 'rgba(255,255,255,0.02)',
        borderRadius: 8,
        border: '1px solid #2a2a2a',
        overflow: 'hidden',
        opacity: 0,
        animation: 'incFadeIn 0.4s ease 0.7s forwards',
        filter: 'saturate(0.3)',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          padding: '8px 12px',
          borderBottom: '1px solid #333',
          background: 'rgba(255,255,255,0.03)',
        }}>
          <div style={{ width: 28 }} />
          <div style={{
            flex: 1,
            fontSize: '0.6rem',
            color: '#666',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            fontFamily: fonts.mono,
          }}>
            Bidder
          </div>
          <div style={{
            fontSize: '0.6rem',
            color: '#666',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            fontFamily: fonts.mono,
            minWidth: 40,
            textAlign: 'right',
          }}>
            Bid
          </div>
          <div style={{ width: 40 }} />
        </div>
        {GOOGLE_BIDDERS.map((bidder, i) => (
          <BidPaddle
            key={bidder.name}
            name={bidder.name}
            bid={bidder.bid}
            color={bidder.color}
            isWinner={i === 0}
            visible={true}
            index={i}
          />
        ))}
      </div>

      <style>{`
        @keyframes incFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

/** CSS-only shield icon with X */
function ShieldX({ delay }: { delay: number }) {
  return (
    <div style={{
      width: 32,
      height: 36,
      position: 'relative',
      opacity: 0,
      animation: `fadeIn 0.3s ease ${delay}s forwards`,
    }}>
      {/* Shield shape */}
      <div style={{
        width: 32,
        height: 36,
        background: 'rgba(239, 68, 68, 0.15)',
        border: '2px solid #ef4444',
        borderRadius: '4px 4px 16px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <span style={{ color: '#ef4444', fontWeight: 700, fontSize: '0.9rem', lineHeight: 1 }}>✕</span>
      </div>
    </div>
  );
}

/** CSS-only padlock icon */
function LockIcon() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 0,
    }}>
      {/* Shackle */}
      <div style={{
        width: 28,
        height: 16,
        border: '3px solid #00BCD4',
        borderBottom: 'none',
        borderRadius: '14px 14px 0 0',
      }} />
      {/* Body */}
      <div style={{
        width: 36,
        height: 24,
        background: 'rgba(0, 188, 212, 0.2)',
        border: '3px solid #00BCD4',
        borderRadius: 4,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        {/* Keyhole */}
        <div style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: '#00BCD4',
        }} />
      </div>
    </div>
  );
}

/** CSS checkmark in circle */
function CheckCircle() {
  return (
    <div style={{
      width: 24,
      height: 24,
      borderRadius: '50%',
      background: '#4CAF50',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginLeft: 4,
    }}>
      <span style={{ color: '#fff', fontSize: '0.75rem', fontWeight: 700, lineHeight: 1 }}>✓</span>
    </div>
  );
}

function TrustTee() {
  return (
    <div style={{ textAlign: 'center', width: '100%' }}>
      {/* Ad blocker shield icons */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: 12,
        marginBottom: 20,
      }}>
        <ShieldX delay={0} />
        <ShieldX delay={0.1} />
        <ShieldX delay={0.2} />
      </div>

      {/* Arrow */}
      <div style={{
        color: '#555',
        fontSize: '1rem',
        marginBottom: 16,
        opacity: 0,
        animation: 'fadeIn 0.3s ease 0.4s forwards',
      }}>
        ↓
      </div>

      {/* Lock + checkmark box */}
      <div style={{
        border: `2px solid ${colors.embedCyan}`,
        borderRadius: 12,
        padding: '20px 16px',
        background: 'rgba(0, 188, 212, 0.04)',
        display: 'inline-block',
        position: 'relative',
        opacity: 0,
        animation: 'fadeScaleIn 0.5s ease 0.5s forwards',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 6,
          marginBottom: 8,
        }}>
          <LockIcon />
          <CheckCircle />
        </div>
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.75rem',
          color: colors.embedCyan,
          marginBottom: 12,
        }}>
          verified: sealed auction
        </div>

        {/* Dr. Chen card inside */}
        <div style={{
          padding: '10px 14px',
          background: 'rgba(76, 175, 80, 0.08)',
          border: `1px solid ${colors.embedGreen}44`,
          borderRadius: 8,
          textAlign: 'left',
          opacity: 0,
          animation: 'fadeIn 0.4s ease 0.9s forwards',
        }}>
          <div style={{
            fontSize: '0.85rem',
            fontWeight: 600,
            color: '#fff',
            marginBottom: 2,
          }}>
            Dr. Chen Sports Biomechanics
          </div>
          <div style={{
            fontSize: '0.7rem',
            color: colors.embedGreen,
          }}>
            Runner knee specialist — eccentric loading
          </div>
        </div>
      </div>

      <div style={{
        fontSize: '0.75rem',
        color: '#888',
        marginTop: 16,
        fontFamily: fonts.mono,
        opacity: 0,
        animation: 'fadeIn 0.3s ease 1.1s forwards',
      }}>
        Not "trust us" — "check our work"
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes fadeScaleIn {
          from { opacity: 0; transform: scale(0.9); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
}
