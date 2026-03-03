import { colors, fonts } from '../theme';

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

function GoogleG() {
  return (
    <span style={{ fontWeight: 800, fontSize: '1.5rem', letterSpacing: '-0.02em' }}>
      <span style={{ color: '#4285F4' }}>G</span>
    </span>
  );
}

function IncumbentsWrong() {
  return (
    <div style={{
      display: 'flex',
      gap: 16,
      width: '100%',
      maxWidth: 440,
    }}>
      {/* Google side */}
      <div style={{
        flex: 1,
        padding: '16px 12px',
        border: `1px solid ${colors.googleRed}44`,
        borderRadius: 8,
        background: 'rgba(255, 68, 68, 0.04)',
        textAlign: 'center',
        opacity: 0,
        animation: 'fadeSlideLeft 0.4s ease 0s forwards',
      }}>
        <GoogleG />
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.65rem',
          color: '#888',
          marginTop: 4,
          marginBottom: 8,
        }}>
          keyword pipeline
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
        }}>
          <span style={{
            fontFamily: fonts.mono,
            fontSize: '0.6rem',
            color: colors.embedBlue,
            opacity: 0.5,
          }}>
            [vec]
          </span>
          {/* CSS X icon */}
          <div style={{
            width: 16,
            height: 16,
            borderRadius: '50%',
            background: 'rgba(255,68,68,0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <span style={{ color: colors.googleRed, fontSize: '0.65rem', fontWeight: 700, lineHeight: 1 }}>✕</span>
          </div>
        </div>
        <div style={{
          fontSize: '0.6rem',
          color: '#555',
          marginTop: 6,
          fontFamily: fonts.mono,
        }}>
          won't build it
        </div>
      </div>

      {/* Ad Networks side */}
      <div style={{
        flex: 1,
        padding: '16px 12px',
        border: '1px solid #44444488',
        borderRadius: 8,
        background: 'rgba(255, 255, 255, 0.02)',
        textAlign: 'center',
        opacity: 0,
        animation: 'fadeSlideRight 0.4s ease 0.2s forwards',
      }}>
        <div style={{
          fontSize: '0.85rem',
          color: '#888',
          marginBottom: 4,
          fontWeight: 600,
        }}>
          Ad Networks
        </div>
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.65rem',
          color: '#888',
          marginBottom: 8,
        }}>
          old pipeline
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
        }}>
          <span style={{
            fontFamily: fonts.mono,
            fontSize: '0.55rem',
            color: colors.googleOrange,
            padding: '2px 6px',
            background: 'rgba(255, 136, 0, 0.1)',
            borderRadius: 4,
            border: `1px solid ${colors.googleOrange}44`,
          }}>
            + "embedding"
          </span>
        </div>
        <div style={{
          fontSize: '0.6rem',
          color: '#555',
          marginTop: 6,
          fontFamily: fonts.mono,
        }}>
          patch, not redesign
        </div>
      </div>

      <style>{`
        @keyframes fadeSlideLeft {
          from { opacity: 0; transform: translateX(-12px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes fadeSlideRight {
          from { opacity: 0; transform: translateX(12px); }
          to { opacity: 1; transform: translateX(0); }
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
