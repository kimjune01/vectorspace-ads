import { colors, fonts } from '../theme';
import { CHAT_MESSAGES, EMBEDDING_RECEIPT } from '../data';
import { Receipt } from './Receipt';

interface Props {
  stepId: string;
}

type SubState = 'dot-intro' | 'dot-brightens' | 'dot-auction' | 'dot-philosophy';

const STEP_TO_SUBSTATE: Record<string, SubState> = {
  'dot-intro': 'dot-intro',
  'dot-brightens': 'dot-brightens',
  'dot-auction': 'dot-auction',
  'dot-philosophy': 'dot-philosophy',
};

function getDotStyle(subState: SubState): { color: string; size: number; shadow: string; glow: boolean } {
  switch (subState) {
    case 'dot-intro':
      return { color: colors.chat.dotGray, size: 12, shadow: 'none', glow: false };
    case 'dot-brightens':
      return { color: colors.chat.dotGreen, size: 14, shadow: `0 0 12px ${colors.chat.dotGreen}`, glow: true };
    case 'dot-auction':
      return { color: colors.chat.dotGreen, size: 200, shadow: 'none', glow: false };
    case 'dot-philosophy':
      return { color: colors.chat.dotGreen, size: 12, shadow: 'none', glow: false };
  }
}

function getPrevVisibleMessages(subState: SubState): number {
  switch (subState) {
    case 'dot-intro': return 0;
    case 'dot-brightens': return 2;
    case 'dot-auction': return 4;
    case 'dot-philosophy': return 5;
  }
}

function getVisibleMessages(subState: SubState): number {
  switch (subState) {
    case 'dot-intro': return 2;
    case 'dot-brightens': return 4;
    case 'dot-auction': return 5;
    case 'dot-philosophy': return 5;
  }
}

const TIMESTAMPS = ['2:32 PM', '2:32 PM', '2:34 PM', '2:34 PM', '2:35 PM'];

export function ChatMockup({ stepId }: Props) {
  const subState = STEP_TO_SUBSTATE[stepId] ?? 'dot-intro';
  const dotStyle = getDotStyle(subState);
  const visibleCount = getVisibleMessages(subState);
  const prevCount = getPrevVisibleMessages(subState);
  const showCard = subState === 'dot-auction';
  const showReceipt = subState === 'dot-auction';
  const isDotExpanded = subState === 'dot-auction';

  return (
    <div style={{
      width: '100%',
      maxWidth: 440,
      display: 'flex',
      flexDirection: 'column',
      gap: 16,
    }}>
    <div style={{
      width: '100%',
      background: colors.chat.bg,
      borderRadius: 12,
      border: '1px solid #2a2a4a',
      overflow: 'hidden',
      position: 'relative',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Chat header */}
      <div style={{
        padding: '10px 16px',
        borderBottom: '1px solid #2a2a4a',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
      }}>
        <div style={{
          width: 28,
          height: 28,
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}>
          <span style={{ color: '#fff', fontSize: '0.6rem', fontWeight: 700 }}>AI</span>
        </div>
        <div>
          <span style={{
            fontSize: '0.8rem',
            fontWeight: 600,
            color: '#e0e0e0',
          }}>
            Assistant
          </span>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}>
            <div style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: '#4CAF50',
            }} />
            <span style={{ fontSize: '0.6rem', color: '#888' }}>Online</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div style={{
        padding: '12px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        maxHeight: isDotExpanded ? 120 : 280,
        overflow: 'hidden',
        transition: 'max-height 0.5s',
      }}>
        {CHAT_MESSAGES.slice(0, visibleCount).map((msg, i) => {
          const isNew = i >= prevCount;
          const staggerDelay = isNew ? `${(i - prevCount) * 0.2}s` : '0s';
          const isUser = msg.role === 'user';
          // Show timestamp if first message or role changed from previous
          const showTimestamp = i === 0 || CHAT_MESSAGES[i - 1].role !== msg.role;
          return (
            <div key={i} style={{
              opacity: isNew ? 0 : 1,
              animation: isNew ? `msgSlideIn 0.4s ease ${staggerDelay} forwards` : 'none',
            }}>
              {/* Timestamp */}
              {showTimestamp && (
                <div style={{
                  textAlign: 'center',
                  fontSize: '0.55rem',
                  color: '#555',
                  margin: '4px 0',
                }}>
                  {TIMESTAMPS[i]}
                </div>
              )}
              <div style={{
                display: 'flex',
                justifyContent: isUser ? 'flex-end' : 'flex-start',
                alignItems: 'flex-start',
                gap: 8,
              }}>
                {/* Assistant avatar */}
                {!isUser && (
                  <div style={{
                    width: 24,
                    height: 24,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    marginTop: 2,
                  }}>
                    <span style={{ color: '#fff', fontSize: '0.5rem', fontWeight: 700 }}>AI</span>
                  </div>
                )}
                <div style={{
                  maxWidth: '75%',
                  padding: '8px 12px',
                  borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                  background: isUser ? '#2563eb' : colors.chat.assistantBubble,
                  fontSize: '0.8rem',
                  lineHeight: 1.5,
                  color: isUser ? '#fff' : '#bbb',
                }}>
                  {msg.text}
                </div>
                {/* User avatar */}
                {isUser && (
                  <div style={{
                    width: 24,
                    height: 24,
                    borderRadius: '50%',
                    background: '#374151',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    marginTop: 2,
                  }}>
                    <span style={{ color: '#9ca3af', fontSize: '0.55rem', fontWeight: 600 }}>U</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Typing indicator — show when new assistant message is about to appear */}
        {visibleCount < CHAT_MESSAGES.length && CHAT_MESSAGES[visibleCount]?.role === 'assistant' && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            opacity: 0,
            animation: `msgSlideIn 0.4s ease ${(visibleCount - prevCount) * 0.2 + 0.3}s forwards`,
          }}>
            <div style={{
              width: 24,
              height: 24,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}>
              <span style={{ color: '#fff', fontSize: '0.5rem', fontWeight: 700 }}>AI</span>
            </div>
            <div style={{
              padding: '8px 14px',
              borderRadius: '12px 12px 12px 2px',
              background: colors.chat.assistantBubble,
              display: 'flex',
              gap: 4,
              alignItems: 'center',
            }}>
              <span className="typing-dot" style={{ animationDelay: '0s' }} />
              <span className="typing-dot" style={{ animationDelay: '0.15s' }} />
              <span className="typing-dot" style={{ animationDelay: '0.3s' }} />
            </div>
          </div>
        )}
      </div>

      {/* Proximity dot / expanded card area */}
      <div style={{
        padding: '8px 16px 8px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 12,
      }}>
        {/* The dot */}
        {!isDotExpanded && (
          <div style={{
            width: dotStyle.size,
            height: dotStyle.size,
            borderRadius: '50%',
            background: dotStyle.color,
            boxShadow: dotStyle.shadow,
            transition: 'all 0.6s ease',
            cursor: 'pointer',
            animation: dotStyle.glow ? 'pulse 2s ease-in-out infinite' : 'none',
          }} />
        )}

        {/* Expanded: Dr. Chen card */}
        {showCard && (
          <div style={{
            width: '100%',
            padding: '14px 16px',
            background: 'rgba(76, 175, 80, 0.08)',
            border: `1px solid ${colors.embedGreen}44`,
            borderRadius: 10,
            opacity: 0,
            animation: 'msgSlideIn 0.5s ease 0.2s forwards',
          }}>
            <div style={{
              fontSize: '0.95rem',
              fontWeight: 600,
              color: '#fff',
              marginBottom: 4,
            }}>
              Dr. Chen Sports Biomechanics
            </div>
            <div style={{
              fontSize: '0.8rem',
              color: colors.embedGreen,
              marginBottom: 6,
            }}>
              Runner knee specialist — eccentric loading
            </div>
            <div style={{
              fontSize: '0.75rem',
              color: '#888',
            }}>
              0.4 mi away · Accepting new patients
            </div>
          </div>
        )}

      </div>

      {/* Input field at bottom */}
      <div style={{
        padding: '8px 12px 12px',
        borderTop: '1px solid #2a2a4a',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          background: '#1e1e3a',
          borderRadius: 20,
          padding: '8px 12px',
          border: '1px solid #333',
        }}>
          <span style={{
            flex: 1,
            fontSize: '0.75rem',
            color: '#555',
            fontFamily: fonts.body,
          }}>
            Message...
          </span>
          <div style={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            background: '#374151',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
          }}>
            <span style={{ color: '#888', fontSize: '0.8rem' }}>↑</span>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { box-shadow: 0 0 6px ${colors.chat.dotGreen}; }
          50% { box-shadow: 0 0 16px ${colors.chat.dotGreen}; }
        }
        @keyframes msgSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .typing-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #666;
          animation: typingBounce 1.2s ease-in-out infinite;
          display: inline-block;
        }
        @keyframes typingBounce {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-4px); opacity: 1; }
        }
      `}</style>
    </div>

      {/* Receipt — below the chat box */}
      {showReceipt && (
        <div style={{
          transform: 'scale(0.85)',
          transformOrigin: 'top center',
          opacity: 0,
          animation: 'msgSlideIn 0.5s ease 0.5s forwards',
        }}>
          <Receipt data={EMBEDDING_RECEIPT} variant="embedding" visible={true} />
        </div>
      )}
    </div>
  );
}
