import { colors, fonts } from '../theme';

interface Props {
  stepId: string;
}

type SubState = 'middlemen' | 'consolidation' | 'degradation' | 'antitrust' | 'chat' | 'closing';

const STEP_TO_SUBSTATE: Record<string, SubState> = {
  'history-consolidation': 'consolidation',
  'history-degradation': 'degradation',
  'history-antitrust': 'antitrust',
  'history-chat': 'chat',
  'history-closing': 'closing',
};

const TIMELINES: Record<SubState, string> = {
  middlemen: '2007–2010',
  consolidation: '2008–2011',
  degradation: '2012–2023',
  antitrust: '2024',
  chat: '2026',
  closing: '2026',
};

interface NodeData {
  label: string;
  role: string;
  isGoogle: boolean;
  isMiddleman: boolean;
  color?: string;
  href?: string;
}

const MIDDLEMAN_COLORS: Record<string, string> = {
  SSP: '#E91E63',
  Exchange: '#673AB7',
  DSP: '#9C27B0',
};

function getNodes(subState: SubState): NodeData[] {
  const base: NodeData[] = [
    { label: 'Publisher', role: 'Seller', isGoogle: false, isMiddleman: false },
    { label: 'SSP', role: 'Supply-side platform', isGoogle: false, isMiddleman: true, href: 'https://en.wikipedia.org/wiki/DoubleClick' },
    { label: 'Exchange', role: 'Ad marketplace', isGoogle: false, isMiddleman: true, href: 'https://en.wikipedia.org/wiki/AdMob' },
    { label: 'DSP', role: 'Demand-side platform', isGoogle: false, isMiddleman: true, href: 'https://en.wikipedia.org/wiki/Invite_Media' },
    { label: 'Advertiser', role: 'Buyer', isGoogle: false, isMiddleman: false },
  ];

  if (subState === 'middlemen') return base;

  return base.map(node => {
    if (node.isMiddleman) {
      return { ...node, isGoogle: true };
    }
    return node;
  });
}

/** Multicolor Google G */
function GoogleLogo({ size = '1.1rem' }: { size?: string }) {
  return (
    <span style={{
      fontWeight: 800,
      fontSize: size,
      fontFamily: fonts.body,
    }}>
      <span style={{ color: '#4285F4' }}>G</span>
    </span>
  );
}

/** Renders as <a> when href is provided, <div> otherwise */
function NodeBox({ href, style, children }: { href?: string; style: React.CSSProperties; children: React.ReactNode }) {
  if (href) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" style={{ ...style, textDecoration: 'none', cursor: 'pointer' }}>
        {children}
      </a>
    );
  }
  return <div style={style}>{children}</div>;
}

export function HistoryPipeline({ stepId }: Props) {
  const subState = STEP_TO_SUBSTATE[stepId] ?? 'consolidation';
  const nodes = getNodes(subState);
  const isDimmed = subState === 'chat' || subState === 'closing';
  const showGavel = subState === 'antitrust';
  const showSocialBubble = subState === 'antitrust';
  const showKeywordBlur = subState === 'degradation';
  const showNewPipeline = subState === 'chat' || subState === 'closing';
  const showGCreep = subState === 'closing';
  const isFirstAppearance = subState === 'consolidation';

  const nodeW = isDimmed ? 160 : 220;
  const nodeH = isDimmed ? 32 : 40;
  const vertGap = 8;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 0,
      width: '100%',
      maxWidth: 440,
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Emoji explosion — bursts when Google buys competitors */}
      {isFirstAppearance && ['💰','💵','🤑','💸','💰','💵','💸','🤑','💰','💵','💸','💰'].map((emoji, i) => {
        const angle = (i / 12) * 360;
        const rad = angle * Math.PI / 180;
        const dist = 120 + (i % 3) * 40;
        const tx = Math.cos(rad) * dist;
        const ty = Math.sin(rad) * dist;
        return (
          <div
            key={`emoji-${i}`}
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              fontSize: ['3rem', '3.9rem', '2.7rem'][i % 3],
              opacity: 0,
              animation: `emojiExplode 2s ease-out 1.8s forwards`,
              zIndex: 10,
              pointerEvents: 'none',
              // @ts-ignore
              '--tx': `${tx}px`,
              '--ty': `${ty}px`,
            } as any}
          >
            {emoji}
          </div>
        );
      })}
      {/* Top row: old pipeline (+ new pipeline when dimmed) */}
      <div style={{
        display: 'flex',
        flexDirection: isDimmed ? 'row' : 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: isDimmed ? 20 : 0,
        width: '100%',
      }}>
        {/* Main pipeline */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 0,
          opacity: isDimmed ? 0.25 : 1,
          transition: 'opacity 0.8s',
          flex: isDimmed ? '0 0 auto' : undefined,
          transform: isDimmed ? 'scale(0.7)' : 'scale(1)',
          transformOrigin: 'center center',
        }}>
          {nodes.map((node, i) => {
            const middlemanColor = MIDDLEMAN_COLORS[node.label] ?? '#888';
            // For consolidation first appearance, start with independent colors
            const showGoogleTransition = isFirstAppearance && node.isMiddleman;
            const nodeColor = node.isGoogle && !showGoogleTransition
              ? colors.googleRed
              : node.isMiddleman
                ? middlemanColor
                : node.label === 'Publisher' ? '#2196F3' : colors.pipeline.advertiser;
            const bgColor = node.isGoogle && !showGoogleTransition
              ? 'rgba(255,68,68,0.08)'
              : node.isMiddleman
                ? `${middlemanColor}12`
                : node.label === 'Publisher' ? 'rgba(33,150,243,0.04)' : 'rgba(76,175,80,0.04)';
            const staggerDelay = isFirstAppearance && node.isMiddleman
              ? `${(i - 1) * 0.2}s`
              : '0s';

            return (
              <div key={node.label + i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                {/* Arrow connector */}
                {i > 0 && (
                  <div style={{
                    color: node.isGoogle && !showGoogleTransition ? colors.googleRed : '#444',
                    fontSize: '0.8rem',
                    lineHeight: 1,
                    height: vertGap + 8,
                    display: 'flex',
                    alignItems: 'center',
                    opacity: isFirstAppearance && node.isMiddleman ? 0 : 1,
                    animation: isFirstAppearance && node.isMiddleman
                      ? `fadeIn 0.3s ease ${staggerDelay} forwards`
                      : 'none',
                  }}>
                    ↓
                  </div>
                )}
                {/* Node */}
                <NodeBox
                  href={node.isGoogle ? node.href : undefined}
                  style={{
                    width: nodeW,
                    height: nodeH,
                    borderRadius: 10,
                    border: `2px solid ${nodeColor}`,
                    background: bgColor,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 8,
                    position: 'relative',
                    transition: 'border-color 0.5s, background 0.5s, width 0.5s, height 0.5s',
                    opacity: isFirstAppearance && node.isMiddleman ? 0 : 1,
                    animation: isFirstAppearance && node.isMiddleman
                      ? `fadeSlideDown 0.4s ease ${staggerDelay} forwards`
                      : 'none',
                    boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
                  }}>
                  {/* Google overlay — fades in at 1.8s for consolidation */}
                  {showGoogleTransition && (
                    <div style={{
                      position: 'absolute',
                      inset: -2,
                      borderRadius: 10,
                      border: `2px solid ${colors.googleRed}`,
                      background: '#140a0c',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 8,
                      opacity: 0,
                      animation: 'consolidateOverlay 0.6s ease 1.8s forwards',
                      zIndex: 1,
                    }}>
                      <GoogleLogo size="1.1rem" />
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontWeight: 600, fontSize: '0.85rem', color: '#fff' }}>
                          Google {node.label}
                        </div>
                        <div style={{ fontSize: '0.65rem', color: '#888', fontFamily: fonts.mono }}>
                          {node.role}
                        </div>
                      </div>
                    </div>
                  )}
                  {/* G logo (for non-transitioning Google nodes) */}
                  {node.isGoogle && !showGoogleTransition && (
                    <GoogleLogo size={isDimmed ? '0.9rem' : '1.1rem'} />
                  )}
                  {/* Label */}
                  <div style={{ textAlign: 'center' }}>
                    <div style={{
                      fontWeight: 600,
                      fontSize: isDimmed ? '0.7rem' : '0.85rem',
                      color: '#fff',
                    }}>
                      {node.isGoogle && !showGoogleTransition ? `Google ${node.label}` : node.label}
                    </div>
                    <div style={{
                      fontSize: isDimmed ? '0.55rem' : '0.65rem',
                      color: '#888',
                      fontFamily: fonts.mono,
                    }}>
                      {node.role}
                    </div>
                  </div>
                </NodeBox>
              </div>
            );
          })}
        </div>

        {/* New pipeline appears alongside the dimmed old one */}
        {showNewPipeline && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 0,
            flex: '0 0 auto',
          }}>
            {['Chat', '???', '???', '???', 'Advertiser'].map((label, i) => {
              const isFilling = showGCreep && i > 0 && i < 4;
              return (
                <div key={`new-${i}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  {i > 0 && (
                    <div style={{
                      color: isFilling ? colors.googleRed : '#444',
                      fontSize: '0.7rem',
                      lineHeight: 1,
                      height: 12,
                      display: 'flex',
                      alignItems: 'center',
                      opacity: 0,
                      animation: `fadeIn 0.3s ease ${i * 0.08}s forwards`,
                    }}>
                      ↓
                    </div>
                  )}
                  <div style={{
                    width: 160,
                    height: 36,
                    borderRadius: 10,
                    border: isFilling
                      ? `2px solid ${colors.googleRed}`
                      : `2px solid ${label === 'Chat' || label === 'Advertiser' ? '#6366f1' : colors.glow}`,
                    background: isFilling
                      ? 'rgba(255,68,68,0.06)'
                      : label === 'Chat' || label === 'Advertiser'
                        ? 'rgba(99,102,241,0.08)'
                        : `rgba(76,175,80,0.06)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 4,
                    opacity: 0,
                    animation: `fadeSlideDown 0.3s ease ${i * 0.08}s forwards`,
                    boxShadow: isFilling
                      ? '0 2px 6px rgba(0,0,0,0.15)'
                      : '0 2px 6px rgba(0,0,0,0.1)',
                  }}>
                    {isFilling && <GoogleLogo size="0.9rem" />}
                    <span style={{
                      fontFamily: fonts.mono,
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      color: isFilling
                        ? colors.googleRed
                        : label === 'Chat' || label === 'Advertiser'
                          ? '#8b8cf8'
                          : colors.glow,
                    }}>
                      {label}
                    </span>
                  </div>
                </div>
              );
            })}
            <div style={{
              fontFamily: fonts.mono,
              fontSize: '0.65rem',
              color: colors.glow,
              marginTop: 8,
              opacity: 0,
              animation: 'fadeIn 0.3s ease 0.4s forwards',
            }}>
              new surface
            </div>
            {showGCreep && (
              <div style={{
                fontFamily: fonts.mono,
                fontSize: '0.65rem',
                color: colors.googleRed,
                marginTop: 4,
                opacity: 0,
                animation: 'fadeIn 0.3s ease 0.5s forwards',
              }}>
                same playbook, different logo
              </div>
            )}
          </div>
        )}
      </div>

      {/* Keyword degradation */}
      {showKeywordBlur && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 4,
          marginTop: 16,
          opacity: 0,
          animation: 'fadeIn 0.4s ease 0.2s forwards',
        }}>
          <span style={{ fontFamily: fonts.mono, fontSize: '0.7rem', color: colors.googleOrange }}>
            "knee pain running downhill"
          </span>
          <span style={{ color: '#555', fontSize: '0.7rem' }}>↓</span>
          <span style={{ fontFamily: fonts.mono, fontSize: '0.7rem', color: colors.googleOrange }}>
            "knee pain running"
          </span>
          <span style={{ color: '#555', fontSize: '0.7rem' }}>↓</span>
          <span style={{ fontFamily: fonts.mono, fontSize: '0.7rem', color: colors.googleOrange }}>
            "knee pain"
          </span>
        </div>
      )}

      {/* Antitrust section — replaces emoji gavel with styled text */}
      {showGavel && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          marginTop: 16,
          opacity: 0,
          animation: 'fadeIn 0.5s ease 0.2s forwards',
        }}>
          {/* CSS scales of justice */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            width: 40,
          }}>
            {/* Balance bar */}
            <div style={{
              width: 32,
              height: 2,
              background: colors.googleRed,
              position: 'relative',
              transform: 'rotate(-8deg)',
            }}>
              {/* Left pan */}
              <div style={{
                position: 'absolute',
                left: -4,
                top: 2,
                width: 12,
                height: 8,
                borderRadius: '0 0 6px 6px',
                border: `2px solid ${colors.googleRed}`,
                borderTop: 'none',
              }} />
              {/* Right pan */}
              <div style={{
                position: 'absolute',
                right: -4,
                top: -6,
                width: 12,
                height: 8,
                borderRadius: '0 0 6px 6px',
                border: `2px solid ${colors.googleRed}`,
                borderTop: 'none',
              }} />
            </div>
            {/* Post */}
            <div style={{ width: 2, height: 12, background: colors.googleRed }} />
            <div style={{ width: 16, height: 2, background: colors.googleRed, borderRadius: 1 }} />
          </div>

          <div>
            <div style={{
              fontFamily: fonts.mono,
              fontSize: '0.8rem',
              color: colors.googleRed,
              fontWeight: 700,
              letterSpacing: '0.05em',
            }}>
              DOJ v. Google
            </div>
            <div style={{
              fontFamily: fonts.mono,
              fontSize: '0.7rem',
              color: colors.googleRed,
              fontWeight: 600,
              marginTop: 2,
            }}>
              MONOPOLY
            </div>
            {showSocialBubble && (
              <div style={{
                fontFamily: fonts.mono, fontSize: '0.65rem', color: '#555',
                marginTop: 4,
                opacity: 0,
                animation: 'fadeIn 0.4s ease 0.6s forwards',
              }}>
                attention, no intent
              </div>
            )}
          </div>
        </div>
      )}

      {/* Timeline marker */}
      <div style={{
        fontFamily: fonts.mono,
        fontSize: '0.75rem',
        color: '#555',
        padding: '4px 12px',
        border: '1px solid #333',
        borderRadius: 12,
        marginTop: 16,
        opacity: 0,
        animation: 'fadeIn 0.3s ease 0.5s forwards',
      }}>
        {TIMELINES[subState]}
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes fadeSlideDown {
          from { opacity: 0; transform: translateY(-8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes consolidateOverlay {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes emojiExplode {
          0% { opacity: 1; transform: translate(-50%, -50%) scale(0); }
          30% { opacity: 1; transform: translate(calc(-50% + var(--tx) * 0.6), calc(-50% + var(--ty) * 0.6)) scale(1.2); }
          100% { opacity: 0; transform: translate(calc(-50% + var(--tx)), calc(-50% + var(--ty))) scale(0.5); }
        }
      `}</style>
    </div>
  );
}
