import { fonts } from '../theme';

const events = [
  { year: '1998', label: 'GoTo.com launches pay-per-click', color: '#4CAF50', side: 'past' as const },
  { year: '2000', label: 'Google launches AdWords', color: '#FF8800', side: 'past' as const },
  { year: '2003', label: 'Google dominates search ads', color: '#FF4444', side: 'past' as const },
  { year: '2008', label: 'Google buys DoubleClick, owns the stack', color: '#FF4444', side: 'past' as const },
  { year: '', label: '', color: 'transparent', side: 'gap' as const },
  { year: '2024', label: 'Chat surfaces emerge', color: '#4CAF50', side: 'now' as const },
  { year: '2026', label: 'Window is open — no protocol yet', color: '#FFB300', side: 'now' as const, highlight: true },
  { year: '202?', label: 'Chatbots get captured', color: '#FF4444', side: 'now' as const, danger: true },
];

export function WindowTimeline() {
  return (
    <div style={{
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 0,
    }}>
      {/* Header */}
      <div style={{
        fontFamily: fonts.mono,
        fontSize: '0.7rem',
        color: '#666',
        letterSpacing: '0.15em',
        marginBottom: 16,
      }}>
        SEARCH ADS: 5 YEARS FROM OPEN TO CAPTURED
      </div>

      {/* Timeline */}
      <div style={{ position: 'relative', width: '100%', maxWidth: 360 }}>
        {events.map((event, i) => {
          if (event.side === 'gap') {
            return (
              <div key={i} style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: 40,
                position: 'relative',
              }}>
                {/* Vertical line continues */}
                <div style={{
                  position: 'absolute',
                  left: 48,
                  top: 0,
                  bottom: 0,
                  width: 2,
                  background: 'repeating-linear-gradient(to bottom, #333 0px, #333 4px, transparent 4px, transparent 8px)',
                }} />
                <div style={{
                  fontFamily: fonts.mono,
                  fontSize: '0.65rem',
                  color: '#555',
                  marginLeft: 80,
                }}>
                  16 years later...
                </div>
              </div>
            );
          }

          return (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 12,
                position: 'relative',
                paddingBottom: i < events.length - 1 ? 0 : 0,
                opacity: 0,
                animation: `timelineReveal 0.4s ease ${i * 0.12}s forwards`,
              }}
            >
              {/* Year label */}
              <div style={{
                fontFamily: fonts.mono,
                fontSize: '0.75rem',
                color: event.highlight ? '#FFB300' : event.danger ? '#FF4444' : '#888',
                width: 36,
                textAlign: 'right',
                flexShrink: 0,
                paddingTop: 4,
                fontWeight: event.highlight || event.danger ? 700 : 400,
              }}>
                {event.year}
              </div>

              {/* Dot + line */}
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                flexShrink: 0,
                width: 14,
              }}>
                <div style={{
                  width: event.highlight ? 14 : 10,
                  height: event.highlight ? 14 : 10,
                  borderRadius: '50%',
                  background: event.color,
                  marginTop: event.highlight ? 2 : 4,
                  flexShrink: 0,
                  boxShadow: event.highlight
                    ? `0 0 12px ${event.color}88`
                    : event.danger
                    ? `0 0 8px ${event.color}66`
                    : 'none',
                }} />
                {i < events.length - 1 && (
                  <div style={{
                    width: 2,
                    height: 28,
                    background: event.side === 'past' ? '#333' : event.danger ? '#FF444444' : '#333',
                  }} />
                )}
              </div>

              {/* Label */}
              <div style={{
                fontSize: '0.85rem',
                color: event.highlight ? '#FFB300' : event.danger ? '#FF4444' : '#bbb',
                paddingTop: 2,
                fontWeight: event.highlight || event.danger ? 600 : 400,
                lineHeight: 1.4,
                animation: event.danger ? 'dangerGlow 2s ease-in-out infinite' : 'none',
              }}>
                {event.label}
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom spacer */}
      <div style={{ height: 8 }} />

      <style>{`
        @keyframes timelineReveal {
          from { opacity: 0; transform: translateX(-8px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes dangerGlow {
          0%, 100% { text-shadow: 0 0 4px rgba(255,68,68,0.3); }
          50% { text-shadow: 0 0 16px rgba(255,68,68,0.8), 0 0 32px rgba(255,68,68,0.4); }
        }
      `}</style>
    </div>
  );
}
