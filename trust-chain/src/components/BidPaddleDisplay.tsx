import { colors, fonts } from '../theme';
import { BidPaddle } from './BidPaddle';
import { QUALITY_SCORE_BIDDERS, GOOGLE_BIDDERS } from '../data';

interface Props {
  stepId: string;
}

export function BidPaddleDisplay({ stepId }: Props) {
  if (stepId === 'history-quality-score') {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        background: 'rgba(255,255,255,0.02)',
        borderRadius: 8,
        border: '1px solid #2a2a2a',
        overflow: 'hidden',
      }}>
        {/* Table header */}
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
            width: 80,
            textAlign: 'right',
          }}>
            Quality
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

        {/* Rows */}
        {QUALITY_SCORE_BIDDERS.map((bidder, i) => (
          <BidPaddle
            key={bidder.name}
            name={bidder.name}
            bid={bidder.bid}
            color={bidder.color}
            isWinner={bidder.wins}
            visible={true}
            index={i}
            qualityScore={bidder.qualityScore}
          />
        ))}

        {/* Timeline marker */}
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.75rem',
          color: '#555',
          padding: '8px 12px',
          textAlign: 'center',
          borderTop: '1px solid #2a2a2a',
        }}>
          2003
        </div>
      </div>
    );
  }

  // wrong-path-auction: the familiar auction display
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      background: 'rgba(255,255,255,0.02)',
      borderRadius: 8,
      border: '1px solid #2a2a2a',
      overflow: 'hidden',
    }}>
      {/* Table header */}
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
        <div style={{
          fontSize: '0.6rem',
          color: '#666',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          fontFamily: fonts.mono,
          width: 40,
          textAlign: 'right',
        }}>
          Status
        </div>
      </div>

      {/* Keyword bin label */}
      <div style={{
        fontSize: '0.6rem',
        letterSpacing: '0.05em',
        fontFamily: fonts.mono,
        padding: '6px 12px',
        borderBottom: '1px solid #222',
        background: 'rgba(255,136,0,0.04)',
        display: 'flex',
        gap: 4,
      }}>
        <span style={{ color: '#888' }}>BIN:</span>
        <span style={{ color: colors.googleOrange }}>"knee pain running"</span>
      </div>

      {/* Rows */}
      {GOOGLE_BIDDERS.map((bidder, i) => {
        const isDrChen = bidder.name.includes('Dr. Chen');
        return (
          <div
            key={bidder.name}
            style={isDrChen ? {
              background: 'rgba(68, 170, 255, 0.12)',
              animation: 'drChenDesaturate 1.2s ease 1.5s forwards',
            } : undefined}
          >
            <BidPaddle
              name={bidder.name}
              bid={bidder.bid}
              color={bidder.color}
              isWinner={i === 0}
              visible={true}
              index={i}
              forceVisible={isDrChen}
            />
          </div>
        );
      })}

      <style>{`
        @keyframes drChenDesaturate {
          0% { filter: none; opacity: 1; background: rgba(68, 170, 255, 0.12); }
          100% { filter: saturate(0) brightness(0.5); opacity: 0.35; background: transparent; }
        }
      `}</style>
    </div>
  );
}
