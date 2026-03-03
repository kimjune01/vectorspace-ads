import { fonts } from '../theme';

interface Props {
  name: string;
  bid: number;
  color: string;
  isWinner: boolean;
  visible: boolean;
  index: number;
  qualityScore?: string;
}

export function BidPaddle({ name, bid, color, isWinner, visible, index, qualityScore }: Props) {
  const delay = `${index * 0.12}s`;
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 0,
      padding: '8px 12px',
      borderLeft: isWinner ? `3px solid ${color}` : '3px solid transparent',
      background: isWinner ? 'rgba(76, 175, 80, 0.06)' : 'transparent',
      borderBottom: '1px solid #222',
      opacity: visible ? 0 : 0,
      animation: visible ? `bidRowSlideIn 0.35s ease ${delay} forwards` : 'none',
    }}>
      {/* Rank */}
      <div style={{
        fontFamily: fonts.mono,
        fontSize: '0.7rem',
        color: isWinner ? color : '#555',
        fontWeight: 600,
        width: 28,
        flexShrink: 0,
      }}>
        #{index + 1}
      </div>

      {/* Name */}
      <div style={{
        flex: 1,
        fontSize: '0.8rem',
        color: isWinner ? '#e0e0e0' : '#777',
        fontWeight: isWinner ? 600 : 400,
        opacity: isWinner ? 1 : 0.7,
      }}>
        {name}
      </div>

      {/* Quality Score (if provided) */}
      {qualityScore && (
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.6rem',
          color: qualityScore === 'high' ? '#4CAF50' : '#FF6666',
          marginRight: 12,
          padding: '2px 6px',
          borderRadius: 3,
          background: qualityScore === 'high' ? 'rgba(76,175,80,0.1)' : 'rgba(255,102,102,0.1)',
        }}>
          QS: {qualityScore}
        </div>
      )}

      {/* Bid */}
      <div style={{
        fontFamily: fonts.mono,
        fontWeight: 700,
        fontSize: '0.9rem',
        color: isWinner ? color : '#666',
        minWidth: 40,
        textAlign: 'right',
      }}>
        ${bid}
      </div>

      {/* Status */}
      <div style={{
        fontFamily: fonts.mono,
        fontSize: '0.6rem',
        color: isWinner ? color : '#555',
        width: 40,
        textAlign: 'right',
        fontWeight: 600,
      }}>
        {isWinner ? 'WON' : ''}
      </div>

      <style>{`
        @keyframes bidRowSlideIn {
          from { opacity: 0; transform: translateX(-12px); }
          to { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  );
}
