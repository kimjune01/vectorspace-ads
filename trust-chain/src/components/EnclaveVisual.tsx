import { colors, fonts } from '../theme';

interface Props {
  stepId: string;
}

export function EnclaveVisual({ stepId }: Props) {
  const sealed = stepId === 'exchange-trust';
  const showReceipt = stepId === 'enclave-proof';

  return (
    <div style={{
      width: '100%',
      maxWidth: 380,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 16,
    }}>
      {/* Inputs flowing in */}
      <div style={{
        display: 'flex',
        gap: 24,
        justifyContent: 'center',
        width: '100%',
      }}>
        <InputArrow label="embedding" sublabel="[0.70, 0.68, ...]" color={colors.embedBlue} />
        <InputArrow label="bids" sublabel="$2, $8, $4 ..." color={colors.embedGreen} />
      </div>

      {/* Arrow down */}
      <div style={{ color: '#555', fontSize: '1.2rem', lineHeight: 1 }}>↓</div>

      {/* The enclave box */}
      <div style={{
        position: 'relative',
        width: '100%',
        border: `2px solid ${sealed ? '#333' : '#00BCD4'}`,
        borderRadius: 12,
        background: sealed ? '#111' : 'rgba(0, 188, 212, 0.04)',
        padding: '28px 20px 20px',
        overflow: 'hidden',
        transition: 'border-color 0.6s, background 0.6s',
      }}>
        {/* Sealed label */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 24,
          background: sealed ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 188, 212, 0.12)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 6,
          transition: 'background 0.6s',
        }}>
          <LockIcon color={sealed ? '#666' : '#00BCD4'} />
          <span style={{
            fontSize: '0.6rem',
            fontFamily: fonts.mono,
            color: sealed ? '#666' : '#00BCD4',
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
            transition: 'color 0.6s',
          }}>
            {sealed ? 'Sealed — contents hidden' : 'Sealed Enclave'}
          </span>
          <LockIcon color={sealed ? '#666' : '#00BCD4'} />
        </div>

        {/* Code inside the box — hidden when sealed */}
        <div style={{
          fontFamily: fonts.mono,
          fontSize: '0.7rem',
          lineHeight: 1.8,
          color: '#888',
          marginTop: 4,
          opacity: sealed ? 0 : 1,
          transition: 'opacity 0.6s',
        }}>
          <div><span style={{ color: '#00BCD4' }}>for</span> bid <span style={{ color: '#00BCD4' }}>in</span> bids {'{'}</div>
          <div style={{ paddingLeft: 16 }}>
            score = <span style={{ color: '#4CAF50' }}>proximity</span>(bid.pos, query)
          </div>
          <div style={{ paddingLeft: 16 }}>
            score -= <span style={{ color: '#FF8800' }}>relocation_fee</span>(bid)
          </div>
          <div>{'}'}</div>
          <div>winner = <span style={{ color: '#4CAF50' }}>max</span>(scores)</div>
          <div>price = <span style={{ color: '#2196F3' }}>second_price</span>(scores)</div>
        </div>

        {/* Question mark when sealed */}
        {sealed && (
          <div style={{
            position: 'absolute',
            top: 24,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '2.5rem',
            color: '#333',
            fontWeight: 700,
            pointerEvents: 'none',
          }}>
            ?
          </div>
        )}

        {/* Walls — visual emphasis that the box is sealed */}
        <div style={{
          position: 'absolute',
          top: 24,
          left: 0,
          bottom: 0,
          width: 2,
          background: sealed ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 188, 212, 0.3)',
          transition: 'background 0.6s',
        }} />
        <div style={{
          position: 'absolute',
          top: 24,
          right: 0,
          bottom: 0,
          width: 2,
          background: sealed ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 188, 212, 0.3)',
          transition: 'background 0.6s',
        }} />
      </div>

      {/* Arrow down */}
      <div style={{ color: '#555', fontSize: '1.2rem', lineHeight: 1 }}>↓</div>

      {/* Output */}
      <div style={{
        display: 'flex',
        gap: 16,
        justifyContent: 'center',
        width: '100%',
        transition: 'opacity 0.6s',
      }}>
        <OutputBox
          label="Winner"
          value={sealed ? 'Dr. Chen — $8' : 'Dr. Chen — $10'}
          color={sealed ? '#FF4444' : colors.embedGreen}
          highlight={sealed}
          highlightColor={sealed ? 'rgba(255, 68, 68, 0.08)' : undefined}
        />
        <OutputBox
          label="Signature"
          value={sealed ? 'none' : (showReceipt ? '✓ verified' : '0x7a3f...c91e')}
          color={sealed ? '#FF4444' : (showReceipt ? '#4CAF50' : '#00BCD4')}
        />
      </div>

      {/* Explanation caption */}
      <div style={{
        fontSize: '0.7rem',
        fontFamily: fonts.mono,
        color: '#555',
        textAlign: 'center',
        lineHeight: 1.5,
        marginTop: 4,
      }}>
        {sealed
          ? 'Inputs go in. What happens inside?'
          : showReceipt
            ? 'The signature proves the code ran unmodified. Nobody picked the winner — the math did.'
            : 'The exchange runs the auction but can\'t see the inputs or rig the output.'
        }
      </div>
    </div>
  );
}

function InputArrow({ label, sublabel, color }: { label: string; sublabel: string; color: string }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 2,
    }}>
      <div style={{
        fontSize: '0.75rem',
        fontFamily: "'JetBrains Mono', monospace",
        color,
        fontWeight: 600,
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '0.6rem',
        fontFamily: "'JetBrains Mono', monospace",
        color: '#555',
      }}>
        {sublabel}
      </div>
    </div>
  );
}

function OutputBox({ label, value, color, highlight, highlightColor }: {
  label: string;
  value: string;
  color: string;
  highlight?: boolean;
  highlightColor?: string;
}) {
  return (
    <div style={{
      flex: 1,
      padding: '10px 12px',
      border: `1px solid ${highlight ? color : '#333'}`,
      borderRadius: 8,
      background: highlight ? (highlightColor ?? 'rgba(76, 175, 80, 0.08)') : 'rgba(255,255,255,0.02)',
      textAlign: 'center',
    }}>
      <div style={{
        fontSize: '0.6rem',
        fontFamily: "'JetBrains Mono', monospace",
        color: '#666',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: 4,
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '0.8rem',
        fontFamily: "'JetBrains Mono', monospace",
        color,
        fontWeight: 600,
      }}>
        {value}
      </div>
    </div>
  );
}

function LockIcon({ color = '#00BCD4' }: { color?: string }) {
  return (
    <svg width="10" height="12" viewBox="0 0 10 12" fill="none">
      <rect x="1" y="5" width="8" height="6" rx="1" fill="none" stroke={color} strokeWidth="1.2" />
      <path d="M3 5V3.5C3 2.12 3.9 1 5 1C6.1 1 7 2.12 7 3.5V5" stroke={color} strokeWidth="1.2" fill="none" />
      <circle cx="5" cy="8.5" r="0.8" fill={color} />
    </svg>
  );
}
