import { colors, fonts } from '../theme';

interface Props {
  keywords: string[];
  discarded: string[];
  showDiscarded: boolean;
}

export function KeywordBin({ keywords, discarded, showDiscarded }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
      {/* Discarded words with red × */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 8,
        justifyContent: 'center',
        opacity: showDiscarded ? 1 : 0,
        transition: 'opacity 0.5s',
      }}>
        {discarded.map((word) => (
          <span
            key={word}
            style={{
              fontFamily: fonts.mono,
              fontSize: '0.85rem',
              color: '#777',
              textDecoration: 'line-through',
              padding: '4px 10px',
              background: 'rgba(255,255,255,0.03)',
              borderRadius: 4,
              border: '1px solid #2a2a2a',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              opacity: showDiscarded ? 0.7 : 0,
              transition: 'opacity 0.8s',
            }}
          >
            {word}
            <span style={{
              color: '#ef4444',
              fontSize: '0.7rem',
              fontWeight: 700,
              lineHeight: 1,
            }}>
              ×
            </span>
          </span>
        ))}
      </div>

      {/* Arrow down */}
      <div style={{ color: '#444', fontSize: '1.2rem' }}>↓</div>

      {/* Bin — styled as a bucket/container */}
      <div style={{
        border: `2px solid ${colors.googleOrange}`,
        borderRadius: 8,
        padding: '12px 20px',
        background: 'rgba(255, 136, 0, 0.06)',
        textAlign: 'center',
        position: 'relative',
      }}>
        {/* Bin header with icon */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 6,
          marginBottom: 10,
        }}>
          {/* Bucket icon - CSS */}
          <div style={{
            width: 16,
            height: 14,
            borderRadius: '0 0 4px 4px',
            border: `2px solid ${colors.googleOrange}`,
            borderTop: 'none',
            position: 'relative',
          }}>
            <div style={{
              position: 'absolute',
              top: -4,
              left: -2,
              right: -2,
              height: 3,
              background: colors.googleOrange,
              borderRadius: 1,
            }} />
          </div>
          <span style={{
            fontSize: '0.65rem',
            color: colors.googleOrange,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            fontWeight: 600,
            fontFamily: fonts.mono,
          }}>
            Keyword Bin
          </span>
        </div>

        {/* Keyword chips */}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
          {keywords.map((kw) => (
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
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <span style={{ fontSize: '0.6rem', opacity: 0.6 }}>#</span>
              {kw}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
