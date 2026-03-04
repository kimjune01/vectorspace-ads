import { useRef } from 'react';
import { fonts } from '../theme';
import { SURPLUS_BAR_CONFIG } from '../data';

interface Props {
  stepId: string;
}

export function SurplusBar({ stepId }: Props) {
  const config = SURPLUS_BAR_CONFIG[stepId];
  const lastConfig = useRef(config);
  const active = !!config;

  // Remember the last non-null config so we can show it desaturated
  if (config) {
    lastConfig.current = config;
  }

  const displayConfig = config ?? lastConfig.current;

  return (
    <div style={{
      width: 140,
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 6,
      padding: '24px 4px',
      opacity: active ? 1 : 0.25,
      filter: active ? 'none' : 'saturate(0)',
      transition: 'opacity 0.5s, filter 0.5s',
      pointerEvents: 'none',
    }}>
      {/* Label */}
      <div style={{
        fontSize: '0.6rem',
        color: '#666',
        fontFamily: fonts.mono,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        textAlign: 'center',
        lineHeight: 1.3,
        flexShrink: 0,
        marginBottom: 2,
      }}>
        Value
      </div>

      {/* Vertical bar with labels to the right */}
      <div style={{
        flex: 1,
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: 0,
        position: 'relative',
      }}>
        {/* The bar itself */}
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: 28,
          borderRadius: 6,
          overflow: 'hidden',
          background: '#1a1a2a',
          display: 'flex',
          flexDirection: 'column',
        }}>
          {(displayConfig ?? []).map((segment, i) => (
            <div
              key={i}
              style={{
                height: `${segment.width * 100}%`,
                background: segment.color,
                transition: 'height 0.8s ease, background 0.5s',
              }}
            />
          ))}
        </div>

        {/* Labels positioned alongside */}
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
        }}>
          {(displayConfig ?? []).map((segment, i) => (
            <div
              key={`label-${i}`}
              style={{
                height: `${segment.width * 100}%`,
                display: 'flex',
                alignItems: 'center',
                paddingLeft: 34,
                minHeight: 0,
              }}
            >
              {segment.width >= 0.12 && (
                <span style={{
                  fontSize: '0.7rem',
                  fontFamily: fonts.mono,
                  color: segment.color,
                  whiteSpace: 'nowrap',
                  lineHeight: 1,
                  background: 'rgba(10, 10, 26, 0.85)',
                  padding: '1px 3px',
                  borderRadius: 2,
                }}>
                  {segment.label}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
