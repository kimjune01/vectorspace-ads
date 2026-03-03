import { fonts } from '../theme';

interface Props {
  stepId: string;
}

interface JsonLine {
  lineNum: number;
  indent: number;
  key?: string;
  value?: string;
  bracket?: string;
  highlight?: boolean;
  comma?: boolean;
}

const JSON_LINES: JsonLine[] = [
  { lineNum: 1, indent: 0, bracket: '{' },
  { lineNum: 2, indent: 1, key: 'type', value: '"bid_request"', comma: true },
  { lineNum: 3, indent: 1, key: 'keywords', value: '["knee", "pain", "running"]', comma: true },
  { lineNum: 4, indent: 1, key: 'category', value: '"IAB7-1"', comma: true },
  { lineNum: 5, indent: 1, key: 'geo', value: '"US-CA"', comma: true },
  { lineNum: 6, indent: 1, key: 'embedding', value: '[0.70, 0.68, ...]', highlight: true },
  { lineNum: 7, indent: 0, bracket: '}' },
];

export function ProtocolForm({ stepId: _stepId }: Props) {
  return (
    <div style={{
      borderRadius: 8,
      overflow: 'hidden',
      maxWidth: 340,
      width: '100%',
      border: '1px solid #333',
      background: '#1e1e2e',
      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    }}>
      {/* Tab header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        background: '#181825',
        borderBottom: '1px solid #333',
        padding: 0,
      }}>
        <div style={{
          padding: '6px 14px',
          fontSize: '0.7rem',
          fontFamily: fonts.mono,
          color: '#cdd6f4',
          background: '#1e1e2e',
          borderBottom: '2px solid #89b4fa',
          borderRight: '1px solid #333',
        }}>
          bid_request.json
        </div>
        <div style={{ flex: 1 }} />
        {/* Window dots */}
        <div style={{ display: 'flex', gap: 5, padding: '0 10px' }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#f38ba8' }} />
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#f9e2af' }} />
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#a6e3a1' }} />
        </div>
      </div>

      {/* Code lines */}
      <div style={{ padding: '8px 0', fontSize: '0.78rem', fontFamily: fonts.mono, lineHeight: 1.7 }}>
        {JSON_LINES.map((line) => (
          <div key={line.lineNum} style={{
            display: 'flex',
            alignItems: 'center',
            background: line.highlight ? 'rgba(137, 180, 250, 0.08)' : 'transparent',
            borderLeft: line.highlight ? '3px solid #89b4fa' : '3px solid transparent',
            padding: '0 12px 0 0',
          }}>
            {/* Line number */}
            <span style={{
              width: 32,
              textAlign: 'right',
              color: '#585b70',
              fontSize: '0.7rem',
              paddingRight: 12,
              userSelect: 'none',
              flexShrink: 0,
            }}>
              {line.lineNum}
            </span>

            {/* Content */}
            <span>
              {'  '.repeat(line.indent)}
              {line.bracket && (
                <span style={{ color: '#cdd6f4' }}>{line.bracket}</span>
              )}
              {line.key && (
                <>
                  <span style={{ color: '#89b4fa' }}>"{line.key}"</span>
                  <span style={{ color: '#cdd6f4' }}>: </span>
                </>
              )}
              {line.value && (
                <span style={{
                  color: line.value.startsWith('"')
                    ? '#a6e3a1'
                    : line.value.startsWith('[')
                      ? '#f9e2af'
                      : '#fab387',
                }}>
                  {line.value}
                </span>
              )}
              {line.comma && <span style={{ color: '#cdd6f4' }}>,</span>}
            </span>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div style={{
        borderTop: '1px solid #333',
        padding: '6px 14px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: '#181825',
      }}>
        <span style={{
          fontSize: '0.6rem',
          fontFamily: fonts.mono,
          color: '#89b4fa',
        }}>
          Meaning arrives intact
        </span>
        <span style={{
          fontSize: '0.55rem',
          fontFamily: fonts.mono,
          color: '#585b70',
        }}>
          JSON · UTF-8
        </span>
      </div>
    </div>
  );
}
