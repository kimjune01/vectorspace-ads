import { colors, fonts } from '../theme';

interface ChainLinkData {
  id: string;
  label: string;
  problem: string;
  solution: string;
  status: 'exists' | 'missing' | 'proven' | 'building' | 'designed';
  statusLabel: string;
  link?: string;
}

interface Props {
  links: ChainLinkData[];
  revealCount: number;
  showLinks?: boolean;
}

const statusColors: Record<string, string> = {
  exists: colors.embedGreen,
  proven: colors.embedBlue,
  building: colors.embedCyan,
  missing: colors.googleOrange,
  designed: '#9C27B0',
};

export function ChainLinks({ links, revealCount, showLinks }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
      {links.map((link, i) => {
        const revealed = i < revealCount;
        const color = statusColors[link.status];
        return (
          <div key={link.id} style={{
            opacity: 0,
            animation: revealed ? `chainLinkReveal 0.4s ease ${i * 0.15}s forwards` : 'none',
          }}>
            <div
              style={{
                border: `2px solid ${revealed ? color : '#222'}`,
                borderRadius: 10,
                padding: '14px 18px',
                background: revealed ? `${color}11` : 'rgba(255,255,255,0.02)',
                transition: 'border-color 0.5s, background 0.5s',
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 6,
              }}>
                <span style={{
                  fontWeight: 600,
                  fontSize: '0.95rem',
                  color: revealed ? '#fff' : '#555',
                }}>
                  {link.label}
                </span>
                <span style={{
                  fontSize: '0.7rem',
                  fontFamily: fonts.mono,
                  color: color,
                  padding: '2px 8px',
                  background: `${color}22`,
                  borderRadius: 10,
                }}>
                  {link.statusLabel}
                </span>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>
                {link.problem}
              </div>
              <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
                <span style={{ color: '#555' }}>→ </span>
                {link.link && showLinks ? (
                  <a href={link.link} style={{ color: color, textDecoration: 'none', borderBottom: `1px solid ${color}44` }}>
                    {link.solution}
                  </a>
                ) : (
                  <span style={{ color: revealed ? '#bbb' : '#666' }}>{link.solution}</span>
                )}
              </div>
            </div>
            {/* Chain connector */}
            {i < links.length - 1 && (
              <div style={{
                textAlign: 'center',
                color: revealed && i + 1 < revealCount ? color : '#222',
                fontSize: '1rem',
                lineHeight: 1,
                transition: 'color 0.5s',
              }}>
                ┃
              </div>
            )}
          </div>
        );
      })}
      <style>{`
        @keyframes chainLinkReveal {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
