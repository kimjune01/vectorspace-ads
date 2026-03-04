import { colors, fonts } from '../theme';

export function CTA() {
  return (
    <div style={{
      maxWidth: 700,
      margin: '0 auto',
      padding: '120px 24px 160px',
      textAlign: 'center',
    }}>
      <h2 style={{
        fontSize: '2.5rem',
        fontWeight: 700,
        color: colors.textBright,
        marginBottom: 16,
        lineHeight: 1.2,
      }}>
        The pieces exist. Nobody has assembled them.
      </h2>
      <p style={{
        color: colors.textDim,
        fontSize: '1.1rem',
        lineHeight: 1.7,
        marginBottom: 40,
      }}>
        I proved that the pieces fit together.
      </p>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
        alignItems: 'center',
        marginBottom: 60,
      }}>
        <CTALink
          href="/vectorspace-ads/"
          label="Try the auction"
          desc="Drag advertisers, drop a query, watch proximity win"
        />
        <CTALink
          href="https://github.com/kimjune01/openauction"
          label="Run the simulation"
          desc="15 agents, 50 trials, measure the surplus shift yourself"
        />
        <CTALink
          href="/vector-space"
          label="Read the math"
          desc="From power diagrams to relocation fees, the full mechanism"
        />
      </div>

      <div style={{
        borderTop: '1px solid #333',
        paddingTop: 40,
      }}>
        <p style={{
          color: colors.textDim,
          fontSize: '1.1rem',
          lineHeight: 1.7,
        }}>
          If you want to strengthen the case —{' '}
          <a
            href="/vector-space"
            style={{
              color: colors.embedBlue,
              textDecoration: 'none',
              borderBottom: `1px solid ${colors.embedBlue}`,
            }}
          >
            let's research together
          </a>
          .<br />
          If you want to make money —{' '}
          <a
            href="/about"
            style={{
              color: colors.embedBlue,
              textDecoration: 'none',
              borderBottom: `1px solid ${colors.embedBlue}`,
            }}
          >
            let's talk
          </a>
          .
        </p>
      </div>

      <div style={{
        color: '#444',
        fontSize: '0.8rem',
        marginTop: 80,
        fontFamily: fonts.mono,
        lineHeight: 1.8,
      }}>
        <p>
          Built by{' '}
          <a href="/" style={{ color: '#666', textDecoration: 'none', borderBottom: '1px solid #444' }}>
            June Kim
          </a>
        </p>
        <p style={{ marginTop: 8 }}>
          Written with{' '}
          <a href="https://claude.ai" style={{ color: '#555', textDecoration: 'none', borderBottom: '1px solid #444' }}>
            Claude Opus 4.6
          </a>
        </p>
        <p style={{ marginTop: 8 }}>
          Scrollytelling inspired by{' '}
          <a href="https://pudding.cool" style={{ color: '#555', textDecoration: 'none', borderBottom: '1px solid #444' }}>
            The Pudding
          </a>
        </p>
      </div>
    </div>
  );
}

function CTALink({ href, label, desc }: { href: string; label: string; desc: string }) {
  return (
    <a
      href={href}
      style={{
        display: 'block',
        padding: '16px 32px',
        border: `1px solid ${colors.embedBlue}`,
        borderRadius: 8,
        color: colors.textBright,
        textDecoration: 'none',
        width: '100%',
        maxWidth: 400,
        transition: 'background 0.2s, border-color 0.2s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'rgba(33, 150, 243, 0.1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
      }}
    >
      <div style={{ fontWeight: 600, fontSize: '1.1rem' }}>{label}</div>
      <div style={{ color: colors.textDim, fontSize: '0.9rem', marginTop: 4 }}>{desc}</div>
    </a>
  );
}
