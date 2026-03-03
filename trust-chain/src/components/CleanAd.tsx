import { fonts } from '../theme';

export function CleanAd() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20 }}>
      {/* Google logo */}
      <div style={{
        fontSize: '1.4rem',
        fontWeight: 700,
        letterSpacing: '-0.02em',
        opacity: 0,
        animation: 'fadeSlideUp 0.3s ease forwards',
      }}>
        <span style={{ color: '#4285F4' }}>G</span>
        <span style={{ color: '#EA4335' }}>o</span>
        <span style={{ color: '#FBBC05' }}>o</span>
        <span style={{ color: '#4285F4' }}>g</span>
        <span style={{ color: '#34A853' }}>l</span>
        <span style={{ color: '#EA4335' }}>e</span>
      </div>

      {/* Search box */}
      <div style={{
        fontFamily: fonts.body,
        fontSize: '0.95rem',
        color: '#222',
        padding: '8px 16px',
        background: '#fff',
        borderRadius: 24,
        border: '1px solid #dfe1e5',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        textAlign: 'center',
        opacity: 0,
        animation: 'fadeSlideUp 0.4s ease 0s forwards',
        minWidth: 280,
      }}>
        knee pain running downhill
      </div>

      {/* Sponsored label */}
      <div style={{
        width: '100%',
        maxWidth: 340,
        opacity: 0,
        animation: 'fadeSlideUp 0.4s ease 0.3s forwards',
      }}>
        <div style={{
          fontSize: '0.65rem',
          color: '#70757a',
          marginBottom: 8,
          fontFamily: fonts.body,
        }}>
          Sponsored
        </div>

        {/* Ad result — styled like real Google Ad */}
        <div style={{ padding: '4px 0' }}>
          {/* URL line with Ad badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            marginBottom: 2,
          }}>
            <span style={{
              fontSize: '0.6rem',
              fontWeight: 700,
              color: '#202124',
              background: '#fff',
              border: '1px solid #dadce0',
              borderRadius: 3,
              padding: '1px 5px',
              lineHeight: 1.4,
              fontFamily: fonts.body,
            }}>
              Ad
            </span>
            <span style={{
              fontSize: '0.72rem',
              color: '#202124',
              fontFamily: fonts.body,
            }}>
              www.portlandsportspt.com
            </span>
          </div>

          {/* Title */}
          <div style={{
            color: '#1a0dab',
            fontSize: '0.95rem',
            fontFamily: 'Georgia, "Times New Roman", serif',
            marginBottom: 3,
            cursor: 'pointer',
            lineHeight: 1.3,
          }}>
            Portland Sports PT — Runner Knee Specialist
          </div>

          {/* Description */}
          <div style={{
            fontSize: '0.78rem',
            color: '#4d5156',
            lineHeight: 1.4,
            fontFamily: fonts.body,
          }}>
            Specializing in eccentric loading injuries for runners. Free consultation. Book online today.
          </div>
        </div>
      </div>

      {/* Timeline marker */}
      <div style={{
        fontFamily: fonts.mono,
        fontSize: '0.75rem',
        color: '#555',
        padding: '4px 12px',
        border: '1px solid #333',
        borderRadius: 12,
        opacity: 0,
        animation: 'fadeSlideUp 0.3s ease 0.6s forwards',
      }}>
        2000
      </div>

      <style>{`
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
