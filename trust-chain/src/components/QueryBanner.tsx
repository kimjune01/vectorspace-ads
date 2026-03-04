import { fonts } from '../theme';

interface Props {
  stepId: string;
}

export function QueryBanner({ stepId }: Props) {
  const isTitle = stepId === 'intro-title';
  const isBareText = stepId === 'intro-1999' || isTitle;
  const showBox = !isBareText;
  const showResults = stepId === 'intro-results' || stepId === 'intro-ads';
  const showBanner = stepId === 'intro-ads';
  const showSponsoredAd = stepId === 'history-overture';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20 }}>
      {/* Google logo text */}
      {showBox && (
        <div style={{
          fontSize: '1.6rem',
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
      )}

      {/* Search box */}
      <div style={{
        fontFamily: fonts.body,
        fontSize: '0.95rem',
        color: showBox ? '#222' : isTitle ? '#333' : '#fff',
        padding: showBox ? '8px 16px' : '8px 0',
        background: showBox ? '#fff' : 'transparent',
        borderRadius: showBox ? 24 : 0,
        border: showBox ? '1px solid #dfe1e5' : '1px solid transparent',
        boxShadow: showBox ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
        textAlign: 'center',
        transition: 'padding 0.5s, background 0.5s, border-color 0.5s',
        animation: 'fadeSlideUp 0.4s ease forwards',
        minWidth: showBox ? 280 : undefined,
      }}>
        knee pain running downhill
      </div>

      {/* Ten blue links */}
      <div style={{
        width: '100%',
        maxWidth: 340,
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        opacity: showResults && !showSponsoredAd ? 1 : 0,
        maxHeight: showSponsoredAd ? 0 : 400,
        overflow: 'hidden',
        transform: showResults && !showSponsoredAd ? 'translateY(0)' : 'translateY(12px)',
        transition: 'opacity 0.4s, transform 0.4s, max-height 0.5s ease',
      }}>
        {[
          { title: 'Knee Pain Symptoms & Causes', url: 'www.webmd.com/knee-pain', desc: 'Common causes of knee pain when running. Learn about patellofemoral syndrome...' },
          { title: "Runner's Knee: Treatment & Prevention", url: 'www.healthline.com/runners-knee', desc: 'Patellofemoral pain syndrome occurs when cartilage under the kneecap...' },
          { title: 'Downhill Running Injuries - Sports Med', url: 'www.sportsmedicinetoday.com/downhill', desc: 'Eccentric loading on the quad tendon during downhill running...' },
        ].map((link, i) => (
          <div key={i} style={{
            padding: '6px 0',
            opacity: showResults ? 1 : 0,
            transition: `opacity 0.4s ${i * 0.15}s`,
          }}>
            <div style={{
              color: '#1a0dab',
              fontSize: '0.9rem',
              textDecoration: 'underline',
              marginBottom: 1,
              fontFamily: 'Georgia, "Times New Roman", serif',
              cursor: 'pointer',
            }}>
              {link.title}
            </div>
            <div style={{
              color: '#006621',
              fontSize: '0.7rem',
              marginBottom: 2,
              fontFamily: fonts.body,
            }}>
              {link.url}
            </div>
            <div style={{ color: '#545454', fontSize: '0.72rem', fontFamily: fonts.body, lineHeight: 1.4 }}>
              {link.desc}
            </div>
          </div>
        ))}
      </div>

      {/* Garish 468x60 banner ad — late-90s style */}
      <div style={{
        width: '100%',
        maxWidth: 320,
        maxHeight: showSponsoredAd ? 0 : 56,
        background: 'linear-gradient(180deg, #FFFF00, #FFD700)',
        border: showSponsoredAd ? 'none' : '3px solid #FF0000',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        opacity: showBanner && !showSponsoredAd ? 1 : 0,
        transform: showBanner && !showSponsoredAd ? 'translateY(0) scale(1)' : 'translateY(16px) scale(0.95)',
        transition: 'opacity 0.4s, transform 0.4s, max-height 0.5s ease, border 0.3s',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 3,
          background: 'repeating-linear-gradient(90deg, #FF0000, #FF0000 4px, #FFFF00 4px, #FFFF00 8px)',
        }} />
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 3,
          background: 'repeating-linear-gradient(90deg, #FF0000, #FF0000 4px, #FFFF00 4px, #FFFF00 8px)',
        }} />
        <div style={{
          fontSize: '0.95rem',
          fontWeight: 900,
          color: '#FF0000',
          textShadow: '1px 1px 0px #000',
          fontFamily: '"Comic Sans MS", "Arial Black", sans-serif',
          letterSpacing: '0.02em',
        }}>
          ★ WIN A FREE iPOD!!! ★
        </div>
        <div style={{
          fontSize: '0.6rem',
          color: '#000',
          fontWeight: 700,
          fontFamily: 'Arial, sans-serif',
        }}>
          {'>>>'} CLICK HERE — You are the 1,000,000th visitor! {'<<<'}
        </div>
      </div>

      {/* Sponsored ad — Overture/early Google ad (history-overture) */}
      <div style={{
        width: '100%',
        maxWidth: 340,
        opacity: showSponsoredAd ? 1 : 0,
        transform: showSponsoredAd ? 'translateY(0)' : 'translateY(12px)',
        transition: 'opacity 0.5s 0.2s, transform 0.5s 0.2s',
        display: showSponsoredAd ? 'block' : 'none',
      }}>
        <div style={{
          fontSize: '0.65rem',
          color: '#70757a',
          marginBottom: 8,
          fontFamily: fonts.body,
        }}>
          Sponsored
        </div>
        <div style={{ padding: '4px 0' }}>
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
      {showSponsoredAd && (
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
      )}

      <style>{`
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
