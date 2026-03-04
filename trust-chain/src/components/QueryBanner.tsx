import { fonts } from '../theme';

interface Props {
  stepId: string;
}

export function QueryBanner({ stepId }: Props) {
  const isTitle = stepId === 'intro-title';
  const is1999 = stepId === 'intro-1999';
  const isBare = isTitle || is1999;
  const showGoogle = !isBare;
  const showBox = !isBare;
  const showImac = stepId === 'intro-1999';
  const showResults = stepId === 'intro-results' || stepId === 'intro-ads';
  const showBanner = stepId === 'intro-ads';
  const showSponsoredAd = stepId === 'history-overture';
  const isEarlyGoogle = stepId === 'intro-search' || stepId === 'intro-results' || stepId === 'intro-ads' || stepId === 'history-overture';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20, position: 'relative' }}>

      {/* Google logo — fades in from intro-search onward */}
      <img
        src={`${import.meta.env.BASE_URL}google-1999.png`}
        alt="Google"
        style={{
          width: 180,
          opacity: showGoogle ? 1 : 0,
          transform: showGoogle ? 'translateY(0)' : 'translateY(12px)',
          transition: 'opacity 0.4s ease, transform 0.4s ease',
          zIndex: 3,
          position: 'relative',
        }}
      />

      {/* Search box + iMac wrapper — iMac sits behind the search bar */}
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        {/* iMac behind the search bar — fades in/out */}
        <img
          src={`${import.meta.env.BASE_URL}imac.png`}
          alt="iMac G3"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -45%)',
            width: 420,
            zIndex: 1,
            pointerEvents: 'none',
            opacity: showImac ? 1 : 0,
            transition: 'opacity 0.6s ease',
          }}
        />

        {/* Yahoo + Ask logos — inside the iMac, above the search text */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
          zIndex: 2,
          position: 'relative',
          opacity: showImac ? 1 : 0,
          height: showImac ? undefined : 0,
          overflow: 'hidden',
          transition: 'opacity 0.5s ease',
          marginBottom: showImac ? 6 : 0,
        }}>
          <img src={`${import.meta.env.BASE_URL}yahoo.png`} alt="Yahoo" style={{ height: 24, imageRendering: 'pixelated' as any }} />
          <img src={`${import.meta.env.BASE_URL}ask.png`} alt="Ask" style={{ height: 24, imageRendering: 'pixelated' as any }} />
          <img src={`${import.meta.env.BASE_URL}aol.png`} alt="AOL" style={{ height: 24, imageRendering: 'pixelated' as any }} />
          <img src={`${import.meta.env.BASE_URL}netscape.webp`} alt="Netscape" style={{ height: 24, imageRendering: 'pixelated' as any }} />
        </div>

        {/* Search box — ALWAYS visible, the constant anchor */}
        <div style={{
          fontFamily: (isTitle || is1999) ? '"Press Start 2P", "Courier New", monospace' : isEarlyGoogle ? '"Times New Roman", Georgia, serif' : fonts.body,
          fontSize: (isTitle || is1999) ? '0.6rem' : showBox ? '0.95rem' : '1.1rem',
          color: showBox ? '#222' : isTitle ? '#555' : '#fff',
          padding: showBox ? '8px 16px' : '0',
          background: showBox ? '#fff' : 'transparent',
          borderRadius: showBox ? (isEarlyGoogle ? 2 : 24) : 0,
          border: showBox ? (isEarlyGoogle ? '2px solid #999' : '1px solid #dfe1e5') : 'none',
          boxShadow: showBox ? (isEarlyGoogle ? 'inset 1px 1px 3px rgba(0,0,0,0.2)' : '0 1px 3px rgba(0,0,0,0.1)') : 'none',
          textAlign: 'center',
          minWidth: showBox ? 280 : undefined,
          zIndex: 2,
          position: 'relative',
          transition: 'all 0.5s ease',
          textShadow: (isTitle || is1999) ? '0 0 6px rgba(255, 255, 255, 0.3)' : 'none',
          imageRendering: (isTitle || is1999) ? 'pixelated' as any : undefined,
        }}>
          knee pain running downhill
        </div>

        {/* "I'm Feeling Lucky" button — early Google era */}
        <div style={{
          display: 'flex',
          gap: 8,
          zIndex: 2,
          position: 'relative',
          opacity: isEarlyGoogle ? 1 : 0,
          height: isEarlyGoogle ? undefined : 0,
          overflow: 'hidden',
          transition: 'opacity 0.4s ease',
          marginTop: isEarlyGoogle ? 8 : 0,
        }}>
          {['Google Search', "I'm Feeling Lucky"].map((label) => (
            <button
              key={label}
              className="retro-btn"
              onClick={(e) => {
                const btn = e.currentTarget;
                btn.style.borderStyle = 'inset';
                btn.style.paddingTop = '5px';
                btn.style.paddingLeft = '13px';
                btn.style.paddingBottom = '3px';
                btn.style.paddingRight = '11px';
                setTimeout(() => {
                  btn.style.borderStyle = 'outset';
                  btn.style.paddingTop = '4px';
                  btn.style.paddingLeft = '12px';
                  btn.style.paddingBottom = '4px';
                  btn.style.paddingRight = '12px';
                  window.open('https://letmegooglethat.com/?q=knee+pain+running+downhill', '_blank');
                }, 150);
              }}
              style={{
                fontFamily: '"Times New Roman", Georgia, serif',
                fontSize: '0.75rem',
                padding: '4px 12px',
                background: '#e0e0e0',
                border: '2px outset #ccc',
                borderRadius: 0,
                cursor: 'pointer',
                color: '#222',
              }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* 1999 props — off to the side, don't affect layout */}
      <img
        src={`${import.meta.env.BASE_URL}swingline.png`}
        alt="Red Swingline stapler"
        style={{
          position: 'absolute',
          right: -20,
          bottom: 60,
          width: 134,
          zIndex: 3,
          pointerEvents: 'none',
          opacity: showImac ? 1 : 0,
          transform: showImac ? 'rotate(-8deg)' : 'rotate(-8deg) translateY(12px)',
          transition: 'opacity 0.6s ease 0.3s, transform 0.6s ease 0.3s',
          filter: 'drop-shadow(0 4px 12px rgba(255,0,0,0.3))',
        }}
      />
      <img
        src={`${import.meta.env.BASE_URL}nbgray.avif`}
        alt=""
        style={{
          position: 'absolute',
          right: 50,
          bottom: 30,
          width: 240,
          zIndex: 3,
          pointerEvents: 'none',
          opacity: showImac ? 1 : 0,
          transform: showImac ? 'rotate(5deg)' : 'rotate(5deg) translateY(12px)',
          transition: 'opacity 0.6s ease 0.4s, transform 0.6s ease 0.4s',
          filter: 'drop-shadow(0 4px 12px rgba(100,100,100,0.4))',
        }}
      />

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
          <a key={i} href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', display: 'block' }}>
          <div style={{
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
          </a>
        ))}
      </div>

      {/* Garish 468x60 banner ad — late-90s style */}
      <a href="https://asdf.com" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', width: '100%', maxWidth: 320, display: 'block' }}>
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
          ★ WIN A FREE WALKMAN!!! ★
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
      </a>

      {/* Sponsored ad — Overture/early Google ad (history-overture) */}
      <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', display: showSponsoredAd ? 'block' : 'none', width: '100%', maxWidth: 340 }}>
      <div style={{
        width: '100%',
        maxWidth: 340,
        opacity: showSponsoredAd ? 1 : 0,
        transform: showSponsoredAd ? 'translateY(0)' : 'translateY(12px)',
        transition: 'opacity 0.5s 0.2s, transform 0.5s 0.2s',
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
      </a>

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
