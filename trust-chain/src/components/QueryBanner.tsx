import { useState, useEffect } from 'react';
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
  const showSponsoredAd = stepId === 'history-overture';
  const isEarlyGoogle = stepId === 'intro-search' || stepId === 'intro-results' || stepId === 'intro-ads' || stepId === 'history-overture';
  const isIntroAds = stepId === 'intro-ads';

  // Simulated click-through animation phases:
  // 0 = normal, 1 = highlight third result, 2 = white blank (page loading),
  // 3 = website renders top-down (stays until user scrolls away)
  const [clickPhase, setClickPhase] = useState(0);
  useEffect(() => {
    if (!isIntroAds) {
      setClickPhase(0);
      return;
    }
    const t1 = setTimeout(() => setClickPhase(1), 400);    // highlight link
    const t2 = setTimeout(() => setClickPhase(2), 1200);   // go white
    const t3 = setTimeout(() => setClickPhase(3), 2400);   // content paints in
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, [isIntroAds]);

  const googleHidden = isIntroAds && clickPhase >= 2;
  const showWhiteBlank = isIntroAds && clickPhase === 2;
  const showWebsite = isIntroAds && clickPhase === 3;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20, position: 'relative' }}>

      {/* Google logo — fades in from intro-search onward, collapses during click-through */}
      <img
        src={`${import.meta.env.BASE_URL}google-1999.png`}
        alt="Google"
        style={{
          width: 180,
          opacity: showGoogle && !googleHidden ? 1 : 0,
          maxHeight: googleHidden ? 0 : 200,
          overflow: 'hidden',
          transform: showGoogle ? 'translateY(0)' : 'translateY(12px)',
          transition: 'opacity 0.15s ease, max-height 0.15s ease, transform 0.4s ease',
          zIndex: 3,
          position: 'relative',
        }}
      />

      {/* Search box + iMac wrapper — iMac sits behind the search bar */}
      <div style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        opacity: googleHidden ? 0 : 1,
        maxHeight: googleHidden ? 0 : 500,
        overflow: googleHidden ? 'hidden' : 'visible',
        transition: 'opacity 0.15s ease, max-height 0.15s ease',
      }}>
        {/* iMac behind the search bar — fades in/out */}
        <img
          src={`${import.meta.env.BASE_URL}imac.png`}
          alt="iMac G3"
          className="imac-img"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -45%)',
            width: 420,
            maxWidth: '120%',
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
          textAlign: isEarlyGoogle ? 'left' : 'center',
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
        className="desktop-prop"
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
        className="desktop-prop"
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
        opacity: showResults && !showSponsoredAd && !googleHidden ? 1 : 0,
        maxHeight: googleHidden ? 0 : (showSponsoredAd ? 0 : 400),
        overflow: 'hidden',
        transform: showResults && !showSponsoredAd ? 'translateY(0)' : 'translateY(12px)',
        transition: 'opacity 0.15s, transform 0.4s, max-height 0.15s ease',
      }}>
          {[
            { title: 'Knee Pain Symptoms & Causes', url: 'www.webmd.com/knee-pain', desc: 'Common causes of knee pain when running. Learn about patellofemoral syndrome...' },
            { title: "Running Injuries Guide", url: 'running.about.com/injuries/knee', desc: 'A complete guide to common running injuries, including knee pain from downhill...' },
            { title: 'Downhill Running & Eccentric Loading', url: 'www.rice.edu/sportsmedicine/downhill', desc: 'Eccentric loading on the quad tendon during downhill running...' },
          ].map((link, i) => (
            <a key={i} href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', display: 'block' }}>
            <div style={{
              padding: '6px 0',
              opacity: showResults ? 1 : 0,
              transition: `opacity 0.4s ${i * 0.15}s, background 0.3s`,
              background: (isIntroAds && clickPhase === 1 && i === 2) ? 'rgba(26, 13, 171, 0.06)' : 'transparent',
              borderRadius: 3,
              paddingLeft: (isIntroAds && clickPhase >= 1 && i === 2) ? 4 : 0,
            }}>
              <div style={{
                color: (isIntroAds && clickPhase >= 1 && i === 2) ? '#551a8b' : '#1a0dab',
                fontSize: '0.9rem',
                textDecoration: 'underline',
                marginBottom: 1,
                fontFamily: 'Georgia, "Times New Roman", serif',
                cursor: 'pointer',
                transition: 'color 0.3s',
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

      {/* Fictional 1999 website — paints top-down after white blank */}
      {isIntroAds && (
        <div style={{
          width: '100%',
          maxWidth: 360,
          overflow: 'hidden',
          pointerEvents: 'none',
          display: (showWhiteBlank || showWebsite) ? 'block' : 'none',
        }}>
          {/* White flash — brief blank before content paints */}
          {showWhiteBlank && <div style={{ background: '#fff', height: 80 }} />}
            {/* Browser chrome — appears instantly */}
            <div style={{
              background: '#c0c0c0',
              border: '2px outset #dfdfdf',
              padding: '3px 6px',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}>
              <span style={{
                background: '#fff',
                border: '1px inset #999',
                padding: '1px 6px',
                flex: 1,
                fontSize: '0.5rem',
                fontFamily: '"Times New Roman", Georgia, serif',
                color: '#000',
              }}>
                http://www.rice.edu/sportsmedicine/eccentric-loading.html
              </span>
            </div>

            {/* Page content — each section paints in top-down with staggered delays */}
            <div style={{
              background: '#e8e8e8',
              fontFamily: '"Times New Roman", Georgia, serif',
              padding: '12px 16px',
              border: '1px solid #999',
              borderTop: 'none',
              color: '#000',
            }}>
              {/* University header — paints first */}
              <div style={{
                borderBottom: '2px solid #003366',
                marginBottom: 10,
                paddingBottom: 6,
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 0s',
              }}>
                <div style={{ fontSize: '0.7rem', color: '#003366', fontWeight: 700, letterSpacing: '0.05em' }}>
                  RICE UNIVERSITY
                </div>
                <div style={{ fontSize: '0.55rem', color: '#666' }}>
                  Department of Sports Medicine
                </div>
              </div>

              {/* Banner ad — paints second */}
              <a href="https://asdf.com" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', display: 'block', pointerEvents: 'auto' }}>
              <div style={{
                width: '100%',
                background: 'linear-gradient(180deg, #FFFF00, #FFD700)',
                border: '3px solid #FF0000',
                textAlign: 'center',
                padding: '4px 0',
                marginBottom: 10,
                position: 'relative',
                overflow: 'hidden',
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 0.3s',
                cursor: 'pointer',
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
                  fontSize: '0.8rem',
                  fontWeight: 900,
                  color: '#FF0000',
                  textShadow: '1px 1px 0px #000',
                  fontFamily: '"Comic Sans MS", "Arial Black", sans-serif',
                  letterSpacing: '0.02em',
                }}>
                  ★ WIN A FREE WALKMAN!!! ★
                </div>
                <div style={{
                  fontSize: '0.5rem',
                  color: '#000',
                  fontWeight: 700,
                  fontFamily: 'Arial, sans-serif',
                }}>
                  {'>>>'} CLICK HERE — You are the 1,000,000th visitor! {'<<<'}
                </div>
              </div>
              </a>

              {/* Article heading — paints third */}
              <h2 style={{
                fontSize: '0.85rem',
                margin: '0 0 8px',
                color: '#003366',
                fontWeight: 700,
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 0.6s',
              }}>
                Eccentric Loading and Downhill Running Injuries
              </h2>

              {/* First paragraph — paints fourth */}
              <p style={{
                fontSize: '0.65rem',
                lineHeight: 1.5,
                margin: '0 0 6px',
                color: '#222',
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 0.8s',
              }}>
                The quadriceps muscle group undergoes eccentric contraction during downhill running, generating forces up to 8x body weight at the patellar tendon insertion.
              </p>

              {/* Knee diagram — loads row-by-row much later (like images always did) */}
              <div style={{
                float: 'right',
                width: 90,
                height: 80,
                marginLeft: 8,
                marginBottom: 4,
                background: '#f0ede4',
                border: '1px solid #999',
                overflow: 'hidden',
                position: 'relative',
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 1.2s',
              }}>
                <div style={{
                  width: '100%',
                  height: '100%',
                  animation: showWebsite ? 'scanlineReveal 2s steps(12) 1.2s forwards' : 'none',
                  clipPath: 'inset(0 0 100% 0)',
                }}>
                  <svg viewBox="0 0 90 80" width="90" height="80" xmlns="http://www.w3.org/2000/svg">
                    <rect width="90" height="80" fill="#f0ede4" />
                    {/* Femur (thigh bone) — thick, coming from upper-left */}
                    <path d="M 20 2 C 22 8, 28 20, 34 30 C 36 34, 38 36, 40 38" stroke="#333" strokeWidth="2.5" fill="none" />
                    <path d="M 28 2 C 30 8, 34 20, 40 30 C 42 33, 43 35, 44 37" stroke="#333" strokeWidth="2.5" fill="none" />
                    {/* Femoral condyle (rounded end) */}
                    <ellipse cx="40" cy="40" rx="8" ry="6" stroke="#333" strokeWidth="1.5" fill="#e8ddd0" />
                    {/* Patella (kneecap) — sits in front */}
                    <ellipse cx="34" cy="39" rx="5" ry="7" stroke="#333" strokeWidth="1.8" fill="#ddd4c4" />
                    {/* Tibia (shin bone) — goes down-right */}
                    <path d="M 38 46 C 40 52, 42 58, 44 68 L 46 78" stroke="#333" strokeWidth="2.5" fill="none" />
                    <path d="M 44 46 C 46 52, 48 58, 50 68 L 52 78" stroke="#333" strokeWidth="2.5" fill="none" />
                    {/* Tibial plateau */}
                    <path d="M 36 45 C 38 44, 44 44, 46 45" stroke="#333" strokeWidth="1.2" fill="none" />
                    {/* Patellar tendon */}
                    <line x1="35" y1="46" x2="40" y2="50" stroke="#993333" strokeWidth="1.2" strokeDasharray="2,1" />
                    {/* Red arrow pointing to tendon */}
                    <line x1="58" y1="48" x2="43" y2="48" stroke="#cc0000" strokeWidth="1" markerEnd="url(#arrowhead)" />
                    <defs>
                      <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="0" refY="2" orient="auto">
                        <polygon points="0 0, 6 2, 0 4" fill="#cc0000" />
                      </marker>
                    </defs>
                    {/* Label */}
                    <text x="62" y="52" fontSize="5" fill="#cc0000" fontFamily="Arial, sans-serif" fontStyle="italic">tendon</text>
                    <text x="45" y="76" textAnchor="middle" fontSize="4.5" fill="#666" fontFamily="Times New Roman, serif">Fig. 1 — lateral view</text>
                  </svg>
                </div>
              </div>

              {/* Second paragraph — paints fifth */}
              <p style={{
                fontSize: '0.65rem',
                lineHeight: 1.5,
                margin: '0 0 8px',
                color: '#222',
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 1.0s',
              }}>
                Repetitive eccentric loading without adequate recovery is the primary mechanism for anterior knee pain in recreational runners...
              </p>

              {/* Print link — paints sixth */}
              <div style={{
                fontSize: '0.55rem',
                color: '#003366',
                textDecoration: 'underline',
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 1.2s',
              }}>
                [Print this article]
              </div>

              {/* Hit counter — paints last */}
              <div style={{
                marginTop: 10,
                borderTop: '1px solid #999',
                paddingTop: 4,
                fontSize: '0.45rem',
                color: '#888',
                textAlign: 'center',
                opacity: showWebsite ? 1 : 0,
                transition: 'opacity 0.1s ease 1.4s',
              }}>
                You are visitor #004,217 to this page | Last updated: March 12, 1999
              </div>
            </div>
          </div>
        )}

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
        @media (max-width: 768px) {
          .desktop-prop {
            display: none !important;
          }
          .imac-img {
            width: 280px !important;
          }
        }
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes scanlineReveal {
          0% { clip-path: inset(0 0 100% 0); }
          8% { clip-path: inset(0 0 92% 0); }
          16% { clip-path: inset(0 0 84% 0); }
          25% { clip-path: inset(0 0 75% 0); }
          33% { clip-path: inset(0 0 67% 0); }
          42% { clip-path: inset(0 0 58% 0); }
          50% { clip-path: inset(0 0 50% 0); }
          58% { clip-path: inset(0 0 42% 0); }
          67% { clip-path: inset(0 0 33% 0); }
          75% { clip-path: inset(0 0 25% 0); }
          83% { clip-path: inset(0 0 17% 0); }
          92% { clip-path: inset(0 0 8% 0); }
          100% { clip-path: inset(0 0 0% 0); }
        }
      `}</style>
    </div>
  );
}
