import { useState, useEffect } from 'react';
import { ScrollySection } from './ScrollySection';
import { Pipeline } from './Pipeline';
import { SurplusBar } from './components/SurplusBar';
import { IntroSteps } from './steps/IntroSteps';
import { HistorySteps } from './steps/HistorySteps';
import { WrongPathSteps } from './steps/WrongPathSteps';
import { FieldSteps } from './steps/FieldSteps';
import { ChatSteps } from './steps/ChatSteps';
import { ChainSteps } from './steps/ChainSteps';
import { CTA } from './steps/CTA';
import { fonts } from './theme';

const allSteps = [
  ...IntroSteps.steps,
  ...HistorySteps.steps,
  ...WrongPathSteps.steps,
  ...FieldSteps.steps,
  ...ChatSteps.steps,
  ...ChainSteps.steps,
];

const stepIds = allSteps.map(s => s.id);

export default function App() {
  const [activeStepId, setActiveStepId] = useState('intro-1999');
  const [showScrollHint, setShowScrollHint] = useState(true);

  // Hide scroll hint after user scrolls past first viewport
  useEffect(() => {
    function onScroll() {
      if (window.scrollY > 100) {
        setShowScrollHint(false);
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const stepIndex = stepIds.indexOf(activeStepId);
  const stepProgress = stepIndex >= 0 ? (stepIndex + 1) / stepIds.length : 0;

  return (
    <div style={{
      fontFamily: fonts.body,
      background: '#0a0a1a',
      color: '#e0e0e0',
      minHeight: '100vh',
    }}>
      <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
        rel="stylesheet"
      />

      {/* Progress bar */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: 3,
        zIndex: 100,
        background: 'rgba(255,255,255,0.05)',
      }}>
        <div style={{
          height: '100%',
          width: `${stepProgress * 100}%`,
          background: 'linear-gradient(90deg, #2196F3, #4CAF50)',
          transition: 'width 0.4s ease',
          borderRadius: '0 2px 2px 0',
        }} />
      </div>

      <ScrollySection
        steps={allSteps}
        onStepChange={setActiveStepId}
        graphic={
          <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'stretch', position: 'relative' }}>
            <SurplusBar stepId={activeStepId} />
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Pipeline stepId={activeStepId} />
            </div>
          </div>
        }
      />
      <CTA />

      {/* Scroll hint — fades out after first scroll */}
      <div style={{
        position: 'fixed',
        bottom: 32,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 6,
        opacity: showScrollHint ? 1 : 0,
        transition: 'opacity 0.6s',
        pointerEvents: 'none',
      }}>
        <span style={{
          fontSize: '0.7rem',
          fontFamily: fonts.mono,
          color: '#555',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
        }}>
          Scroll
        </span>
        <div className="scroll-arrow" style={{
          color: '#555',
          fontSize: '1.2rem',
          lineHeight: 1,
        }}>
          ↓
        </div>
      </div>

      <style>{`
        .scroll-arrow {
          animation: scrollBounce 2s ease-in-out infinite;
        }
        @keyframes scrollBounce {
          0%, 100% { transform: translateY(0); opacity: 0.5; }
          50% { transform: translateY(6px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
