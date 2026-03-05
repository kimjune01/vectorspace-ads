import { useEffect, useRef } from 'react';
import type { ReactNode, CSSProperties } from 'react';

export interface StepData {
  id: string;
  content: ReactNode;
}

interface Props {
  steps: StepData[];
  onStepChange: (stepId: string) => void;
  graphic: ReactNode;
}

const containerStyle: CSSProperties = {
  display: 'flex',
  position: 'relative',
  maxWidth: 1200,
  margin: '0 auto',
  padding: '0 20px',
};

const stickyStyle: CSSProperties = {
  position: 'sticky',
  top: 0,
  height: '100vh',
  width: '55%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
};

const stepsContainerStyle: CSSProperties = {
  width: '40%',
  marginLeft: '5%',
};

const stepStyle: CSSProperties = {
  minHeight: '80vh',
  display: 'flex',
  alignItems: 'center',
  padding: '40px 0',
};

// Mobile styles applied via media query in a style tag
const mobileCSS = `
@media (max-width: 768px) {
  .scrolly-container {
    flex-direction: column !important;
  }
  .sticky-graphic {
    width: 100% !important;
    height: 40vh !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 10;
    background: #0a0a1a !important;
    border-bottom: 1px solid #1a1a2a !important;
    overflow: hidden !important;
  }
  .sticky-graphic > * {
    transform: scale(0.75);
    transform-origin: center center;
  }
  .text-steps {
    width: 100% !important;
    margin-left: 0 !important;
    padding: 0 16px !important;
  }
  .step-item {
    min-height: 60vh !important;
  }
  .surplus-bar-wrapper,
  .surplus-bar-wrapper > * {
    display: none !important;
  }
}
`;

export function ScrollySection({ steps, onStepChange, graphic }: Props) {
  const stepRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const isMobile = window.matchMedia('(max-width: 768px)').matches;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const index = stepRefs.current.indexOf(entry.target as HTMLDivElement);
            if (index !== -1) {
              onStepChange(steps[index].id);
            }
          }
        }
      },
      {
        rootMargin: isMobile ? '-30% 0px -30% 0px' : '-40% 0px -40% 0px',
        threshold: 0,
      }
    );

    const currentRefs = stepRefs.current;
    currentRefs.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => {
      currentRefs.forEach((ref) => {
        if (ref) observer.unobserve(ref);
      });
    };
  }, [steps.length, onStepChange]);

  return (
    <>
      <style>{mobileCSS}</style>
      <main id="scrolly-content" className="scrolly-container" style={containerStyle} aria-label="The Journey of Internet Ads">
        <div className="sticky-graphic" style={stickyStyle} aria-hidden="true">
          {graphic}
        </div>
        <div className="text-steps" style={stepsContainerStyle}>
          {steps.map((step, i) => (
            <div
              key={step.id}
              ref={(el) => { stepRefs.current[i] = el; }}
              className="step-item"
              style={stepStyle}
            >
              <div style={{ width: '100%' }}>{step.content}</div>
            </div>
          ))}
        </div>
      </main>
    </>
  );
}
