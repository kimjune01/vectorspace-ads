import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';

function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.7, maxWidth: 440 }}>
      {children}
    </div>
  );
}

export const IntroSteps = {
  steps: [
    {
      id: 'intro-1999',
      content: (
        <StepText>
          <h2 style={{ fontSize: '2rem', fontWeight: 700, color: '#fff', marginBottom: 16, lineHeight: 1.3 }}>
            It's 1999. Your knee hurts.
          </h2>
          <p style={{ color: '#bbb' }}>
            You just started running. Something's wrong when you go downhill. You don't know what to call it.
          </p>
        </StepText>
      ),
    },
    {
      id: 'intro-search',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            You try this new search engine.
          </h2>
          <p style={{ color: '#bbb' }}>
            Your friend told you about Google. You type:
          </p>
          <p style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '1.1rem',
            color: '#FFD700',
            padding: '12px 16px',
            background: 'rgba(255, 215, 0, 0.08)',
            borderRadius: 8,
            border: '1px solid rgba(255, 215, 0, 0.2)',
            marginTop: 12,
          }}>
            knee pain running downhill
          </p>
        </StepText>
      ),
    },
    {
      id: 'intro-results',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            It works.
          </h2>
          <p style={{ color: '#bbb' }}>
            Ten blue links. No ads. The third link is a sports medicine article that describes your exact problem. You print it out and bring it to your doctor.
          </p>
        </StepText>
      ),
    },
    {
      id: 'intro-ads',
      content: (
        <StepText>
          <p style={{ color: '#888' }}>
            But Google is burning through VC money. It needs a business model. And the web already has ads — banner ads, flashing rectangles, untargeted. The same ad whether you're reading about knees or cars.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The problem isn't that ads exist. Ads are how the web stays free. Ads are how a specialist in Portland finds the runner who needs her. The problem is that these ads have <em style={{ color: '#fff' }}>no idea who you are</em>.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
