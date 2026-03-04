import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';

function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.45, maxWidth: 480 }}>
      {children}
    </div>
  );
}

export const IntroSteps = {
  steps: [
    {
      id: 'intro-title',
      content: (
        <StepText>
          <h1 style={{
            fontFamily: 'Georgia, "Times New Roman", serif',
            fontStyle: 'italic',
            fontSize: '2.2rem',
            fontWeight: 400,
            color: '#fff',
            lineHeight: 1.2,
            marginBottom: 20,
          }}>
            The Journey of Internet Ads
          </h1>
          <p style={{ color: '#bbb', fontSize: '1.15rem', lineHeight: 1.6 }}>
            By the end of this, you'll understand exactly how ads should work on chatbots, and why no one has built it yet.
          </p>
          <p style={{ color: '#666', marginTop: 20, fontSize: '0.95rem', lineHeight: 1.6 }}>
            How search ads went from useful to exploitative, why chatbots can't repeat the same pattern, and the mechanism that fixes it.
          </p>
          <p style={{ color: '#555', marginTop: 24, fontSize: '0.85rem' }}>
            ↓ Scroll to begin
          </p>
        </StepText>
      ),
    },
    {
      id: 'intro-1999',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16, lineHeight: 1.3 }}>
            It's 1999. Your knee hurts.
          </h2>
          <p style={{ color: '#bbb' }}>
            You just started running. Something's wrong when you go downhill. You try to look for it on the internet, but it's clumsy.
          </p>
        </StepText>
      ),
    },
    {
      id: 'intro-search',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            You try this new search engine.
          </h2>
          <p style={{ color: '#bbb' }}>
            Your friend told you about Google. You type:
          </p>
          <p style={{
            fontFamily: 'Georgia, "Times New Roman", serif',
            fontStyle: 'italic',
            fontSize: '1.1rem',
            color: '#bbb',
            padding: '4px 0 4px 16px',
            borderLeft: '2px solid #444',
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
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
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
            But Google is burning through VC money. It needs a business model. The web already has ads: flashing banners, pop-ups, untargeted. Google does something different: small text ads, matched to your search query. It works.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The problem isn't that ads exist. Ads are how a specialist in Portland finds the runner who needs her. The problem is that these ads have <strong style={{ color: '#fff', fontWeight: 500 }}>no idea what you really need</strong>.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
