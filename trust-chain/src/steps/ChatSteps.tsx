import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';

function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.7, maxWidth: 440 }}>
      {children}
    </div>
  );
}

export const ChatSteps = {
  steps: [
    {
      id: 'dot-intro',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            So what does the user actually see?
          </h2>
          <p style={{ color: '#bbb' }}>
            All this mechanism — embeddings, sigma, relocation fees, sealed auctions — is invisible. The user never sees any of it. They see one thing: a dot.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            You open a chat. You start typing. A small dot appears in the margin. Barely visible. Dim gray. It means: <em>there's expertise somewhere in this space, but it's far away. Your query is vague.</em>
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            The chatbot responds normally. No ad. No interruption.
          </p>
        </StepText>
      ),
    },
    {
      id: 'dot-brightens',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            The dot warms.
          </h2>
          <p style={{ color: '#bbb' }}>
            You add more detail. The dot brightens slightly. Warmer. Closer. Your intent is getting more specific, and somewhere in the embedding field, an advertiser's circle overlaps with where you're standing.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The dot glows. Bright. Close. It's not selling anything. It's not a link. It's an ambient signal: <em style={{ color: '#4CAF50' }}>someone nearby can help with exactly this.</em>
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            You tap it.
          </p>
        </StepText>
      ),
    },
    {
      id: 'dot-auction',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Now — and only now — the auction fires.
          </h2>
          <p style={{ color: '#bbb' }}>
            The chatbot sends your embedding into the sealed auction. The TEE runs the match. Dr. Chen wins — three steps away, $2 bid. Metro Orthopedic, far across the field, never had a chance.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            You tapped because you were curious. The match is real because the math is real. The chatbot can prove — cryptographically — that it didn't choose Dr. Chen. The sealed auction did.
          </p>
        </StepText>
      ),
    },
    {
      id: 'dot-philosophy',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Trust isn't promised. It's earned.
          </h2>
          <p style={{ color: '#bbb' }}>
            One dot. One tap. One verifiable match. That's the entire ad surface.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            No banner. No "sponsored" label buried in the results. No ad that pretends to be an answer. A proximity signal that earns trust by being right, consistently, over weeks and months — until tapping the dot is something you <em>want</em> to do, not something you endure.
          </p>
          <p style={{ color: '#00BCD4', marginTop: 12 }}>
            The alternative is much worse. Without an honest ad layer, chatbots either stay unprofitable — or they quietly sell your intent to the same keyword machine. The dot isn't optional. It's the only way this doesn't end like last time.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
