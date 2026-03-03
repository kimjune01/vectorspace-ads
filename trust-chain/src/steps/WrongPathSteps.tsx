import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';

function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.7, maxWidth: 440 }}>
      {children}
    </div>
  );
}

export const WrongPathSteps = {
  steps: [
    {
      id: 'wrong-path-bins',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            You're in the bin.
          </h2>
          <p style={{ color: '#bbb' }}>
            Your query — all its nuance — got compressed into{' '}
            <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#FF8800' }}>knee pain running</span>{' '}
            and thrown into a pen. You're crowded in with everyone who ever typed anything vaguely knee-related.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Dr. Chen specializes in eccentric loading for downhill runners. Metro Orthopedic does general consultations. In the bin, they're identical. The one with more budget wins.
          </p>
        </StepText>
      ),
    },
    {
      id: 'wrong-path-auction',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Same bin, same outcome.
          </h2>
          <p style={{ color: '#bbb' }}>
            Every advertiser who bought "knee pain" competes for you. Dr. Chen — the specialist who actually treats downhill runners — is drowned out by bigger budgets.
          </p>
        </StepText>
      ),
    },
    {
      id: 'wrong-path-receipt',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            The old economics repeat.
          </h2>
          <p style={{ color: '#666', marginTop: 12, fontStyle: 'italic' }}>
            This is where the story usually ends.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
