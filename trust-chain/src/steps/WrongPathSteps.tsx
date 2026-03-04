import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';


function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.45, maxWidth: 480 }}>
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
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            You're thrown into the bin.
          </h2>
          <p style={{ color: '#bbb' }}>
            Your intent, all its nuance, got compressed into{' '}
            <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#b89a6a' }}>knee pain running</span>{' '}
            and thrown into a pen. You're crowded in with everyone whose query matched the same three keywords — no matter how different their actual problem is.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Different people, different problems. The ad system sees one bin. More people in the bin means more competition, higher bids, more revenue.
          </p>
        </StepText>
      ),
    },
    {
      id: 'wrong-path-auction',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            Same bin, same outcome.
          </h2>
          <p style={{ color: '#bbb' }}>
            Dr. Chen could pay $10 for a downhill runner. That patient converts. But the bin doesn't send her downhill runners. It sends everyone who typed "knee." She converts one in twenty. At that rate she can only afford $2.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Metro converts one in five. "General consultation" matches more of the bin. Low resolution rewards the generalist and prices out the specialist.
          </p>
        </StepText>
      ),
    },
    {
      id: 'wrong-path-receipt',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            The old economics repeat.
          </h2>
          <p style={{ color: '#bbb' }}>
            Google keeps <a href="https://digiday.com/media/the-rundown-u-s-v-google-ad-tech-antitrust-trial-by-numbers-so-far/" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>37 cents of every ad dollar</a> across three layers it controls. Internal documents show CPCs doubled between 2013 and 2020 through deliberate <a href="https://searchengineland.com/doj-google-search-ad-price-manipulation-440207" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>"tunings"</a>, and advertisers can't even see <a href="https://www.seerinteractive.com/insights/google-ads-removes-search-terms-for-28-percent-of-paid-search-budgets" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>28% of the queries</a> their money is spent on.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            But look at where the value was lost. It wasn't in the auction. It was in the signal when your full question got crushed into three keywords.
          </p>
        </StepText>
      ),
    },
    {
      id: 'resolution-keywords',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            This is what keywords see.
          </h2>
          <p style={{ color: '#bbb' }}>
            Three words. Three blocks of color. Everything else (the downhill, the asymmetry, the biomechanics) thrown away. The ad system is matching you at the lowest resolution possible.
          </p>
        </StepText>
      ),
    },
    {
      id: 'resolution-embeddings',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            This is what an embedding sees.
          </h2>
          <p style={{ color: '#bbb' }}>
            The same query. Hundreds of semantic dimensions (anatomy, biomechanics, activity, condition) all encoded as a single coordinate. Not every word survives, but the meaning does. Keywords couldn't carry it.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
