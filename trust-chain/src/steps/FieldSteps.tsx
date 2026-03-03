import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';

function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.7, maxWidth: 440 }}>
      {children}
    </div>
  );
}

function Label({ children }: { children: ReactNode }) {
  return (
    <p style={{ color: '#888', marginBottom: 12, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
      {children}
    </p>
  );
}

export const FieldSteps = {
  steps: [
    {
      id: 'gate-opens',
      content: (
        <StepText>
          <Label>What if</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            What if the bin wasn't a bin?
          </h2>
          <p style={{ color: '#bbb' }}>
            The chatbot doesn't extract keywords. It produces an embedding vector — a coordinate in meaning-space that represents <em>everything</em> you said.
          </p>
          <p style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '1.05rem',
            color: '#4CAF50',
            padding: '12px 16px',
            background: 'rgba(76, 175, 80, 0.08)',
            borderRadius: 8,
            border: '1px solid rgba(76, 175, 80, 0.2)',
            marginTop: 12,
          }}>
            "my knee hurts when I run downhill but not uphill"
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Every word stays. The coordinate lands not in a bin, but in an open field. And in the field, distance is meaning.
          </p>
        </StepText>
      ),
    },
    {
      id: 'protocol-gap',
      content: (
        <StepText>
          <Label>The problem</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            There's one problem.
          </h2>
          <p style={{ color: '#bbb' }}>
            Today's ad protocol can't carry this. OpenRTB — the standard behind every real-time bid request — has fields for keywords and category codes. No field for a vector.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The full meaning would die at the protocol boundary. The fix is one new field:{' '}
            <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#2196F3' }}>
              embedding: [0.70, 0.68, ...]
            </span>
            . That's the bottleneck.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            But assume it exists. Here's what the field looks like.
          </p>
        </StepText>
      ),
    },
    {
      id: 'sigma-intro',
      content: (
        <StepText>
          <Label>So</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Each advertiser claims a spot.
          </h2>
          <p style={{ color: '#bbb' }}>
            Dr. Chen treats runners with eccentric loading injuries — your exact problem. She sets a narrow sigma. A small, bright circle: <em>"I'm excellent at this one thing."</em>
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Metro Orthopedic treats everything knee-related. Wide sigma. Big, dim circle: <em>"I can handle most knee problems, none of them exceptionally."</em>
          </p>
        </StepText>
      ),
    },
    {
      id: 'sigma-incentives',
      content: (
        <StepText>
          <Label>The question</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            What stops Metro from claiming to be a specialist too?
          </h2>
          <p style={{ color: '#bbb' }}>
            If Metro claims a narrow sigma to look like a specialist in <em>your</em> injury, they miss every query outside that tiny circle. Their volume collapses.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            If Dr. Chen claims a wide sigma to win more auctions, she pays for impressions from people she can't help — post-surgical rehab, arthritis, knee replacements. Her budget burns on bad matches.
          </p>
          <p style={{ color: '#4CAF50', marginTop: 12, fontWeight: 600 }}>
            The sigma that makes you the most money is the one that matches your actual expertise. The honest signal <em>is</em> the greedy signal.
          </p>
        </StepText>
      ),
    },
    {
      id: 'keywords-tiny-circles',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Here's the trick: keywords already work this way.
          </h2>
          <p style={{ color: '#bbb' }}>
            A keyword is just an embedding with sigma near zero. The tightest possible circle. A single point.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The old system isn't being replaced — it's being <em>generalized</em>. Keywords still work. They're tiny circles in the same field. The difference is that now sigma can grow.
          </p>
        </StepText>
      ),
    },
    {
      id: 'hotelling',
      content: (
        <StepText>
          <Label>The risk</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            If everyone can move freely, everyone moves to the center.
          </h2>
          <p style={{ color: '#bbb' }}>
            Dr. Chen sees that most queries cluster around "knee pain." She's tempted to broaden her positioning — drift away from "downhill eccentric loading" toward the generic center where the traffic is.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            So is every other specialist. They all abandon their niches and crowd the center. The field collapses back into a bin.
          </p>
          <p style={{ color: '#FF8800', marginTop: 12 }}>
            You lose Dr. Chen. She's standing in the same crowd as Metro Orthopedic again, competing on budget instead of expertise.
          </p>
        </StepText>
      ),
    },
    {
      id: 'relocation-fee',
      content: (
        <StepText>
          <Label>The fix</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            It costs money to move.
          </h2>
          <p style={{ color: '#bbb' }}>
            A relocation fee: shift your position, pay proportional to how far you moved. Small adjustments are cheap. Abandoning your niche to chase the center is expensive.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Dr. Chen runs the numbers. Staying where she is — "downhill running biomechanics" — costs nothing. Drifting to "general knee pain" would cost more than the extra impressions are worth. She stays put.
          </p>
          <p style={{ color: '#4CAF50', marginTop: 12, fontSize: '1rem' }}>
            Simulations confirm it: with relocation fees, specialist surplus flips from negative to positive. Dr. Chen earns more per impression than Metro.
          </p>
        </StepText>
      ),
    },
    {
      id: 'everyone-wins',
      content: (
        <StepText>
          <Label>The result</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Greed aligns.
          </h2>
          <p style={{ color: '#bbb' }}>
            Dr. Chen stays in her niche because moving is expensive and her niche is where she converts best. Metro stays wide because their generalist model needs volume. And you — you get Dr. Chen, three steps away, for $2 instead of Metro for $8.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Nobody is being altruistic. The mechanism makes the selfish move the fair move.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The economics look like 2003 again. Before the middlemen. Before the consolidation. Before the degradation — not by going backwards, but by building forward with a better signal.
          </p>
        </StepText>
      ),
    },
    {
      id: 'exchange-trust',
      content: (
        <StepText>
          <Label>One problem remains</Label>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            What keeps the exchange honest?
          </h2>
          <p style={{ color: '#bbb' }}>
            The mechanism is sound. The incentives align. But the exchange runs the auction. What stops it from peeking at the bids and rigging the outcome? What stops it from selling your embedding to a data broker?
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Twenty years of ad tech taught users one lesson: whoever runs the auction eventually exploits it. Google proved that. Promising "we won't" isn't enough.
          </p>
        </StepText>
      ),
    },
    {
      id: 'enclave-proof',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#00BCD4', marginBottom: 16 }}>
            The auction runs inside sealed hardware.
          </h2>
          <p style={{ color: '#bbb' }}>
            A TEE — trusted execution environment — is a chip-level enclave. The auction code runs inside it. The exchange can trigger the computation, but it can't read the inputs, alter the logic, or change the output.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The result comes out with a cryptographic signature — proof that the code ran exactly as published, unmodified. Not a promise. A receipt you can verify.
          </p>
          <p style={{ color: '#00BCD4', marginTop: 12, fontWeight: 600 }}>
            The user doesn't trust the company. They verify the math.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
