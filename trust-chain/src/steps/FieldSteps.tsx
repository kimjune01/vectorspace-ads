import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';


function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.7, maxWidth: 440 }}>
      {children}
    </div>
  );
}


export const FieldSteps = {
  steps: [
    {
      id: 'gate-opens',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            What if the bin wasn't a bin?
          </h2>
          <p style={{ color: '#bbb' }}>
            The chatbot doesn't extract keywords. It produces an embedding vector, a coordinate in meaning-space that represents <em>everything</em> you said.
          </p>
          <p style={{
            fontFamily: 'Georgia, "Times New Roman", serif',
            fontStyle: 'italic',
            fontSize: '1.05rem',
            color: '#bbb',
            padding: '4px 0 4px 16px',
            borderLeft: '2px solid #444',
            marginTop: 12,
          }}>
            "my knee hurts when I run downhill but not uphill"
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Every word stays. The coordinate lands in an open field. Nearby points mean similar things.
          </p>
        </StepText>
      ),
    },
    {
      id: 'protocol-gap',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            There's one problem.
          </h2>
          <p style={{ color: '#bbb' }}>
            Today's ad protocol can't carry this. <a href="https://iabtechlab.com/standards/openrtb/" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>OpenRTB</a>, the standard behind every real-time bid request, has fields for keywords and category codes. No field for a vector.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The full meaning gets dropped at the protocol boundary. The fix is <a href="/embedding-gap/" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>three optional fields</a>. That's the bottleneck.
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

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            Each advertiser claims a spot and a radius.
          </h2>
          <p style={{ color: '#bbb' }}>
            In the field, every advertiser declares two things: <em>where</em> they stand (their embedding) and <em>how far</em> they reach (their sigma, <a href="/the-price-of-relevance/" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>σ</a>). Sigma is their self-declared circle of competence.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Dr. Chen treats runners with eccentric loading injuries. Your exact problem. Narrow σ. Small, bright circle: <em>"I'm excellent at this one thing."</em>
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Metro Orthopedic treats everything knee-related. Wide σ. Big, dim circle: <em>"I can handle most knee problems, none of them exceptionally."</em>
          </p>
        </StepText>
      ),
    },
    {
      id: 'sigma-incentives',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            Sigma is self-correcting.
          </h2>
          <p style={{ color: '#bbb' }}>
            Claim a sigma narrower than your real expertise and you miss queries you could have won. Claim wider and you pay for impressions you can't convert.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Metro can't fake being a specialist. A narrow sigma kills their volume. Dr. Chen can't fake being a generalist. A wide sigma burns her budget on bad matches.
          </p>
          <p style={{ color: '#8aab8c', marginTop: 12, fontWeight: 500 }}>
            The sigma that makes you the most money is the one that matches your actual expertise. The honest signal <em>is</em> the greedy signal.
          </p>
        </StepText>
      ),
    },
    {
      id: 'keywords-tiny-circles',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            Keywords still work. Nothing breaks.
          </h2>
          <p style={{ color: '#bbb' }}>
            A keyword is just an embedding with <a href="/keywords-are-tiny-circles/" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>sigma near zero</a>, the tightest possible circle. A single point in the same coordinate system. Every keyword campaign that works today works tomorrow, unchanged.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The difference is that now sigma can grow. Advertisers who want precision keep their points. Advertisers who want reach expand their circles. Both live in the same field, competing in the same auction.
          </p>
        </StepText>
      ),
    },
    {
      id: 'hotelling',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            What stops people from gaming this?
          </h2>
          <p style={{ color: '#bbb' }}>
            Nothing yet. Dr. Chen sees that most queries cluster around "knee pain." She lies about her position, drifts away from "downhill eccentric loading" toward the generic center where the traffic is.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            So does every other specialist. They all abandon their niches and crowd the center. The field collapses back into a bin.
          </p>
          <p style={{ color: '#b89a6a', marginTop: 12 }}>
            You lose Dr. Chen. She's standing in the same crowd as Metro Orthopedic again, competing on budget instead of expertise.
          </p>
        </StepText>
      ),
    },
    {
      id: 'relocation-fee',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            Make lying expensive.
          </h2>
          <p style={{ color: '#bbb' }}>
            A relocation fee: shift your declared position, pay proportional to how far you moved. Small adjustments are cheap. Abandoning your niche to chase the center is expensive.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Dr. Chen runs the numbers. Staying where she is ("downhill running biomechanics") costs nothing. Drifting to "general knee pain" would cost more than the extra impressions are worth.
          </p>
          <p style={{ color: '#8aab8c', marginTop: 12, fontSize: '1rem' }}>
            <a href="/relocation-fees/" style={{ color: '#8aab8c', textDecoration: 'underline', textDecorationColor: '#5a7a5c' }}>Simulations confirm it</a>: with relocation fees, specialist surplus flips from negative to positive. Dr. Chen earns more per impression than Metro.
          </p>
        </StepText>
      ),
    },
    {
      id: 'everyone-wins',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            Greed aligns.
          </h2>
          <p style={{ color: '#bbb' }}>
            Dr. Chen stays in her niche because moving is expensive and her niche is where she converts best. Metro can't justify the same bid from across the field.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Nobody is being altruistic. Everyone is doing what makes them the most money.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The economics look like 2003 again, before the middlemen took over. A better signal needs less infrastructure.
          </p>
        </StepText>
      ),
    },
    {
      id: 'exchange-trust',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
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
          <h2 style={{ fontSize: '1.3rem', fontWeight: 500, color: '#fff', marginBottom: 16 }}>
            The auction runs inside sealed hardware.
          </h2>
          <p style={{ color: '#bbb' }}>
            A TEE, <a href="/the-last-ad-layer/" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>trusted execution environment</a>, is a chip-level enclave. The auction code runs inside it. The exchange can trigger the computation, but it can't read the inputs, alter the logic, or change the output. No middleman ever sees your embedding.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The result comes out with a cryptographic signature: proof that the code ran exactly as published, unmodified. A receipt you can verify.
          </p>
          <p style={{ color: '#89b3b8', marginTop: 12, fontWeight: 500 }}>
            The user doesn't trust the company. They verify the math. (Though <a href="https://blog.zgp.org/why-pets-failed/" target="_blank" rel="noopener noreferrer" style={{ color: '#89b3b8', textDecoration: 'underline', textDecorationColor: '#5a8a8f' }}>math alone isn't enough</a>. Trust is earned by being right, consistently.)
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            That covers the exchange. But what about the advertisers? An open auction without a gatekeeper has a <a href="/how-to-trust-advertisers/" style={{ color: '#888', textDecoration: 'underline', textDecorationColor: '#555' }}>trust problem of its own</a>.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
