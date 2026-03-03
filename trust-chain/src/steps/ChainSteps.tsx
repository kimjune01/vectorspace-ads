import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';

function StepText({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: '1.25rem', lineHeight: 1.7, maxWidth: 440 }}>
      {children}
    </div>
  );
}

export const ChainSteps = {
  steps: [
    {
      id: 'chatbots-stuck',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            The chatbot providers have the richest intent signal ever — but can't use it to cover costs.
          </h2>
          <p style={{ color: '#bbb' }}>
            Every conversation carries an embedding. Every question is intent in vector form. ChatGPT, Claude, Perplexity — they have the richest signal any ad system has ever seen, and no way to connect it to advertisers.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Perplexity tried ads. Users revolted. Anthropic promised no ads at all. Billions of high-intent queries, every day, with full semantic resolution. Thrown away — because there's no protocol, no auction, and no trust layer to carry it.
          </p>
        </StepText>
      ),
    },
    {
      id: 'incumbents-wrong',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            The ad industry is building the wrong thing.
          </h2>
          <p style={{ color: '#bbb' }}>
            Google won't build it — an embedding auction would cannibalize the keyword monopoly. Every improvement to matching quality is a threat to a $250 billion business.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            SSPs and DSPs are reaching for the wrong piece — bolting embeddings onto the old infrastructure. Embed the page content, score it with cosine similarity, run the same auction. It's a better keyword, not a new system.
          </p>
        </StepText>
      ),
    },
    {
      id: 'the-chain',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Each piece exists. Nobody has connected them.
          </h2>
          <p style={{ color: '#bbb' }}>
            The intent signal. The auction mechanism. The trust layer. The incentive design. The user experience. Each one solves a real problem. Each one is being built in isolation.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            The chatbot providers have the signal. The ad industry has the infrastructure. The cryptography community has the enclaves. Nobody is looking at the whole system.
          </p>
        </StepText>
      ),
    },
    {
      id: 'the-surface',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Nobody owns the chat surface. Yet.
          </h2>
          <p style={{ color: '#bbb' }}>
            ChatGPT, Claude, Perplexity, Gemini, open-source models — they all produce embeddings. The distribution is fragmented, which means it's <em>available</em>. No single company controls it.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            An open protocol — one that carries embeddings, runs the auction in a TEE, and enforces honest positioning — would connect all of them. Not another platform. A standard.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The last time a surface shift this large happened, Google captured it and spent twenty years extracting the surplus. This time, the protocol can be open from day one.
          </p>
        </StepText>
      ),
    },
    {
      id: 'zoom-surveillance',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Keyword ads needed surveillance to work. Embedding ads don't.
          </h2>
          <p style={{ color: '#bbb' }}>
            The entire behavioral targeting industry — cookies, fingerprinting, cross-site tracking, data brokers — exists because keywords couldn't capture intent. So the industry tracked <em>people</em> instead.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            An embedding auction matches on what you <em>said</em>, not what you <em>did</em>. The vector comes from the conversation. No tracking pixel. No profile. No third party sees it. The sealed enclave runs the auction and forgets.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            The ad system that made the internet hostile to privacy was never the only option. It was just the first one that worked.
          </p>
        </StepText>
      ),
    },
    {
      id: 'zoom-businesses',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Dr. Chen is one specialist. There are millions.
          </h2>
          <p style={{ color: '#bbb' }}>
            The immigration lawyer who speaks Tagalog. The HVAC tech who specializes in century homes. The tutor who teaches calculus to kids with ADHD. Every one of them is invisible to keyword ads — too niche to win a bid, too specific to fit a category.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            An embedding auction makes them discoverable for the first time — matched to the people who actually need them, at a price they can afford. Not because someone tracked a user across the internet, but because the user described exactly what they needed.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            That's not an optimization. That's a different economy.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
