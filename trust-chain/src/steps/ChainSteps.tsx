import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';
import { Cite } from '../components/Cite';

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
            Chatbot providers have the best ad signal ever built — and no way to use it.
          </h2>
          <p style={{ color: '#bbb' }}>
            Every conversation already produces an embedding — intent in vector form. ChatGPT, Claude, Perplexity all have it. None of them can connect it to advertisers.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Perplexity tried ads. Users revolted<Cite href="https://futurism.com/artificial-intelligence/openai-perplexity-admits-ai-adverts-mistake" n={9} />. Anthropic promised no ads at all<Cite href="https://www.anthropic.com/news/claude-is-a-space-to-think" n={10} />. The signal exists. There's just no pipe to carry it<Cite href="/the-last-signal/" n={31} />.
          </p>
        </StepText>
      ),
    },
    {
      id: 'incumbents-wrong',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            The adtech industry is building around the wrong abstraction.
          </h2>
          <p style={{ color: '#bbb' }}>
            Google's "AI Max" uses AI to broad-match your exact-match keywords<Cite href="https://www.seroundtable.com/google-ads-ai-max-broad-matchifies-40562.html" n={15} />. The Trade Desk's Kokai<Cite href="https://www.thetradedesk.com/press-room/the-trade-desk-launches-kokai-a-new-media-buying-platform-that-brings-the-full-power-of-ai-to-digital-marketing" n={23} /> optimizes bids within the same OpenRTB pipe. More AI, same primitives.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Cognitiv<Cite href="https://www.globenewswire.com/news-release/2024/06/11/2896871/0/en/PubMatic-Is-The-First-SSP-To-Make-Cognitiv-ContextGPT-4-Available-To-Buyers.html" n={16} />, Seedtag<Cite href="https://www.seedtag.com/contextual-ai/" n={24} />, GumGum<Cite href="https://www.businesswire.com/news/home/20200115005272/en/GumGums-Groundbreaking-Contextual-Analysis-Solution-for-Digital-Publishers-Makes-Official-Debut" n={25} /> — they all use embeddings to <em>read</em> page content. But the output collapses to a deal ID or a segment code before it enters the bid stream. The vector never reaches the auction. OpenRTB 2.6 has fields for keywords and category codes. No field for a vector.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Everyone is using AI to optimize the selection of the same old primitives. Nobody is changing the pipe.
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
      id: 'zoom-independence',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Without revenue, chatbots become subsidiaries.
          </h2>
          <p style={{ color: '#bbb' }}>
            OpenAI loses $5 billion a year<Cite href="https://fortune.com/2025/11/12/openai-cash-burn-rate-annual-losses-2028-profitable-2030-financial-documents/" n={12} />. Anthropic burns $3 billion<Cite href="https://cybernews.com/ai-news/openai-anthropic-profit-revenue-ai/" n={13} />. Perplexity spends more on compute than it earns<Cite href="https://sacra.com/c/perplexity/" n={14} />. None of them are profitable. They survive on venture capital, and VC money has an expiration date.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            When it runs out, the exits are acquisition or subsidy. Microsoft absorbs OpenAI. Google absorbs Gemini's costs because it owns the ad engine. Independent AI companies without a revenue model become features inside megacorps — or instruments of governments that fund them.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            An ad layer that works would let chatbots pay for themselves. Without one, every model becomes a feature inside a company that can afford to lose money on it.
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
            An embedding auction makes them discoverable for the first time — matched to the people who actually need them, at a price they can afford, because the user described exactly what they needed.
          </p>
        </StepText>
      ),
    },
    {
      id: 'zoom-jobs',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Each dot is a business. Each business supports jobs.
          </h2>
          <p style={{ color: '#bbb' }}>
            Dr. Chen has a receptionist, a billing coder, a PT aide. The HVAC tech has an apprentice and a dispatcher. The tutor hires other tutors when demand grows.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            These are the people who do what AI can't — hands-on, local, specialized work that requires showing up. When they get discovered, they hire.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            That's not a better ad. That's a better labor market.
          </p>
        </StepText>
      ),
    },
    {
      id: 'closing',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Your knee still hurts. But this time, Dr. Chen is one click away.
          </h2>
          <p style={{ color: '#bbb' }}>
            You didn't search a keyword. You described your problem. The auction didn't sell you to the highest bidder — it matched you to the nearest expert. Nobody tracked you. Nobody profiled you. The chatbot proved the match was honest, and you tapped because you wanted to.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Remember when you could just ask a question and get help?
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
