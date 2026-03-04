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
            Chatbot providers have the best ad signal ever built — and no way to use it.
          </h2>
          <p style={{ color: '#bbb' }}>
            Every conversation already produces an embedding — intent in vector form. ChatGPT, Claude, Perplexity all have it. None of them can connect it to advertisers.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Perplexity tried ads. Users <a href="https://futurism.com/artificial-intelligence/openai-perplexity-admits-ai-adverts-mistake" target="_blank" rel="noopener noreferrer" style={{ color: '#888', textDecoration: 'underline', textDecorationColor: '#555' }}>revolted</a>. Anthropic <a href="https://www.anthropic.com/news/claude-is-a-space-to-think" target="_blank" rel="noopener noreferrer" style={{ color: '#888', textDecoration: 'underline', textDecorationColor: '#555' }}>promised no ads at all</a>. The signal exists. There's just <a href="/the-last-signal/" style={{ color: '#888', textDecoration: 'underline', textDecorationColor: '#555' }}>no pipe to carry it</a>.
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
            Google's "AI Max" uses AI to <a href="https://www.seroundtable.com/google-ads-ai-max-broad-matchifies-40562.html" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>broad-match your exact-match keywords</a>. The Trade Desk's <a href="https://www.thetradedesk.com/press-room/the-trade-desk-launches-kokai-a-new-media-buying-platform-that-brings-the-full-power-of-ai-to-digital-marketing" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>Kokai</a> optimizes bids within the same OpenRTB pipe. More AI, same primitives.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            <a href="https://www.globenewswire.com/news-release/2024/06/11/2896871/0/en/PubMatic-Is-The-First-SSP-To-Make-Cognitiv-ContextGPT-4-Available-To-Buyers.html" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>Cognitiv</a>, <a href="https://www.seedtag.com/contextual-ai/" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>Seedtag</a>, <a href="https://www.businesswire.com/news/home/20200115005272/en/GumGums-Groundbreaking-Contextual-Analysis-Solution-for-Digital-Publishers-Makes-Official-Debut" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>GumGum</a> — they all use embeddings to <em>read</em> page content. But the output collapses to a deal ID or a segment code before it enters the bid stream. The vector never reaches the auction. OpenRTB 2.6 has fields for keywords and category codes. No field for a vector.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Google won't fix this. A precise signal means fewer competitors per query, lower bids, and transparent matching — all of which erode the margin that vague keywords protect. The better signal is the one they can't afford to carry.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Everyone else is using AI to optimize the selection of the same old primitives. Nobody is changing the pipe.
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
            OpenAI loses <a href="https://fortune.com/2025/11/12/openai-cash-burn-rate-annual-losses-2028-profitable-2030-financial-documents/" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>$5 billion a year</a>. Anthropic burns <a href="https://cybernews.com/ai-news/openai-anthropic-profit-revenue-ai/" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>$3 billion</a>. Perplexity <a href="https://sacra.com/c/perplexity/" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>spends more on compute than it earns</a>. None of them are profitable. They survive on venture capital, and VC money has an expiration date.
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
    {
      id: 'what-needs-to-happen',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            <a href="/who-builds-it/" style={{ color: '#fff', textDecoration: 'none', borderBottom: '1px solid #888' }}>Here's what needs to happen.</a>
          </h2>
          <p style={{ color: '#bbb' }}>
            <strong style={{ color: '#fff' }}>IAB Tech Lab</strong> — add three fields to OpenRTB: <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#4CAF50', fontSize: '0.95em' }}>embedding</span>, <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#4CAF50', fontSize: '0.95em' }}>embedding_model</span>, <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#4CAF50', fontSize: '0.95em' }}>sigma</span>.<br />
            <a href="/embedding-gap/" style={{ color: '#888', textDecoration: 'none', borderBottom: '1px solid #555' }}>The $200 Billion Bottleneck</a>
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            <strong style={{ color: '#fff' }}>A chatbot provider</strong> — carry the embedding to the ad boundary.<br />
            <a href="/perplexity-was-right-to-kill-ads/" style={{ color: '#888', textDecoration: 'none', borderBottom: '1px solid #555' }}>Perplexity Was Right to Kill Ads</a>
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            <strong style={{ color: '#fff' }}>An exchange</strong> — add embedding parameters to the existing auction. CloudX already runs attested auctions in a TEE — the scoring function just needs three optional fields.<br />
            <a href="/letter-to-cloudx/" style={{ color: '#888', textDecoration: 'none', borderBottom: '1px solid #555' }}>An Open Letter to CloudX</a>
          </p>
          <p style={{ color: '#4CAF50', marginTop: 16, fontWeight: 600 }}>
            Three engineering decisions. Not a moonshot.
          </p>
        </StepText>
      ),
    },
    {
      id: 'the-window',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            The window won't stay open.
          </h2>
          <p style={{ color: '#bbb' }}>
            Every month without an open protocol is another month for a platform to capture the chat surface and lock it down. Google is already embedding ads in AI Overviews. They don't need a better ad system. They need this one to not exist.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The last time an open protocol could have prevented a monopoly was 1998. The web was open. Search was open. Ads weren't. One company filled the gap and spent twenty years extracting the surplus.
          </p>
          <p style={{ color: '#FF8800', marginTop: 12, fontWeight: 600 }}>
            We're at that moment again. This time, build the pipe before someone owns it.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
