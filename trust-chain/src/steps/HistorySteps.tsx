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


export const HistorySteps = {
  steps: [
    {
      id: 'history-overture',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Someone connects ads to intent.
          </h2>
          <p style={{ color: '#bbb' }}>
            2000. A company called Overture<Cite href="https://en.wikipedia.org/wiki/Yahoo_Native" n={1} /> lets businesses bid on search keywords. You search "knee pain" — a physio ad appears. You click, they pay. No click, no charge.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            For the first time, the ad <em>is the answer to your question</em>. That's not an interruption — it's a service.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-quality-score',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            The highest bidder always wins.
          </h2>
          <p style={{ color: '#bbb' }}>
            Overture has a flaw: a hedge fund running knee-brace arbitrage can outbid the physical therapist who actually helps runners.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Google fixes this. AdWords adds Quality Score — your ad's <em>relevance</em> affects your rank, not just your bid. A helpful ad with a lower bid beats a spammy ad with a higher one.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            This is the mechanism that makes the system fair. Remember it — it matters later.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-middlemen',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Quality Score works — on Google. But most of the web isn't Google.
          </h2>
          <p style={{ color: '#bbb' }}>
            Millions of sites have readers but can't sell their own ads. And advertisers want to reach people across thousands of sites, not just on search.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Supply-side platforms represent publishers. Demand-side platforms bid for advertisers. An ad exchange connects them in real time. Each solves a real problem. Each takes a cut.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-consolidation',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            One company buys all three layers.
          </h2>
          <p style={{ color: '#bbb' }}>
            2008: DoubleClick<Cite href="https://en.wikipedia.org/wiki/DoubleClick" n={2} /> (ad server). 2009: AdMob<Cite href="https://en.wikipedia.org/wiki/AdMob" n={3} /> (mobile). 2010: Invite Media<Cite href="https://en.wikipedia.org/wiki/Invite_Media" n={4} /> (DSP). Each acquisition fills a gap. Each makes sense alone.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            But now Google operates the tool advertisers use to buy, the tool publishers use to sell, <em>and</em> the exchange in the middle. The referee is also both teams.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-degradation',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Then Google starts making the matching worse on purpose.
          </h2>
          <p style={{ color: '#bbb' }}>
            Broad match becomes the default — you bid on "running knee specialist," Google matches you to "knee pain," "leg doctor," "orthopedic surgery." More advertisers per auction means higher bids.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Search term reports get cut. "Exact match" is redefined to not mean exact. Manual bidding is replaced by Google's algorithm.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The ads aren't irrelevant because ads are inherently bad. They're irrelevant because <em style={{ color: '#2196F3' }}>Google makes more money when they're vague</em>.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-antitrust',
      content: (
        <StepText>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            A court finally looks inside.
          </h2>
          <p style={{ color: '#bbb' }}>
            2024. The DOJ's<Cite href="https://en.wikipedia.org/wiki/United_States_v._Google_LLC_(2023)" n={5} /> most effective exhibit is a Monopoly board with every road leading to Google.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The judge rules: <span style={{ color: '#2196F3', fontWeight: 600 }}>monopoly</span><Cite href="https://en.wikipedia.org/wiki/United_States_v._Google_LLC_(2020)" n={6} />. But the ruling changes nothing. Google has the search box. 90% of searches start there<Cite href="https://gs.statcounter.com/search-engine-market-share" n={7} />. The advertisers can't leave.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            The only players who ever escaped built their own distribution — Facebook, Instagram, TikTok. But social ads target <em>who you are</em>, not <em>what you need</em>. There's no intent.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-chat',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Google can't be forced to change. But it can be routed around.
          </h2>
          <p style={{ color: '#bbb' }}>
            It's 2026. Your knee still hurts. And millions of people are asking chatbots the questions they used to type into Google. You open a chat:
          </p>
          <p style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '1.05rem',
            color: '#FFD700',
            padding: '12px 16px',
            background: 'rgba(255, 215, 0, 0.08)',
            borderRadius: 8,
            border: '1px solid rgba(255, 215, 0, 0.2)',
            marginTop: 12,
          }}>
            "my knee hurts when I run downhill but not uphill"
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Every word carries meaning. And for the first time in twenty years, the surface isn't Google's.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-closing',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Google is already absorbing the new surface.
          </h2>
          <p style={{ color: '#bbb' }}>
            AI Overviews — Google's chatbot answers — now sit above the search results, absorbing clicks that used to go to organic links. The playbook is the same: capture the new surface, bolt on the old keyword plumbing, keep the bins.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Surfaces don't stay unclaimed. The last time one appeared — search, in 1999 — one company captured it in three years. If the chat surface repeats the same pattern — keywords, bins, highest-bidder-wins — you get the same result with a different logo.
          </p>
          <p style={{ color: '#bbb', marginTop: 12, fontWeight: 600 }}>
            Here's what that looks like.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
