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
            1998. A company called GoTo.com<Cite href="https://en.wikipedia.org/wiki/Pay-per-click" n={1} /> lets businesses bid on search keywords. You search "knee pain" — a physio ad appears. You click, they pay. No click, no charge.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            For the first time, the ad <em>is the answer to your question</em>.
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
            Google fixes this. AdWords adds Quality Score<Cite href="https://en.wikipedia.org/wiki/Quality_Score" n={11} /> — your ad's <em>relevance</em> affects your rank, not just your bid. A helpful ad with a lower bid beats a spammy ad with a higher one.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            This is the part that matters later.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-consolidation',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            An ecosystem appears. One company buys it all.
          </h2>
          <p style={{ color: '#bbb' }}>
            Publishers need help selling ads. Advertisers need help buying them. Middlemen appear — SSPs, DSPs, exchanges — each solving a real problem, each taking a cut.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Then Google acquires them. 2008: DoubleClick<Cite href="https://en.wikipedia.org/wiki/DoubleClick" n={2} />. 2009: AdMob<Cite href="https://en.wikipedia.org/wiki/AdMob" n={3} />. 2010: Invite Media<Cite href="https://en.wikipedia.org/wiki/Invite_Media" n={4} />. Now Google operates the tool advertisers use to buy, the tool publishers use to sell, <em>and</em> the exchange in the middle.
          </p>
        </StepText>
      ),
    },
    {
      id: 'history-degradation',
      content: (
        <StepText>

          <h2 style={{ fontSize: '1.6rem', fontWeight: 600, color: '#fff', marginBottom: 16 }}>
            Then the matching gets worse. On purpose.
          </h2>
          <p style={{ color: '#bbb' }}>
            You bid on "running knee specialist." Google shows your ad to people searching "leg doctor"<Cite href="https://searchengineland.com/google-ads-broad-match-default-new-search-campaigns-444003" n={17} />. You ask which searches triggered your ad. Google stops telling you<Cite href="https://searchengineland.com/google-ads-to-limit-search-terms-reporting-citing-privacy-340137" n={18} />. You set "exact match." Google redefines exact to mean approximate<Cite href="https://searchengineland.com/googles-exact-match-close-variants-expand-again-now-include-same-meaning-variations-305056" n={19} />.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Each change makes the auction less precise and more crowded. More advertisers competing per query means higher bids. <strong style={{ color: '#fff' }}>Google makes more money when the matching is vague.</strong>
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
            The judge rules: <strong style={{ color: '#fff' }}>monopoly</strong><Cite href="https://en.wikipedia.org/wiki/United_States_v._Google_LLC_(2020)" n={6} />. But the ruling changes nothing. Google has the search box. 90% of searches start there<Cite href="https://gs.statcounter.com/search-engine-market-share" n={7} />. The advertisers can't leave.
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
            And for the first time in twenty years, the surface isn't Google's.
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
            AI Overviews — Google's chatbot answers — now sit above the search results, absorbing clicks that used to go to organic links<Cite href="https://searchengineland.com/google-ai-overviews-drive-drop-organic-paid-ctr-464212" n={21} />. The playbook is the same: capture the new surface, bolt on the old keyword plumbing, keep the bins.
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
