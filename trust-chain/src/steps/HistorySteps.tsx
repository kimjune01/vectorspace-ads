import type { ReactNode } from 'react';
import type { StepData } from '../ScrollySection';


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
            1998. A company called <a href="https://en.wikipedia.org/wiki/Pay-per-click" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>GoTo.com</a> lets businesses bid on search keywords. You search "knee pain." A physio ad appears. You click, they pay. No click, no charge.
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
            GoTo.com (renamed Overture, then acquired by Yahoo) has a flaw: a hedge fund running knee-brace arbitrage can outbid the physical therapist who actually helps runners.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Google fixes this. AdWords adds <a href="https://en.wikipedia.org/wiki/Quality_Score" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>Quality Score</a>, where your ad's <em>relevance</em> affects your rank, not just your bid. A helpful ad with a lower bid beats a spammy ad with a higher one.
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
            Publishers need help selling ads. Advertisers need help buying them. Middlemen appear: SSPs, DSPs, exchanges. Each solving a real problem, each taking a cut.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Then Google acquires them. 2008: <a href="https://en.wikipedia.org/wiki/DoubleClick" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>DoubleClick</a>. 2009: <a href="https://en.wikipedia.org/wiki/AdMob" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>AdMob</a>. 2010: <a href="https://en.wikipedia.org/wiki/Invite_Media" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>Invite Media</a>. Now Google operates the tool advertisers use to buy, the tool publishers use to sell, <em>and</em> the exchange in the middle.
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
            Google <a href="https://searchengineland.com/google-enhanced-campaigns-now-includes-close-variants-mandatory-200068" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>forced</a> close variants, <a href="https://searchengineland.com/google-ads-close-variants-reordered-words-added-removed-function-words-303145" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>reordered</a> your keywords, <a href="https://searchengineland.com/googles-exact-match-close-variants-expand-again-now-include-same-meaning-variations-305056" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>redefined</a> exact match, <a href="https://searchengineland.com/google-ads-to-limit-search-terms-reporting-citing-privacy-340137" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>hid</a> your search terms, <a href="https://searchengineland.com/google-expands-phrase-match-to-include-broad-match-modifier-traffic-345874" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>killed</a> broad match modifier, and <a href="https://searchengineland.com/google-ads-broad-match-default-new-search-campaigns-444003" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>defaulted</a> every new campaign to broad match.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            Each change made the matching vaguer and the auction more crowded. More advertisers per query means higher bids. <strong style={{ color: '#fff' }}>Google makes more money when the matching is vague.</strong>
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
            2024. The <a href="https://en.wikipedia.org/wiki/United_States_v._Google_LLC_(2023)" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>DOJ's</a> most effective exhibit is a Monopoly board with every road leading to Google.
          </p>
          <p style={{ color: '#bbb', marginTop: 12 }}>
            The judge rules: <a href="https://en.wikipedia.org/wiki/United_States_v._Google_LLC_(2020)" target="_blank" rel="noopener noreferrer" style={{ color: '#fff', fontWeight: 700, textDecoration: 'underline', textDecorationColor: '#888' }}>monopoly</a>. But the ruling changes nothing. Google has the search box. <a href="https://gs.statcounter.com/search-engine-market-share" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>90%</a> of searches start there. The advertisers can't leave.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            The only players who ever escaped built their own distribution: Facebook, Instagram, TikTok. But social ads target <em>who you are</em>, not <em>what you need</em>. There's no intent.
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
            AI Overviews, Google's chatbot answers, now sit above the search results, <a href="https://searchengineland.com/google-ai-overviews-drive-drop-organic-paid-ctr-464212" target="_blank" rel="noopener noreferrer" style={{ color: '#bbb', textDecoration: 'underline', textDecorationColor: '#666' }}>absorbing clicks</a> that used to go to organic links. The playbook is the same: capture the new surface, bolt on the old keyword plumbing, keep the bins.
          </p>
          <p style={{ color: '#888', marginTop: 12 }}>
            Surfaces don't stay unclaimed. The last time one appeared (search, in 1999) one company captured it in three years. If the chat surface repeats the same pattern, you get the same result with a different logo.
          </p>
          <p style={{ color: '#bbb', marginTop: 12, fontWeight: 600 }}>
            Here's what that looks like.
          </p>
        </StepText>
      ),
    },
  ] satisfies StepData[],
};
