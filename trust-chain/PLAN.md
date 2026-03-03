# Trust Chain — Scrollytelling Visualization for HN

## Context
Build a standalone scrollytelling page that contrasts Google Search keyword ads vs. AI chat embedding ads. The visual metaphor is a **trust chain** — follow a query link by link through each system and watch where surplus gets extracted vs. preserved. Pre-baked example: "my knee hurts when I run downhill but not uphill."

Goal: A top-tier Show HN post that makes the case for embedding-space ad auctions without requiring any adtech knowledge. Should feel like a Magic School Bus journey through the ad pipeline.

## Page: `/trust-chain/` on kimjune01.github.io

## Tech Stack
- **New Vite + React + TypeScript project** at `/Users/junekim/Documents/vectorspace-ads/trust-chain/`
- **motion** (~8KB) for scroll-driven animations via `useScroll`
- **lenis** (~3KB) for smooth scrolling
- Inline CSS (matches existing prototype style)
- Deploy: `pnpm build` → copy `dist/` to blog repo's `/trust-chain/` directory
- Base path: `/trust-chain/`

## Narrative Structure (Scroll Sections)

### Act 1: The Query
- User sees a chat-style text box. Pre-baked text types out: **"my knee hurts when I run downhill but not uphill"**
- Visual: The words appear one by one, then glow/pulse — this is rich intent. A photograph.
- The query then splits into two paths. Fork in the road.

### Act 2: Google Search — The Broken Chain

**Link 1: Understanding (Google)**
- The query enters Google's search box
- Visual: The photograph gets run through a shredder. Out come keyword tokens: `knee` `pain` `running`
- "Downhill but not uphill" — discarded. "2am" context — discarded.
- Visual metaphor: cattle into bins. The query token falls into a pen labeled "knee pain running" with thousands of other queries
- Label: **Google** — owns the front door

**Link 2: Transmission (OpenRTB)**
- The keyword gets packaged into a bid request form
- Visual: A shipping form with fixed fields. Category: "Health > Orthopedic." There's literally no box for "what they actually meant"
- The form travels to the ad exchange
- Label: **The Protocol** — OpenRTB has no embedding field

**Link 3: The Auction (Google Ads)**
- The bid request arrives. All advertisers who bid on "knee pain running" are in the same room
- Visual: Shouting match. Big Orthopedic Chain holds up $8. Amazon Knee Brace holds up $5. Generic PT Clinic holds up $4. Sports biomechanics specialist holds up $2 — drowned out.
- Google runs the auction AND owns the search engine AND owns the ad network
- Label: **Google** — the auctioneer is also the house. DOJ ruled this a monopoly.

**Link 4: The Result**
- Big Orthopedic Chain wins. Generic "book a consultation" page.
- Visual: Receipt. $8 click. Google took ~$2.50 (30%). Intermediaries took another cut. The chain cost $8 and delivered a generic answer.
- The sports PT who actually treats runners? Couldn't afford to compete on "knee pain."

### Act 3: AI Chat — The Trust Chain

**Link 1: Understanding (The Chatbot)**
- Same query enters a chatbot
- Visual: The photograph stays intact. The chatbot understands: eccentric loading, anterior knee, runner, specific biomechanical pattern
- The full intent becomes a vector — a precise coordinate on a map, not a bin label
- Visual metaphor: pin drop on a map vs. cattle into a bin

**Link 2: Transmission (Open Protocol)**
- The vector travels intact through an open protocol
- Visual: The shipping form has a new field: "embedding vector." The full meaning arrives at the exchange
- No one strips it. No one collapses it to a category code
- Label: **The Protocol** — the vector field that doesn't exist yet in OpenRTB

**Link 3: The Auction (Independent Exchange)**
- The vector enters an embedding space where advertisers have planted their flags
- Visual: A map. Advertisers are positioned by what they actually do. The query's pin drop lands. Closest flag wins.
- No shouting match. Proximity decides. The sports PT who specializes in runners is RIGHT THERE.
- The exchange doesn't own the publisher. The auctioneer isn't the house.
- Visual: Reuse/adapt the existing power diagram canvas from the prototype
- Label: **Independent Exchange** — auction runs in a TEE, cryptographically attested

**Link 4: Privacy**
- The intent vector is sensitive — it reveals what you need
- Visual: The vector enters a locked vault (TEE). The math runs inside. The result comes out. Nobody saw the vector.
- Label: **Trusted Execution** — the auction is private and verifiable

**Link 5: The Result**
- Sports PT specializing in runners wins with a $2 bid
- Visual: Receipt side-by-side with the Google receipt. $2 vs $8. No middleman stack. The right answer won because it was closest, not loudest.

### Act 4: The Pieces Are Converging
- Zoom out. Show the four links of the trust chain as puzzle pieces:
  1. **Intent Extraction** — every LLM already does this
  2. **Open Protocol** — OpenRTB needs an embedding field (this doesn't exist yet)
  3. **Spatial Auction** — power diagrams, not keyword buckets (the math works — try the prototype)
  4. **Private Execution** — TEEs, confidential computing (being built)
- Each piece snaps together visually
- Link to the interactive prototype at `/vectorspace-ads/` — "see the auction math yourself"
- Link to the blog series at `/vector-space` — "read the full argument"

### Act 5: CTA
- "This is buildable. The pieces exist. Nobody's assembled them."
- Links: GitHub (openauction simulation), blog series, prototype

## Visual Design Principles
- Dark background (like the formula tooltip in the prototype: `#1a1a2e`)
- Bold, high-contrast text that appears on scroll
- Visual metaphors rendered as simple illustrated animations (not photographs)
- Each intermediary/player labeled with their name and their take rate
- Receipts as actual receipt-styled elements (monospace, dotted borders)
- The Google side should feel cluttered, extractive, noisy
- The embedding side should feel clean, direct, elegant

## Data Structures → Visual Metaphors
| Data Structure | Visual Metaphor |
|---|---|
| Raw query | Photograph — full color, full detail |
| Keywords/tokens | Cattle sorted into bins |
| Category code (IAB) | Shipping form with limited fields |
| Embedding vector | Pin drop on a map |
| Keyword auction | Shouting match in a pen |
| Embedding auction | Map with flags — closest wins |
| Intermediaries | Tollbooths on the road |
| TEE | Locked vault — math in, result out |

## File Structure
```
/Users/junekim/Documents/vectorspace-ads/trust-chain/
├── package.json          # vite, react, motion, lenis
├── vite.config.ts        # base: '/trust-chain/'
├── tsconfig.json
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx           # scroll container, section orchestration
    ├── sections/
    │   ├── QuerySection.tsx      # Act 1: the query types out
    │   ├── GoogleChain.tsx       # Act 2: keyword pipeline
    │   ├── EmbeddingChain.tsx    # Act 3: embedding pipeline
    │   ├── Convergence.tsx       # Act 4: pieces coming together
    │   └── CTA.tsx               # Act 5: call to action
    ├── components/
    │   ├── ScrollSection.tsx     # wrapper: ties scroll position to content
    │   ├── Receipt.tsx           # receipt-style cost breakdown
    │   ├── ChainLink.tsx         # individual link in the trust chain
    │   ├── QueryDisplay.tsx      # typing animation for the query
    │   ├── BinSort.tsx           # cattle-into-bins animation
    │   ├── MapPinDrop.tsx        # pin drop on embedding map
    │   ├── ShoutingMatch.tsx     # keyword auction visual
    │   └── AuctionMap.tsx        # embedding auction (adapted from prototype canvas)
    ├── data/
    │   └── scenarios.ts          # pre-baked query/advertiser/outcome data
    └── styles.ts                 # shared style constants
```

## Pre-baked Scenario Data (`scenarios.ts`)

Primary example (shown on scroll):
```ts
{
  query: "my knee hurts when I run downhill but not uphill",
  keywords: ["knee", "pain", "running"],  // what Google extracts
  discarded: ["downhill", "not uphill", "biomechanical pattern"],
  iabCategory: "Health > Orthopedic",

  googleAuction: {
    advertisers: [
      { name: "Metro Orthopedic Group", bid: 8.00, type: "chain", wins: true },
      { name: "Amazon Knee Braces", bid: 5.00, type: "ecommerce" },
      { name: "CityPT Physical Therapy", bid: 4.00, type: "generic" },
      { name: "Dr. Chen Sports Biomechanics", bid: 2.00, type: "specialist" },
    ],
    platformCut: 0.30,  // Google's take
    result: "Generic orthopedic consultation page. $8 click.",
  },

  embeddingAuction: {
    advertisers: [
      { name: "Dr. Chen Sports Biomechanics", position: [0.72, 0.65], bid: 2.00, wins: true },
      { name: "Metro Orthopedic Group", position: [0.3, 0.4], bid: 8.00 },
      { name: "RunLab Gait Analysis", position: [0.65, 0.55], bid: 3.00 },
      { name: "CityPT Physical Therapy", position: [0.4, 0.5], bid: 4.00 },
    ],
    queryPosition: [0.70, 0.68],
    platformCut: 0.05,
    result: "Sports PT specializing in runners. $2 click.",
  },
}
```

## Build & Deploy
```bash
cd /Users/junekim/Documents/vectorspace-ads/trust-chain
pnpm install
pnpm build
# copy dist/ to blog repo
cp -r dist/* /Users/junekim/Documents/kimjune01.github.io/trust-chain/
```

## Verification
- Page loads in < 1 second on a cold cache
- Scroll through all 5 acts — animations trigger at correct scroll positions
- Google side feels cluttered and extractive; embedding side feels clean and direct
- Receipts are legible and the cost difference is viscerally obvious
- Links to prototype and blog series work
- Mobile-responsive (HN readers on phones)
- No console errors
