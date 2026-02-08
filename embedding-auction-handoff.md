# Embedding-Based Ad Auction Mechanisms: Project Handoff

## Context

This document specifies three deliverables exploring a novel approach to ad auction design for AI chatbot platforms (specifically OpenAI's ChatGPT ads rollout, but applicable to any LLM-based product). The core insight is that when ad targeting shifts from discrete keywords to continuous embedding spaces, the entire auction mechanism must be rethought — and the right framework comes from computational geometry (power diagrams / weighted Voronoi tessellations), not from extending keyword auctions.

**Why this matters now:** OpenAI announced in January 2026 that it will begin testing ads in ChatGPT for free and Go tier users in the US. Their internal strategy is called "intent-based monetization." They have 800M monthly users. They are building their ad platform from scratch. The auction mechanism design problem described here is one they will need to solve, and the public literature has no solution.

---

## The Core Problem

### Keywords vs. Embeddings

In Google's keyword auction, "mesothelioma lawyer" is a discrete biddable unit. Market mechanics are clean: second-price auction, clear winner determination, predictable budget spend.

In an embedding-based system (ChatGPT conversations, contextual targeting), there are no discrete units. A user's conversation traces a trajectory through high-dimensional embedding space. An "impression" is a point in this continuous space. Advertisers don't want a keyword — they want a *region* of this space.

### The Geometric Framing

The natural analog to keyword auctions in continuous space is a **power diagram** (additively weighted Voronoi diagram):

- Each advertiser i has a center point c_i (their ideal customer embedding), a bid b_i, and a value function v_i(x) describing how much they value an impression at point x.
- The simplest case: isotropic Gaussian value functions v_i(x) = b_i · exp(-||x - c_i||² / σ²). With shared σ, boundaries between advertisers are hyperplanes and the partition is a power diagram — tractable and well-studied.
- More realistic: anisotropic Gaussians where each advertiser has a covariance matrix Σ_i (they care more about some dimensions). Boundaries become quadric surfaces.
- Most realistic: mixture-of-Gaussian preferences ("fitness moms OR tech execs"). Boundaries are level sets of Gaussian sums — arbitrarily complex. Clean geometry breaks down.

### The Dimensionality Problem

Real ad targeting isn't just semantic similarity. The full space includes:
- Semantic content (embedding dimensions)
- Temporal (time of day, seasonality)
- Demographic (age, income, location)
- Psychographic (interests, values)
- Intent (browsing vs. purchase-ready)
- Device/context

Advertisers bid on high-dimensional manifolds, not spheres. Two advertisers might compete on weekday mornings but not weekends.

### Unsolved Problems

1. **Representation**: How do advertisers express preferences over continuous high-dimensional space tractably?
2. **Market clearing**: How do you run incentive-compatible auctions when bids are over overlapping fuzzy regions?
3. **Budget pacing**: How do advertisers predict spend when win rate depends on complex geometry of competing value functions?
4. **Computational tractability**: Real-time auctions need fast winner determination across billions of impressions.

### The Real Estate Analogy

The embedding space of a major LLM platform is like real estate at the scale of a country:
- **Regions** have different "property values" based on user traffic density
- **Zoning**: OpenAI excludes politics, health, mental health from ads — exclusion zones
- **Leasing**: Advertisers lease regions; their territory expands/contracts based on bid weight (the power diagram)
- **Property value shifts**: Cultural trends, seasons, news events change traffic through regions
- **Generative ads**: The same product gets different pitches depending on location in intent space — the ad creative is a function of position

---

## Prior Art Summary

### What Exists (Separate Silos)

**1. Neural Auction Design (DNA, CGA)**
- DNA (Liu et al., KDD 2021): End-to-end learned auction using set encoder + monotonic neural net + differentiable sorting. Deployed at Alibaba/Taobao.
- CGA (2024): Extends DNA with permutation-level externalities via autoregressive decoder.
- **Gap**: These take a pre-filtered candidate set and learn auction mechanics. They decouple targeting from the auction. The embedding space is used upstream for retrieval, then a standard-ish auction runs on the shortlist. They never address bidding on regions of continuous space.

**2. Mechanism Design via Optimal Transport (Daskalakis, Deckelbaum, Tzamos, EC 2013, MIT)**
- Framework connecting optimal mechanism design to optimal transport theory for multi-item auctions with continuous type spaces.
- Dual of revenue-maximization = continuous analog of minimum-weight bipartite matching.
- Strong duality theorems, closed-form solutions for several settings.
- **Gap**: Oriented toward multi-item auctions (selling bundles of goods), not spatial targeting where bidders want fuzzy regions of impressions.

**3. Hotelling / Spatial Competition Models**
- Irmen and Thisse (1998): "Competition in Multi-characteristics Spaces: Hotelling was Almost Right" — firms differentiate maximally on one dominant dimension, minimally on others.
- Voronoi games / competitive facility location: Game theory on spatial competition, often NP-complete.
- **Gap**: Assumes firms choose locations strategically; in ad auctions, "locations" (content embeddings) are exogenous impressions. Models studied mainly in 2-3 dimensions, not hundreds.

**4. Industry Practice (Contextual Targeting)**
- Seedtag: "Neuro-contextual" targeting using embeddings for interest/emotion/intent. No published auction mechanics.
- GumGum/Verity: CV + NLP for contextual classification at sub-10ms latency. Feeds into standard RTB auctions.
- Industry standard: IAB content taxonomy (~700 categories). Everyone discretizes the embedding space into coarse segments, runs standard auctions on cells, accepts boundary inefficiencies.

**5. Auctions with LLM Summaries (Google Research, 2024)**
- Models ad formats as continuous space (word count per ad in LLM summary).
- Shows continuous format space is easier to optimize than combinatorial discrete case.
- Small but real example of continuous auction design outperforming discrete.

**6. Optimal Auctions through Deep Learning (Dütting et al., ICML 2019)**
- Models auctions as neural networks; optimal design as constrained learning.
- Recovers known analytical solutions and finds novel mechanisms for unsolved settings.
- Extended by many follow-ups (PreferenceNet, context-integrated transformers, etc.).

### What Doesn't Exist

**The specific synthesis**: A representation of advertiser preferences as regions in embedding space + an incentive-compatible auction mechanism for clearing overlapping fuzzy bids + a budget management framework for geometric competition. This requires computational geometry × mechanism design × ML and doesn't appear in public literature.

### Key Papers to Cite

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Daskalakis, Deckelbaum, Tzamos "Mechanism Design via Optimal Transport" | EC 2013 | Optimal transport ↔ mechanism design duality |
| Dütting et al. "Optimal Auctions through Deep Learning" | ICML 2019 / CACM 2021 | Neural network auction design |
| Liu et al. "Neural Auction (DNA)" | KDD 2021 | End-to-end learned ad auctions |
| Zhu et al. "CGA with Permutation-level Externalities" | 2024 | State-of-art neural ad auction |
| Irmen & Thisse "Competition in Multi-characteristics Spaces" | JET 1998 | Spatial competition in high dimensions |
| Myerson "Optimal Auction Design" | MOR 1981 | Foundation of auction theory |
| Aggarwal et al. "Simple Mechanisms for Welfare Maximization in Rich Advertising Auctions" | NeurIPS 2022 | Rich auction settings |
| Shen, Tang, Zuo "Automated Mechanism Design via Neural Networks" | AAMAS 2019 | AI-driven mechanism discovery |
| Google Research "Auctions with LLM Summaries" | 2024 | Continuous format space auctions |

### OpenAI Ads Context

- Announced January 16, 2026: testing ads in US for free and Go tiers
- Internal strategy: "intent-based monetization"
- Ads at bottom of answers, based on current conversation context
- No ads near politics, health, mental health
- Exploring "generative ads" where ChatGPT creates the ad copy itself
- Exploring "sponsors" for GPT store (e.g., sauce brand sponsors a cooking GPT)
- Committed to answer independence (ads don't influence responses)
- 800M monthly users, $20B annualized revenue, $1.4T infrastructure commitments
- Plus/Pro/Business/Enterprise remain ad-free
- Previously pulled a shopping feature in Dec 2025 after user backlash

---

## Deliverable 1: Academic Paper

### Title (working)
"Power Diagrams for Embedding-Based Ad Auctions: Mechanism Design in Continuous Intent Space"

### Target Venues
- EC (ACM Conference on Economics and Computation)
- KDD (if more empirical)
- AAAI (if more AI-focused)
- Could also work as a workshop paper at EC or NeurIPS

### Structure

**Abstract**: Frame the problem (keyword auctions → embedding auctions), introduce power diagram framework, state main results (tractability conditions, incentive properties, approximation bounds).

**1. Introduction**
- Shift from keyword to embedding-based targeting (contextualized by OpenAI, Perplexity, etc. entering ads)
- Why existing auction mechanisms don't transfer
- The real estate analogy as intuition
- Contribution summary

**2. Model**
- Impression space X ⊆ R^d (embedding space)
- N advertisers, each with value function v_i: X → R+
- Value function parameterized by (c_i, b_i, Σ_i) — center, bid scalar, covariance
- Impression arrival process μ (distribution over X)
- Define the allocation problem: partition X into regions, one per advertiser (+ unallocated)

**3. Isotropic Case: Power Diagrams**
- v_i(x) = b_i · exp(-||x - c_i||² / σ²), shared σ
- Show boundaries are hyperplanes: the partition is a power diagram
- Properties: VCG-like payment rules preserve incentive compatibility
- Budget pacing: advertiser i's spend = b_i · ∫_{V_i} μ(x) dx, computable via integration over Voronoi cells
- This section should have clean theorems

**4. Anisotropic Case: Quadric Boundaries**
- v_i(x) = b_i · exp(-(x - c_i)^T Σ_i^{-1} (x - c_i))
- Boundaries are quadric surfaces (ellipsoids, hyperboloids)
- Still computable but structure is more complex
- Show conditions under which incentive compatibility holds or approximately holds
- Discuss the "factorized bidding" approximation (bid separately per dimension)

**5. Mixture-of-Gaussians: When Geometry Breaks Down**
- Advertiser with multimodal preferences
- Boundaries are level sets of Gaussian mixture sums — arbitrarily complex
- Propose approximation: represent each mixture component as a separate "virtual advertiser" with linked budgets
- Connection to combinatorial auction literature

**6. Computational Aspects**
- Real-time winner determination: for a given impression x, find argmax_i v_i(x). This is a nearest-neighbor-like query in weighted space.
- Pre-computation: build spatial index (kd-tree, ball tree, or approximate NN with bid-weighting)
- For isotropic case: standard power diagram algorithms apply
- For anisotropic: reduction to multiplicatively-weighted Voronoi with Mahalanobis distance
- Discuss latency constraints (sub-10ms, per GumGum's architecture)

**7. Budget Pacing and Spend Prediction**
- Key difficulty: advertiser's spend depends on the geometry of all competitors' bids
- In power diagram case: Monte Carlo estimation by sampling from μ
- Dynamic case: as bids change, regions shift — connection to competitive facility location dynamics
- Propose a "bid landscape" tool that shows an advertiser their estimated region and spend

**8. Experiments**
- Synthetic: generate impression embeddings from known distribution, create advertiser profiles, run auctions under different mechanisms (GSP-on-discretized, VCG-on-discretized, power-diagram-native). Compare revenue, welfare, advertiser utility.
- Semi-synthetic: use real embedding distributions (e.g., embed a corpus of web queries or conversation excerpts), create plausible advertiser profiles, compare approaches.
- Metrics: platform revenue, social welfare, incentive compatibility (regret), computational latency, budget prediction accuracy.

**9. Discussion**
- Connection to OpenAI's "intent-based monetization"
- The generative ads dimension: ad creative as function of position
- Privacy implications: embeddings encode user state
- Limitations: high-dimensional Voronoi computation, approximation quality
- Future work: dynamic bidding, learning value functions, multi-slot extensions

**10. Conclusion**

### Key Mathematical Results to Prove

1. **Isotropic power diagram is incentive-compatible** under VCG-like payments (likely follows from standard results but needs to be shown for the continuous setting)
2. **Budget prediction bound**: under isotropic Gaussians with known μ, advertiser spend can be estimated within ε with O(1/ε²) samples
3. **Approximation theorem**: any Lipschitz value function can be ε-approximated by a mixture of K isotropic Gaussians, where K depends on the Lipschitz constant and ε
4. **Computational complexity**: winner determination for isotropic case is O(log N) per impression via spatial indexing; anisotropic case is O(N) worst-case but O(log N) amortized with preprocessing

### Notation Conventions

- x ∈ R^d: impression embedding
- c_i ∈ R^d: advertiser i's center
- b_i ∈ R+: bid scalar
- Σ_i ∈ R^{d×d}: covariance matrix (positive definite)
- v_i(x): value function
- V_i: allocated region (Voronoi cell) for advertiser i
- μ: impression distribution
- p_i: payment per impression won

---

## Deliverable 2: Blog Post

### Title
"The Geometry of AI Advertising: How Power Diagrams Could Replace Keyword Auctions"

### Target Audience
Technical product people, adtech engineers, ML engineers at AI companies, investors in adtech. Should be accessible to someone who understands embeddings and has heard of auction theory but isn't a specialist in either.

### Tone
Analytical, concrete, forward-looking. Not hype. Show the problem is real and hard, show the geometric intuition clearly, make it obvious this is a gap worth filling.

### Structure (~2500 words)

**Opening hook**: OpenAI just announced ads in ChatGPT. They have 800M users. They're building their ad platform from scratch. But there's a fundamental problem: the entire machinery of online advertising was built for keywords, and LLM conversations don't have keywords.

**Section 1: Keywords Had Clean Markets**
- "mesothelioma lawyer" costs $200/click because the auction mechanics are clean
- Keyword = discrete biddable unit
- Second-price auction clears efficiently
- Budget forecasting is tractable (count keyword searches × your win rate × your bid)

**Section 2: Embeddings Break Everything**
- A ChatGPT conversation isn't a keyword — it's a trajectory through continuous space
- An "impression" is a point in high-dimensional space
- Advertisers don't want a keyword — they want a *region*
- Two advertisers might compete for "running shoes" conversations but not "trail running philosophy" conversations
- The dimensionality isn't just semantic: time, demographics, intent, device all matter

**Section 3: The Geometric Solution (Power Diagrams)**
- Visual intuition: imagine a 2D map where each advertiser "owns" territory
- Higher bids = bigger territory (like inflating a balloon)
- The boundary between two advertisers is where their bid-adjusted distances are equal
- This is a power diagram — a well-studied object in computational geometry
- Include a diagram (2D power diagram with 4-5 advertisers, different bid weights)

**Section 4: Why It Gets Hard**
- Isotropic case (same reach in all directions) is tractable
- Anisotropic case (care about some dimensions more) gives quadric boundaries — harder but computable
- Multimodal preferences ("fitness moms OR tech execs") break clean geometry entirely
- Real-time constraint: you have <10ms to determine the winner

**Section 5: The Real Estate Analogy**
- OpenAI's embedding space is like a country's real estate
- Regions have different property values based on user traffic
- Zoning: politics, health excluded from ads
- Leasing, not buying: your territory shifts as competitors bid
- Generative ads: the same product gets different pitches depending on "neighborhood"

**Section 6: What Industry Does Today (and Why It's Suboptimal)**
- Everyone discretizes: IAB taxonomy, ~700 categories
- Standard RTB auctions on cells
- Boundary inefficiencies: users near category boundaries get mismatched ads
- The value of embedding-based targeting is precisely the nuance that discretization throws away

**Section 7: The Open Research Problem**
- Need a representation of "regions of interest" that is: expressive, structured for market clearing, transparent for budget management
- This requires synthesis of computational geometry + mechanism design + ML
- Nobody has published this synthesis
- The buyer is identifiable: OpenAI, Perplexity, Anthropic, any AI company considering ads

**Closing**: The company that solves this owns the auction infrastructure for the next generation of advertising. The math exists in pieces. Someone needs to assemble it.

### Visual Assets Needed
1. Side-by-side: keyword auction (discrete) vs embedding auction (continuous) — show the conceptual difference
2. 2D power diagram with 4-5 advertisers at different bids — show how bid weight affects territory size
3. Anisotropic case: elliptical regions showing an advertiser who cares more about one dimension
4. The "real estate map" metaphor: embedding space colored by advertiser territory, with traffic density as a heatmap overlay

---

## Deliverable 3: Interactive Prototype

### What to Build

A browser-based interactive demo (React/JSX artifact or standalone HTML) that lets users explore the power diagram auction concept visually.

### Core Features

**1. 2D Embedding Space Visualization**
- A 2D canvas representing a slice of embedding space
- X-axis: e.g., "intent" (browsing ← → purchase-ready)
- Y-axis: e.g., "topic" (fitness ← → nutrition)
- Background heatmap showing impression density μ(x) — denser in some regions than others

**2. Advertisers as Draggable Points**
- 4-5 pre-configured advertisers (e.g., Nike, Whole Foods, Peloton, GNC, Fitbit)
- Each has a position (center), bid (adjustable slider), and reach/σ (adjustable slider)
- Color-coded

**3. Power Diagram Overlay**
- Real-time computation and rendering of the Voronoi/power diagram partition
- Each advertiser's territory filled with their color (semi-transparent)
- Boundaries clearly drawn
- As user adjusts bids or drags centers, the diagram updates in real-time

**4. Metrics Panel**
- For each advertiser: estimated impressions (integral of μ over their region), estimated spend (impressions × payment), territory area
- Platform metrics: total revenue, welfare
- Budget efficiency: cost per impression for each advertiser

**5. Anisotropic Mode (toggle)**
- When enabled, each advertiser gets directional sliders (σ_x, σ_y) so they can stretch their value function along one axis
- Boundaries become curved (elliptical regions)
- Demonstrate how an advertiser can specialize along one dimension

**6. Impression Stream (optional but good)**
- Animate dots (impressions) arriving according to μ
- Each dot lands in an advertiser's territory and gets colored
- Shows the dynamic nature of the allocation

### Technical Approach

- Use D3.js or Canvas 2D for the visualization
- Power diagram computation: for 2D with <10 advertisers, brute-force evaluation on a grid is fast enough (evaluate v_i(x) for all i at each pixel, assign to argmax)
- For the isotropic Gaussian case: v_i(x) = b_i · exp(-||x - c_i||² / σ_i²), so argmax_i v_i(x) = argmax_i [log(b_i) - ||x - c_i||² / σ_i²]
- Impression density: pre-defined 2D Gaussian mixture (3-4 clusters representing common conversation regions)
- Integration for metrics: sum pixel values within each region

### Starter Data

```
Advertisers:
  Nike:       center=(0.6, 0.3), bid=5.0, σ=0.3, color=#FF6B35
  Whole Foods: center=(0.3, 0.7), bid=3.0, σ=0.25, color=#4CAF50
  Peloton:    center=(0.5, 0.5), bid=4.0, σ=0.2, color=#2196F3
  GNC:        center=(0.7, 0.7), bid=2.5, σ=0.35, color=#9C27B0
  Fitbit:     center=(0.4, 0.3), bid=3.5, σ=0.25, color=#FF9800

Impression density (Gaussian mixture):
  Cluster 1: center=(0.5, 0.4), weight=0.4, σ=0.15  (general fitness intent)
  Cluster 2: center=(0.3, 0.6), weight=0.3, σ=0.1   (nutrition-focused)
  Cluster 3: center=(0.7, 0.5), weight=0.3, σ=0.12  (purchase-ready fitness)
```

### UX Flow

1. User sees the 2D space with advertisers placed and power diagram computed
2. User drags an advertiser's center → diagram updates in real-time → metrics update
3. User adjusts bid slider → territory expands/contracts → see how bid affects region size and spend
4. User toggles anisotropic mode → sees elliptical reach controls
5. Optional: user enables impression stream → sees dots flowing and being allocated

### What This Demonstrates

- The core insight: bids create geometric territories in embedding space
- How competition shapes boundaries (drag two advertisers close together → see the border)
- The budget prediction problem (adjust your bid → see how your estimated spend changes non-linearly because it depends on competitors)
- Why this is different from keyword auctions (there are no discrete boundaries — everything is continuous)

---

## Implementation Notes for Claude Code

### Paper
- Use LaTeX. Target EC/KDD formatting.
- The mathematical results in Section 3 (isotropic case) should be provably correct — this is the core contribution.
- Experiments can be synthetic for now. Use Python (numpy, scipy) for computation, matplotlib for figures.
- Generate the power diagram figures programmatically so they can be reused in the blog post.

### Blog Post
- Markdown output. Should stand alone without the paper.
- Diagrams can be generated as SVGs or PNGs from the same code as the paper figures.
- Link to the prototype at the end.

### Prototype
- Single-file React (JSX) or HTML+JS artifact.
- Prioritize interactivity and real-time responsiveness over visual polish.
- The power diagram computation should be correct (argmax of value functions on a grid).
- Mobile-friendly is nice but not required.
- Should work as a standalone demo that someone could share in a tweet.

### Shared Code
- The power diagram computation (evaluate value functions, determine boundaries) should be a shared module used by both the paper's experiments and the prototype.
- In Python for the paper, in JavaScript for the prototype — but the algorithm is the same.

### Priority Order
1. **Prototype** first — it's the most shareable and validates the visual intuition
2. **Blog post** second — references the prototype, sets up the framing
3. **Paper** third — requires the most rigor and iteration
