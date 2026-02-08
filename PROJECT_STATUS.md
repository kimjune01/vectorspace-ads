# Embedding-Based Ad Auction PoC — Project Status

## Overview & Goals

Build a PoC establishing that the natural allocation mechanism in embedding space is a **power diagram** (additively weighted Voronoi tessellation), where bids shape territory boundaries. Three deliverables:

1. **Blog post** — SEO/discoverability entry point (~2500 words)
2. **Research paper** — depth, rigor, unexplored territory pointers
3. **Interactive prototype** — visual demonstration

## Current Phase

**All phases complete** — initial versions of all deliverables are done.

### Completed
- [x] Set up Python environment (uv) with numpy, scipy, matplotlib
- [x] Build core computation module (`shared/auction.py`)
- [x] Generate 5 blog diagrams (`shared/diagrams/`)
- [x] Write blog post (~2270 words, `blog/post.md`)
- [x] Create LaTeX paper skeleton with all 10 sections (`paper/main.tex`)
- [x] Build + run paper experiments (`paper/experiments/run_comparison.py`)
- [x] Build interactive React+Vite+TypeScript prototype (`prototype/`)

### Potential Follow-ups
- [ ] Polish blog post SEO metadata and publish
- [ ] Flesh out paper proofs and discussion sections
- [ ] Add impression stream animation to prototype
- [ ] Cross-check Python and TypeScript produce identical results for same inputs
- [ ] Add mobile responsiveness to prototype

## Deliverable Status

| Deliverable | Status | Location | Notes |
|-------------|--------|----------|-------|
| Blog Post | **Done** | `blog/post.md` | ~2270 words, 5 diagrams, SEO-optimized headings |
| Research Paper | **Done** | `paper/main.tex` | 10 sections, 3 theorems, 1 proposition, bibliography |
| Paper Experiments | **Done** | `paper/experiments/` | GSP vs VCG vs Power Diagram, 50 trials, figures generated |
| Interactive Prototype | **Done** | `prototype/` | React+Vite+TS, canvas rendering, drag+slider interaction |
| Shared Computation | **Done** | `shared/auction.py` | Power diagram, density, metrics, colormap |
| Blog Diagrams | **Done** | `shared/diagrams/` | 5 PNGs at 200 DPI |

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-07 | Blog first, paper second, prototype third | Blog for SEO discoverability, paper for rigor, prototype for demo |
| 2026-02-07 | Power diagram via grid evaluation | Brute-force argmax on grid is fast enough for 5 advertisers in 2D |
| 2026-02-07 | Canvas rendering for prototype | Perf-critical heatmap/diagram rendering, no heavy viz lib needed |
| 2026-02-07 | 200px grid resolution for prototype | Balances render speed with visual quality for real-time interaction |

## File Structure

```
vectorspace-ads/
├── PROJECT_STATUS.md
├── embedding-auction-handoff.md
├── pyproject.toml
├── blog/
│   └── post.md                          # ~2270 word blog post
├── paper/
│   ├── main.tex                         # LaTeX paper (10 sections)
│   ├── references.bib                   # Bibliography
│   ├── experiments/
│   │   └── run_comparison.py            # Synthetic comparison experiment
│   └── figures/
│       └── comparison_results.png       # Experiment results chart
├── prototype/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx                      # Main app layout
│   │   ├── AuctionCanvas.tsx            # Canvas rendering + drag interaction
│   │   ├── ControlPanel.tsx             # Sliders, metrics, anisotropic toggle
│   │   ├── auction.ts                   # Power diagram computation (TypeScript)
│   │   ├── data.ts                      # Default advertisers + clusters
│   │   └── types.ts                     # TypeScript interfaces
│   └── dist/                            # Production build
└── shared/
    ├── auction.py                       # Core computation (Python)
    └── diagrams/
        ├── generate_all.py              # Diagram generation script
        ├── 01_keywords_vs_embeddings.png
        ├── 02_power_diagram.png
        ├── 03_bid_change.png
        ├── 04_anisotropic.png
        └── 05_density_overlay.png
```

## Research References

See `embedding-auction-handoff.md` for full reference list.
