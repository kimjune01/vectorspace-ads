# Power Diagrams for Embedding-Based Ad Auctions

Embedding-based ad auctions using power diagrams (additively weighted Voronoi tessellations) as the allocation mechanism for continuous intent space.

Blog post: [The Geometry of AI Advertising](https://www.june.kim/2026/02/07/power-diagrams-ad-auctions/)

## Structure

```
shared/          Core Python: auction computation, diagram generation
paper/           LaTeX paper with proofs and experiments
prototype/       Interactive React + TypeScript explorer
blog/            Blog post source
```

## Setup

### Python (paper, diagrams, experiments)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python shared/diagrams/generate_all.py
uv run python paper/experiments/run_comparison.py
```

### Prototype

```bash
cd prototype
pnpm install
pnpm dev
```

### Paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Key idea

The welfare-maximizing allocation for isotropic Gaussian value functions over a continuous impression space is a power diagram. The winner at each point is:

```
i*(x) = argmax_i [ log(b_i) - ||x - c_i||² / σ² ]
```

This reduces mechanism design to computational geometry: allocation is power diagram construction, payments are Voronoi cell integration.
