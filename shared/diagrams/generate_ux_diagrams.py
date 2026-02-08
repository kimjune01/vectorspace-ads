"""
Generate blog diagrams for the spatial ad buying UX post.

Diagrams:
1. Keywords vs embedding space (discrete grid vs continuous point cloud)
2. Hill-climbing sequence: 3-panel showing locus moving through space with arrows
3. Restriction zones: power diagram with gray exclusion areas
4. Competitive heatmap
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
from shared.auction import (
    Advertiser,
    DEFAULT_ADVERTISERS,
    DEFAULT_CLUSTERS,
    make_grid,
    compute_value_functions,
    compute_allocation,
    compute_impression_density,
    compute_metrics,
    hex_to_rgb,
    make_colormap,
)

# Consistent styling
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

RESOLUTION = 500
OUTPUT_DIR = os.path.dirname(__file__)


def add_advertiser_markers(ax, advertisers, fontsize=9):
    """Add labeled markers for each advertiser on the plot."""
    for adv in advertisers:
        cx, cy = adv.center
        ax.plot(cx, cy, "o", color="white", markersize=10, markeredgecolor="black",
                markeredgewidth=1.5, zorder=10)
        ax.annotate(
            f"{adv.name}\n(${adv.bid:.1f})",
            (cx, cy),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.85),
            zorder=11,
        )


def draw_boundaries(ax, allocation, xs, ys, color="black", linewidth=1.2):
    """Draw territory boundaries on the allocation map."""
    h, w = allocation.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:-1, :] |= allocation[:-1, :] != allocation[1:, :]
    boundary[:, :-1] |= allocation[:, :-1] != allocation[:, 1:]
    by, bx = np.where(boundary)
    ax.scatter(xs[bx], ys[by], c=color, s=0.15, alpha=0.6, zorder=5)


def diagram_1_keywords_vs_embeddings():
    """Side-by-side: keyword auctions (discrete) vs embedding auctions (continuous)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Keywords as discrete grid
    keywords = [
        ("running\nshoes", 0.3, 0.7, "#FF6B35", 5.0),
        ("protein\npowder", 0.7, 0.8, "#4CAF50", 3.0),
        ("fitness\ntracker", 0.5, 0.4, "#2196F3", 4.0),
        ("yoga\nmat", 0.2, 0.3, "#9C27B0", 2.5),
        ("gym\nmembership", 0.8, 0.3, "#FF9800", 3.5),
        ("meal\nprep", 0.4, 0.9, "#E91E63", 2.0),
        ("weight\nloss", 0.6, 0.6, "#00BCD4", 4.5),
        ("marathon\ntraining", 0.15, 0.55, "#795548", 3.0),
        ("keto\ndiet", 0.85, 0.65, "#607D8B", 2.8),
    ]

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    for kw, x, y, color, bid in keywords:
        size = 80 + bid * 30
        ax1.scatter(x, y, s=size, c=color, marker="s", edgecolors="black",
                    linewidths=1.5, zorder=5, alpha=0.9)
        ax1.annotate(kw, (x, y), ha="center", va="center", fontsize=7,
                     fontweight="bold", color="white", zorder=6)

    for i in np.arange(0.05, 1.0, 0.1):
        ax1.axhline(i, color="#E0E0E0", linewidth=0.5, linestyle="--", alpha=0.5)
        ax1.axvline(i, color="#E0E0E0", linewidth=0.5, linestyle="--", alpha=0.5)

    ax1.set_title("Keyword Auctions: Discrete Biddable Units")
    ax1.set_xlabel("Keyword Space")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(0.5, 0.02, "Each square = one biddable keyword",
             ha="center", fontsize=9, style="italic", transform=ax1.transAxes)

    # Right: Embedding space with continuous power diagram
    xx, yy, xs, ys = make_grid(RESOLUTION)
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy)
    allocation = compute_allocation(values)
    rgb = make_colormap(DEFAULT_ADVERTISERS, allocation)

    ax2.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", alpha=0.6, aspect="auto")
    draw_boundaries(ax2, allocation, xs, ys, color="black", linewidth=1.5)
    add_advertiser_markers(ax2, DEFAULT_ADVERTISERS, fontsize=8)

    # Scatter many small dots to show continuous nature
    np.random.seed(42)
    pts = np.random.rand(200, 2)
    ax2.scatter(pts[:, 0], pts[:, 1], c="#333", s=2, alpha=0.15, zorder=3)

    ax2.set_title("Embedding Auctions: Continuous Territory")
    ax2.set_xlabel("Embedding Space")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.5, 0.02, "Advertisers bid on regions, not keywords",
             ha="center", fontsize=9, style="italic", transform=ax2.transAxes)

    fig.suptitle("From Keywords to Embeddings: The Auction Design Shift",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "06_keywords_vs_embeddings_ux.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def diagram_2_hill_climbing():
    """3-panel hill-climbing sequence showing locus moving with direction arrows."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    xx, yy, xs, ys = make_grid(RESOLUTION)
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy)
    allocation = compute_allocation(values)
    rgb = make_colormap(DEFAULT_ADVERTISERS, allocation)

    # Three steps of a hill-climbing sequence
    steps = [
        {"locus": (0.5, 0.5), "label": 'Step 1: "fitness shoppers"',
         "target_label": "gym membership\nshoppers",
         "candidates": [(0.3, 0.3, "CrossFit"), (0.1, 0.5, "marathon\ntraining"),
                        (0.7, 0.7, "healthy\nmeal prep"), (0.5, 0.7, "fitness\ntracker deals")]},
        {"locus": (0.5, 0.7), "label": "Step 2: Clicked 'fitness tracker deals'",
         "target_label": "fitness tracker\ndeal seekers",
         "candidates": [(0.3, 0.7, "home gym\nequipment"), (0.5, 0.9, "fitness\nsubscriptions"),
                        (0.7, 0.5, "meal kit\nswitchers"), (0.1, 0.7, "running shoe\ncomparison")]},
        {"locus": (0.3, 0.7), "label": "Step 3: Clicked 'home gym equipment'",
         "target_label": "home gym\nequipment shoppers",
         "candidates": [(0.3, 0.9, "premium\nequipment"), (0.1, 0.7, "running shoe\ncomparison"),
                        (0.5, 0.5, "gym membership"), (0.7, 0.7, "meal prep\nbuyers")]},
    ]

    # Draw breadcrumb trail across all panels
    trail = [(0.5, 0.5), (0.5, 0.7), (0.3, 0.7)]

    for idx, (ax, step) in enumerate(zip(axes, steps)):
        ax.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", alpha=0.5, aspect="auto")
        draw_boundaries(ax, allocation, xs, ys, color="gray", linewidth=0.8)

        lx, ly = step["locus"]

        # Breadcrumb trail
        for i in range(idx + 1):
            bx, by = trail[i]
            if i < idx:
                # Past position
                ax.plot(bx, by, "o", color="#999", markersize=8, alpha=0.5, zorder=8)
            if i > 0:
                # Line connecting steps
                ax.plot([trail[i-1][0], trail[i][0]], [trail[i-1][1], trail[i][1]],
                        "--", color="#666", linewidth=1.5, alpha=0.5, zorder=7)

        # Current locus
        ax.plot(lx, ly, "o", color="#1a1a2e", markersize=14, zorder=12)
        ax.plot(lx, ly, "o", color="white", markersize=10, zorder=12)
        ax.plot(lx, ly, "o", color="#2196F3", markersize=6, zorder=13)

        # Target label
        ax.annotate(
            step["target_label"], (lx, ly),
            textcoords="offset points", xytext=(0, 18),
            ha="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd", edgecolor="#90caf9", alpha=0.95),
            zorder=14,
        )

        # Draw candidate arrows
        for cx, cy, clabel in step["candidates"]:
            ax.annotate(
                "", xy=(cx, cy), xytext=(lx, ly),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5, linestyle="--"),
                zorder=9,
            )
            ax.annotate(
                clabel, (cx, cy),
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=7, color="#555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.85),
                zorder=10,
            )

        ax.set_title(step["label"], fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Semantic Hill-Climbing: Navigating Embedding Space Through Language",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, "07_hill_climbing_sequence.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def diagram_3_restriction_zones():
    """Power diagram with gray exclusion areas for brand safety."""
    fig, ax = plt.subplots(figsize=(10, 8))

    xx, yy, xs, ys = make_grid(RESOLUTION)
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy)
    allocation = compute_allocation(values)
    rgb = make_colormap(DEFAULT_ADVERTISERS, allocation)

    ax.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", alpha=0.65, aspect="auto")
    draw_boundaries(ax, allocation, xs, ys)
    add_advertiser_markers(ax, DEFAULT_ADVERTISERS)

    # Restriction zones
    zones = [
        {"label": "Mental Health\n(Restricted)", "x0": 0.0, "y0": 0.82, "x1": 0.15, "y1": 1.0},
        {"label": "Political Content\n(Restricted)", "x0": 0.85, "y0": 0.0, "x1": 1.0, "y1": 0.18},
    ]

    for zone in zones:
        w = zone["x1"] - zone["x0"]
        h = zone["y1"] - zone["y0"]
        rect = Rectangle(
            (zone["x0"], zone["y0"]), w, h,
            facecolor="gray", alpha=0.6, zorder=6,
            hatch="///", edgecolor="darkgray", linewidth=2,
        )
        ax.add_patch(rect)
        cx = (zone["x0"] + zone["x1"]) / 2
        cy = (zone["y0"] + zone["y1"]) / 2
        ax.text(cx, cy, zone["label"], ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
                zorder=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Topic Dimension  (fitness ← → nutrition)", fontsize=12)
    ax.set_ylabel("Intent Dimension  (browsing ← → purchase-ready)", fontsize=12)
    ax.set_title("Brand Safety as Geometry: Restriction Zones in Embedding Space", fontsize=14)

    patches = [mpatches.Patch(color=adv.color, label=f"{adv.name}") for adv in DEFAULT_ADVERTISERS]
    patches.append(mpatches.Patch(facecolor="gray", hatch="///", label="Restricted Zone"))
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

    path = os.path.join(OUTPUT_DIR, "08_restriction_zones.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def diagram_4_competitive_heatmap():
    """Competitive intensity heatmap: where are the most contested regions?"""
    fig, ax = plt.subplots(figsize=(10, 8))

    xx, yy, xs, ys = make_grid(RESOLUTION)
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy)
    allocation = compute_allocation(values)

    # Competitive intensity: difference between best and second-best scores
    # Smaller difference = more competitive
    sorted_values = np.sort(values, axis=0)
    intensity = sorted_values[-1] - sorted_values[-2]  # gap between 1st and 2nd
    # Invert: high intensity = small gap = competitive
    max_gap = intensity.max()
    competitive = 1.0 - (intensity / max_gap)

    # Display competitive heatmap
    im = ax.imshow(competitive, extent=[0, 1, 0, 1], origin="lower",
                   cmap="YlOrRd", alpha=0.75, aspect="auto")
    draw_boundaries(ax, allocation, xs, ys, color="black", linewidth=1.5)
    add_advertiser_markers(ax, DEFAULT_ADVERTISERS)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("← Low Competition        High Competition →", fontsize=10)
    cbar.set_ticks([])

    # Overlay density contours
    density = compute_impression_density(DEFAULT_CLUSTERS, xx, yy)
    density_norm = density / density.max()
    ax.contour(xx, yy, density_norm, levels=5, colors="white", linewidths=0.8, alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Topic Dimension  (fitness ← → nutrition)", fontsize=12)
    ax.set_ylabel("Intent Dimension  (browsing ← → purchase-ready)", fontsize=12)
    ax.set_title("Competitive Heatmap: Where Territory Boundaries Are Most Contested",
                 fontsize=14, fontweight="bold")

    ax.text(0.02, 0.02,
            "White contours show impression density.\n"
            "Red = tight competition (similar bids).\n"
            "Yellow = one advertiser dominates.",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    path = os.path.join(OUTPUT_DIR, "09_competitive_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating UX blog diagrams...")
    diagram_1_keywords_vs_embeddings()
    diagram_2_hill_climbing()
    diagram_3_restriction_zones()
    diagram_4_competitive_heatmap()
    print("\nAll UX diagrams generated!")
