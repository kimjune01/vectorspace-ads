"""
Generate all blog visual assets.

Diagrams:
1. Keywords (discrete grid) vs Embeddings (continuous gradient) — side-by-side
2. 2D power diagram with 5 advertisers, color-coded territories, bid labels
3. Same diagram with one bid changed — showing territory shift
4. Anisotropic case — elliptical regions
5. Impression density heatmap overlaid on power diagram
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
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
    # Detect boundaries: where adjacent pixels have different owners
    h, w = allocation.shape
    boundary = np.zeros((h, w), dtype=bool)

    # Check horizontal and vertical neighbors
    boundary[:-1, :] |= allocation[:-1, :] != allocation[1:, :]
    boundary[:, :-1] |= allocation[:, :-1] != allocation[:, 1:]

    # Plot boundary points
    by, bx = np.where(boundary)
    ax.scatter(xs[bx], ys[by], c=color, s=0.15, alpha=0.6, zorder=5)


def diagram_1_keywords_vs_embeddings():
    """Side-by-side: keyword auctions (discrete) vs embedding auctions (continuous)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Keywords as discrete grid
    np.random.seed(42)
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

    # Draw grid lines to emphasize discreteness
    for i in np.arange(0.05, 1.0, 0.1):
        ax1.axhline(i, color="#E0E0E0", linewidth=0.5, linestyle="--", alpha=0.5)
        ax1.axvline(i, color="#E0E0E0", linewidth=0.5, linestyle="--", alpha=0.5)

    ax1.set_title("Keyword Auctions: Discrete Biddable Units")
    ax1.set_xlabel("Keyword Space")
    ax1.set_ylabel("")
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

    ax2.set_title("Embedding Auctions: Continuous Territory")
    ax2.set_xlabel("Embedding Space")
    ax2.set_ylabel("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.5, 0.02, "Advertisers bid on regions, not keywords",
             ha="center", fontsize=9, style="italic", transform=ax2.transAxes)

    fig.suptitle("From Keywords to Embeddings: The Auction Design Shift",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "01_keywords_vs_embeddings.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def diagram_2_power_diagram():
    """2D power diagram with 5 advertisers, color-coded territories."""
    fig, ax = plt.subplots(figsize=(10, 8))

    xx, yy, xs, ys = make_grid(RESOLUTION)
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy)
    allocation = compute_allocation(values)
    rgb = make_colormap(DEFAULT_ADVERTISERS, allocation)

    ax.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", alpha=0.65, aspect="auto")
    draw_boundaries(ax, allocation, xs, ys)
    add_advertiser_markers(ax, DEFAULT_ADVERTISERS)

    # Legend
    patches = [mpatches.Patch(color=adv.color, label=f"{adv.name} (bid=${adv.bid:.1f}, σ={adv.sigma})")
               for adv in DEFAULT_ADVERTISERS]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Topic Dimension  (fitness ← → nutrition)", fontsize=12)
    ax.set_ylabel("Intent Dimension  (browsing ← → purchase-ready)", fontsize=12)
    ax.set_title("Power Diagram: How Bids Create Territory in Embedding Space", fontsize=14)

    path = os.path.join(OUTPUT_DIR, "02_power_diagram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def diagram_3_bid_change():
    """Same diagram with Nike's bid increased — showing territory shift."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    xx, yy, xs, ys = make_grid(RESOLUTION)

    # Original
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy)
    alloc = compute_allocation(values)
    rgb = make_colormap(DEFAULT_ADVERTISERS, alloc)
    ax1.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", alpha=0.65, aspect="auto")
    draw_boundaries(ax1, alloc, xs, ys)
    add_advertiser_markers(ax1, DEFAULT_ADVERTISERS, fontsize=8)
    ax1.set_title("Before: Nike bids $5.0", fontsize=13)
    ax1.set_xlabel("Topic")
    ax1.set_ylabel("Intent")

    # Modified: Nike raises bid to $10
    modified = [Advertiser(a.name, a.center, a.bid, a.sigma, a.color) for a in DEFAULT_ADVERTISERS]
    modified[0] = Advertiser("Nike", (0.6, 0.3), bid=10.0, sigma=0.3, color="#FF6B35")

    values2 = compute_value_functions(modified, xx, yy)
    alloc2 = compute_allocation(values2)
    rgb2 = make_colormap(modified, alloc2)
    ax2.imshow(rgb2, extent=[0, 1, 0, 1], origin="lower", alpha=0.65, aspect="auto")
    draw_boundaries(ax2, alloc2, xs, ys)
    add_advertiser_markers(ax2, modified, fontsize=8)
    ax2.set_title("After: Nike raises bid to $10.0", fontsize=13)
    ax2.set_xlabel("Topic")
    ax2.set_ylabel("Intent")

    # Compute area change
    metrics1 = compute_metrics(DEFAULT_ADVERTISERS, alloc, compute_impression_density(DEFAULT_CLUSTERS, xx, yy), values)
    metrics2 = compute_metrics(modified, alloc2, compute_impression_density(DEFAULT_CLUSTERS, xx, yy), values2)
    area1 = metrics1["per_advertiser"][0]["territory_area"]
    area2 = metrics2["per_advertiser"][0]["territory_area"]

    fig.suptitle(
        f"Higher Bid → Bigger Territory: Nike's area {area1:.1%} → {area2:.1%}",
        fontsize=15, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "03_bid_change.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def diagram_4_anisotropic():
    """Anisotropic case — elliptical regions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    xx, yy, xs, ys = make_grid(RESOLUTION)

    # Isotropic
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy, anisotropic=False)
    alloc = compute_allocation(values)
    rgb = make_colormap(DEFAULT_ADVERTISERS, alloc)
    ax1.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", alpha=0.65, aspect="auto")
    draw_boundaries(ax1, alloc, xs, ys)
    add_advertiser_markers(ax1, DEFAULT_ADVERTISERS, fontsize=8)
    ax1.set_title("Isotropic: Equal reach in all directions", fontsize=13)
    ax1.set_xlabel("Topic")
    ax1.set_ylabel("Intent")

    # Anisotropic: give advertisers different x/y reaches
    aniso_advertisers = [
        Advertiser("Nike", (0.6, 0.3), 5.0, 0.3, "#FF6B35", sigma_x=0.45, sigma_y=0.15),
        Advertiser("Whole Foods", (0.3, 0.7), 3.0, 0.25, "#4CAF50", sigma_x=0.15, sigma_y=0.4),
        Advertiser("Peloton", (0.5, 0.5), 4.0, 0.2, "#2196F3", sigma_x=0.3, sigma_y=0.3),
        Advertiser("GNC", (0.7, 0.7), 2.5, 0.35, "#9C27B0", sigma_x=0.2, sigma_y=0.5),
        Advertiser("Fitbit", (0.4, 0.3), 3.5, 0.25, "#FF9800", sigma_x=0.35, sigma_y=0.15),
    ]

    values2 = compute_value_functions(aniso_advertisers, xx, yy, anisotropic=True)
    alloc2 = compute_allocation(values2)
    rgb2 = make_colormap(aniso_advertisers, alloc2)
    ax2.imshow(rgb2, extent=[0, 1, 0, 1], origin="lower", alpha=0.65, aspect="auto")
    draw_boundaries(ax2, alloc2, xs, ys)
    add_advertiser_markers(ax2, aniso_advertisers, fontsize=8)

    # Draw ellipses to show directional reach
    for adv in aniso_advertisers:
        ellipse = plt.matplotlib.patches.Ellipse(
            adv.center, width=adv.effective_sigma_x * 2, height=adv.effective_sigma_y * 2,
            fill=False, edgecolor=adv.color, linewidth=2, linestyle="--", alpha=0.8, zorder=8,
        )
        ax2.add_patch(ellipse)

    ax2.set_title("Anisotropic: Directional reach (ellipses)", fontsize=13)
    ax2.set_xlabel("Topic")
    ax2.set_ylabel("Intent")

    fig.suptitle("Isotropic vs. Anisotropic: Advertisers Can Specialize Along Dimensions",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "04_anisotropic.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def diagram_5_density_overlay():
    """Impression density heatmap overlaid on power diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))

    xx, yy, xs, ys = make_grid(RESOLUTION)
    values = compute_value_functions(DEFAULT_ADVERTISERS, xx, yy)
    allocation = compute_allocation(values)
    density = compute_impression_density(DEFAULT_CLUSTERS, xx, yy)
    rgb = make_colormap(DEFAULT_ADVERTISERS, allocation)

    # Show territory as base layer
    ax.imshow(rgb, extent=[0, 1, 0, 1], origin="lower", alpha=0.4, aspect="auto")

    # Overlay density as contour
    density_display = density / density.max()  # normalize for display
    contour = ax.contourf(xx, yy, density_display, levels=15, cmap="hot", alpha=0.5)
    ax.contour(xx, yy, density_display, levels=8, colors="black", linewidths=0.5, alpha=0.4)

    draw_boundaries(ax, allocation, xs, ys, color="white", linewidth=2)
    add_advertiser_markers(ax, DEFAULT_ADVERTISERS)

    # Colorbar for density
    cbar = fig.colorbar(contour, ax=ax, shrink=0.8, label="Impression Density")
    cbar.set_ticks([])
    cbar.set_label("← Low Traffic        High Traffic →", fontsize=10)

    # Mark cluster centers
    for j, cl in enumerate(DEFAULT_CLUSTERS):
        ax.plot(*cl.center, "x", color="white", markersize=12, markeredgewidth=2.5, zorder=12)
        labels = ["General\nFitness", "Nutrition\nFocused", "Purchase\nReady"]
        ax.annotate(
            labels[j], cl.center, textcoords="offset points", xytext=(12, -8),
            fontsize=8, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            zorder=13,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Topic Dimension  (fitness ← → nutrition)", fontsize=12)
    ax.set_ylabel("Intent Dimension  (browsing ← → purchase-ready)", fontsize=12)
    ax.set_title("Territory + Impression Density: Not All Territory Is Equally Valuable",
                 fontsize=14, fontweight="bold")

    # Compute and show weighted metrics
    metrics = compute_metrics(DEFAULT_ADVERTISERS, allocation, density, values)
    info_text = "Impression Share:\n"
    for m in metrics["per_advertiser"]:
        info_text += f"  {m['name']}: {m['impressions']:.1%}\n"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    path = os.path.join(OUTPUT_DIR, "05_density_overlay.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating blog diagrams...")
    diagram_1_keywords_vs_embeddings()
    diagram_2_power_diagram()
    diagram_3_bid_change()
    diagram_4_anisotropic()
    diagram_5_density_overlay()
    print("\nAll diagrams generated!")
