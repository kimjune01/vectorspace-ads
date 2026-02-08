"""
Core computation for embedding-based ad auctions via power diagrams.

The key insight: in continuous embedding space, the natural allocation mechanism
is a power diagram (additively weighted Voronoi tessellation). For isotropic
Gaussian value functions v_i(x) = b_i * exp(-||x - c_i||^2 / sigma_i^2),
the winner at point x is:

    argmax_i [log(b_i) - ||x - c_i||^2 / sigma_i^2]

This module provides grid-based evaluation for visualization and metrics.
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class Advertiser:
    """An advertiser with a position, bid, and reach in embedding space."""
    name: str
    center: tuple[float, float]  # (x, y) position in [0, 1]^2
    bid: float                    # bid scalar b_i > 0
    sigma: float                  # isotropic reach parameter
    color: str                    # hex color for rendering
    # Anisotropic parameters (optional)
    sigma_x: float | None = None
    sigma_y: float | None = None

    @property
    def effective_sigma_x(self) -> float:
        return self.sigma_x if self.sigma_x is not None else self.sigma

    @property
    def effective_sigma_y(self) -> float:
        return self.sigma_y if self.sigma_y is not None else self.sigma


@dataclass
class ImpressionCluster:
    """A cluster in the impression density Gaussian mixture."""
    center: tuple[float, float]
    weight: float
    sigma: float


# Default advertisers from the handoff spec
DEFAULT_ADVERTISERS = [
    Advertiser("Nike", (0.6, 0.3), bid=5.0, sigma=0.3, color="#FF6B35"),
    Advertiser("Whole Foods", (0.3, 0.7), bid=3.0, sigma=0.25, color="#4CAF50"),
    Advertiser("Peloton", (0.5, 0.5), bid=4.0, sigma=0.2, color="#2196F3"),
    Advertiser("GNC", (0.7, 0.7), bid=2.5, sigma=0.35, color="#9C27B0"),
    Advertiser("Fitbit", (0.4, 0.3), bid=3.5, sigma=0.25, color="#FF9800"),
]

# Default impression density clusters
DEFAULT_CLUSTERS = [
    ImpressionCluster((0.5, 0.4), weight=0.4, sigma=0.15),  # general fitness intent
    ImpressionCluster((0.3, 0.6), weight=0.3, sigma=0.1),   # nutrition-focused
    ImpressionCluster((0.7, 0.5), weight=0.3, sigma=0.12),  # purchase-ready fitness
]


def make_grid(resolution: int = 500) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Create a 2D evaluation grid over [0, 1]^2.

    Returns:
        xx, yy: meshgrid arrays (resolution x resolution)
        xs, ys: 1D coordinate arrays (resolution,)
    """
    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, xs, ys


def compute_value_functions(
    advertisers: list[Advertiser],
    xx: NDArray,
    yy: NDArray,
    anisotropic: bool = False,
) -> NDArray:
    """Compute log-value function for each advertiser on the grid.

    For isotropic case:
        score_i(x) = log(b_i) - ||x - c_i||^2 / sigma_i^2

    For anisotropic case:
        score_i(x) = log(b_i) - (x - cx)^2/sx^2 - (y - cy)^2/sy^2

    Returns:
        values: (N, H, W) array of scores for each advertiser
    """
    n = len(advertisers)
    h, w = xx.shape
    values = np.full((n, h, w), -np.inf)

    for i, adv in enumerate(advertisers):
        cx, cy = adv.center
        dx = xx - cx
        dy = yy - cy

        if anisotropic:
            sx = adv.effective_sigma_x
            sy = adv.effective_sigma_y
            values[i] = np.log(adv.bid) - dx**2 / sx**2 - dy**2 / sy**2
        else:
            values[i] = np.log(adv.bid) - (dx**2 + dy**2) / adv.sigma**2

    return values


def compute_allocation(values: NDArray) -> NDArray:
    """Compute the power diagram allocation (winner at each grid point).

    Returns:
        allocation: (H, W) array of winner indices (0-indexed)
    """
    return np.argmax(values, axis=0)


def compute_impression_density(
    clusters: list[ImpressionCluster],
    xx: NDArray,
    yy: NDArray,
) -> NDArray:
    """Compute impression density as a Gaussian mixture.

    Returns:
        density: (H, W) array, normalized to integrate to 1 over the grid
    """
    density = np.zeros_like(xx)
    for cluster in clusters:
        cx, cy = cluster.center
        dx = xx - cx
        dy = yy - cy
        density += cluster.weight * np.exp(-(dx**2 + dy**2) / (2 * cluster.sigma**2))

    # Normalize so it integrates to 1 over the grid
    total = density.sum()
    if total > 0:
        density /= total

    return density


def compute_metrics(
    advertisers: list[Advertiser],
    allocation: NDArray,
    density: NDArray,
    values: NDArray,
) -> dict:
    """Compute auction metrics for each advertiser and the platform.

    Returns dict with:
        - per_advertiser: list of dicts with impressions, territory_area, spend, cpi
        - platform_revenue: total platform revenue
        - social_welfare: total welfare
    """
    n = len(advertisers)
    h, w = allocation.shape
    pixel_area = 1.0 / (h * w)  # area of each pixel in [0,1]^2

    results = []
    total_revenue = 0.0
    total_welfare = 0.0

    for i, adv in enumerate(advertisers):
        mask = allocation == i
        territory_pixels = mask.sum()
        territory_area = territory_pixels * pixel_area

        # Weighted impressions (density-weighted)
        impressions = density[mask].sum()

        # For VCG-like pricing, the payment is approximately the second-highest
        # value at each point. For simplicity, we use a CPM-like estimate:
        # spend = bid * impressions (simplified)
        spend = adv.bid * impressions

        cpi = spend / impressions if impressions > 0 else 0.0

        results.append({
            "name": adv.name,
            "impressions": float(impressions),
            "territory_area": float(territory_area),
            "spend": float(spend),
            "cpi": float(cpi),
        })

        total_revenue += spend
        total_welfare += impressions * adv.bid

    return {
        "per_advertiser": results,
        "platform_revenue": float(total_revenue),
        "social_welfare": float(total_welfare),
    }


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color string to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def make_colormap(advertisers: list[Advertiser], allocation: NDArray) -> NDArray:
    """Create an RGB image from the allocation map.

    Returns:
        rgb: (H, W, 3) array with advertiser colors
    """
    h, w = allocation.shape
    rgb = np.zeros((h, w, 3))

    for i, adv in enumerate(advertisers):
        mask = allocation == i
        r, g, b = hex_to_rgb(adv.color)
        rgb[mask] = [r, g, b]

    return rgb
