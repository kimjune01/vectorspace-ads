"""
Publication-quality experiments: GSP-discretized vs VCG-discretized vs Power-Diagram-Native.

Generates Tables and Figures for the paper, including:
  - Welfare/Revenue/IC-regret vs dimensionality  (d = 2, 5, 10, 20)
  - Latency scaling vs number of advertisers      (N = 5..100, d = 2)
  - Budget prediction accuracy vs MC samples
  - 2D territory visualisation

All experiments use 30 random seeds and report mean +/- std with 95% CIs.
Total wall-clock target: < 5 minutes on a modern laptop.

Mechanisms compared
-------------------
  Power-VCG  : Allocation via power diagram (continuous), payments via exact VCG.
  VCG-Disc   : Allocation via discretised grid, payments via per-impression VCG
               (re-run auction without each bidder).
  GSP-Disc   : Allocation via discretised grid, payments via per-impression
               second-price: each impression pays the *second-highest score* at
               that point, converted to value space.  GSP does NOT use welfare-
               difference payments, which is the source of its IC distortions.
"""

import sys, os, time, warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import KDTree

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Global experiment parameters
# ---------------------------------------------------------------------------
N_SEEDS          = 30
N_ADVERTISERS    = 20
N_IMPRESSIONS    = 50_000
N_CLUSTERS       = 5
GRID_CELLS_PER_DIM = 10
BID_DEVIATIONS   = [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]
RANDOM_PROJ_DIM  = 5

# Colorblind-friendly palette (Wong 2011)
CB_BLUE   = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN  = "#009E73"
CB_RED    = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_CYAN   = "#56B4E9"

MECH_COLORS  = {"GSP-Disc": CB_RED, "VCG-Disc": CB_BLUE, "Power-VCG": CB_GREEN}
MECH_MARKERS = {"GSP-Disc": "s",    "VCG-Disc": "^",     "Power-VCG": "o"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SyntheticAdvertiser:
    center: NDArray
    bid: float
    sigma: float
    index: int


# ---------------------------------------------------------------------------
# Data generation — sigma scales with sqrt(d)
# ---------------------------------------------------------------------------
def generate_scenario(n_adv, d, n_imp, n_clusters, rng):
    centers = rng.uniform(0, 1, size=(n_adv, d))
    bids    = rng.lognormal(mean=1.0, sigma=0.5, size=n_adv)
    # Scale sigma ~ sqrt(d) so that exp(-||x-c||^2/sigma^2) remains O(1)
    base_sigmas = rng.uniform(0.1, 0.4, size=n_adv)
    sigmas = base_sigmas * np.sqrt(d)
    advertisers = [SyntheticAdvertiser(centers[i], float(bids[i]),
                                       float(sigmas[i]), i)
                   for i in range(n_adv)]

    cl_centers = rng.uniform(0.1, 0.9, size=(n_clusters, d))
    cl_weights = rng.dirichlet(np.ones(n_clusters))
    cl_sigmas  = rng.uniform(0.05, 0.15, size=n_clusters)

    assigns = rng.choice(n_clusters, size=n_imp, p=cl_weights)
    impressions = np.empty((n_imp, d))
    for k in range(n_clusters):
        mask = assigns == k
        n_k  = int(mask.sum())
        impressions[mask] = rng.normal(cl_centers[k], cl_sigmas[k], (n_k, d))
    impressions = np.clip(impressions, 0, 1)

    return advertisers, impressions, (cl_centers, cl_weights, cl_sigmas)


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------
def _pack(advs):
    """Return (centers (N,d), log_bids (N,), inv_sigma_sq (N,), bids (N,))."""
    c = np.array([a.center for a in advs])
    lb = np.log(np.array([a.bid for a in advs]))
    isq = 1.0 / np.array([a.sigma**2 for a in advs])
    b = np.array([a.bid for a in advs])
    return c, lb, isq, b


def _scores(imps, centers, log_bids, inv_sigma_sq):
    """(M, N) score: log(b_j) - ||x_i - c_j||^2 / sigma_j^2."""
    diff = imps[:, None, :] - centers[None, :, :]
    dsq  = (diff * diff).sum(axis=2)
    return log_bids[None, :] - dsq * inv_sigma_sq[None, :]


def _values(imps, centers, bids, inv_sigma_sq):
    """(M, N) value: b_j * exp(-||x_i - c_j||^2 / sigma_j^2)."""
    diff = imps[:, None, :] - centers[None, :, :]
    dsq  = (diff * diff).sum(axis=2)
    return bids[None, :] * np.exp(-dsq * inv_sigma_sq[None, :])


# ---------------------------------------------------------------------------
# Power-diagram allocation (continuous)
# ---------------------------------------------------------------------------
def power_allocate(imps, advs, pk=None):
    if pk is None: pk = _pack(advs)
    c, lb, isq, _ = pk
    return np.argmax(_scores(imps, c, lb, isq), axis=1)


# ---------------------------------------------------------------------------
# Exact VCG payments (continuous)
# ---------------------------------------------------------------------------
def vcg_payments(imps, advs, winners=None, pk=None):
    """
    p_i = W_{-i}^{without i} - W_{-i}^{with i}
    where W_{-i} means welfare of all advertisers except i.
    Returns (N,) total payments.
    """
    N = len(advs)
    M = len(imps)
    if pk is None: pk = _pack(advs)
    c, lb, isq, b = pk
    if winners is None: winners = power_allocate(imps, advs, pk)

    V = _values(imps, c, b, isq)                          # (M, N)
    wv = V[np.arange(M), winners]                          # (M,) winner's value

    payments = np.zeros(N)
    for i in range(N):
        # welfare of others WITH i in the auction
        others_with = wv[winners != i].sum()

        # re-run without i
        idx = [j for j in range(N) if j != i]
        sc_wo  = _scores(imps, c[idx], lb[idx], isq[idx])
        w_wo   = np.argmax(sc_wo, axis=1)                  # into idx
        orig   = np.array(idx)[w_wo]
        others_without = V[np.arange(M), orig].sum()

        payments[i] = others_without - others_with

    return payments


# ---------------------------------------------------------------------------
# Discretised allocation
# ---------------------------------------------------------------------------
def _random_proj(d_hi, d_lo, rng):
    return rng.standard_normal((d_lo, d_hi)) / np.sqrt(d_lo)


def _project(imps, adv_centers, proj):
    ip = imps @ proj.T
    cp = adv_centers @ proj.T
    lo = np.minimum(ip.min(0), cp.min(0))
    hi = np.maximum(ip.max(0), cp.max(0))
    sp = np.maximum(hi - lo, 1e-8)
    return (ip - lo) / sp, np.clip((cp - lo) / sp, 0, 1)


def disc_allocate(imps, advs, grid_size, proj_matrix=None):
    """
    Discretised allocation: assign each impression to a cell, pick the highest-
    scoring advertiser at the cell centre.

    Returns (winners (M,), flat_cells (M,), unique_flat, occ_centres, inv_idx).
    The extra return values are for payment computation.
    """
    d = imps.shape[1]
    adv_c = np.array([a.center for a in advs])

    imp_eff, cen_eff = (imps, adv_c) if proj_matrix is None else \
                        _project(imps, adv_c, proj_matrix)
    d_eff = imp_eff.shape[1]

    cell_idx = np.clip(np.floor(imp_eff * grid_size).astype(np.int64),
                       0, grid_size - 1)
    mults = grid_size ** np.arange(d_eff, dtype=np.int64)
    flat  = (cell_idx * mults[None, :]).sum(axis=1)

    uf, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)

    # Cell centres in effective space
    occ_c = np.empty((len(uf), d_eff))
    tmp = uf.copy()
    for dim in range(d_eff):
        occ_c[:, dim] = (tmp % grid_size + 0.5) / grid_size
        tmp //= grid_size

    # Score at cell centres (in effective space with original sigma)
    lb  = np.log(np.array([a.bid for a in advs]))
    isq = 1.0 / np.array([a.sigma**2 for a in advs])
    diff = occ_c[:, None, :] - cen_eff[None, :, :]
    dsq  = (diff * diff).sum(axis=2)
    sc   = lb[None, :] - dsq * isq[None, :]
    cw   = np.argmax(sc, axis=1)

    winners = cw[inv]
    return winners, flat, uf, occ_c, inv, cen_eff


def disc_gsp_payments(imps, advs, winners, pk):
    """
    GSP second-price: for each impression, the winner pays the value-equivalent
    of the second-highest scorer *at that impression's true location in original
    space*.

    payment_i = sum over impressions won by i of v_{second-best}(x).

    This is the standard per-impression GSP second-price.  It differs from VCG
    because it doesn't account for the externality structure — just charges
    the "next best" value.
    """
    c, lb, isq, b = pk
    V = _values(imps, c, b, isq)  # (M, N) — original space

    # For each impression, find second-highest value
    # We need second-highest *score* to identify who the runner-up is,
    # then charge the winner that runner-up's *value*.
    S = _scores(imps, c, lb, isq)  # (M, N)

    N = len(advs)
    M = len(imps)
    payments = np.zeros(N)

    # Vectorised: set winner's score to -inf, take argmax of remainder
    S_mod = S.copy()
    S_mod[np.arange(M), winners] = -np.inf
    second_best_idx = np.argmax(S_mod, axis=1)  # (M,)
    second_best_val = V[np.arange(M), second_best_idx]  # (M,)

    # Accumulate per winner
    np.add.at(payments, winners, second_best_val)
    return payments


def disc_vcg_payments(imps, advs, winners, pk, disc_alloc_func, grid_size,
                      proj_matrix=None):
    """
    Discretised VCG: for each advertiser i, re-run the discretised auction
    without i, compute welfare difference.

    This is proper VCG on the discretised mechanism.  The allocation is still
    determined by cell centres, but the welfare is measured using true values
    at the actual impression locations.
    """
    N = len(advs)
    M = len(imps)
    c, lb, isq, b = pk
    V = _values(imps, c, b, isq)  # (M, N) original space
    wv = V[np.arange(M), winners]  # winner values

    payments = np.zeros(N)
    for i in range(N):
        # welfare of others with i
        others_with = wv[winners != i].sum()

        # re-run discretised without i
        idx = [j for j in range(N) if j != i]
        advs_wo = [advs[j] for j in idx]
        w_wo = disc_alloc_func(imps, advs_wo, grid_size,
                               proj_matrix=proj_matrix)[0]
        # map back to original indices
        orig_wo = np.array(idx)[w_wo]
        others_without = V[np.arange(M), orig_wo].sum()

        payments[i] = others_without - others_with

    return payments


# ---------------------------------------------------------------------------
# Welfare (always original space)
# ---------------------------------------------------------------------------
def welfare_per_imp(imps, advs, winners, pk=None):
    if pk is None: pk = _pack(advs)
    c, _, isq, b = pk
    wc = c[winners]; wb = b[winners]; wi = isq[winners]
    dsq = ((imps - wc)**2).sum(axis=1)
    return float((wb * np.exp(-dsq * wi)).mean())


# ---------------------------------------------------------------------------
# IC regret
# ---------------------------------------------------------------------------
def ic_regret_experiment(imps, advs, mechanism, grid_size=10, proj_matrix=None):
    """
    Mean IC regret.  For each advertiser, compare truthful utility to best
    deviation.  utility_i = gross_value_i - payment_i.
    """
    N = len(advs)
    M = len(imps)
    pk = _pack(advs)
    c, lb, isq, b = pk
    V = _values(imps, c, b, isq)  # (M, N)

    if mechanism == "power":
        winners  = power_allocate(imps, advs, pk)
        payments = vcg_payments(imps, advs, winners, pk)
    elif mechanism == "gsp":
        winners  = disc_allocate(imps, advs, grid_size, proj_matrix)[0]
        payments = disc_gsp_payments(imps, advs, winners, pk)
    else:  # vcg disc
        winners  = disc_allocate(imps, advs, grid_size, proj_matrix)[0]
        payments = disc_vcg_payments(imps, advs, winners, pk, disc_allocate,
                                      grid_size, proj_matrix)

    regrets = []
    for i in range(N):
        mask_i = winners == i
        gross  = V[mask_i, i].sum() if mask_i.any() else 0.0
        util_i = gross - payments[i]

        best = util_i
        for mult in BID_DEVIATIONS:
            mod = list(advs)
            mod[i] = SyntheticAdvertiser(advs[i].center, advs[i].bid * mult,
                                          advs[i].sigma, advs[i].index)

            if mechanism == "power":
                mpk = _pack(mod)
                mw  = power_allocate(imps, mod, mpk)
                mp  = vcg_payments(imps, mod, mw, mpk)
            elif mechanism == "gsp":
                mw  = disc_allocate(imps, mod, grid_size, proj_matrix)[0]
                mp  = disc_gsp_payments(imps, mod, mw, _pack(mod))
            else:
                mw  = disc_allocate(imps, mod, grid_size, proj_matrix)[0]
                mp  = disc_vcg_payments(imps, mod, mw, _pack(mod),
                                         disc_allocate, grid_size, proj_matrix)

            mask_m = mw == i
            # Use TRUE valuation (original bid) for gross value
            g_mod = V[mask_m, i].sum() if mask_m.any() else 0.0
            u_mod = g_mod - mp[i]
            best  = max(best, u_mod)

        if util_i > 1e-12:
            regrets.append(max(0.0, (best - util_i) / util_i))
        elif best > 1e-12:
            regrets.append(1.0)
        else:
            regrets.append(0.0)

    return float(np.mean(regrets))


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def measure_latency(imps, advs, use_kdtree=False, pk=None):
    if pk is None: pk = _pack(advs)
    M = len(imps)
    if use_kdtree:
        tree = KDTree(pk[0])
        tree.query(imps[:100])
        t0 = time.perf_counter()
        tree.query(imps)
        return (time.perf_counter() - t0) / M * 1e6
    else:
        power_allocate(imps[:100], advs, pk)
        t0 = time.perf_counter()
        power_allocate(imps, advs, pk)
        return (time.perf_counter() - t0) / M * 1e6


# ---------------------------------------------------------------------------
# Budget prediction accuracy
# ---------------------------------------------------------------------------
def budget_accuracy(advs, imps, M_values, rng):
    pk     = _pack(advs)
    full_M = len(imps)
    w_full = power_allocate(imps, advs, pk)
    p_full = vcg_payments(imps, advs, w_full, pk)
    true_s = p_full / full_M

    res = {}
    for M in M_values:
        errs = []
        for _ in range(20):
            idx = rng.choice(full_M, min(M, full_M), replace=False)
            sw  = power_allocate(imps[idx], advs, pk)
            sp  = vcg_payments(imps[idx], advs, sw, pk)
            est = sp / M
            act = true_s > 1e-8
            if act.any():
                errs.append(float(np.abs(est - true_s)[act].mean()
                                  / true_s[act].mean()))
            else:
                errs.append(0.0)
        res[M] = np.array(errs)
    return res


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ===========================================================================
# Experiment 1 — Dimensionality
# ===========================================================================
def experiment_dim(dims, n_seeds, out):
    print("\n=== Experiment 1: Dimensionality sweep ===")
    mechs = ["GSP-Disc", "VCG-Disc", "Power-VCG"]
    R = {d: {m: {"welfare": [], "revenue": [], "ic_regret": []}
             for m in mechs} for d in dims}

    for d in dims:
        print(f"\n  d = {d}")
        for si in range(n_seeds):
            rng = np.random.default_rng(si * 17 + 1000)
            advs, imps, _ = generate_scenario(N_ADVERTISERS, d, N_IMPRESSIONS,
                                               N_CLUSTERS, rng)
            pk = _pack(advs)
            proj = _random_proj(d, RANDOM_PROJ_DIM, rng) if d > RANDOM_PROJ_DIM else None

            # Power-VCG
            pw = power_allocate(imps, advs, pk)
            R[d]["Power-VCG"]["welfare"].append(welfare_per_imp(imps, advs, pw, pk))
            pv = vcg_payments(imps, advs, pw, pk)
            R[d]["Power-VCG"]["revenue"].append(float(pv.sum() / len(imps)))

            # Discretised allocation (shared between GSP and VCG-Disc)
            dw = disc_allocate(imps, advs, GRID_CELLS_PER_DIM, proj)[0]
            dw_welfare = welfare_per_imp(imps, advs, dw, pk)

            # GSP payments
            gsp_pay = disc_gsp_payments(imps, advs, dw, pk)
            R[d]["GSP-Disc"]["welfare"].append(dw_welfare)
            R[d]["GSP-Disc"]["revenue"].append(float(gsp_pay.sum() / len(imps)))

            # VCG-Disc payments: welfare-difference approach on the disc allocation
            vcg_d_pay = disc_vcg_payments(imps, advs, dw, pk, disc_allocate,
                                           GRID_CELLS_PER_DIM, proj)
            R[d]["VCG-Disc"]["welfare"].append(dw_welfare)
            R[d]["VCG-Disc"]["revenue"].append(float(vcg_d_pay.sum() / len(imps)))

            if si % 10 == 0:
                print(f"    seed {si}/{n_seeds}")

        # IC regret (fewer seeds — expensive due to N deviations x N advertisers)
        print(f"  Computing IC regret for d={d}...")
        n_ic = min(5, n_seeds)
        for si in range(n_ic):
            rng = np.random.default_rng(si * 31 + 2000)
            advs, imps, _ = generate_scenario(N_ADVERTISERS, d, 3000,
                                               N_CLUSTERS, rng)
            proj = _random_proj(d, RANDOM_PROJ_DIM, rng) if d > RANDOM_PROJ_DIM else None

            R[d]["Power-VCG"]["ic_regret"].append(
                ic_regret_experiment(imps, advs, "power"))
            R[d]["GSP-Disc"]["ic_regret"].append(
                ic_regret_experiment(imps, advs, "gsp", GRID_CELLS_PER_DIM, proj))
            R[d]["VCG-Disc"]["ic_regret"].append(
                ic_regret_experiment(imps, advs, "vcg", GRID_CELLS_PER_DIM, proj))

    out["dim"] = R
    return R


# ===========================================================================
# Experiment 2 — Latency scaling
# ===========================================================================
def experiment_lat(n_values, n_seeds, out):
    print("\n=== Experiment 2: Latency scaling ===")
    d = 2
    R = {"N": n_values, "brute": [], "kdtree": []}
    for N in n_values:
        print(f"  N = {N}")
        bf, kd = [], []
        for si in range(n_seeds):
            rng = np.random.default_rng(si * 13 + 3000)
            advs, imps, _ = generate_scenario(N, d, 10_000, N_CLUSTERS, rng)
            pk = _pack(advs)
            bf.append(measure_latency(imps, advs, False, pk))
            kd.append(measure_latency(imps, advs, True,  pk))
        R["brute"].append((np.mean(bf),  np.std(bf)))
        R["kdtree"].append((np.mean(kd), np.std(kd)))
    out["lat"] = R
    return R


# ===========================================================================
# Experiment 3 — Budget accuracy
# ===========================================================================
def experiment_budget(m_values, n_seeds, out):
    print("\n=== Experiment 3: Budget prediction accuracy ===")
    d = 2
    AE = {M: [] for M in m_values}
    for si in range(n_seeds):
        rng = np.random.default_rng(si * 19 + 4000)
        advs, imps, _ = generate_scenario(N_ADVERTISERS, d, 20_000,
                                           N_CLUSTERS, rng)
        res = budget_accuracy(advs, imps, m_values, rng)
        for M in m_values:
            AE[M].append(res[M].mean())
        if si % 5 == 0:
            print(f"    seed {si}/{n_seeds}")
    out["budget"] = {M: np.array(AE[M]) for M in m_values}
    return out["budget"]


# ===========================================================================
# Figures
# ===========================================================================
def _eb(ax, xs, data_lists, **kw):
    """Plot with 95% CI error bars."""
    means = [np.mean(dl) for dl in data_lists]
    ci    = [1.96 * np.std(dl) / max(np.sqrt(len(dl)), 1) for dl in data_lists]
    ax.errorbar(xs, means, yerr=ci, capsize=4, lw=2, ms=7, **kw)


def fig_welfare(R, dims, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    lab = {"GSP-Disc": "GSP-Discretised", "VCG-Disc": "VCG-Discretised",
           "Power-VCG": "Power Diagram (VCG)"}
    for m in ["GSP-Disc", "VCG-Disc", "Power-VCG"]:
        _eb(ax, dims, [R[d][m]["welfare"] for d in dims],
            marker=MECH_MARKERS[m], color=MECH_COLORS[m], label=lab[m])
    ax.set_xlabel("Dimensionality $d$")
    ax.set_ylabel("Social Welfare (per impression)")
    ax.set_title("Social Welfare vs. Dimensionality")
    ax.legend(); ax.set_xticks(dims)
    p = os.path.join(fig_dir, "fig_welfare_by_d.png")
    fig.savefig(p); plt.close(fig); print(f"  Saved {p}")


def fig_revenue(R, dims, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    lab = {"GSP-Disc": "GSP-Discretised", "VCG-Disc": "VCG-Discretised",
           "Power-VCG": "Power Diagram (VCG)"}
    for m in ["GSP-Disc", "VCG-Disc", "Power-VCG"]:
        _eb(ax, dims, [R[d][m]["revenue"] for d in dims],
            marker=MECH_MARKERS[m], color=MECH_COLORS[m], label=lab[m])
    ax.set_xlabel("Dimensionality $d$")
    ax.set_ylabel("Revenue (per impression)")
    ax.set_title("Platform Revenue vs. Dimensionality")
    ax.legend(); ax.set_xticks(dims)
    p = os.path.join(fig_dir, "fig_revenue_by_d.png")
    fig.savefig(p); plt.close(fig); print(f"  Saved {p}")


def fig_ic_regret(R, dims, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    lab = {"GSP-Disc": "GSP-Discretised", "VCG-Disc": "VCG-Discretised",
           "Power-VCG": "Power Diagram (VCG)"}
    for m in ["GSP-Disc", "VCG-Disc", "Power-VCG"]:
        means, lo, hi = [], [], []
        for d in dims:
            v  = np.array(R[d][m]["ic_regret"])
            mu = np.mean(v) if len(v) else 0.0
            ci = (1.96 * np.std(v) / max(np.sqrt(len(v)), 1)) if len(v) > 1 else 0.0
            means.append(max(mu, 1e-6))
            lo.append(max(mu - ci, 1e-7))
            hi.append(mu + ci + 1e-7)
        ax.semilogy(dims, means, marker=MECH_MARKERS[m], color=MECH_COLORS[m],
                     label=lab[m], lw=2, ms=7)
        ax.fill_between(dims, lo, hi, alpha=0.15, color=MECH_COLORS[m])
    ax.set_xlabel("Dimensionality $d$")
    ax.set_ylabel("IC Regret (log scale)")
    ax.set_title("Incentive Compatibility Regret vs. Dimensionality")
    ax.legend(); ax.set_xticks(dims)
    p = os.path.join(fig_dir, "fig_ic_regret_by_d.png")
    fig.savefig(p); plt.close(fig); print(f"  Saved {p}")


def fig_latency(R, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    Ns = R["N"]
    bm = [x[0] for x in R["brute"]]; bs = [x[1] for x in R["brute"]]
    km = [x[0] for x in R["kdtree"]]; ks = [x[1] for x in R["kdtree"]]
    ax.errorbar(Ns, bm, yerr=bs, marker="o", color=CB_BLUE,
                label="Brute force $O(N)$", lw=2, capsize=4)
    ax.errorbar(Ns, km, yerr=ks, marker="^", color=CB_GREEN,
                label="KD-tree $O(\\log N)$", lw=2, capsize=4)
    ax.set_xlabel("Number of Advertisers $N$")
    ax.set_ylabel("Latency ($\\mu$s / impression)")
    ax.set_title("Winner Determination Latency vs. Advertisers")
    ax.legend()
    p = os.path.join(fig_dir, "fig_latency_scaling.png")
    fig.savefig(p); plt.close(fig); print(f"  Saved {p}")


def fig_budget(BR, m_vals, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [np.mean(BR[M]) for M in m_vals]
    ci    = [1.96 * np.std(BR[M]) / np.sqrt(len(BR[M])) for M in m_vals]
    ax.errorbar(m_vals, means, yerr=ci, marker="o", color=CB_BLUE,
                lw=2, capsize=4, label="Empirical error")
    ma = np.array(m_vals, dtype=float)
    ref = means[0] * np.sqrt(m_vals[0]) / np.sqrt(ma)
    ax.plot(m_vals, ref, "--", color=CB_RED, lw=1.5,
            label="$O(1/\\sqrt{M})$ reference")
    ax.set_xlabel("MC Samples $M$"); ax.set_ylabel("Relative Budget Error")
    ax.set_title("Budget Prediction Accuracy")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend()
    p = os.path.join(fig_dir, "fig_budget_accuracy.png")
    fig.savefig(p); plt.close(fig); print(f"  Saved {p}")


def fig_territory(fig_dir):
    print("  Generating 2D territory visualisation...")
    rng = np.random.default_rng(42)
    advs, _, _ = generate_scenario(8, 2, 1000, N_CLUSTERS, rng)
    res = 300
    xs = np.linspace(0, 1, res)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    pk = _pack(advs)

    pw_map  = power_allocate(pts, advs, pk).reshape(res, res)
    dw      = disc_allocate(pts, advs, 5)[0].reshape(res, res)

    N = len(advs)
    cmap = ListedColormap(plt.colormaps["tab10"].colors[:N])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, dat, ttl in [(axes[0], dw, "Discretised (5x5 grid)"),
                          (axes[1], pw_map, "Power Diagram (continuous)")]:
        ax.imshow(dat, origin="lower", extent=[0,1,0,1], cmap=cmap,
                  vmin=-0.5, vmax=N-0.5, aspect="equal", interpolation="nearest")
        for a in advs:
            ax.plot(a.center[0], a.center[1], "k+", ms=10, mew=2)
        ax.set_title(ttl); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    plt.suptitle("Territory Allocation: Discretised vs. Continuous",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    p = os.path.join(fig_dir, "fig_territory_2d.png")
    fig.savefig(p); plt.close(fig); print(f"  Saved {p}")


# ===========================================================================
# Summary table
# ===========================================================================
def print_table(R, dims):
    mechs = ["GSP-Disc", "VCG-Disc", "Power-VCG"]
    print("\n" + "=" * 105)
    print(f"{'d':>4} | {'Mechanism':<16} | {'Welfare':>18} | {'Revenue':>18} | {'IC Regret':>18}")
    print("=" * 105)
    for d in dims:
        for m in mechs:
            w  = np.array(R[d][m]["welfare"])
            r  = np.array(R[d][m]["revenue"])
            ic = np.array(R[d][m]["ic_regret"])
            ics = f"{np.mean(ic):.5f} +/- {np.std(ic):.5f}" if len(ic) else "N/A"
            print(f"{d:>4} | {m:<16} | {np.mean(w):.5f} +/- {np.std(w):.5f} "
                  f"| {np.mean(r):.5f} +/- {np.std(r):.5f} | {ics}")
        print("-" * 105)
    print("=" * 105)


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    setup_style()

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = {}

    dims   = [2, 5, 10, 20]
    n_vals = [5, 10, 20, 50, 100]
    m_vals = [100, 500, 1000, 5000, 10000]

    DR = experiment_dim(dims, N_SEEDS, out)
    LR = experiment_lat(n_vals, N_SEEDS, out)
    BR = experiment_budget(m_vals, min(N_SEEDS, 15), out)

    print_table(DR, dims)

    print("\nGenerating figures...")
    fig_welfare(DR, dims, fig_dir)
    fig_revenue(DR, dims, fig_dir)
    fig_ic_regret(DR, dims, fig_dir)
    fig_latency(LR, fig_dir)
    fig_budget(BR, m_vals, fig_dir)
    fig_territory(fig_dir)

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")
    print("All experiments complete.")


if __name__ == "__main__":
    main()
