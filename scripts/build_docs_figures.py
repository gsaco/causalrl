"""Build documentation figures for the CausalRL website."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import sys  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from crl.assumptions import AssumptionSet  # noqa: E402
from crl.assumptions_catalog import (  # noqa: E402
    MARKOV,
    OVERLAP,
    SEQUENTIAL_IGNORABILITY,
    BEHAVIOR_POLICY_KNOWN,
)
from crl.benchmarks.bandit_synth import (  # noqa: E402
    SyntheticBandit,
    SyntheticBanditConfig,
)
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig  # noqa: E402
from crl.diagnostics.ess import effective_sample_size  # noqa: E402
from crl.estimands.policy_value import PolicyValueEstimand  # noqa: E402
from crl.estimators.dr import (  # noqa: E402
    DoublyRobustEstimator,
    DRCrossFitConfig,
)
from crl.estimators.importance_sampling import (  # noqa: E402
    ISEstimator,
    PDISEstimator,
    WISEstimator,
)
from crl.estimators.utils import compute_action_probs  # noqa: E402
from crl.sensitivity.bandits import sensitivity_bounds  # noqa: E402
from crl.utils.seeding import set_seed  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from crl_plotting import apply_style, save_figure_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets" / "figures"

PALETTE = {
    "primary": "0.15",
    "secondary": "0.35",
    "accent": "0.55",
    "muted": "0.65",
    "light": "0.9",
}


def _plot_estimator_comparison(
    labels: list[str],
    values: np.ndarray,
    ci: list[tuple[float, float] | None],
    *,
    truth: float | None,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(labels))

    ax.scatter(values, y, color=PALETTE["primary"], s=45, zorder=3)
    for idx, bounds in enumerate(ci):
        if bounds is None:
            continue
        ax.errorbar(
            values[idx],
            y[idx],
            xerr=[[values[idx] - bounds[0]], [bounds[1] - values[idx]]],
            fmt="none",
            ecolor=PALETTE["muted"],
            elinewidth=1.6,
            capsize=3,
            zorder=2,
        )
    if truth is not None:
        ax.axvline(truth, color=PALETTE["accent"], linestyle="--", linewidth=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Estimated policy value")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def _plot_histogram(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    logy: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(values, bins=40, color=PALETTE["primary"], alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    return fig


def _plot_ess_by_time(ess: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(
        np.arange(ess.size),
        ess,
        marker="o",
        color=PALETTE["secondary"],
        linewidth=2.0,
    )
    ax.set_title("Effective sample size by time step")
    ax.set_xlabel("Time step")
    ax.set_ylabel("ESS")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    return fig


def _plot_sensitivity(
    gammas: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(gammas, lower, color=PALETTE["primary"], label="Lower")
    ax.plot(gammas, upper, color=PALETTE["accent"], linestyle="--", label="Upper")
    ax.fill_between(gammas, lower, upper, color=PALETTE["light"], alpha=1.0)
    ax.set_title("Sensitivity bounds (bandit)")
    ax.set_xlabel("Sensitivity parameter gamma")
    ax.set_ylabel("Policy value")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def build_bandit_figures() -> None:
    bench = SyntheticBandit(
        SyntheticBanditConfig(seed=0, num_contexts=6, num_actions=4)
    )
    data = bench.sample(num_samples=1200, seed=1)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN]),
    )
    estimators = [
        ISEstimator(estimand),
        WISEstimator(estimand),
    ]

    labels = []
    values = []
    ci = []
    for estimator in estimators:
        report = estimator.estimate(data)
        labels.append(report.metadata.get("estimator", type(estimator).__name__))
        values.append(report.value)
        ci.append(report.ci)

    fig = _plot_estimator_comparison(
        labels,
        np.asarray(values),
        ci,
        truth=bench.true_policy_value(bench.target_policy),
        title="Bandit OPE estimates with uncertainty",
    )
    save_figure_bundle(fig, OUT_DIR, "bandit_estimator_comparison")
    plt.close(fig)

    target_probs = bench.target_policy.action_prob(data.contexts, data.actions)
    ratios = target_probs / data.behavior_action_probs

    fig = _plot_histogram(
        ratios,
        title="Overlap diagnostics (target / behavior)",
        xlabel="Importance ratio",
        ylabel="Count",
    )
    save_figure_bundle(fig, OUT_DIR, "bandit_overlap_ratios")
    plt.close(fig)

    fig = _plot_histogram(
        ratios * data.rewards,
        title="Importance-weighted rewards (log scale)",
        xlabel="Weighted reward",
        ylabel="Count",
        logy=True,
    )
    save_figure_bundle(fig, OUT_DIR, "bandit_weighted_rewards")
    plt.close(fig)

    gammas = np.linspace(1.0, 3.0, 9)
    bounds = sensitivity_bounds(data, bench.target_policy, gammas)
    fig = _plot_sensitivity(bounds.gammas, bounds.lower, bounds.upper)
    save_figure_bundle(fig, OUT_DIR, "bandit_sensitivity_bounds")
    plt.close(fig)


def build_mdp_figures() -> None:
    bench = SyntheticMDP(
        SyntheticMDPConfig(seed=1, num_states=6, num_actions=3, horizon=5)
    )
    data = bench.sample(num_trajectories=200, seed=2)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=data.discount,
        horizon=data.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, MARKOV]),
    )
    estimators = [
        ISEstimator(estimand),
        WISEstimator(estimand),
        PDISEstimator(estimand),
        DoublyRobustEstimator(
            estimand, config=DRCrossFitConfig(num_folds=2, num_iterations=3, seed=0)
        ),
    ]

    labels = []
    values = []
    ci = []
    for estimator in estimators:
        report = estimator.estimate(data)
        labels.append(report.metadata.get("estimator", type(estimator).__name__))
        values.append(report.value)
        ci.append(report.ci)

    fig = _plot_estimator_comparison(
        labels,
        np.asarray(values),
        ci,
        truth=bench.true_policy_value(bench.target_policy),
        title="MDP OPE estimates with uncertainty",
    )
    save_figure_bundle(fig, OUT_DIR, "mdp_estimator_comparison")
    plt.close(fig)

    target_probs = compute_action_probs(
        bench.target_policy, data.observations, data.actions
    )
    ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
    cum_weights = np.cumprod(ratios, axis=1)
    ess = np.array(
        [effective_sample_size(cum_weights[:, t]) for t in range(data.horizon)]
    )

    fig = _plot_ess_by_time(ess)
    save_figure_bundle(fig, OUT_DIR, "mdp_ess_by_time")
    plt.close(fig)


def main() -> None:
    set_seed(0)
    apply_style()
    build_bandit_figures()
    build_mdp_figures()


if __name__ == "__main__":
    main()
