from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import (
    BOUNDED_REWARDS,
    MARKOV,
    OVERLAP,
    SEQUENTIAL_IGNORABILITY,
)
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.confounded_bandit import ConfoundedBandit, ConfoundedBanditConfig
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.confounding.proximal import ProximalBanditEstimator
from crl.diagnostics.plots import plot_ratio_histogram, plot_weight_histogram
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.bootstrap import BootstrapConfig
from crl.estimators.fqe import FQEConfig, FQEEstimator
from crl.estimators.high_confidence import HighConfidenceISEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator
from crl.estimators.mis import MarginalizedImportanceSamplingEstimator
from crl.estimators.utils import compute_action_probs
from crl.ope import evaluate
from crl.sensitivity.bandits import sensitivity_bounds
from crl.viz import save_figure
from crl.viz.plots import (
    plot_effective_sample_size,
    plot_estimator_comparison,
    plot_overlap_diagnostics,
    plot_sensitivity_curve,
)


def _weights_histograms(outdir: Path, n_samples: int, seed: int) -> None:
    bench = SyntheticBandit(SyntheticBanditConfig(seed=7))
    data = bench.sample(num_samples=n_samples, seed=seed)

    target_probs = bench.target_policy.action_prob(data.contexts, data.actions)
    ratios = target_probs / data.behavior_action_probs

    fig_w = plot_weight_histogram(
        ratios, bins=40, xlabel=r"$\hat{w}$", title=None
    )
    save_figure(fig_w, outdir / "weights_hist")

    fig_r = plot_ratio_histogram(
        ratios, bins=40, xlabel=r"$\hat{\nu}$", title=None
    )
    save_figure(fig_r, outdir / "ratio_hist")


def _quickstart_bandit(outdir: Path) -> None:
    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = benchmark.sample(num_samples=1_000, seed=1)
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    report = evaluate(
        dataset=dataset,
        policy=benchmark.target_policy,
        estimators=["is", "wis", "double_rl"],
        seed=0,
    )

    fig = report.plot_estimator_comparison(truth=true_value)
    save_figure(fig, outdir / "quickstart_bandit_estimator_comparison")

    weights = (
        benchmark.target_policy.action_prob(dataset.contexts, dataset.actions)
        / dataset.behavior_action_probs
    )
    fig_w = report.plot_importance_weights(weights, logy=True)
    save_figure(fig_w, outdir / "quickstart_bandit_weights")


def _quickstart_mdp(outdir: Path) -> None:
    benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
    dataset = benchmark.sample(num_trajectories=300, seed=1)
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    report = evaluate(
        dataset=dataset,
        policy=benchmark.target_policy,
        estimators=["is", "wis", "pdis", "dr", "wdr", "mrdr", "fqe"],
        seed=0,
    )

    fig = report.plot_estimator_comparison(truth=true_value)
    save_figure(fig, outdir / "quickstart_mdp_estimator_comparison")


def _diagnostics_overlap_ess(outdir: Path) -> None:
    benchmark = SyntheticMDP(SyntheticMDPConfig(seed=3, horizon=6))
    dataset = benchmark.sample(num_trajectories=400, seed=4)

    target_probs = compute_action_probs(
        benchmark.target_policy, dataset.observations, dataset.actions
    )
    ratios = np.where(
        dataset.mask, target_probs / dataset.behavior_action_probs, 1.0
    )

    fig_overlap = plot_overlap_diagnostics(
        target_probs, dataset.behavior_action_probs, mask=dataset.mask
    )
    save_figure(fig_overlap, outdir / "diagnostics_overlap")

    fig_ess = plot_effective_sample_size(ratios, by_time=True)
    save_figure(fig_ess, outdir / "diagnostics_ess")


def _confidence_intervals(outdir: Path) -> None:
    bandit = SyntheticBandit(SyntheticBanditConfig(seed=20))
    bandit_data = bandit.sample(num_samples=1_000, seed=21)

    bandit_estimand = PolicyValueEstimand(
        policy=bandit.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet(
            [SEQUENTIAL_IGNORABILITY, OVERLAP, BOUNDED_REWARDS]
        ),
    )

    is_report = ISEstimator(bandit_estimand).estimate(bandit_data)
    hcope_report = HighConfidenceISEstimator(bandit_estimand).estimate(bandit_data)

    bandit_rows = [
        {"estimator": "IS", "value": is_report.value, "ci": is_report.ci},
        {"estimator": "HCOPE", "value": hcope_report.value, "ci": hcope_report.ci},
    ]
    fig_bandit = plot_estimator_comparison(bandit_rows)
    save_figure(fig_bandit, outdir / "ci_bandit_hcope")

    mdp = SyntheticMDP(SyntheticMDPConfig(seed=30, horizon=4))
    mdp_data = mdp.sample(num_trajectories=60, seed=31)

    mdp_estimand = PolicyValueEstimand(
        policy=mdp.target_policy,
        discount=mdp_data.discount,
        horizon=mdp_data.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    fqe_config = FQEConfig(
        num_epochs=2,
        num_iterations=2,
        bootstrap=True,
        bootstrap_config=BootstrapConfig(num_bootstrap=20, method="trajectory", seed=2),
    )

    fqe_report = FQEEstimator(mdp_estimand, config=fqe_config).estimate(mdp_data)

    mdp_rows = [
        {
            "estimator": "FQE (bootstrap)",
            "value": fqe_report.value,
            "ci": fqe_report.ci,
        }
    ]
    fig_mdp = plot_estimator_comparison(mdp_rows)
    save_figure(fig_mdp, outdir / "ci_mdp_fqe_bootstrap")


def _sensitivity_bandits(outdir: Path) -> None:
    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=10))
    dataset = benchmark.sample(num_samples=1_500, seed=11)

    gammas = np.linspace(1.0, 3.0, 15)
    bounds = sensitivity_bounds(dataset, benchmark.target_policy, gammas)

    rows = [
        {"gamma": g, "lower": lo, "upper": up}
        for g, lo, up in zip(bounds.gammas, bounds.lower, bounds.upper)
    ]
    fig = plot_sensitivity_curve(rows)
    save_figure(fig, outdir / "sensitivity_bandits_curve")


def _long_horizon_mis_vs_is(outdir: Path) -> None:
    benchmark = SyntheticMDP(SyntheticMDPConfig(seed=40, horizon=10))
    dataset = benchmark.sample(num_trajectories=300, seed=41)
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    estimators = [
        ISEstimator(estimand),
        PDISEstimator(estimand),
        MarginalizedImportanceSamplingEstimator(estimand),
    ]

    rows = []
    for estimator in estimators:
        report = estimator.estimate(dataset)
        rows.append(
            {
                "estimator": report.metadata["estimator"],
                "value": report.value,
                "ci": report.ci,
            }
        )

    fig = plot_estimator_comparison(rows, truth=true_value)
    save_figure(fig, outdir / "long_horizon_mis_vs_is")


def _proximal_confounded(outdir: Path) -> None:
    benchmark = ConfoundedBandit(ConfoundedBanditConfig(seed=7))
    prox_data = benchmark.sample(num_samples=2_000, seed=8)
    logged_data = prox_data.to_logged_dataset()
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    is_report = ISEstimator(estimand).estimate(logged_data)
    prox_report = ProximalBanditEstimator(benchmark.target_policy).estimate(prox_data)

    rows = [
        {"estimator": "IS", "value": is_report.value, "ci": is_report.ci},
        {"estimator": "Proximal", "value": prox_report, "ci": None},
    ]
    fig = plot_estimator_comparison(rows, truth=true_value)
    save_figure(fig, outdir / "proximal_confounded_bandit")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate journal-ready figures for docs/papers."
    )
    parser.add_argument("--outdir", type=str, default="docs/assets/figures")
    parser.add_argument("--n", type=int, default=1000, help="Samples for histograms")
    parser.add_argument("--seed", type=int, default=11, help="Seed for histograms")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except Exception:
        pass

    _weights_histograms(outdir, n_samples=args.n, seed=args.seed)
    _quickstart_bandit(outdir)
    _quickstart_mdp(outdir)
    _diagnostics_overlap_ess(outdir)
    _confidence_intervals(outdir)
    _sensitivity_bandits(outdir)
    _long_horizon_mis_vs_is(outdir)
    _proximal_confounded(outdir)


if __name__ == "__main__":
    main()
