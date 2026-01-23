from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.ope import evaluate


def test_evaluate_bandit_default():
    bench = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = bench.sample(num_samples=200, seed=1)
    report = evaluate(dataset=dataset, policy=bench.target_policy)
    df = report.to_dataframe()
    assert not df.empty


def test_evaluate_mdp_named_estimators():
    bench = SyntheticMDP(SyntheticMDPConfig(seed=1, horizon=3))
    dataset = bench.sample(num_trajectories=50, seed=2)
    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, MARKOV]),
    )
    report = evaluate(
        dataset=dataset,
        policy=bench.target_policy,
        estimand=estimand,
        estimators=["is", "wdr", "mrdr"],
    )
    df = report.to_dataframe()
    assert {"IS", "WDR", "MRDR"}.issubset(set(df["estimator"]))
