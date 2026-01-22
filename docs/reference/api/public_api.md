# Public API

This page lists the stable, supported import paths for CausalRL. Anything not
listed here may change without notice.

## Core entry points

- `crl.evaluate` - end-to-end OPE pipeline returning an `OpeReport`.
- `crl.ope.evaluate` - same entry point, explicit module import.
- `crl.ope.OpeReport` - aggregate report for an OPE run.
- `crl.set_seed` - seed Python/NumPy/torch RNGs.

## Estimands and assumptions

- `crl.PolicyValueEstimand`
- `crl.PolicyContrastEstimand`
- `crl.SensitivityPolicyValueEstimand`
- `crl.ProximalPolicyValueEstimand`
- `crl.Assumption`
- `crl.AssumptionSet`

## Datasets

- `crl.data.BanditDataset`
- `crl.data.LoggedBanditDataset`
- `crl.data.TrajectoryDataset`
- `crl.data.TransitionDataset`

## Behavior policy estimation

- `crl.fit_behavior_policy`
- `crl.BehaviorPolicyFit`

## Policies

- `crl.policies.Policy`
- `crl.policies.StochasticPolicy`
- `crl.policies.CallablePolicy`
- `crl.policies.TabularPolicy`
- `crl.policies.TorchMLPPolicy`

## Estimator selection

- `crl.select_estimator`
- `crl.SelectionResult`

## Dataset adapters

- `crl.load_d4rl_dataset`
- `crl.load_rl_unplugged_dataset`

## Estimators and reports

- `crl.estimators.OPEEstimator`
- `crl.estimators.EstimatorReport`
- `crl.estimators.UncertaintySummary`
- `crl.estimators.DiagnosticsConfig`
- `crl.estimators.ISEstimator`
- `crl.estimators.WISEstimator`
- `crl.estimators.PDISEstimator`
- `crl.estimators.DoublyRobustEstimator`
- `crl.estimators.WeightedDoublyRobustEstimator`
- `crl.estimators.MAGICEstimator`
- `crl.estimators.MRDREstimator`
- `crl.estimators.MarginalizedImportanceSamplingEstimator`
- `crl.estimators.FQEEstimator`
- `crl.estimators.DualDICEEstimator`
- `crl.estimators.GenDICEEstimator`
- `crl.estimators.DoubleRLEstimator`
- `crl.estimators.DRLEstimator`
- `crl.estimators.HighConfidenceISEstimator`

## API namespace

- `crl.api` re-exports the stable surface for convenience in scripts.
