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
- `crl.Assumption`
- `crl.AssumptionSet`

## Datasets

- `crl.data.LoggedBanditDataset`
- `crl.data.TrajectoryDataset`

## Estimators and reports

- `crl.estimators.OPEEstimator`
- `crl.estimators.EstimatorReport`
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
- `crl.estimators.DoubleRLEstimator`
- `crl.estimators.HighConfidenceISEstimator`

## API namespace

- `crl.api` re-exports the stable surface for convenience in scripts.
