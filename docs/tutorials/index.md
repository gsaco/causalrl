# Tutorials

Use these tutorials as a guided path through the core workflows.

- [Quickstart (Bandit)](quickstart_bandit.md)
- [Quickstart (MDP)](quickstart_mdp.md)
- [Diagnostics](diagnostics.md)
- [Confidence Intervals](confidence_intervals.md)
- [Sensitivity Analysis](sensitivity.md)
- [Long-Horizon MIS vs IS](long_horizon_mis.md)
- [Proximal OPE (Advanced)](proximal_ope.md)

## Notebook gallery

The repository notebooks are paired with `.py` scripts via Jupytext, so you can
review diffs cleanly and run them as scripts. They include end-to-end pipelines,
diagnostics deep dives, and advanced identification tutorials.

Suggested entry points:

- `00_introduction.ipynb` — estimand-first tour with report export
- `01_estimands_and_assumptions.ipynb` — assumptions enforcement and overlap stress test
- `02_bandit_ope_walkthrough.ipynb` — diagnostics-driven bandit OPE
- `03_mdp_ope_walkthrough.ipynb` — trajectory estimators and horizon effects
- `04_confidence_intervals_and_hcope.ipynb` — CIs and high-confidence bounds
- `05_sensitivity_analysis_bandits.ipynb` — bandit sensitivity curves
- `06_sensitivity_unobserved_confounding.ipynb` — sequential Gamma bounds
- `07_proximal_ope_confounded_pomdp.ipynb` — proximal OPE with proxies
- `08_estimator_selection_and_debugging.ipynb` — selection heuristics and playbook
- `10_bandit_ope_end_to_end.ipynb` — research-grade workflow with sensitivity
- `11_mdp_long_horizon_comparison.ipynb` — long-horizon estimator comparison
