.PHONY: test docs benchmarks-smoke benchmarks-full notebooks-smoke

test:
	python -m pytest

docs:
	mkdocs build --strict

benchmarks-smoke:
	python -c "from crl.benchmarks.harness import run_all_benchmarks; run_all_benchmarks(num_samples=100, num_trajectories=50, seed=0)"

benchmarks-full:
	python -c "from crl.benchmarks.harness import run_all_benchmarks; run_all_benchmarks(num_samples=5000, num_trajectories=2000, seed=0)"

notebooks-smoke:
	python notebooks/00_introduction.py
	python notebooks/02_bandit_ope_walkthrough.py
	python notebooks/03_mdp_ope_walkthrough.py
