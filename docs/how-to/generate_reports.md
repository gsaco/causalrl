# Generate Reports

Use `OpeReport` to save HTML reports and figures.

```python
report = evaluate(dataset=dataset, policy=policy)
report.plot_estimator_comparison()
report.save_html("report.html")
```

The HTML report embeds figures as base64 so it is portable.

Sample artifact: `docs/assets/reports/intro_bandit_report.html`.
