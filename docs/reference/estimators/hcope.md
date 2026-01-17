# High-Confidence OPE Lower Bounds

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Bounded rewards

## Formula

Using an empirical Bernstein bound, compute a lower confidence bound for
importance-sampled returns:

$\text{LCB} = \bar X - \sqrt{\frac{2 \hat\sigma^2 \log(2/\delta)}{n}} - \frac{7 R_{\max} \log(2/\delta)}{3(n-1)}$.

## Failure modes

- Conservative when variance is large.
- Requires a valid reward bound.

## References

- Thomas et al. (2015)

## Notebook

- [04_confidence_intervals_safe_selection.ipynb](../../notebooks/04_confidence_intervals_safe_selection.ipynb)
