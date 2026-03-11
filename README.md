# continuum

Continuum normalization routine in JAX, robust to outlier spikes (e.g. cosmic rays) and missing data. The algorithm is the following:
1. If `pre_smooth=True` (needed if there are missing values, recommended for low S/N), first smooth the spectrum with a GP using [`smolgp`](https://smolgp.readthedocs.io/en/latest/).
2. On the flux or smoothed flux, compute two rolling maxima: one with width `window_narrow` (should be ~a few times the typical line FWHM) and one with width `window_wide` (should be ~a few times the broadest features, or ~5-10x `window_narrow`).
3. Take the average of these two rolling maxima, called `envelope`, as our initial guess
4. Define a series of weights
    - `w_agree` : the ratio between the narrow and wide rolling maximum. Tells us what parts of the spectrum are not inside deep/broad features, so we can trust them more.
    - `w_deriv` : penalize high-derivative regions (steep parts of spectral lines/discontinuities)
    - `w_line` : downweight pixels with low flux relative to `envelope` (deep lines) by a factor of `line_suppress_factor`.
    - `w_edge` : boost the weight of the pixels within `edge_pixels` pixels of either edge multiplicitively by `edge_boost`. Helps pin the continuum to the edges
    - The overall weight is the product of these weights.
5. Use the overall weight to fit a weighted penalised P-spline with `n_knots` knots, where the penalty term is the second derivative weighted by the `rigidity` parameter. Smaller `rigidity` provides a more flexible model, while larger values enforces a stricter smoothness.
    - You can use the `fit_target` parameter to define which quantity to optimize the spline with respect to. The default is the narrow rolling maximum (`fit_target='narrow'`), other options are the wide rolling maximum (`'wide'`), the envelope (`'envelope'`), or the flux itself (`'flux'`).

## Example usage

Let `wave`, `flux`, and `ferr` be 2D arrays of shape `(n_orders, n_pixels)` containing the wavelength, flux, and flux uncertainty values repsectively.

```
from continuum import fit_continuum

#################   PARAMETERS TUNED TO KPF  #################
n_reject = 1 # number of max pixels to toss from each rolling max window
n_knots  = 40 # 30-50 is good
rigidity = 50 # 50-100 is good
window_narrow = 30 # pixels
window_wide   = 300 # pixels
edge_pixels = 150 # pixels
edge_boost = 1e2 
line_suppress_factor = 20

# Use JAX to vmap over all the orders in a KPF spectrum
@jax.jit
def normalize_order(wave, flux, err):
    return fit_continuum(wave, flux, err,
                         window_narrow=window_narrow, window_wide=window_wide,
                         n_reject=n_reject, n_knots=n_knots, rigidity=rigidity, 
                         line_suppress_factor=line_suppress_factor,
                         edge_boost=edge_boost, edge_pixels=edge_pixels,
                         pre_smooth=False,
                         )

continua = jax.vmap(normalize_order)(wave, flux, ferr)
```

<img width="1012" height="530" alt="image" src="https://github.com/user-attachments/assets/9257f0c0-4e45-424d-8957-c7e4e55b6e95" />
<img width="1012" height="530" alt="image" src="https://github.com/user-attachments/assets/d6f2812d-1691-4244-aca9-ef734bb6837f" />
<img width="990" height="530" alt="image" src="https://github.com/user-attachments/assets/20e82ec7-d905-49c2-99d5-f13e44382af0" />
