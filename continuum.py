import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=("window", "n_reject"))
def rolling_maximum(
    flux: jnp.ndarray,
    window: int,
    n_reject: int = 1,
) -> jnp.ndarray:
    """
    Rolling maximum robust to narrow outliers (e.g. cosmic rays or hot pixels).

    A spike of width w pixels can occupy at most w of the top ranks inside
    any window that contains it, so setting n_reject >= w makes the
    estimator completely blind to those spikes.

    Requires: window > n_reject + 1

    Parameters:
        flux     : (n_pixels,) input spectrum
        window   : rolling window width (in pixels)
        n_reject : the number of largest values to ignore in each window

    Returns:
        Array (n_pixels,) of the rolling maximum with the same length as the input.
    """
    assert window > n_reject + 1, (
        "window must be larger than n_reject+1 to have anything in the window!"
    )
    n = len(flux)
    half = window // 2

    # Pad with edge values on either side before passing into scan
    padded = jnp.pad(flux, (half, window - half), mode="edge")

    # Initialise the buffer with the window centred on pixel 0
    init_buf = jax.lax.dynamic_slice(padded, (0,), (window,))

    def scan(buf, i):
        val = jnp.sort(buf)[::-1][n_reject]
        new_buf = jnp.roll(buf, -1).at[-1].set(padded[i + window])
        return new_buf, val

    _, result = jax.lax.scan(scan, init_buf, jnp.arange(n))
    return result


def percentiles(x):
    """
    Compute the percentile rank of each element in x,
    i.e. the fraction of elements less than it.
    """
    ranks = jnp.argsort(jnp.argsort(x))
    p = (ranks + 0.5) / x.size
    return p


def logistic_weights(p, p0=0.9, softness=0.03):
    """
    Logistic function centered at p0 with softness
    parameter to control the steepness of the transition.
    """
    w = 1.0 / (1.0 + jnp.exp((p - p0) / softness))
    return w


@partial(
    jax.jit,
    static_argnames=(
        "window_narrow",
        "window_wide",
        "n_reject",
        "n_knots",
        "rigidity",
        "edge_boost",
        "edge_pixels",
        "line_suppress_factor",
        "p0_deriv",
        "softness_deriv",
        "p0_edge",
        "softness_edge",
        "fit_target",
        "pre_smooth",
    ),
)
def fit_continuum(
    wave: jnp.ndarray,
    flux: jnp.ndarray,
    err: jnp.ndarray,
    window_narrow: int = 30,
    window_wide: int = 200,
    n_reject: int = 1,
    n_knots: int = 30,
    rigidity: float = 1e2,
    edge_boost: float = 200,
    edge_pixels: int = 200,
    line_suppress_factor: float = 0.0,
    p0_deriv: float = 0.84,
    softness_deriv: float = 0.01,
    p0_edge: float = 0.05,
    softness_edge: float = 0.05,
    fit_target: str = "narrow",
    pre_smooth: bool = False,
) -> tuple:
    """
    Estimate the continuum of a single echelle order using
    a weighted penalised P-spline fit to the envelope defined
    by a narrow and wide rolling maximum. The weights boost
    pixels where the narrow and wide passes agree, and supresses
    pixels with strong local derivatives or deep lines. The edges
    of the order are boosted to ensure the spline is anchored there.
    The spline is penalized by its second derivative to enforce the
    user's desired balance between smoothness and flexibility.

    Generally, the parameters a user will need to tune are
        window_narrow : set to a few times the typical FWHM of lines in the spectrum
        window_wide   : set to a few times the width of the widest features in the spectrum
                        (e.g. broad lines or molecular bands)
        n_reject      : set to the maximum expected width (in pixels) of outliers such as cosmic rays
        n_knots       : can be larger at higher spectral resolution or if more flexibility is needed
                        to capture a complex blaze shape, but beware of overfitting and ringing artifacts!
        rigidity      : smaller values allow the spline to fit more closely to the envelope,
                        while larger values enforce more smoothness. Tune jointly with n_knots.
        edge_boost    : ~10-100 or more for wide orders with "bent" edges
        edge_pixels   : ~width from the edge where the "bending" is seen (if applicable)

    Parameters:
        wave             : (n_pixels,) wavelength array corresponding to flux
        flux             : (n_pixels,) input spectrum
        err              : (n_pixels,) uncertainty on flux
        window_narrow    : narrow rolling-max window (in pixels, ~few times typical FWHM)
        window_wide      : wide rolling-max window   (in pixels, ~few times widest features)
        n_reject         : number of outlier pixels trimmed from the rolling maximum (e.g. to mitigate cosmic rays)
        n_knots          : number of evenly-spaced spline knots
        rigidity         : second-difference penalty weight (smaller = more flexible)
        edge_boost       : multiplicative weight boost at order boundaries
        edge_pixels      : number of pixels at each end to receive edge_boost
        line_suppress_factor : exponent applied to (flux/envelope)**line_suppress_factor for downweighting pixels inside lines
        p0_deriv         : percentile of |dflux| at which to place the knee of the logistic function for derivative weighting
        softness_deriv   : softness parameter for the logistic function for derivative weighting
        p0_edge          : percentile of (1 - w_agree) at which to place the knee of the logistic function for edge weighting
        softness_edge    : softness parameter for the logistic function for edge weighting
        fit_target       : which array to fit the spline to:
                            'narrow' = the rolling max with `window_narrow` (default)
                            'wide'   = the rolling max with `window_wide`
                            'envelope' = the average of the narrow and wide rolling max
                            'flux'   = the original flux
    Returns:
        continuum    : (n_pixels,) fitted continuum
    """
    n = len(flux)
    x = jnp.linspace(-1.0, 1.0, n)

    if pre_smooth:
        import smolgp

        # Step 0: Use a GP to fit a smooth flux model
        # This mitigates the impact of noise on the rolling maximum
        # and its derivatives, plus filters over any NaNs in the input flux.
        missing = ~jnp.isfinite(flux)
        f_safe = jnp.where(missing, 0.0, flux)  # doesn't matter since err=inf
        e_safe = jnp.where(missing, 1e9, err)  # large error -> downweight in GP fit
        med = jnp.median(f_safe)  # scale to ~1 for GP fit
        kernel = smolgp.kernels.Exp(sigma=0.1, scale=1)
        gp = smolgp.GaussianProcess(kernel, X=wave, noise=jnp.power(e_safe / med, 2))
        f_smooth = gp.predict(wave, f_safe / med, return_var=False)
        flux = f_smooth  # * med
    else:
        med = 1.0
    ######################################################################

    # Step 1: Rolling maxima and envelope
    roll_narrow = rolling_maximum(flux, window_narrow, n_reject)
    roll_wide = rolling_maximum(flux, window_wide, n_reject)
    # envelope    = jnp.maximum(roll_narrow, roll_wide)
    envelope = (roll_narrow + roll_wide) / 2
    ######################################################################

    # Step 2: Weights for p-spline
    # (a) Agreement between narrow and wide passes
    w_agree = roll_narrow / (roll_wide + 1e-30)  # ∈ (0, 1]

    # (b) Derivative weighting: lets us know where we're
    #     inside a line to strongly downweight those pixels
    df = jnp.diff(flux / jnp.median(flux), n=1)
    df = jnp.concatenate([df[:1], df])  # pad to original length
    adf = jnp.abs(df)
    adf /= jnp.max(adf)
    p = percentiles(adf)
    w_deriv = logistic_weights(p, p0=p0_deriv, softness=softness_deriv)

    # (c) Line-depth weighting: suppress pixels where flux ≪ envelope
    line_ratio = flux / (envelope + 1e-30)
    w_line = jnp.power(line_ratio, line_suppress_factor)
    w_line *= jnp.where(
        flux < 0.1 * jnp.median(envelope), 0.0, 1.0
    )  # hard cutoff for very deep lines to prevent ringing artifacts

    # (d) Edge anchor boost
    left_score = jnp.mean(w_agree[:edge_pixels])
    right_score = jnp.mean(w_agree[edge_pixels:])
    left_edge_weight = edge_boost * logistic_weights(
        1 - left_score, p0=p0_edge, softness=softness_edge
    )
    right_edge_weight = edge_boost * logistic_weights(
        1 - right_score, p0=p0_edge, softness=softness_edge
    )
    w_edge = (
        jnp.ones(n)
        .at[:edge_pixels]
        .set(left_edge_weight)
        .at[-edge_pixels:]
        .set(right_edge_weight)
    )

    weights = w_agree * w_deriv * w_line * w_edge
    ######################################################################

    # Step 3: Fit the P-spline using these weights
    # (a) Initialize spline basis  B : (n, n_knots)
    # B[p, k] encodes how much knot k contributes to pixel p.
    knots = jnp.linspace(-1.0, 1.0, n_knots)
    x_norm = (x - knots[0]) / (knots[-1] - knots[0]) * (n_knots - 1)
    idx = jnp.clip(jnp.floor(x_norm).astype(jnp.int32), 0, n_knots - 2)
    t = x_norm - idx  # fractional offset ∈ [0, 1)

    pix = jnp.arange(n)
    B = jnp.zeros((n, n_knots)).at[pix, idx].set(1.0 - t).at[pix, idx + 1].add(t)

    # (b) Second-difference smoothness matrix  D2 : (n_knots-2, n_knots)
    # For a B-spline, the second derivative at knot k is proportional to
    #   a[k+2] - 2 a[k+1] + a[k],
    # ||D2 a||² penalises curvature of the knot values.
    rows = jnp.arange(n_knots - 2)
    D2 = (
        jnp.zeros((n_knots - 2, n_knots))
        .at[rows, rows]
        .set(1.0)
        .at[rows, rows + 1]
        .set(-2.0)
        .at[rows, rows + 2]
        .set(1.0)
    )

    # (c) Closed-form P-spline solve
    # See: https://psplines.bitbucket.io/Support/WhyPsplines.pdf
    # Unweighted solution is
    #   (B.T @ B + λ D.T @ D) a = B.T y
    # where a = spline coefficients at knot positions.
    # To make weighted,
    #   (B.T @ W @ B + λ D.T @ D) a = B.T @ W @ y

    # Data to fit spline to
    y = {"narrow": roll_narrow, "envelope": envelope, "flux": flux, "wide": roll_wide}[
        fit_target
    ]

    # Solve with jnp.linalg.solve (one O(K^3) call)
    WB = B * weights[:, None]
    A = WB.T @ B + rigidity * D2.T @ D2
    b = WB.T @ y
    a = jnp.linalg.solve(A, b)
    continuum = B @ a

    return continuum * med
