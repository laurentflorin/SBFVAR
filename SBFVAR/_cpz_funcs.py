"""
Helper routines for the Chan, Poon & Zhu (2024) mixed-frequency approach.

This module contains faithful Python/NumPy/SciPy ports of the MATLAB reference
implementation ("WeeklyVARcode") provided with

    Chan, J., Poon, A. & Zhu, D. (2024). "High-dimensional conditionally
    Gaussian state space models with missing data." Journal of Econometrics
    236, 105468.

The routines here are deliberately kept separate from the Schorfheide-Song
(2015) estimator that lives in :mod:`SBFVAR._estimation`.  The two approaches
are physically isolated; nothing in this module is imported by the SS path.

Each function documents the MATLAB routine it corresponds to:

===========================  ==================================================
MATLAB routine               Python port
===========================  ==================================================
``get_resid_var.m``          :func:`get_resid_var`
``construct_minnesota.m``    :func:`construct_minnesota`
``sample_CSV.m``             :func:`sample_CSV`
``vec`` (builtin)            :func:`vec`
(selection matrices in       :func:`build_frequency_ratios`,
 ``MFVAR.m``)                :func:`build_stacked_data`,
                             :func:`build_selection_matrices`
===========================  ==================================================
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, solve_triangular


# Periods-per-year used to derive integer conversion ratios between adjacent
# frequencies.  These match the conventions already used by the Schorfheide-Song
# path (``freq_ratio_list = [12, 4]`` for Q/M/W, i.e. 4 weeks per month and 12
# weeks per quarter) and by ``aggregate`` in :mod:`SBFVAR._estimation`.
PERIODS_PER_YEAR = {"Y": 1, "Q": 4, "M": 12, "W": 48, "D": 240}


def vec(a):
    """Column-major (Fortran-order) vectorisation, equivalent to MATLAB ``vec``."""
    return np.reshape(np.asarray(a), (-1,), order="F")


def build_frequency_ratios(frequencies):
    """Return integer conversion ratios for an arbitrary list of frequencies.

    Parameters
    ----------
    frequencies : list of str
        Frequency codes ordered lowest to highest, e.g. ``["Q", "M", "W"]``.

    Returns
    -------
    ratios_adjacent : list of int
        ``ratios_adjacent[k]`` is the number of ``frequencies[k+1]`` periods per
        one ``frequencies[k]`` period (length ``L-1``).
    ratios_to_highest : list of int
        ``ratios_to_highest[k]`` is the number of highest-frequency periods per
        one ``frequencies[k]`` period (length ``L``); the last entry is ``1``.

    Notes
    -----
    Generalises the hard-coded ``freq_ratio_list = [12, 4]`` of the SS path to
    an arbitrary number of frequencies.
    """
    for f in frequencies:
        if f not in PERIODS_PER_YEAR:
            raise ValueError(
                f"Unsupported frequency '{f}'. Supported: {list(PERIODS_PER_YEAR)}"
            )
    ppy = [PERIODS_PER_YEAR[f] for f in frequencies]
    L = len(frequencies)
    ratios_adjacent = []
    for k in range(L - 1):
        hi, lo = ppy[k + 1], ppy[k]
        if hi % lo != 0:
            raise ValueError(
                f"Non-integer ratio between '{frequencies[k]}' and "
                f"'{frequencies[k + 1]}'."
            )
        ratios_adjacent.append(hi // lo)
    ppy_high = ppy[-1]
    ratios_to_highest = [ppy_high // p for p in ppy]
    return ratios_adjacent, ratios_to_highest


def get_resid_var(tmpY):
    """AR(4) residual variances used to scale the Minnesota prior.

    Port of ``get_resid_var.m``.  For each column an AR(4) with intercept is
    fitted by OLS and the mean squared residual returned.

    Parameters
    ----------
    tmpY : ndarray, shape (T, n)
        A (single-frequency, gap-free) data block.

    Returns
    -------
    sig2 : ndarray, shape (n,)
    """
    tmpY = np.asarray(tmpY, dtype=float)
    T, n = tmpY.shape
    sig2 = np.zeros(n)
    for i in range(n):
        y = tmpY[:, i]
        if T <= 5:
            # Not enough observations for AR(4); fall back to the sample var.
            sig2[i] = np.var(y) if y.size else 1.0
            continue
        Z = np.column_stack([
            np.ones(T - 4),
            y[3:-1], y[2:-2], y[1:-3], y[0:-4],
        ])
        target = y[4:]
        tmpb, *_ = np.linalg.lstsq(Z, target, rcond=None)
        resid = target - Z @ tmpb
        sig2[i] = np.mean(resid ** 2)
    # Guard against degenerate (zero) variances that would break the prior.
    sig2[sig2 <= 0] = 1e-8
    return sig2


def construct_minnesota(sig2, n, lag, theta):
    """Build the Minnesota prior precision ``invVbeta``.

    Faithful port of ``construct_minnesota.m``.  ``sig2`` are the AR(4)
    residual variances (see :func:`get_resid_var`) supplied in the stacked
    variable order (high-frequency variables first, then lower frequencies).

    Parameters
    ----------
    sig2 : ndarray, shape (n,)
        AR(4) residual variances, one per variable.
    n : int
        Number of variables in the stacked system.
    lag : int
        Number of VAR lags.
    theta : sequence of float, length 4
        ``[overall_tightness, cross_shrinkage, const_scale, lag_decay]``.

    Returns
    -------
    invVbeta : scipy.sparse.csc_matrix, shape (dim, dim)
        Diagonal prior term, ``dim = n * (n * lag + 1)``.  The per-variable
        block ordering matches the coefficient vector ``beta = [const; lag_1;
        ...; lag_p]`` stacked column-major over variables.
    """
    sig2 = np.asarray(sig2, dtype=float).ravel()
    AR_s2 = sig2  # diagonal of the residual-variance matrix
    k = n * lag + 1

    sigma_const = np.zeros(n)
    Pi_pv = np.zeros((n * (k - 1),))  # diagonal of the coefficient block
    co = 0
    for i in range(n):
        sigma_const[i] = AR_s2[i] * theta[2]
        for l in range(1, lag + 1):
            for j in range(n):
                if i == j:
                    Pi_pv[co] = theta[0] / (l ** theta[3])
                else:
                    Pi_pv[co] = (
                        AR_s2[i] / AR_s2[j] * theta[0] * theta[1] / (l ** theta[3])
                    )
                co += 1

    # Reshape the coefficient diagonal to (k-1, n) column-major and stack the
    # constant term on top of each variable's block, then vectorise.
    Pi_block = Pi_pv.reshape((k - 1, n), order="F")
    stacked = np.vstack([sigma_const[np.newaxis, :], Pi_block])  # (k, n)
    diag_vals = vec(stacked)  # length n*k = dim
    # ``invVbeta`` is the prior *precision* (inverse of the prior variances just
    # built), added to the likelihood precision in the beta update.
    diag_vals = 1.0 / diag_vals
    return sp.diags(diag_vals, format="csc")


def sample_CSV(s2, rho, sigh2, h, n, is_forced_accept=False):
    """Common stochastic-volatility sampler (accept/reject + MH).

    Faithful port of ``sample_CSV.m``.  Samples the log-volatility path ``h``
    of a single common stochastic-volatility process given the sum of squared
    (whitened) residuals ``s2``.

    Parameters
    ----------
    s2 : ndarray, shape (T,)
        Sum over variables of squared whitened residuals per period.
    rho : float
        AR(1) persistence of the log-volatility.
    sigh2 : float
        Innovation variance of the log-volatility.
    h : ndarray, shape (T,)
        Current log-volatility path.
    n : int
        Number of variables (cross-sectional dimension).
    is_forced_accept : bool
        Force acceptance of the proposal (used on the very first draw).

    Returns
    -------
    h : ndarray, shape (T,)
    is_accept : int
    """
    s2 = np.asarray(s2, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel().copy()
    T = s2.shape[0]
    is_accept = 0

    # Hrho = I - rho * subdiagonal
    Hrho = sp.eye(T, format="csc") - rho * sp.diags(
        np.ones(T - 1), -1, shape=(T, T), format="csc"
    )
    dvec = np.concatenate([[(1 - rho ** 2) / sigh2], np.ones(T - 1) / sigh2])
    HiSH = (Hrho.T @ sp.diags(dvec) @ Hrho).tocsc()

    # Newton iterations to locate the conditional mode.
    errh = 1.0
    ht = h.copy()
    Kh = None
    while errh > 1e-3:
        eht = np.exp(ht)
        sieht = s2 / eht
        fh = -n / 2.0 + 0.5 * sieht
        Gh = 0.5 * sieht
        Kh = (HiSH + sp.diags(Gh)).tocsc()
        newht = sp.linalg.spsolve(Kh, fh + Gh * ht)
        errh = np.max(np.abs(newht - ht))
        ht = newht

    Kh_dense = Kh.toarray()
    Kh_dense = 0.5 * (Kh_dense + Kh_dense.T)
    CKh = cholesky(Kh_dense, lower=True)
    HiSH_dense = HiSH.toarray()

    def _logpost(x):
        return (-0.5 * x @ (HiSH_dense @ x) - n / 2.0 * np.sum(x)
                - 0.5 * np.exp(-x) @ s2)

    # AR (accept/reject) step.
    hstar = ht.copy()
    logc = (-0.5 * hstar @ (HiSH_dense @ hstar) - n / 2.0 * np.sum(hstar)
            - 0.5 * np.exp(-hstar) @ s2 + np.log(3.0))
    flag = 0
    hc = ht.copy()
    while flag == 0:
        # hc = ht + CKh'\randn  (solve upper-triangular system CKh' x = z)
        z = np.random.standard_normal(T)
        hc = ht + solve_triangular(CKh.T, z, lower=False)
        alpARc = (_logpost(hc) + 0.5 * (hc - ht) @ (Kh_dense @ (hc - ht)) - logc)
        if alpARc > np.log(np.random.rand()):
            flag = 1

    # MH step.
    alpAR = (_logpost(h) + 0.5 * (h - ht) @ (Kh_dense @ (h - ht)) - logc)
    if alpAR < 0:
        alpMH = 1.0
    elif alpARc < 0:
        alpMH = -alpAR
    else:
        alpMH = alpARc - alpAR
    if alpMH > np.log(np.random.rand()) or is_forced_accept:
        h = hc
        is_accept = 1

    return h, is_accept


def build_stacked_data(datasets, ratios_to_highest):
    """Embed all frequencies into one high-frequency data matrix with NaNs.

    Generalises the ``dataset`` construction of ``MFVAR.m`` to an arbitrary
    number of frequencies.  Variables are stacked **high-frequency first**
    (weekly, then monthly, then quarterly, ... down to the lowest frequency).

    Parameters
    ----------
    datasets : list of ndarray
        Transformed data blocks ordered **lowest to highest** frequency
        (``datasets[0]`` is the lowest frequency).  These are already
        log/÷100 transformed.
    ratios_to_highest : list of int
        Highest-frequency periods per one period of each frequency
        (see :func:`build_frequency_ratios`), same ordering as ``datasets``.

    Returns
    -------
    Yraw : ndarray, shape (T, n)
        Stacked high-frequency data, high-frequency variables first, with NaN
        wherever a (lower-frequency) variable is not observed.
    block_info : list of dict
        One entry per frequency, ordered **high to low**, each with keys
        ``level`` (index into ``datasets``), ``col_start``, ``n_vars``,
        ``ratio`` (highest-frequency periods per period), ``n_periods``.
    """
    L = len(datasets)
    n_vars = [d.shape[1] for d in datasets]
    n = int(sum(n_vars))

    # Number of complete highest-frequency periods consistent across all
    # frequencies (truncate ragged edges, mirroring the SS kron/truncate logic).
    T = min(datasets[k].shape[0] * ratios_to_highest[k] for k in range(L))

    Yraw = np.full((T, n), np.nan)
    block_info = []
    col = 0
    # High-to-low ordering: iterate levels from highest (L-1) down to lowest (0).
    for level in range(L - 1, -1, -1):
        d = np.asarray(datasets[level], dtype=float)
        r = ratios_to_highest[level]
        nk = n_vars[level]
        n_periods = T // r
        for p in range(min(n_periods, d.shape[0])):
            row = (p + 1) * r - 1  # last high-frequency period of block p
            Yraw[row, col:col + nk] = d[p, :]
        block_info.append({
            "level": level,
            "col_start": col,
            "n_vars": nk,
            "ratio": r,
            "n_periods": n_periods,
        })
        col += nk
    return Yraw, block_info


def build_selection_matrices(Yraw, block_info, datasets, lag):
    """Build ``vecY``, ``M_o``, ``M_u``, ``M_a`` and ``Y_con``.

    Faithful, N-frequency generalisation of the selection-matrix construction
    in ``MFVAR.m``.  All lower-frequency variables are treated as
    intertemporal-aggregation ("flow") constraints handled through ``M_a`` (the
    Mariano-Murasawa tent weights ``[1:m, m-1:-1:1]/m``), matching the
    ``temp_agg`` convention of the package; the highest-frequency block is fully
    observed.

    Parameters
    ----------
    Yraw : ndarray, shape (T, n)
        Stacked data from :func:`build_stacked_data`.
    block_info : list of dict
        Block metadata (high-to-low) from :func:`build_stacked_data`.
    datasets : list of ndarray
        Transformed data blocks ordered lowest-to-highest (for ``Y_con`` values).
    lag : int
        VAR lag order (unused here but kept for signature symmetry).

    Returns
    -------
    out : dict
        With keys ``vecY`` (ndarray), ``M_o``/``M_u``/``M_a`` (csc matrices),
        ``Y_con`` (ndarray), ``n``, ``T``, ``n_high``, ``n_low``,
        ``low_order`` (list of dicts describing the low-frequency blocks in the
        latent-vector ordering).
    """
    T, n = Yraw.shape
    # block_info[0] is the highest frequency.
    n_high = block_info[0]["n_vars"]
    n_low = n - n_high

    # --- observed / missing selection (M_o, M_u) -------------------------
    # vec ordering is time-major: for each t the n variables in high-to-low
    # order.  The first n_high entries per time are observed, the remaining
    # n_low are latent (missing).
    nT = n * T
    N_m = n_low * T

    obs_rows = np.concatenate([
        np.arange(t * n, t * n + n_high) for t in range(T)
    ]) if n_high > 0 else np.array([], dtype=int)
    mis_rows = np.concatenate([
        np.arange(t * n + n_high, (t + 1) * n) for t in range(T)
    ]) if n_low > 0 else np.array([], dtype=int)

    n_obs = obs_rows.shape[0]
    M_o = sp.csc_matrix(
        (np.ones(n_obs), (obs_rows, np.arange(n_obs))), shape=(nT, n_obs)
    )
    M_u = sp.csc_matrix(
        (np.ones(N_m), (mis_rows, np.arange(N_m))), shape=(nT, N_m)
    )

    # vecY = observed high-frequency values, stacked time-major.
    Yfull_vec = Yraw.reshape((nT,), order="C")  # row t block of n vars
    vecY = np.asarray(M_o.T @ np.nan_to_num(Yfull_vec)).ravel()

    # --- intertemporal-aggregation constraints (M_a, Y_con) --------------
    # Low-frequency blocks are ordered high-to-low starting after the highest
    # frequency block.  Their offsets *within the latent (n_low) ordering* are
    # the column offsets minus n_high.
    low_blocks = block_info[1:]
    rows_ma = []
    cols_ma = []
    vals_ma = []
    y_con_parts = []
    con_col = 0
    for blk in low_blocks:
        level = blk["level"]
        m = blk["ratio"]  # highest-frequency periods per low-frequency period
        nk = blk["n_vars"]
        off = blk["col_start"] - n_high  # offset in the latent (n_low) ordering
        d = np.asarray(datasets[level], dtype=float)
        n_periods = blk["n_periods"]
        # tent weights [1..m, m-1..1] / m, length 2m-1
        w = np.concatenate([np.arange(1, m + 1), np.arange(m - 1, 0, -1)]) / m
        for i in range(1, min(n_periods, d.shape[0])):
            # 0-indexed high-frequency start of the tent for observation i
            start = (i - 1) * m + 1
            for a in range(nk):
                for wi, wv in enumerate(w):
                    t = start + wi
                    if t < 0 or t >= T:
                        continue
                    # latent-vector row for variable (off+a) at time t
                    rows_ma.append(t * n_low + off + a)
                    cols_ma.append(con_col + a)
                    vals_ma.append(wv)
            y_con_parts.append(d[i, :])
            con_col += nk

    if con_col > 0:
        M_a = sp.csc_matrix(
            (vals_ma, (rows_ma, cols_ma)), shape=(N_m, con_col)
        )
        Y_con = np.concatenate(y_con_parts)
    else:
        M_a = sp.csc_matrix((N_m, 0))
        Y_con = np.zeros(0)

    low_order = []
    for blk in low_blocks:
        low_order.append({
            "level": blk["level"],
            "off": blk["col_start"] - n_high,
            "n_vars": blk["n_vars"],
        })

    return {
        "vecY": vecY,
        "M_o": M_o,
        "M_u": M_u,
        "M_a": M_a,
        "Y_con": Y_con,
        "n": n,
        "T": T,
        "n_high": n_high,
        "n_low": n_low,
        "low_order": low_order,
    }


def build_A_matrices(T, lag, n):
    """Return the list of time-selection matrices ``A`` used to form ``C``.

    Port of the ``A`` cell array in ``MFVAR.m``/``Sample_latent_Y_approx.m``.
    ``A[0]`` selects the contemporaneous ``y_t`` and ``A[j]`` (j>=1) selects
    ``y_{t-j}``; each has shape ``(T-lag, T)``.
    """
    Tnew = T - lag
    A = []
    # A0 = [0_{Tnew x lag}, I_{Tnew}]
    A0 = sp.hstack([sp.csc_matrix((Tnew, lag)), sp.eye(Tnew, format="csc")]).tocsc()
    A.append(A0)
    for j in range(1, lag + 1):
        Aj = sp.hstack([
            sp.csc_matrix((Tnew, lag - j)),
            sp.eye(Tnew, format="csc"),
            sp.csc_matrix((Tnew, j)),
        ]).tocsc()
        A.append(Aj)
    return A
