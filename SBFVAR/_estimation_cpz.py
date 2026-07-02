"""
Chan, Poon & Zhu (2024) mixed-frequency estimator for SBFVAR.

This module is a Python/NumPy/SciPy port of the MATLAB reference code
(``MFVAR.m`` + ``Sample_latent_Y_approx.m`` + ``sample_CSV.m`` +
``construct_minnesota.m``), generalised from the fixed weekly/monthly/quarterly
layout to an arbitrary number of frequencies.

The whole system is treated as **one large stacked conditionally-Gaussian
state-space model** (all variables stacked, missing data handled through the
selection matrices ``M_o``/``M_u``/``M_a`` and intertemporal-constraint
aggregation, with a single common stochastic-volatility process).

The estimator is intentionally isolated from the Schorfheide-Song (2015) path in
:mod:`SBFVAR._estimation`.  After sampling, the posterior draws are re-packed
into the *same* attribute shapes that the existing ``forecast``/``aggregate``/
``to_excel`` methods consume, so those downstream methods keep working unchanged
for the ``method="chan_poon_zhu"`` path.
"""

import copy
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.stats import wishart
from tqdm import tqdm

from ._cpz_funcs import (
    build_frequency_ratios,
    build_stacked_data,
    build_selection_matrices,
    build_A_matrices,
    construct_minnesota,
    get_resid_var,
    sample_CSV,
)


def _sample_latent_Y(h, invSig, Sig_chol, betaT, A, M_o, M_u, M_a, Y_con,
                     vecY, n, T, lag, n_low, ridge_dim):
    """Draw the latent high-frequency states given the stacked precision.

    Port of ``Sample_latent_Y_approx.m``.  The Gaussian draw
    ``x ~ N(Knew^{-1} b, Knew^{-1})`` is produced with the perturbation
    (Papandreou-Yuille / RUE) sampler so that only sparse LU solves are needed
    (no sparse Cholesky dependency):

        x = Knew^{-1} [ sum_k F_k^T W_k (c_k + w_k) ],   w_k ~ N(0, W_k^{-1}).

    Returns the reconstructed ``Y_new`` of shape ``(T, n)``.
    """
    Tnew = T - lag
    I_n = sp.eye(n, format="csc")

    # C = kron(A0, I) - sum_j kron(A_j, beta_j)
    C = sp.kron(A[0], I_n, format="csc")
    for j in range(1, lag + 1):
        beta_j = betaT[:, 1 + (j - 1) * n: 1 + j * n]
        C = C - sp.kron(A[j], sp.csc_matrix(beta_j), format="csc")
    C = C.tocsc()

    Cu = (C @ M_u).tocsc()

    exph_neg = np.exp(-h)
    invSigma = sp.kron(sp.diags(exph_neg), sp.csc_matrix(invSig), format="csc")

    bigK = (Cu.T @ invSigma).tocsc()
    K = (bigK @ Cu).tocsc()

    # Initial-condition ridge on the earliest latent entries.
    if ridge_dim > 0:
        ridge = sp.csc_matrix(
            (100.0 * np.ones(ridge_dim),
             (np.arange(ridge_dim), np.arange(ridge_dim))),
            shape=K.shape,
        )
        K = (K + ridge).tocsc()

    intercept = betaT[:, 0]
    m_vec = np.tile(intercept, Tnew) - np.asarray(C @ (M_o @ vecY)).ravel()

    iW = 1e10
    N_m = M_u.shape[1]
    Ncon = M_a.shape[1]

    Knew = (K + iW * (M_a @ M_a.T)).tocsc()

    # ---- perturbation sampler right-hand side ----
    # term 1: F1 = Cu, W1 = invSigma, w1 ~ N(0, invSigma^{-1})
    Z = np.random.standard_normal((Tnew, n))
    w1_blocks = (Z @ Sig_chol.T) * np.exp(0.5 * h)[:, None]
    w1 = w1_blocks.reshape((n * Tnew,))
    rhs = bigK @ (m_vec + w1)

    # term 2: ridge prior N(0, (1/100) I) on the first ``ridge_dim`` entries
    if ridge_dim > 0:
        w2 = np.random.standard_normal(ridge_dim) / math.sqrt(100.0)
        rhs[:ridge_dim] += 100.0 * w2

    # term 3: hard aggregation constraints, W3 = iW I, w3 ~ N(0, (1/iW) I)
    if Ncon > 0:
        w3 = np.random.standard_normal(Ncon) / math.sqrt(iW)
        rhs = rhs + iW * (M_a @ (Y_con + w3))

    lu = splu(Knew)
    vecY_u = lu.solve(rhs)

    full = np.asarray(M_o @ vecY + M_u @ vecY_u).ravel()
    Y_new = full.reshape((T, n))
    return Y_new


def fit_cpz(self, mufbvar_data, hyp, var_of_interest=None, temp_agg="mean",
            check_explosive=True, return_mdd=False, max_it_explosive=1000,
            **kwargs):
    """Estimate the mixed-frequency VAR using the Chan, Poon & Zhu approach.

    Implements the stacked conditionally-Gaussian state-space sampler with a
    single common stochastic-volatility process, generalised to an arbitrary
    number of frequencies.

    Parameters
    ----------
    mufbvar_data : sbfvar_data
        Prepared data object.
    hyp : ndarray
        Hyperparameter vector ``[lambda1, ..., lambda5]``.  The Minnesota prior
        maps these to ``theta = [lambda1, lambda2, lambda4, lambda3]`` =
        ``[overall_tightness, cross_shrinkage, const_scale, lag_decay]``.
    var_of_interest, temp_agg, check_explosive, max_it_explosive
        Kept for signature compatibility with the SS ``fit``.
    return_mdd : bool
        The CPZ path is tuned via RMSE (not MDD); when ``True`` this returns
        ``np.nan`` so MDD-based callers degrade gracefully.

    Returns
    -------
    float or None
    """
    self.hyp = np.asarray(hyp, dtype=float)
    self.temp_agg = temp_agg
    self.var_of_interest = var_of_interest
    self.method = "chan_poon_zhu"
    assert temp_agg in ("mean", "sum"), (
        f"Invalid temp_agg: {temp_agg}. Choose 'mean' or 'sum'."
    )

    frequencies = list(mufbvar_data.frequencies)
    self.frequencies = frequencies
    L = len(frequencies)

    # Transformed data blocks, ordered lowest -> highest frequency.
    datasets = [np.asarray(mufbvar_data.YQ0_list[0], dtype=float)]
    datasets += [np.asarray(a, dtype=float) for a in mufbvar_data.YM0_list]

    ratios_adjacent, ratios_to_highest = build_frequency_ratios(frequencies)

    # Combined (high -> low) variable metadata reused by downstream methods.
    varlist = np.asarray(mufbvar_data.varlist_list[-1])
    select_combined = np.asarray(mufbvar_data.select_list[-1])

    p = int(self.nlags)
    lag = p

    # ---- build the stacked system --------------------------------------
    Yraw, block_info = build_stacked_data(datasets, ratios_to_highest)
    sel = build_selection_matrices(Yraw, block_info, datasets, lag)
    n = sel["n"]
    T = sel["T"]
    n_high = sel["n_high"]
    n_low = sel["n_low"]
    Tnew = T - lag

    Nw = n_high
    Nq = datasets[0].shape[1]           # lowest-frequency block
    Nm = n_low - Nq                     # all intermediate frequencies
    Ntotal = n

    rqw = ratios_to_highest[0]          # lowest -> highest ratio
    rmw = ratios_to_highest[1] if L > 1 else 1

    print(" ", end="\n")
    print("Multiple Frequency SBFVAR (Chan, Poon & Zhu): Fitting", end="\n")
    print(f"Stacked system: n={n} variables, T={T} high-frequency periods, "
          f"lags={lag}", end="\n")

    A = build_A_matrices(T, lag, n)
    M_o = sel["M_o"]
    M_u = sel["M_u"]
    M_a = sel["M_a"]
    Y_con = sel["Y_con"]
    vecY = sel["vecY"]
    ridge_dim = min(M_u.shape[1], n_low * lag)

    # ---- Minnesota prior -----------------------------------------------
    # AR(4) residual variances in the stacked (high -> low) variable order.
    sig2_parts = []
    for blk in block_info:
        sig2_parts.append(get_resid_var(datasets[blk["level"]]))
    sig2 = np.concatenate(sig2_parts)
    theta = [float(self.hyp[0]), float(self.hyp[1]),
             float(self.hyp[3]), float(self.hyp[2])]
    invVbeta = construct_minnesota(sig2, n, lag, theta).tocsc()
    dim = n * (n * lag + 1)

    # ---- MCMC bookkeeping ----------------------------------------------
    total = int(self.nsim)
    Burn = int(round(self.nburn_perc * total))
    Sample = max(total - Burn, 1)
    thin = int(self.thining)
    ndraws = int(math.ceil(Sample / thin))

    Phip = np.zeros((ndraws, Ntotal * p + 1, Ntotal))
    Sigmap = np.zeros((ndraws, Ntotal, Ntotal))
    YYactsim_list = np.full((ndraws, rqw + 1, Ntotal), np.nan)
    XXactsim_list = np.full((ndraws, rqw + 1, Ntotal * p + 1), np.nan)
    lstate_list = np.zeros((ndraws, T - rqw, n_low))
    mh = np.zeros((ndraws, Tnew))       # stochastic-volatility path draws

    # ---- initial values (mirroring MFVAR.m) ----------------------------
    h = np.zeros(Tnew)
    rho = 0.9
    sigh2 = 0.1
    # Small random start for the VAR coefficients; scaled down by ``n * 10`` so
    # the initial companion matrix is comfortably non-explosive.
    beta = np.random.standard_normal((n, n * lag + 1)).T / (n * 10.0)  # (k, n)
    invSig = np.eye(n)

    store_idx = 0
    accept_count = 0
    it_total = Burn + Sample
    for it in tqdm(range(it_total)):
        betaT = beta.T  # (n, k): [intercept | lag1 | ... | lagp]
        Sig = np.linalg.inv(invSig)
        Sig = 0.5 * (Sig + Sig.T)
        try:
            Sig_chol = np.linalg.cholesky(Sig)
        except np.linalg.LinAlgError:
            Sig_chol = np.linalg.cholesky(Sig + 1e-10 * np.eye(n))

        # (a) latent states
        Y_new_full = _sample_latent_Y(
            h, invSig, Sig_chol, betaT, A, M_o, M_u, M_a, Y_con, vecY,
            n, T, lag, n_low, ridge_dim,
        )

        # regressors X (intercept first), targets Y_new
        X = np.ones((Tnew, n * lag + 1))
        for j in range(1, lag + 1):
            X[:, 1 + (j - 1) * n: 1 + j * n] = Y_new_full[lag - j: T - j, :]
        Y_new = Y_new_full[lag:, :]

        # (b) sample beta
        Dexp = np.exp(-h)
        XtD = X.T * Dexp                    # (k, Tnew)
        XtDX = XtD @ X                       # (k, k)
        Kbeta = np.kron(invSig, XtDX) + invVbeta.toarray()
        Kbeta = 0.5 * (Kbeta + Kbeta.T)
        rhs = (X.T * Dexp) @ Y_new @ invSig  # (k, n)
        mu = np.linalg.solve(Kbeta, rhs.reshape((dim,), order="F"))
        U = np.linalg.cholesky(Kbeta).T      # upper
        beta_vec = mu + np.linalg.solve(U, np.random.standard_normal(dim))
        beta = beta_vec.reshape((n * lag + 1, n), order="F")

        # (c) sample invSig (Wishart)
        err = Y_new - X @ beta
        scale_inv = 100.0 * np.eye(n) + (err.T * Dexp) @ err
        scale_inv = 0.5 * (scale_inv + scale_inv.T)
        Swish = np.linalg.inv(scale_inv)
        Swish = 0.5 * (Swish + Swish.T)
        invSig = wishart.rvs(df=Tnew + n + 3, scale=Swish)
        invSig = 0.5 * (invSig + invSig.T)

        # (d) common stochastic volatility
        try:
            chol_invSig = np.linalg.cholesky(invSig).T
        except np.linalg.LinAlgError:
            chol_invSig = np.linalg.cholesky(invSig + 1e-10 * np.eye(n)).T
        s2 = np.sum((err @ chol_invSig) ** 2, axis=1)
        h, is_accept = sample_CSV(s2, rho, sigh2, h, n, is_forced_accept=True)
        accept_count += is_accept
        eh = h[1:] - rho * h[:-1]
        sigh2 = 1.0 / np.random.gamma(
            10.0 + Tnew / 2.0, 1.0 / (0.004 + np.sum(eh ** 2) / 2.0)
        )
        K_rho = h[:-1] @ h[:-1] / sigh2 + 100.0
        rho = (h[:-1] @ h[1:] / sigh2) / K_rho + np.random.standard_normal() / math.sqrt(K_rho)

        # ---- store retained draws --------------------------------------
        if it >= Burn and ((it - Burn) % thin == 0) and store_idx < ndraws:
            d = store_idx
            # Phi in SS layout: [lag1; ...; lagp; const]
            Phi = np.vstack([beta[1:, :], beta[0:1, :]])
            Phip[d, :, :] = Phi
            scale_vol = math.exp(float(h[-1]))
            Sigmap[d, :, :] = Sig * scale_vol

            YYactsim_list[d, :, :] = Y_new_full[-(rqw + 1):, :]
            xx = np.concatenate(
                [Y_new_full[-1 - j, :] for j in range(1, lag + 1)] + [[1.0]]
            )
            XXactsim_list[d, -1, :] = xx
            lstate_list[d, :, :] = Y_new_full[rqw:, n_high:]
            mh[d, :] = h
            store_idx += 1

    # trim in case fewer draws stored than allocated
    valid = list(range(store_idx))

    # ---- expose SS-compatible attributes -------------------------------
    self.Phip = Phip
    self.Sigmap = Sigmap
    self.YYactsim_list = YYactsim_list
    self.XXactsim_list = XXactsim_list
    self.lstate_list = lstate_list
    self.mh = mh
    self.valid_draws = valid
    self.explosive_counter = 0
    self.nv = Ntotal
    self.Nw = Nw
    self.Nm = Nm
    self.Nq = Nq
    self.nburn = 0
    self.freq_ratio = list(mufbvar_data.freq_ratio_list)
    self.rqw = rqw
    self.rmw = rmw
    self.YMX_list = mufbvar_data.YMX_list
    self.varlist = varlist
    self.select = select_combined
    self.select_w = select_combined[:Nw]
    self.select_m_q = select_combined[Nw:]

    # weekly (highest-frequency) history and datetime index truncated to T
    self.input_data_W = np.asarray(datasets[-1], dtype=float)[:T].copy()
    self.input_data_M = np.asarray(datasets[1], dtype=float) if L > 1 else None
    self.input_data_Q = np.asarray(datasets[0], dtype=float)
    idx = copy.deepcopy(mufbvar_data.index_list[-1])
    self.index_list = [idx[:T]]
    self.input_index_M_W = mufbvar_data.input_data
    self.input_index_Q = mufbvar_data.input_data_Q

    print(f"CPZ sampler finished. SV acceptance rate: "
          f"{accept_count / max(it_total, 1):.3f}", end="\n")

    if return_mdd:
        return np.nan
    return None


def forecast_cpz(self, H, conditionals=None):
    """Forecast for the Chan, Poon & Zhu path.

    Because :func:`fit_cpz` stores its posterior draws in the same attribute
    shapes the Schorfheide-Song forecaster consumes, this simply delegates to
    the shared forecasting implementation.
    """
    return self._forecast_ss(H, conditionals)
