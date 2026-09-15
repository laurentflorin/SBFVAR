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
            agg_identity="mean", **kwargs):
    """Estimate the mixed-frequency VAR using the Chan, Poon & Zhu approach.

    Implements the stacked conditionally-Gaussian state-space sampler with a
    single common stochastic-volatility process, generalised to an arbitrary
    number of frequencies.

    Parameters
    ----------
    mufbvar_data : sbfvar_data
        Prepared data object.
    hyp : ndarray
        Hyperparameter vector in the Chan-Poon-Zhu reference order
        ``[kappa1, kappa2, kappa3, kappa4]`` =
        ``[own-lag tightness, cross shrinkage, constant scale, lag-decay
        exponent]``, e.g. the reference values ``[0.2**2, 0.5**2, 100, 2]``.
        A fifth entry, if present, is ignored (public-API compatibility).
        A SIX-entry vector ``[k1, k2, k3, k4, mu_soc, mu_dio]`` additionally
        activates sum-of-coefficients dummies (tightness ``mu_soc``, one row
        per variable, built from each variable's own-frequency pre-sample
        mean) and a dummy-initial-observation row (tightness ``mu_dio``);
        zero disables either block, and the dummies inform the coefficient
        draw only (not the Wishart update of ``invSig`` nor the volatility
        step).

        .. warning:: Versions <= 0.1.3 transposed the last two entries
           internally (``theta = [hyp[0], hyp[1], hyp[3], hyp[2]]``), so the
           reference vector above was executed as constant scale 2 and
           lag-decay exponent 100 -- the latter pins every lag beyond the
           first to zero (prior variance ``kappa1 / l**100``).  Runs made
           with those versions are therefore effectively VAR(1) fits
           regardless of the nominal lag order.  The executed ``theta`` is
           now printed at fit time and stored as ``self.cpz_theta``.
    var_of_interest, temp_agg, check_explosive, max_it_explosive
        Kept for signature compatibility with the SS ``fit``.
    return_mdd : bool
        When ``True``, return the **conditional marginal data density** of the
        stacked CPZ system: at every retained draw the closed-form Gaussian
        marginal likelihood ``ln p(Y_completed | invSig, h)`` is evaluated
        with the VAR coefficients ``beta`` integrated out analytically under
        the Minnesota prior ``N(0, Vbeta)``,

        ``ln p = -nT'/2 ln(2pi) + T'/2 ln|invSig| - n/2 sum_t h_t
        + 1/2 ln|invVbeta| - 1/2 ln|Kbeta| - 1/2 (S - b' Kbeta^{-1} b)``,

        where ``Kbeta = kron(invSig, X'DX) + invVbeta`` is the posterior
        precision of ``vec(beta)``, ``b`` its linear term, and
        ``S = sum_t e^{-h_t} y_t' invSig y_t``.  The returned scalar is the
        MEAN of this quantity over the retained draws; the per-draw values are
        stored in ``self.cpz_mdd_draws``.  Like the Schorfheide-Song path's
        ``mdd_`` objective, this conditions on the drawn latent states (it is
        a plug-in conditional marginal likelihood, not the observed-data MDD),
        but unlike the SS objective it averages over retained draws instead of
        using the final draw only.  Its Occam term
        ``1/2 ln|invVbeta| - 1/2 ln|Kbeta|`` penalises the VAR dimension, so
        it is usable for lag-order selection at fixed hyperparameters.

    Returns
    -------
    float or None
    """
    self.hyp = np.asarray(hyp, dtype=float)
    self.temp_agg = temp_agg
    self.var_of_interest = var_of_interest
    self.method = "chan_poon_zhu"
    # Recorded so aggregate() can check it agrees with what it does to the
    # forecast path; see build_selection_matrices for why that matters.
    self.cpz_agg_identity = agg_identity
    if temp_agg == "sum":
        raise ValueError(
            "method='chan_poon_zhu' currently supports temp_agg='mean' only; "
            "sum aggregation constraints are not implemented in the CPZ path."
        )
    if temp_agg != "mean":
        raise ValueError(f"Invalid temp_agg: {temp_agg}. Choose 'mean'.")

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
    # Transformation flags per low-frequency block, in block_info's high-to-low
    # order. datasets is lowest-to-highest, so level 0 is the quarterly block
    # (select_q) and level L >= 1 is select_m_list[L - 1]; the constraint
    # builder needs them to give levels a mean identity even when growth
    # rates get the tent.
    # A data object without the flags (minimal test doubles) is treated as
    # all-growth, which under the mean identity changes nothing: mean weights
    # apply to growth and level alike. Only the tent needs the distinction.
    select_q = getattr(mufbvar_data, "select_q", None)
    select_m = getattr(mufbvar_data, "select_m_list", None)
    select_low = None
    if select_q is not None and select_m is not None:
        select_low = []
        for blk in block_info[1:]:
            lvl = blk["level"]
            flags = select_q[0] if lvl == 0 else select_m[lvl - 1]
            select_low.append(np.asarray(flags).ravel())
    sel = build_selection_matrices(Yraw, block_info, datasets, lag,
                                   select_low=select_low,
                                   agg_identity=agg_identity)
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
    ridge_dim = sel["ridge_dim"]

    # ---- Minnesota prior -----------------------------------------------
    # AR(4) residual variances in the stacked (high -> low) variable order.
    sig2_parts = []
    for blk in block_info:
        sig2_parts.append(get_resid_var(datasets[blk["level"]]))
    sig2 = np.concatenate(sig2_parts)
    # CPZ's MATLAB-style Minnesota prior in the reference order
    # [own tightness, cross shrinkage, constant scale, lag-decay exponent].
    # (Versions <= 0.1.3 transposed the last two entries; see the docstring.)
    theta = [float(self.hyp[0]), float(self.hyp[1]),
             float(self.hyp[2]), float(self.hyp[3])]
    self.cpz_theta = list(theta)
    print(f"Minnesota theta [own, cross, const, lag-decay]: {theta}",
          end="\n")
    invVbeta = construct_minnesota(sig2, n, lag, theta).tocsc()
    dim = n * (n * lag + 1)
    # Prior log-determinant for the conditional-MDD evaluation (diagonal).
    logdet_invVbeta = float(np.sum(np.log(invVbeta.diagonal())))
    cpz_mdd_draws = []

    # ---- optional sum-of-coefficients / dummy-initial-observation priors
    # Activated by a 6-entry hyp vector [k1, k2, k3, k4, mu_soc, mu_dio];
    # zero tightness disables the corresponding block. The dummies enter the
    # beta step as extra observation rows with unit volatility weight (they
    # do not enter the Wishart update of invSig or the SV step), i.e. as a
    # Gaussian prior-mean/precision augmentation conditional on invSig.
    hyp_arr = np.asarray(self.hyp, dtype=float).ravel()
    mu_soc = float(hyp_arr[4]) if hyp_arr.size >= 6 else 0.0
    mu_dio = float(hyp_arr[5]) if hyp_arr.size >= 6 else 0.0
    if mu_soc < 0 or mu_dio < 0:
        raise ValueError("mu_soc and mu_dio must be non-negative.")
    self.cpz_mu_soc, self.cpz_mu_dio = mu_soc, mu_dio
    k_reg = n * lag + 1
    dummy_y, dummy_x = [], []
    if mu_soc > 0 or mu_dio > 0:
        # Pre-sample means per variable, taken from each variable's OWN
        # frequency data (fixed across iterations), stacked in block order.
        ybar = np.concatenate([
            np.nanmean(np.asarray(datasets[blk["level"]], dtype=float), axis=0)
            for blk in block_info])
        if mu_soc > 0:
            for i in range(n):
                y_row = np.zeros(n)
                y_row[i] = mu_soc * ybar[i]
                x_row = np.zeros(k_reg)
                for l in range(lag):
                    x_row[1 + l * n + i] = mu_soc * ybar[i]
                dummy_y.append(y_row)
                dummy_x.append(x_row)
        if mu_dio > 0:
            y_row = mu_dio * ybar
            x_row = np.zeros(k_reg)
            x_row[0] = mu_dio
            for l in range(lag):
                x_row[1 + l * n: 1 + (l + 1) * n] = mu_dio * ybar
            dummy_y.append(y_row)
            dummy_x.append(x_row)
    if dummy_y:
        Xd = np.vstack(dummy_x)
        Yd = np.vstack(dummy_y)
        XdtXd = Xd.T @ Xd
        XdtYd = Xd.T @ Yd
        print(f"CPZ dummy priors active: mu_soc={mu_soc}, mu_dio={mu_dio} "
              f"({Yd.shape[0]} dummy rows)", end="\n")
    else:
        XdtXd = XdtYd = None

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
        if XdtXd is not None:
            XtDX = XtDX + XdtXd
        Kbeta = np.kron(invSig, XtDX) + invVbeta.toarray()
        Kbeta = 0.5 * (Kbeta + Kbeta.T)
        rhs = (X.T * Dexp) @ Y_new @ invSig  # (k, n)
        if XdtYd is not None:
            rhs = rhs + XdtYd @ invSig
        b_vec = rhs.reshape((dim,), order="F")
        mu = np.linalg.solve(Kbeta, b_vec)
        U = np.linalg.cholesky(Kbeta).T      # upper
        beta_vec = mu + np.linalg.solve(U, np.random.standard_normal(dim))
        beta = beta_vec.reshape((n * lag + 1, n), order="F")

        # Conditional MDD ln p(Y_new | invSig, h) with beta integrated out
        # (see the fit_cpz docstring); evaluated at the same (invSig, h) the
        # beta step conditioned on, for retained draws only.
        if return_mdd and it >= Burn and ((it - Burn) % thin == 0):
            sign_isig, logdet_invSig = np.linalg.slogdet(invSig)
            if sign_isig <= 0:
                raise np.linalg.LinAlgError(
                    "invSig draw is not positive definite in the conditional-"
                    "MDD evaluation."
                )
            S_quad = float(np.sum((Y_new @ invSig) * Y_new * Dexp[:, None]))
            logdet_Kbeta = 2.0 * float(np.sum(np.log(np.diag(U))))
            if XdtXd is None:
                logdet_P0 = logdet_invVbeta
                quad0 = 0.0
            else:
                # Dummy-augmented prior: N(m0, P0^{-1}) with
                # P0 = invVbeta + kron(invSig, Xd'Xd), P0 m0 = b0.
                P0 = invVbeta.toarray() + np.kron(invSig, XdtXd)
                P0 = 0.5 * (P0 + P0.T)
                b0 = (XdtYd @ invSig).reshape((dim,), order="F")
                C0 = np.linalg.cholesky(P0)
                logdet_P0 = 2.0 * float(np.sum(np.log(np.diag(C0))))
                z0 = np.linalg.solve(C0, b0)
                quad0 = float(z0 @ z0)   # b0' P0^{-1} b0 = m0' P0 m0
            cond_mdd = (
                -0.5 * n * Tnew * math.log(2.0 * math.pi)
                + 0.5 * Tnew * logdet_invSig
                - 0.5 * n * float(np.sum(h))
                + 0.5 * logdet_P0
                - 0.5 * logdet_Kbeta
                - 0.5 * (S_quad + quad0 - float(b_vec @ mu))
            )
            cpz_mdd_draws.append(cond_mdd)

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

    self.cpz_mdd_draws = np.asarray(cpz_mdd_draws, dtype=float)
    if return_mdd:
        if self.cpz_mdd_draws.size == 0:
            raise RuntimeError(
                "return_mdd=True but no retained draws produced a "
                "conditional-MDD value."
            )
        return float(np.mean(self.cpz_mdd_draws))
    return None


def forecast_cpz(self, H, conditionals=None):
    """Forecast for the Chan, Poon & Zhu path.

    Because :func:`fit_cpz` stores its posterior draws in the same attribute
    shapes the Schorfheide-Song forecaster consumes, this simply delegates to
    the shared forecasting implementation.
    """
    return self._forecast_ss(H, conditionals)
