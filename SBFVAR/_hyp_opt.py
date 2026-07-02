import numpy as np


def _estim(self, mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg):
    """Call fit() with return_mdd=True, temporarily overriding nsim."""
    original_nsim = self.nsim
    self.nsim = nsim
    mdd = None
    try:
        mdd = self.fit(mufbvar_data, hyp_list, var_of_interest=var_of_interest,
                       temp_agg=temp_agg, return_mdd=True)
    except NameError:
        # fit() raises NameError('No Stable VAR at j=0') after exhausting
        # 100 full MCMC restarts due to explosive VAR draws.  Return the
        # penalty value so the optimiser can continue rather than crashing.
        print("No stable VAR found after maximum restarts, returning penalty value.")
        return -1e16
    finally:
        self.nsim = original_nsim
    # NaN-handling: prevent NaN/inf from propagating to Mango's optimiser.
    # Mirrors the same guard used in MBFVAR/_hyp_opt.py.
    if np.isnan(mdd) or np.isinf(mdd):
        print("MDD is NaN or inf, returning penalty value.")
        return -1e16
    return mdd


def update_hyperparameters(self, mufbvar_data, pbounds, init_points, n_iter, nsim,
                           var_of_interest=None, temp_agg='mean', save=False, name="hyp.txt"):
    '''
    Uses Bayesian optimization to find hyperparameters with the highest MDD.

    NOTE: This function calls the main fit() function with return_mdd=True,
    ensuring hyperparameters are optimized using the same model as estimation.

    Parameters
    ----------
    mufbvar_data : sbfvar_data class object
    pbounds : dict
        Boundaries for each hyperparameter (lambda1, lambda2, lambda4, lambda5)
    init_points : int
        Number of random exploration steps
    n_iter : int
        Number of Bayesian optimization steps
    nsim : int
        Number of draws in each estimation
    var_of_interest : list or None
    temp_agg : str
    save : bool
    name : str

    Returns
    -------
    hyp : list
    '''
    from bayes_opt import BayesianOptimization

    def calc_mdd_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        hyp_list = [lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]
        return _estim(self, mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)

    optimizer = BayesianOptimization(
        f=calc_mdd_1,
        pbounds=pbounds,
        verbose=2,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    hyp_opt = optimizer.max
    values = list(hyp_opt["params"].values())
    hyp = [values[0], values[1], 1, values[2], values[3]]

    if save:
        with open(name, 'w') as f:
            print(hyp, file=f)

    return hyp


def update_hyperparameters_mango(self, mufbvar_data, param_space, init_points, n_iter,
                                  nsim, njobs, var_of_interest=None, temp_agg='mean',
                                  save=False, name="hyp.txt"):
    '''
    Uses Mango Bayesian optimization to find hyperparameters with the highest MDD.

    NOTE: This function calls the main fit() function with return_mdd=True,
    ensuring hyperparameters are optimized using the same model as estimation.

    Parameters
    ----------
    mufbvar_data : sbfvar_data class object
    param_space : dict
    init_points : int
    n_iter : int
    nsim : int
    njobs : int
    var_of_interest : list or None
    temp_agg : str
    save : bool
    name : str

    Returns
    -------
    hyp : list
    '''
    from mango import scheduler, Tuner

    @scheduler.parallel(n_jobs=njobs)
    def calc_mdd_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        hyp_list = [lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]
        return _estim(self, mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg)

    conf_dict = dict(
        num_iteration=n_iter,
        initial_random=init_points,
    )

    tuner = Tuner(param_space, calc_mdd_1, conf_dict)
    results = tuner.maximize()
    best_params = results["best_params"]

    values = list(best_params.values())
    hyp = [values[0], values[1], 1, values[2], values[3]]

    if save:
        with open(name, 'w') as f:
            print(hyp, file=f)

    return hyp


def _rmse_holdout(self, mufbvar_data_in, hyp_list, H, nsim, var_of_interest,
                  temp_agg, method, h_eval, n_eval):
    '''
    Rolling-origin holdout RMSE for a single hyperparameter vector.

    Fits the requested ``method`` on an in-sample subset (dropping the last
    ``h_eval`` lowest-frequency observations), forecasts, aggregates to the
    lowest frequency, and compares the aggregated forecast against the held-out
    actuals.  Numerical failures return a large penalty so the optimiser keeps
    running rather than crashing joblib.
    '''
    import numpy as _np

    # Import lazily to avoid a circular import at module load time.
    from . import sbfvar_data

    try:
        frequencies = list(mufbvar_data_in.frequencies)
        # Raw (untransformed) frames, ordered lowest -> highest frequency.
        raw = [mufbvar_data_in.input_data_Q] + list(mufbvar_data_in.input_data)
        trans = [_np.asarray(mufbvar_data_in.select_q[0])]
        trans += [_np.asarray(s) for s in mufbvar_data_in.select_m_list]

        low = raw[0]
        if n_eval <= 0 or n_eval >= low.shape[0]:
            return 1e10
        cutoff = low.index[-n_eval]

        # Trim every frequency to strictly before the first held-out period.
        raw_in = [df.loc[df.index < cutoff] for df in raw]
        data_in = sbfvar_data(list(raw_in), list(trans), frequencies)

        # Fit / forecast / aggregate on the in-sample data.
        original_nsim = self.nsim
        self.nsim = nsim
        try:
            self.fit(data_in, hyp_list, var_of_interest=var_of_interest,
                     temp_agg=temp_agg, check_explosive=False, method=method)
            self.forecast(H)
            self.aggregate(frequency=frequencies[0])
        finally:
            self.nsim = original_nsim

        fcst = self.YY_mean_agg.copy()
        # Compare on the held-out lowest-frequency actuals.
        actual = low.iloc[-n_eval:]
        cols = list(actual.columns)
        if var_of_interest is not None:
            cols = [c for c in cols if c in var_of_interest] or cols

        fcst_idx = fcst.index.to_timestamp() if hasattr(fcst.index, "to_timestamp") else fcst.index
        fcst.index = fcst_idx

        errs = []
        for date in actual.index:
            if date in fcst.index:
                for c in cols:
                    if c in fcst.columns:
                        errs.append((float(fcst.loc[date, c]) - float(actual.loc[date, c])) ** 2)
        if not errs:
            return 1e10
        rmse = float(_np.sqrt(_np.mean(errs)))
        if _np.isnan(rmse) or _np.isinf(rmse):
            return 1e10
        return rmse
    except Exception as exc:  # noqa: BLE001 - guard optimiser against crashes
        print(f"RMSE evaluation failed ({exc}); returning penalty value.")
        return 1e10


def update_hyperparameters_mango_rmse(self, mufbvar_data_in, param_space, H,
                                      init_points, n_iter, nsim, njobs,
                                      var_of_interest=None, temp_agg='mean',
                                      method='chan_poon_zhu', h_eval=None,
                                      n_eval=1, save=False, name="hyp.txt"):
    '''
    Mango RMSE-based hyperparameter tuner (used for the Chan, Poon & Zhu path).

    Unlike :func:`update_hyperparameters_mango`, this tuner does not use the
    marginal data density.  Instead it uses a rolling-origin holdout: for each
    candidate hyperparameter vector it fits on an in-sample subset, forecasts,
    aggregates to the lowest frequency and minimises the RMSE against the
    held-out actuals.

    Parameters
    ----------
    mufbvar_data_in : sbfvar_data
        Prepared (full-sample) data object.
    param_space : dict
        Mango parameter space over ``lambda1_1, lambda2_1, lambda4_1,
        lambda5_1``.
    H : int
        Forecast horizon (highest frequency) used for each evaluation.  When
        ``None`` it is derived from ``n_eval`` lowest-frequency periods.
    init_points : int
        Number of initial random evaluations.
    n_iter : int
        Number of Bayesian optimisation iterations.
    nsim : int
        Number of MCMC draws per evaluation.
    njobs : int
        Number of parallel jobs.
    var_of_interest : list or None
    temp_agg : str
    method : str
        Estimation method to tune (defaults to ``'chan_poon_zhu'``).
    h_eval : int or None
        Number of highest-frequency periods to forecast for evaluation; if
        ``None`` it is derived from ``n_eval`` lowest-frequency periods.
    n_eval : int
        Number of lowest-frequency periods to hold out.
    save : bool
    name : str

    Returns
    -------
    hyp : list
    '''
    from mango import scheduler, Tuner
    from ._cpz_funcs import build_frequency_ratios

    # Highest-frequency periods per one lowest-frequency period.
    _, ratios_to_highest = build_frequency_ratios(list(mufbvar_data_in.frequencies))
    ratio_low_high = ratios_to_highest[0]
    holdout_hf = h_eval if h_eval is not None else n_eval * ratio_low_high

    @scheduler.parallel(n_jobs=njobs)
    def calc_rmse_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        hyp_list = [lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]
        return _rmse_holdout(self, mufbvar_data_in, hyp_list, holdout_hf, nsim,
                             var_of_interest, temp_agg, method, holdout_hf,
                             n_eval)

    conf_dict = dict(
        num_iteration=n_iter,
        initial_random=init_points,
    )

    tuner = Tuner(param_space, calc_rmse_1, conf_dict)
    results = tuner.minimize()
    best_params = results["best_params"]

    values = list(best_params.values())
    hyp = [best_params["lambda1_1"], best_params["lambda2_1"], 1,
           best_params["lambda4_1"], best_params["lambda5_1"]]

    if save:
        with open(name, 'w') as f:
            print(hyp, file=f)

    return hyp
