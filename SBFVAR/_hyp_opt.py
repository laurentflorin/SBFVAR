import numpy as np


def _estim(self, mufbvar_data, hyp_list, nsim, var_of_interest, temp_agg):
    """Call fit() with return_mdd=True, temporarily overriding nsim."""
    original_nsim = self.nsim
    self.nsim = nsim
    try:
        mdd = self.fit(mufbvar_data, hyp_list, var_of_interest=var_of_interest,
                       temp_agg=temp_agg, return_mdd=True)
    finally:
        self.nsim = original_nsim
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
