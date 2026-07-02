"""Example: Chan, Poon & Zhu (2024) mixed-frequency VAR with the SBFVAR package.

This mirrors ``python_example.py`` but selects the Chan, Poon & Zhu (CPZ)
conditionally-Gaussian estimator with common stochastic volatility instead of
the default Schorfheide-Song (2015) approach.  The public API is identical: the
only change is passing ``method="chan_poon_zhu"`` to :func:`fit`.
"""

import SBFVAR
import pandas as pd
import numpy as np


# Preparations
# ---------------------

io_data = "examples/hist_small.xlsx"

# Model Specification
H = 96          # forecast horizon (highest frequency)
nsim = 1000     # number of draws from the posterior density
nburn = 0.5     # fraction of draws to discard as burn-in
nlags = 4       # number of lags
thining = 1     # thinning

hyp = [0.09, 4.3, 1, 2.7, 4.3]  # hyperparameters, see documentation for details

frequencies = ["Q", "M", "W"]   # frequencies (lowest -> highest)


# Load the data
# --------------
data = []
for freq in frequencies:
    data_temp = pd.read_excel(io_data, sheet_name=freq, index_col=0)
    data.append(data_temp)

# Transformations
trans = [np.array((1, 1)), np.array((1, 1)), np.array((1, 1))]


# Initialize data class
mufbvar_data = SBFVAR.sbfvar_data(data, trans, frequencies)


# Fit and Forecast
# --------------------

# Initialize model class
model = SBFVAR.multifrequency_var(nsim, nburn, nlags, thining)

# Estimate the model using the Chan, Poon & Zhu approach
model.fit(
    mufbvar_data,
    hyp=hyp,
    var_of_interest=None,
    temp_agg="mean",
    check_explosive=False,
    method="chan_poon_zhu",
)

# Create forecasts in the highest frequency
model.forecast(H)
model.to_excel("cpz_test.xlsx")

# Aggregate to the lowest frequency
model.aggregate(frequency="Q")
model.to_excel("cpz_test_q.xlsx", agg=True)


# Hyperparameter tuning (optional)
# --------------------------------
# The CPZ path is tuned by minimising an out-of-sample RMSE on held-out
# lowest-frequency observations (rather than the marginal data density used by
# the Schorfheide-Song tuner).
#
# param_space = {
#     "lambda1_1": [0.01, 1.0],
#     "lambda2_1": [1.0, 10.0],
#     "lambda4_1": [1.0, 5.0],
#     "lambda5_1": [1.0, 10.0],
# }
# best_hyp = model.update_hyperparameters_mango_rmse(
#     mufbvar_data,
#     param_space,
#     H=None,        # derived from n_eval when None
#     init_points=5,
#     n_iter=20,
#     nsim=200,
#     njobs=1,
#     n_eval=4,      # hold out the last 4 quarters
# )
# print("Best hyperparameters:", best_hyp)
