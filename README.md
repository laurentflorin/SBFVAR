# SBFVAR

**Single Base Frequency Bayesian VAR** — a Python package for handling, disaggregating, and forecasting mixed-frequency time-series data using Bayesian Vector Autoregression models in a state-space framework.

---

## Features

- **Mixed-frequency modelling**: combine weekly, monthly, and quarterly data in a single VAR model
- **State-space Kalman filter / smoother**: handles ragged-edge data and temporal aggregation constraints automatically
- **Minnesota prior**: Bayesian shrinkage with fully configurable hyperparameters
- **Bayesian hyperparameter optimisation**: automatic tuning via `arm-mango` or `bayesian-optimization`
- **Forecasting with conditionals**: impose hard constraints on any variable over the forecast horizon
- **Temporal aggregation**: convert high-frequency draws to any lower frequency
- **Excel & pickle export**: publish results in `.xlsx` or save/restore full model objects
- **R integration**: use from R via `reticulate`

---

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/laurentflorin/SBFVAR.git
```

### Development install

```bash
git clone https://github.com/laurentflorin/SBFVAR.git
cd SBFVAR
pip install -e .
```

### Build requirements

The package contains C++ extensions (pybind11 + Eigen) that are compiled from source at install time. You need:

- A C++14-compatible compiler (GCC ≥ 7, Clang ≥ 5, MSVC 2017+)
- Python headers (usually included with your Python installation)
- `pybind11` and `numpy` (installed automatically as build dependencies)

The vendored Eigen headers (`SBFVAR/third_party/eigen`) are included in the repository, so no separate Eigen installation is required.

---

## Quick Start

```python
import pandas as pd
import numpy as np
import SBFVAR

# 1. Load data — one DataFrame per frequency (lowest to highest)
df_quarterly = pd.read_excel("data.xlsx", sheet_name="quarterly", index_col=0, parse_dates=True)
df_monthly   = pd.read_excel("data.xlsx", sheet_name="monthly",   index_col=0, parse_dates=True)

# 2. Specify transformations per variable per frequency
#    0 = take log, 1 = divide by 100
trans_quarterly = np.array([0, 0, 1])   # e.g. GDP (log), CPI (log), interest rate (/100)
trans_monthly   = np.array([0, 1])      # e.g. IP (log), unemployment rate (/100)

# 3. Initialise the data object
data = SBFVAR.sbfvar_data(
    data=[df_quarterly, df_monthly],
    trans=[trans_quarterly, trans_monthly],
    frequencies=["Q", "M"],
)

# 4. Create and fit the model
model = SBFVAR.multifrequency_var(nsim=1000, nburn_perc=0.2, nlags=4, thining=1)

hyp = np.array([0.1, 0.5, 1.0, 100.0, 1.0])  # λ1 … λ5

model.fit(
    mufbvar_data=data,
    hyp=hyp,
    var_of_interest=None,
    temp_agg="mean",
    check_explosive=True,
)

# 5. Forecast 8 periods ahead
model.forecast(H=8, conditionals=None)

# 6. Export results to Excel
model.to_excel("forecast_output.xlsx", agg=True)

# 7. Aggregate forecast to quarterly frequency
model.aggregate(frequency="Q")

# 8. Save the fitted model
model.save("sbfvar_model")
```

---

## Package Architecture

```
SBFVAR/
├── __init__.py              # Public API: multifrequency_var, sbfvar_data
├── _estimation.py           # fit(), forecast(), aggregate() implementations
├── _save.py                 # to_excel(), save() implementations
├── mfbvar_funcs.py          # Helper functions (calc_yyact, is_explosive, …)
├── cholcov/                 # C++ extension: Cholesky / eigendecomposition
│   ├── cholcov.cpp
│   ├── cholcov_bindings.cpp
│   └── cholcov.h
├── inverse/                 # C++ extension: matrix inversion (Eigen)
│   └── matrix_inversion.cpp
├── pseudo_inverse/          # C++ extension: Moore-Penrose pseudo-inverse
│   ├── pseudo_inverse.cpp
│   └── pseudo_inverse_bindings.cpp
├── solve/                   # C++ extension: linear system solver (Eigen)
│   └── solve.cpp
└── third_party/
    └── eigen/               # Vendored Eigen linear algebra library
```

---

## API Reference

### `SBFVAR.sbfvar_data(data, trans, frequencies)`

Prepares and transforms raw data for use with `multifrequency_var`.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `list[pd.DataFrame]` | One DataFrame per frequency, ordered lowest to highest. Each DataFrame must have a `DatetimeIndex`. |
| `trans` | `list[np.ndarray]` | One integer array per frequency. `0` = take log; `1` = divide by 100. |
| `frequencies` | `list[str]` | Frequency codes ordered lowest to highest. Supported: `"Y"`, `"Q"`, `"M"`, `"W"`, `"D"`. |

**Key attributes after construction**

| Attribute | Description |
|-----------|-------------|
| `YDATA_list` | Transformed, combined data arrays at the highest frequency |
| `varlist_list` | Variable names at each frequency level |
| `Nm_list` | Number of high-frequency variables per level |
| `Nq_list` | Number of low-frequency variables per level |
| `freq_ratio_list` | Frequency conversion ratios |

---

### `SBFVAR.multifrequency_var(nsim, nburn_perc, nlags, thining)`

Creates a Bayesian mixed-frequency VAR model instance.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `nsim` | `int` | Total number of MCMC draws |
| `nburn_perc` | `float` | Proportion of draws to discard as burn-in (0–1) |
| `nlags` | `int` | Number of lags in the VAR (at the highest frequency) |
| `thining` | `int` | Thinning factor — keep every *n*-th draw |

---

### `model.fit(mufbvar_data, hyp, var_of_interest=None, temp_agg='mean', max_it_explosive=1000, check_explosive=True)`

Estimates the model via MCMC.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mufbvar_data` | `sbfvar_data` | — | Prepared data object |
| `hyp` | `np.ndarray` | — | Hyperparameter vector `[λ1, λ2, λ3, λ4, λ5]` |
| `var_of_interest` | `str \| None` | `None` | Variable name to track separately |
| `temp_agg` | `str` | `'mean'` | Temporal aggregation rule: `'mean'` or `'sum'` |
| `max_it_explosive` | `int` | `1000` | Maximum retries when an explosive draw is detected |
| `check_explosive` | `bool` | `True` | Whether to reject explosive VAR draws |

---

### `model.forecast(H, conditionals=None)`

Generates out-of-sample forecasts.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `H` | `int` | — | Forecast horizon (in units of the highest frequency) |
| `conditionals` | `dict \| None` | `None` | Dictionary `{variable_name: array_of_length_H}` imposing hard constraints on specific variables over the forecast horizon |

---

### `model.aggregate(frequency, reset_index=True)`

Aggregates forecast draws to a lower frequency.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frequency` | `str` | — | Target frequency: `"Y"`, `"Q"`, `"M"`, `"W"`, or `"D"` |
| `reset_index` | `bool` | `True` | Whether to reset the time index after aggregation |

---

### `model.to_excel(filename, agg=False, include_metadata=True)`

Exports forecast results to an Excel workbook.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | — | Output file path (`.xlsx` extension added automatically if absent) |
| `agg` | `bool` | `False` | If `True`, also exports aggregated (lower-frequency) results |
| `include_metadata` | `bool` | `True` | Include model metadata sheet |

---

### `model.save(filename='sbfvar_model')`

Serialises the fitted model to a pickle file for later use.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | `'sbfvar_model'` | Output file path (`.pkl` extension added automatically if absent) |

---

## Hyperparameters

The `hyp` vector passed to `fit()` contains five Minnesota-prior hyperparameters:

| Symbol | Position | Description |
|--------|----------|-------------|
| λ1 | `hyp[0]` | Overall tightness — controls how much the prior shrinks coefficients towards zero. Smaller values → more shrinkage. |
| λ2 | `hyp[1]` | Cross-variable shrinkage — relative weight on coefficients of other variables versus own lags. |
| λ3 | `hyp[2]` | Lag decay — controls how quickly prior tightness increases with lag order. |
| λ4 | `hyp[3]` | Exogenous/constant tightness. |
| λ5 | `hyp[4]` | Covariance prior scaling. |

---

## Data Transformations

The `trans` arrays control per-variable pre-processing applied inside `sbfvar_data`:

| Code | Transformation |
|------|---------------|
| `0` | Natural logarithm (`np.log`) |
| `1` | Divide by 100 |

Transformations are applied once, in-place, before model estimation. Forecasts are returned in the transformed space; back-transformation must be performed by the user.

---

## R Integration

Use `reticulate` to call SBFVAR from R or RStudio.

```r
library(reticulate)

# Import the package
sbfvar <- reticulate::import("SBFVAR")
np     <- reticulate::import("numpy")
pd     <- reticulate::import("pandas")

# Load data
df_q <- pd$read_excel("data.xlsx", sheet_name = "quarterly",
                       index_col = 0L, parse_dates = TRUE)
df_m <- pd$read_excel("data.xlsx", sheet_name = "monthly",
                       index_col = 0L, parse_dates = TRUE)

trans_q <- np$array(c(0L, 0L, 1L))
trans_m <- np$array(c(0L, 1L))

# Prepare data
data <- sbfvar$sbfvar_data(
  data        = list(df_q, df_m),
  trans       = list(trans_q, trans_m),
  frequencies = list("Q", "M")
)

# Fit
hyp   <- np$array(c(0.1, 0.5, 1.0, 100.0, 1.0))
model <- sbfvar$multifrequency_var(nsim = 1000L, nburn_perc = 0.2,
                                   nlags = 4L, thining = 1L)
model$fit(mufbvar_data = data, hyp = hyp)

# Forecast and export
model$forecast(H = 8L)
model$to_excel("forecast_output.xlsx", agg = TRUE)
```

### RStudio Setup

To use SBFVAR in RStudio you need a Python virtual environment with the package installed.

1. In RStudio click **Tools → Global Options**

   ![Global Options](./readme_images/global_options.png)

2. Go to **Python** and click **Select…**

   ![Python Options](./readme_images/python.png)

3. Go to the **Virtual Environments** tab and select the environment that contains your Python installation

   ![Virtual Environments](./readme_images/virtenv.png)

4. Install SBFVAR into that environment:

   ```bash
   pip install git+https://github.com/laurentflorin/SBFVAR.git
   ```

---

## Examples

See the [`examples/`](./examples/) directory for complete worked examples in Python and R.

---

## Requirements

- Python ≥ 3.8
- numpy
- scipy
- pandas
- matplotlib
- tqdm
- plotly
- fanchart
- openpyxl
- xlsxwriter
- bayesian-optimization
- arm-mango
- scikit-learn
- datetime

**Build-time only** (not required at runtime):
- pybind11
- A C++14-compatible compiler

---

## How It Works

SBFVAR implements a **state-space mixed-frequency Bayesian VAR** with a single base frequency. The key idea is to embed all variables — regardless of their native observation frequency — into a common high-frequency state vector. Low-frequency observations are modelled as temporal aggregates (means or sums) of the latent high-frequency state via the measurement equation.

1. **State equation** — a VAR(*p*) governs the dynamics of the latent high-frequency state vector.
2. **Measurement equation** — observed low- and high-frequency data are linked to the state via frequency-specific aggregation matrices (Kalman filter).
3. **Minnesota prior** — provides Bayesian shrinkage on the VAR coefficients, with hyperparameters λ1–λ5 controlling the strength and pattern of shrinkage.
4. **MCMC sampling** — the posterior is explored via Gibbs sampling: draw VAR coefficients and error covariance jointly, then run the Kalman filter/smoother to draw the latent state path, iterating until convergence.
5. **Forecasting** — the companion-form VAR is propagated forward; optional conditional forecasts impose exact constraints on selected variables.

---

## Citation

If you use SBFVAR in your research, please cite:

> Florin, L. (2024). *Real-Time Forecasting with Multiple Frequency VARs*. Swiss Federal Finance Administration.

A pre-print / working paper is available in [`docs/`](./docs/).

---

## License

This project is licensed under the MIT License.

---

## Author

**Laurent Florin** — [laurent.florin@efv.admin.ch](mailto:laurent.florin@efv.admin.ch)  
Swiss Federal Finance Administration (EFV)

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/laurentflorin/SBFVAR).