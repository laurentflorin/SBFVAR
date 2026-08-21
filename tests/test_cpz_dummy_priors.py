"""Regression tests for the CPZ sum-of-coefficients / dummy-initial-
observation priors added in SBFVAR 0.1.5 (6-entry hyp vector).

1. A 6-entry hyp with mu_soc = mu_dio = 0 must reproduce the 4-entry
   behaviour bit-for-bit (same seed, identical stored draws): the dummy
   machinery must be inert when disabled.
2. Positive tightness must actually change the posterior draws and must
   pull the sum of own-lag coefficients toward the sum-of-coefficients
   target relative to the no-dummy fit (checked in expectation over stored
   draws on a persistent synthetic series).
"""

import unittest

import numpy as np
import pandas as pd


def _real_sbfvar():
    import importlib
    import sys
    mod = sys.modules.get("SBFVAR")
    if mod is not None and not hasattr(mod, "sbfvar_data"):
        for k in [k for k in list(sys.modules)
                  if k == "SBFVAR" or k.startswith("SBFVAR.")]:
            del sys.modules[k]
        mod = None
    return mod or importlib.import_module("SBFVAR")


def _make_data(seed=0, persistent=False):
    SBFVAR = _real_sbfvar()
    rng = np.random.default_rng(seed)
    t_q = 24
    idx_q = pd.date_range("2000-01-01", periods=t_q, freq="QS")
    idx_m = pd.date_range("2000-01-01", periods=3 * t_q, freq="MS")
    idx_w = pd.date_range("2000-01-07", periods=12 * t_q, freq="7D")

    def series(n):
        e = rng.standard_normal(n) * 0.1
        if not persistent:
            return e
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.8 * x[t - 1] + e[t]
        return x + 0.5

    dq = pd.DataFrame({"Q1": series(t_q)}, index=idx_q)
    dm = pd.DataFrame({"M1": series(3 * t_q)}, index=idx_m)
    dw = pd.DataFrame({"W1": series(12 * t_q)}, index=idx_w)
    trans = [np.ones(1, dtype=int)] * 3
    return SBFVAR.sbfvar_data([dq, dm, dw], trans, ["Q", "M", "W"])


def _fit(hyp, nsim=12, nlags=2, seed=0, persistent=False):
    SBFVAR = _real_sbfvar()
    model = SBFVAR.multifrequency_var(nsim, 0.5, nlags, 1, seed=seed)
    model.fit(_make_data(persistent=persistent), hyp=hyp,
              method="chan_poon_zhu")
    return model


class TestDummyPriors(unittest.TestCase):
    def test_zero_tightness_is_bit_identical_to_length4(self):
        m4 = _fit([0.04, 0.25, 100, 2])
        m6 = _fit([0.04, 0.25, 100, 2, 0.0, 0.0])
        np.testing.assert_array_equal(m4.Phip, m6.Phip)
        np.testing.assert_array_equal(m4.Sigmap, m6.Sigmap)
        self.assertEqual(m6.cpz_mu_soc, 0.0)
        self.assertEqual(m6.cpz_mu_dio, 0.0)

    def test_positive_tightness_changes_draws(self):
        m0 = _fit([0.04, 0.25, 100, 2], persistent=True)
        m1 = _fit([0.04, 0.25, 100, 2, 5.0, 5.0], persistent=True)
        self.assertFalse(np.array_equal(m0.Phip, m1.Phip))

    def test_soc_pulls_lag_sum_toward_unity(self):
        # On persistent data with a mean well away from zero, a strong SOC
        # prior should move each equation's sum of own-lag coefficients
        # closer to 1 than the plain-Minnesota fit (posterior-mean check).
        m0 = _fit([0.04, 0.25, 100, 2], nsim=40, persistent=True)
        m1 = _fit([0.04, 0.25, 100, 2, 50.0, 0.0], nsim=40, persistent=True)

        def own_lag_sums(model, n=3, lags=2):
            # Phi rows: [lag1 (n), lag2 (n), const]; columns = equations
            phi = model.Phip.mean(axis=0)
            return np.array([sum(phi[l * n + i, i] for l in range(lags))
                             for i in range(n)])

        d0 = np.abs(own_lag_sums(m0) - 1.0)
        d1 = np.abs(own_lag_sums(m1) - 1.0)
        self.assertLess(d1.mean(), d0.mean())


if __name__ == "__main__":
    unittest.main()
