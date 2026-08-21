"""Regression test: fit_cpz must execute the Minnesota theta in the
Chan-Poon-Zhu reference order [own, cross, const, lag-decay].

Versions <= 0.1.3 transposed the last two entries, so the reference vector
[0.04, 0.25, 100, 2] was executed with a lag-decay exponent of 100, pinning
every lag beyond the first to zero. This test runs a micro fit on synthetic
two-frequency data and checks both the stored ``cpz_theta`` and the implied
prior variance of a lag-2 own coefficient.
"""

import unittest

import numpy as np
import pandas as pd


def _real_sbfvar():
    """Import the real installed package even if a sibling test replaced
    sys.modules['SBFVAR'] with a bare stub for standalone module loading."""
    import importlib
    import sys
    mod = sys.modules.get("SBFVAR")
    if mod is not None and not hasattr(mod, "sbfvar_data"):
        for k in [k for k in list(sys.modules)
                  if k == "SBFVAR" or k.startswith("SBFVAR.")]:
            del sys.modules[k]
        mod = None
    return mod or importlib.import_module("SBFVAR")


def _make_data():
    SBFVAR = _real_sbfvar()
    rng = np.random.default_rng(0)
    t_q = 24
    idx_q = pd.date_range("2000-01-01", periods=t_q, freq="QS")
    idx_m = pd.date_range("2000-01-01", periods=3 * t_q, freq="MS")
    idx_w = pd.date_range("2000-01-07", periods=12 * t_q, freq="7D")
    dq = pd.DataFrame({"Q1": rng.standard_normal(t_q) * 0.1}, index=idx_q)
    dm = pd.DataFrame({"M1": rng.standard_normal(3 * t_q) * 0.1}, index=idx_m)
    dw = pd.DataFrame({"W1": rng.standard_normal(12 * t_q) * 0.1}, index=idx_w)
    trans = [np.ones(1, dtype=int), np.ones(1, dtype=int),
             np.ones(1, dtype=int)]
    return SBFVAR.sbfvar_data([dq, dm, dw], trans, ["Q", "M", "W"])


class TestThetaMapping(unittest.TestCase):
    def test_reference_order_is_executed(self):
        SBFVAR = _real_sbfvar()
        from SBFVAR._cpz_funcs import construct_minnesota

        data_in = _make_data()
        model = SBFVAR.multifrequency_var(10, 0.5, 3, 1, seed=0)
        hyp = [0.2 ** 2, 0.5 ** 2, 100, 2]
        model.fit(data_in, hyp=hyp, method="chan_poon_zhu")

        np.testing.assert_allclose(model.cpz_theta, [0.04, 0.25, 100.0, 2.0],
                                   rtol=1e-12)

        # The implied prior variance of an own-lag-l coefficient must decay
        # as kappa1 / l**2, not kappa1 / l**100.
        iv = construct_minnesota(np.ones(2), 2, 3, model.cpz_theta).diagonal()
        variances = 1.0 / iv
        # layout per equation: [const, lag1 (n), lag2 (n), lag3 (n)]
        own_l1 = variances[1]          # equation 1, lag 1, own coefficient
        own_l2 = variances[3]          # equation 1, lag 2, own coefficient
        const_v = variances[0]
        self.assertAlmostEqual(own_l1, 0.04, places=12)
        self.assertAlmostEqual(own_l2, 0.04 / 4.0, places=12)
        self.assertAlmostEqual(const_v, 100.0, places=9)


if __name__ == "__main__":
    unittest.main()
