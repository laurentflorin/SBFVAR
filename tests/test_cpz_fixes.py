import importlib.util
import sys
import types
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def _load_cpz_modules():
    root = Path(__file__).resolve().parents[1] / "SBFVAR"
    pkg = types.ModuleType("SBFVAR")
    pkg.__path__ = [str(root)]
    sys.modules["SBFVAR"] = pkg

    out = {}
    for name in ("_cpz_funcs", "_estimation_cpz"):
        spec = importlib.util.spec_from_file_location(
            f"SBFVAR.{name}", root / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"SBFVAR.{name}"] = mod
        spec.loader.exec_module(mod)
        out[name] = mod
    return out["_cpz_funcs"], out["_estimation_cpz"]


CPZ_FUNCS, CPZ_EST = _load_cpz_modules()


class TestCPZFixes(unittest.TestCase):
    def test_high_frequency_nan_is_latent_not_observed_zero(self):
        q = np.array([[1.0], [2.0]])
        m = np.arange(10.0, 16.0).reshape(-1, 1)
        w = np.arange(24.0).reshape(-1, 1)
        w[5, 0] = np.nan

        yraw, block_info = CPZ_FUNCS.build_stacked_data([q, m, w], [12, 4, 1])
        sel = CPZ_FUNCS.build_selection_matrices(yraw, block_info, [q, m, w], 1)

        missing_hf_row = 5 * sel["n"]
        obs_rows = set(sel["M_o"].nonzero()[0])
        latent_rows = set(sel["M_u"].nonzero()[0])

        self.assertNotIn(missing_hf_row, obs_rows)
        self.assertIn(missing_hf_row, latent_rows)
        np.testing.assert_array_equal(
            sel["vecY"], yraw[:, 0][np.isfinite(yraw[:, 0])]
        )

    def test_lower_frequency_nan_skips_aggregation_constraint(self):
        q = np.array([[1.0], [np.nan]])
        m = np.arange(10.0, 16.0).reshape(-1, 1)
        m[2, 0] = np.nan
        w = np.arange(24.0).reshape(-1, 1)

        yraw, block_info = CPZ_FUNCS.build_stacked_data([q, m, w], [12, 4, 1])
        sel = CPZ_FUNCS.build_selection_matrices(yraw, block_info, [q, m, w], 1)

        self.assertFalse(np.isnan(sel["Y_con"]).any())
        self.assertEqual(sel["M_a"].shape[1], 4)
        np.testing.assert_array_equal(sel["Y_con"], np.array([11.0, 13.0, 14.0, 15.0]))

    def test_get_resid_var_is_nan_safe(self):
        y = np.arange(12.0).reshape(-1, 1)
        y[6, 0] = np.nan
        sig2 = CPZ_FUNCS.get_resid_var(y)

        self.assertEqual(sig2.shape, (1,))
        self.assertTrue(np.isfinite(sig2).all())
        self.assertGreater(sig2[0], 0.0)

        sparse = np.array([[np.nan], [1.0], [np.nan], [np.nan], [np.nan]])
        sparse_sig2 = CPZ_FUNCS.get_resid_var(sparse)
        self.assertTrue(np.isfinite(sparse_sig2).all())
        self.assertGreater(sparse_sig2[0], 0.0)

    def test_minnesota_returns_precision(self):
        inv_vbeta = CPZ_FUNCS.construct_minnesota(
            np.array([2.0, 8.0]), n=2, lag=1, theta=[0.5, 0.2, 10.0, 1.0]
        )
        expected = np.array([0.05, 2.0, 40.0, 0.0125, 2.5, 2.0])
        np.testing.assert_allclose(inv_vbeta.diagonal(), expected)

    def test_cpz_rejects_sum_aggregation(self):
        class Model:
            pass

        model = Model()
        with self.assertRaisesRegex(ValueError, "temp_agg='mean' only"):
            CPZ_EST.fit_cpz(model, object(), hyp=[0.1, 0.5, 1.0, 10.0, 1.0],
                            temp_agg="sum")

    def test_tiny_complete_cpz_smoke(self):
        class Data:
            pass

        class Model:
            pass

        np.random.seed(123)
        q = np.linspace(0.10, 0.20, 2).reshape(-1, 1)
        m = np.linspace(0.05, 0.16, 6).reshape(-1, 1)
        w = np.linspace(0.01, 0.24, 24).reshape(-1, 1)

        data = Data()
        data.frequencies = ["Q", "M", "W"]
        data.YQ0_list = deque([q])
        data.YM0_list = deque([m, w])
        data.varlist_list = deque([
            np.array(["m", "q"]),
            np.array(["w", "m", "q"]),
        ])
        data.select_list = deque([np.array([1, 1]), np.array([1, 1, 1])])
        data.freq_ratio_list = [12, 4]
        data.YMX_list = deque([
            pd.DataFrame(m, columns=["m"]),
            pd.DataFrame(w, columns=["w"]),
        ])
        data.input_data = data.YMX_list.copy()
        data.input_data_Q = pd.DataFrame(q, columns=["q"])
        data.index_list = deque([
            pd.date_range("2020-01-31", periods=6, freq="ME"),
            pd.date_range("2020-01-05", periods=24, freq="W"),
        ])

        model = Model()
        model.nsim = 2
        model.nburn_perc = 0.0
        model.nlags = 1
        model.thining = 1

        CPZ_EST.fit_cpz(
            model,
            data,
            hyp=[0.1, 0.5, 1.0, 10.0, 1.0],
            check_explosive=False,
        )

        self.assertEqual(len(model.valid_draws), 2)
        self.assertTrue(np.isfinite(model.Phip).all())
        self.assertTrue(np.isfinite(model.Sigmap).all())
        self.assertTrue(np.isfinite(model.lstate_list).all())


if __name__ == "__main__":
    unittest.main()
