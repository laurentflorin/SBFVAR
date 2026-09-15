"""
The intertemporal-aggregation identity of the CPZ path.

For three complete 192-origin runs every quarterly SBF-CPZ forecast was
emitted at 1/12 of its value and every monthly level at 1/4: the constraint
used Mariano-Murasawa tent weights (column sum m) on every low-frequency
variable, and aggregate() then took a simple mean of the latent path. The
unemployment rate stepped from 5.57 to 1.37 at the first forecast row, and
the paper's 35-40% CPZ margin over the SBF-VAR was that factor of twelve.
These tests pin the identity so the constraint and the aggregation cannot
disagree again.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load():
    root = Path(__file__).resolve().parents[1] / "SBFVAR"
    pkg = types.ModuleType("SBFVAR")
    pkg.__path__ = [str(root)]
    sys.modules["SBFVAR"] = pkg
    spec = importlib.util.spec_from_file_location(
        "SBFVAR._cpz_funcs", root / "_cpz_funcs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["SBFVAR._cpz_funcs"] = mod
    spec.loader.exec_module(mod)
    return mod


CPZ = _load()


def _system(select_q, select_m, agg_identity):
    """One quarterly and one monthly variable over 24 weeks (2 quarters,
    6 months), no missing values, so every constraint column is present."""
    q = np.array([[1.0], [2.0]])
    m = np.arange(10.0, 16.0).reshape(-1, 1)
    w = np.arange(24.0).reshape(-1, 1)
    yraw, block_info = CPZ.build_stacked_data([q, m, w], [12, 4, 1])
    return CPZ.build_selection_matrices(
        yraw, block_info, [q, m, w], 1,
        select_low=[np.array([select_m]), np.array([select_q])],
        agg_identity=agg_identity), block_info


def _column_sums(sel):
    return np.asarray(sel["M_a"].sum(axis=0)).ravel()


class TestAggregationIdentity(unittest.TestCase):

    def test_mean_identity_weights_sum_to_one_for_every_variable(self):
        """The observation is the period mean of the latent, so the
        constraint weights must sum to one -- for growth rates and levels
        alike. This is the identity MBFVAR's CPZ path uses."""
        for select_q, select_m in ((1, 1), (0, 0), (1, 0)):
            sel, _ = _system(select_q, select_m, "mean")
            with self.subTest(select_q=select_q, select_m=select_m):
                np.testing.assert_allclose(_column_sums(sel), 1.0)

    def test_tent_identity_sums_to_ratio_for_growth_and_one_for_levels(self):
        """The tent is a per-period growth identity: its weights sum to the
        frequency ratio. A level is not the tent sum of anything, so levels
        keep the mean identity even when growth rates get the tent."""
        sel, block_info = _system(1, 0, "tent")
        sums = _column_sums(sel)
        # low blocks run high-to-low: monthly (level, mean) then quarterly
        # (growth, tent); the monthly block contributes 6 columns, the
        # quarterly one only observation 1 (the tent needs period -1).
        n_month = block_info[1]["n_periods"]
        np.testing.assert_allclose(sums[:n_month], 1.0)
        np.testing.assert_allclose(sums[n_month:], 12.0)

    def test_tent_cannot_constrain_observation_zero_but_mean_can(self):
        sel_mean, _ = _system(1, 1, "mean")
        sel_tent, _ = _system(1, 1, "tent")
        # mean: 6 monthly + 2 quarterly; tent: 5 monthly + 1 quarterly
        self.assertEqual(sel_mean["M_a"].shape[1], 8)
        self.assertEqual(sel_tent["M_a"].shape[1], 6)

    def test_mean_window_is_the_observations_own_period(self):
        """Weights for quarterly observation i must sit on weeks 12i..12i+11
        of that variable's latent rows, not straddle the previous quarter."""
        sel, block_info = _system(1, 1, "mean")
        n = sel["n"]
        n_high = sel["n_high"]
        q_off = [b for b in block_info if b["level"] == 0][0]["col_start"] - n_high
        M_a = sel["M_a"].tocsc()
        # the last constraint column is quarterly observation 1
        col = M_a.getcol(M_a.shape[1] - 1)
        latent_rows = col.nonzero()[0]
        # map latent index back to (week, variable)
        mis_rows = sel["M_u"].nonzero()[0]
        full_rows = mis_rows[latent_rows]
        weeks = sorted(set(full_rows // n))
        self.assertEqual(weeks, list(range(12, 24)))
        self.assertTrue(all((full_rows % n) - n_high == q_off))

    def test_unknown_identity_and_wrong_flag_length_raise(self):
        q = np.array([[1.0], [2.0]]); m = np.arange(10.0, 16.0).reshape(-1, 1)
        w = np.arange(24.0).reshape(-1, 1)
        yraw, block_info = CPZ.build_stacked_data([q, m, w], [12, 4, 1])
        with self.assertRaises(ValueError):
            CPZ.build_selection_matrices(yraw, block_info, [q, m, w], 1,
                                         agg_identity="sum")
        with self.assertRaises(ValueError):
            CPZ.build_selection_matrices(
                yraw, block_info, [q, m, w], 1,
                select_low=[np.array([1, 1]), np.array([1])],
                agg_identity="mean")

    def test_omitting_flags_is_all_growth_which_the_mean_identity_ignores(self):
        q = np.array([[1.0], [2.0]]); m = np.arange(10.0, 16.0).reshape(-1, 1)
        w = np.arange(24.0).reshape(-1, 1)
        yraw, block_info = CPZ.build_stacked_data([q, m, w], [12, 4, 1])
        sel_none = CPZ.build_selection_matrices(yraw, block_info, [q, m, w], 1,
                                                agg_identity="mean")
        sel_flags, _ = _system(0, 0, "mean")
        np.testing.assert_allclose(
            sel_none["M_a"].toarray(), sel_flags["M_a"].toarray())


if __name__ == "__main__":
    unittest.main()
