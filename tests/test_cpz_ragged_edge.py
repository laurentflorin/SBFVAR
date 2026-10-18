"""
The CPZ path keeps the ragged edge (SBFVAR 0.2.5).

Before 0.2.5 ``build_stacked_data`` cut the stacked system at the length
covered by complete periods of every frequency, i.e. at the end of the last
released quarter. The CPZ SBF-VAR therefore never saw the weekly and monthly
data of the current quarter: in the paper's real-time runs its quarterly
average of the federal funds rate missed the observed data by a median 0.08
points at origins where the whole quarter was observed, while the SS sampler
and the MBF-VAR's CPZ path reproduce it. These tests pin the fix on the
package's ragged-edge fixture, whose last quarter has weekly and monthly data
but no quarterly observation.
"""
import contextlib
import io
import unittest

import numpy as np

from _real_package import real_sbfvar

SBFVAR = real_sbfvar()
from SBFVAR import _cpz_funcs as CPZ  # noqa: E402
from test_ss_state_layout import ragged_edge_data  # noqa: E402

HYP = [0.2 ** 2, 0.5 ** 2, 100, 2]


class TestStackedSystemKeepsTheTail(unittest.TestCase):
    def test_system_runs_to_the_last_week_and_constrains_only_released_values(self):
        q = np.array([[1.0], [2.0]])                 # two released quarters (24 weeks)
        m = np.arange(10.0, 17.0).reshape(-1, 1)     # seven months (28 weeks)
        w = np.arange(30.0).reshape(-1, 1)           # thirty weeks
        yraw, info = CPZ.build_stacked_data([q, m, w], [12, 4, 1])
        self.assertEqual(yraw.shape[0], 30)
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(yraw[:, 1])), [3, 7, 11, 15, 19, 23, 27])
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(yraw[:, 2])), [11, 23])
        sel = CPZ.build_selection_matrices(yraw, info, [q, m, w], 1, agg_identity="mean")
        # one constraint per released value: 7 months + 2 quarters
        self.assertEqual(len(np.asarray(sel["Y_con"]).ravel()), 9)

    def test_lower_frequency_data_beyond_the_highest_are_refused(self):
        with self.assertRaises(ValueError):
            CPZ.build_stacked_data([np.ones((3, 1)), np.ones((9, 1)), np.ones((30, 1))], [12, 4, 1])


class TestRaggedEdgeFit(unittest.TestCase):
    """A real CPZ fit on the fixture: the weekly observations of the last,
    quarterly-unobserved quarter are honoured and the monthly ones are exact
    4-week means of their latent path. (The forecast step is not run here:
    the shared SS forecaster extends the weekly date index with a loop that
    does not terminate on this fixture's synthetic dates; on the canonical
    weekly grid of the real data it does, and the real-data check of the
    quarterly output is in the paper repository.)"""

    @classmethod
    def setUpClass(cls):
        cls.data = ragged_edge_data()
        model = SBFVAR.multifrequency_var(40, 0.5, 4, 1, seed=3)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(cls.data, hyp=HYP, var_of_interest=None, temp_agg="mean",
                      method="chan_poon_zhu")
        cls.model = model

    def test_the_whole_weekly_sample_is_used(self):
        n_weeks = np.asarray(self.data.YM0_list[-1]).shape[0]
        self.assertEqual(self.model.input_data_W.shape[0], n_weeks)

    def test_weekly_observations_hold_in_every_draw(self):
        m = self.model
        tail = m.YYactsim_list[:, 1:, :m.Nw]
        np.testing.assert_allclose(tail, np.broadcast_to(m.input_data_W[-m.rqw:], tail.shape), atol=1e-10)

    def test_monthly_latent_averages_to_the_monthly_observations(self):
        m = self.model
        months = np.asarray(m.input_data_M, dtype=float)[-3:, 0]
        for d in range(len(m.valid_draws)):
            lat = m.lstate_list[d][-12:, 0]            # monthly latent, last 12 weeks
            # The CPZ latent sampler (a port of Sample_latent_Y_approx.m)
            # holds the aggregation constraint to ~1e-5 everywhere in the
            # sample (median 5e-6, max 3e-5 on this fixture, data s.d. 0.03);
            # under 0.2.4 these three months were outside the system.
            np.testing.assert_allclose(lat.reshape(3, 4).mean(axis=1), months, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
