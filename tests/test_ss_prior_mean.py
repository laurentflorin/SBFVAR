"""
A per-variable prior mean for the Schorfheide-Song prior (SBFVAR 0.2.3).

The SS prior's dummy observations centre every variable's own first lag on
a random walk and add sum-of-coefficients dummies that push towards unit
roots. That suits levels and year-on-year growth rates. For period-on-period
growth rates the usual centring is white noise for the growth series, with
persistent ones such as interest rates kept at a random walk (Banbura,
Giannone and Reichlin, 2010). ``prior_mean`` provides that; ``None`` must
leave the prior, and therefore every existing run, exactly as it was.
"""
import contextlib
import io
import unittest

import numpy as np

from _real_package import real_sbfvar

SBFVAR = real_sbfvar()
from SBFVAR.mfbvar_funcs import (prior_mean_vector, resolve_prior_mean,  # noqa: E402
                                 varprior)
from test_ss_state_layout import ragged_edge_data  # noqa: E402


@contextlib.contextmanager
def silence_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def varprior_022(nv, nlags, nex, hyp, premom):
    """The prior of SBFVAR 0.2.2, verbatim, as the regression reference."""
    lambda1, lambda2, lambda3, lambda4, lambda5 = hyp[0], hyp[1], int(hyp[2]), hyp[3], hyp[4]
    dsize = nex + (nlags + lambda3 + 1) * nv
    breakss = np.zeros((5, 1))
    ydu = np.zeros((int(dsize), int(nv)))
    xdu = np.zeros((int(dsize), int(nv * nlags + nex)))
    sig = np.diag(premom[:, 1])
    ydu[range(nv), :] = lambda1 * sig
    xdu[:nv, :sig.shape[1]] = lambda1 * sig
    breakss[0] = nv
    if nlags > 1:
        ydu[int(breakss[0, 0]):(nv * nlags), :] = np.zeros(((nlags - 1) * nv, nv))
        j = 1
        while j <= nlags - 1:
            xdu[int(breakss[0, 0]) + (j - 1) * nv:int(breakss[0, 0]) + j * nv] = np.hstack((
                np.zeros((nv, j * nv)), lambda1 * sig * ((j + 1) ** lambda2),
                np.zeros((nv, (nlags - 1 - j) * nv + nex))))
            j = j + 1
        breakss[1, 0] = breakss[0, 0] + (nlags - 1) * nv
    else:
        breakss[1, 0] = breakss[0, 0]
    ydu[int(breakss[1, 0]):int(breakss[1, 0]) + lambda3 * nv, :] = np.kron(np.ones((lambda3, 1)), sig)
    breakss[2, 0] = breakss[1, 0] + lambda3 * nv
    lammean = lambda4 * premom[:, 0]
    ydu[int(breakss[2, 0]), :] = lammean
    xdu[int(breakss[2, 0]), :] = np.hstack((np.squeeze(np.kron(np.ones((1, nlags)), lammean)), lambda4))
    breakss[3] = breakss[2, 0] + 1
    mumean = np.diag(lambda5 * premom[:, 0])
    ydu[int(breakss[3, 0]):int(breakss[3, 0]) + nv, :] = mumean
    xdu[int(breakss[3, 0]):int(breakss[3, 0]) + nv, :] = np.hstack((
        np.squeeze(np.kron(np.ones((1, nlags)), mumean)), np.zeros((nv, nex))))
    return ydu, xdu


class TestVarpriorPriorMean(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.nv, self.p = 5, 4
        self.hyp = [0.7, 1.3, 1, 2.1, 0.9]
        self.premom = np.column_stack((rng.normal(size=self.nv), rng.uniform(0.5, 2, self.nv)))

    def test_default_is_the_022_prior_bit_for_bit(self):
        y0, x0 = varprior_022(self.nv, self.p, 1, self.hyp, self.premom)
        for pm in (None, np.ones(self.nv)):
            y, x = varprior(self.nv, self.p, 1, self.hyp, self.premom, prior_mean=pm)
            self.assertTrue(np.array_equal(y, y0) and np.array_equal(x, x0))

    def test_first_lag_dummy_encodes_the_requested_mean(self):
        delta = np.array([0.0, 1.0, 0.0, 0.5, 1.0])
        y, x = varprior(self.nv, self.p, 1, self.hyp, self.premom, prior_mean=delta)
        # rows 0..nv-1 are the first-lag dummies: y = lambda1*sigma_i*delta_i
        # on the diagonal against x = lambda1*sigma_i, i.e. own-lag mean delta_i
        np.testing.assert_allclose(np.diag(y[:self.nv]) / np.diag(x[:self.nv, :self.nv]), delta)
        self.assertEqual(np.count_nonzero(y[:self.nv] - np.diag(np.diag(y[:self.nv]))), 0)

    def test_sum_of_coefficients_dummy_drops_out_for_white_noise_series(self):
        delta = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        y, x = varprior(self.nv, self.p, 1, self.hyp, self.premom, prior_mean=delta)
        y0, x0 = varprior_022(self.nv, self.p, 1, self.hyp, self.premom)
        soc = slice(y.shape[0] - self.nv, y.shape[0])           # the last nv rows
        for i in range(self.nv):
            if delta[i] == 0:
                self.assertEqual(np.count_nonzero(y[soc][i]), 0)
                self.assertEqual(np.count_nonzero(x[soc][i]), 0)
            else:
                np.testing.assert_array_equal(y[soc][i], y0[soc][i])
                np.testing.assert_array_equal(x[soc][i], x0[soc][i])
        # everything between the first-lag and sum-of-coefficients blocks is untouched
        mid = slice(self.nv, y.shape[0] - self.nv)
        np.testing.assert_array_equal(y[mid], y0[mid])
        np.testing.assert_array_equal(x[mid], x0[mid])

    def test_vector_checks(self):
        with self.assertRaises(ValueError):
            prior_mean_vector([1.0, 0.0], 3)
        with self.assertRaises(ValueError):
            prior_mean_vector([1.0, np.nan, 0.0], 3)


class TestResolvePriorMean(unittest.TestCase):
    varlist = ["WGS10YR", "FEDFUNDS", "INDPRO", "UNRATE", "GDPC1"]

    def test_by_name_with_unnamed_variables_kept_at_one(self):
        v = resolve_prior_mean({"INDPRO": 0, "GDPC1": 0}, self.varlist)
        np.testing.assert_array_equal(v, [1, 1, 0, 1, 0])

    def test_a_misspelt_name_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_prior_mean({"GDPC": 0}, self.varlist)

    def test_none_stays_none(self):
        self.assertIsNone(resolve_prior_mean(None, self.varlist))


class TestFitWithPriorMean(unittest.TestCase):
    """End to end on the package's small ragged-edge fixture (w_1, m_1, q_1)."""

    def fit(self, **kw):
        model = SBFVAR.multifrequency_var(12, 0.5, 12, 1, seed=5)
        with silence_output():
            model.fit(ragged_edge_data(), [0.09, 4.3, 1, 2.7, 4.3], max_it_explosive=10, **kw)
        return model

    def test_default_and_explicit_none_draw_the_same_chain(self):
        a, b = self.fit(), self.fit(prior_mean=None)
        self.assertTrue(np.array_equal(np.array(a.Phip), np.array(b.Phip)))

    def test_white_noise_centring_changes_the_chain_and_is_recorded(self):
        a, b = self.fit(), self.fit(prior_mean={"m_1": 0, "q_1": 0})
        self.assertFalse(np.array_equal(np.array(a.Phip), np.array(b.Phip)))
        np.testing.assert_array_equal(b.prior_mean, [1, 0, 0])

    def test_refused_on_the_cpz_path(self):
        model = SBFVAR.multifrequency_var(12, 0.5, 12, 1, seed=5)
        with self.assertRaises(ValueError):
            model.fit(ragged_edge_data(), [0.09, 4.3, 1, 2.7], method="chan_poon_zhu",
                      prior_mean={"q_1": 0})


if __name__ == "__main__":
    unittest.main()
