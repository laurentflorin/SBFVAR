"""
The exact accept-reject Metropolis-Hastings volatility step (SBFVAR 0.2.7).

Up to 0.2.6 the CPZ fit forced acceptance of the volatility proposal on every
iteration, as the replication driver MFVAR.m does. That skips the MH
correction of the algorithm described in Chan (2020, Online Appendix A) and
cited by Chan, Poon and Zhu (2023): the draws then follow the accept/reject
envelope, not the conditional posterior. 0.2.7 forces acceptance only on the
first iteration, where h starts at zero. The tests check the step against the
exact conditional of a two-period problem computed on a grid (and that forced
acceptance misses it), and that the fit forces acceptance once.
"""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from _real_package import real_sbfvar

SBFVAR = real_sbfvar()
from SBFVAR import _cpz_funcs as C  # noqa: E402
import SBFVAR._estimation_cpz as E  # noqa: E402
from test_ss_state_layout import ragged_edge_data  # noqa: E402

RHO, SIGH2, N = 0.5, 4.0, 1
S2 = np.array([0.02, 30.0])      # a calm period and a shock: far from Gaussian


def exact_moments():
    g = np.linspace(-15, 15, 1201)
    h1, h2 = np.meshgrid(g, g, indexing="ij")
    logp = (-0.5 * ((1 - RHO ** 2) * h1 ** 2 + (h2 - RHO * h1) ** 2) / SIGH2
            - N / 2 * (h1 + h2) - 0.5 * (np.exp(-h1) * S2[0] + np.exp(-h2) * S2[1]))
    w = np.exp(logp - logp.max())
    w /= w.sum()
    return (w * h1).sum(), (w * h2).sum()


def chain(forced, n_iter=30000, burn=2000, seed=0):
    np.random.seed(seed)
    h = np.zeros(2)
    out = np.empty((n_iter - burn, 2))
    for k in range(n_iter):
        h, _ = C.sample_CSV(S2, RHO, SIGH2, h, N, is_forced_accept=forced)
        if k >= burn:
            out[k - burn] = h
    batches = out[: len(out) // 50 * 50].reshape(50, -1, 2).mean(axis=1)
    return out.mean(axis=0), batches.std(axis=0, ddof=1) / np.sqrt(50)


class TestTheStepIsExact(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.exact = np.array(exact_moments())
        cls.armh = chain(False)
        cls.forced = chain(True)

    def test_the_mh_step_recovers_the_exact_conditional(self):
        mean, se = self.armh
        self.assertTrue(np.all(np.abs(mean - self.exact) < 4 * se), (mean, self.exact, se))

    def test_forced_acceptance_does_not(self):
        mean, se = self.forced
        self.assertGreater(np.abs(mean[1] - self.exact[1]), 4 * se[1], (mean, self.exact, se))


class TestTheFitForcesOnlyTheFirstDraw(unittest.TestCase):

    def test_forced_once_then_exact(self):
        flags = []
        real = E.sample_CSV

        def spy(*a, is_forced_accept=False, **k):
            flags.append(bool(is_forced_accept))
            return real(*a, is_forced_accept=is_forced_accept, **k)

        model = SBFVAR.multifrequency_var(30, 0.5, 4, 1, seed=3)
        with mock.patch.object(E, "sample_CSV", spy), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(ragged_edge_data(), hyp=[0.04, 0.25, 100, 2], var_of_interest=None,
                      temp_agg="mean", method="chan_poon_zhu")
        self.assertEqual(flags[0], True)
        self.assertFalse(any(flags[1:]))
        self.assertGreater(len(flags), 1)
        self.assertTrue(0.0 < model.csv_accept_rate <= 1.0)


if __name__ == "__main__":
    unittest.main()
