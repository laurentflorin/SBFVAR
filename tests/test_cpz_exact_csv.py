"""
The CPZ volatility step: forced acceptance (the default) and the exact step.

The CPZ fit forces acceptance of the accept/reject draw on every iteration,
as the replication driver MFVAR.m of Chan, Poon and Zhu (2023) does. That
skips the MH correction of the accept-reject Metropolis-Hastings algorithm of
Chan (2020, Online Appendix A), so the draws follow the Gaussian
approximation truncated at its envelope, not the conditional posterior.
SBFVAR 0.2.7 made the exact step the default. On the weekly system it rejects
almost every proposal and the volatility path does not move, so 0.2.8
restored forced acceptance as the default (the paper's choice, disclosed
there) and keeps the exact step behind ``_cpz_funcs.CSV_EXACT``.

The tests check the step against the exact conditional of a two-period
problem computed on a grid: the exact step recovers it and forced acceptance
misses it, which is the price the default pays. They also check which draws
the fit forces in each mode.
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


class TestWhichDrawsTheFitForces(unittest.TestCase):

    def flags(self, exact):
        flags = []
        real = E.sample_CSV

        def spy(*a, is_forced_accept=False, **k):
            flags.append(bool(is_forced_accept))
            return real(*a, is_forced_accept=is_forced_accept, **k)

        model = SBFVAR.multifrequency_var(30, 0.5, 4, 1, seed=3)
        with mock.patch.object(E, "sample_CSV", spy), \
                mock.patch.object(C, "CSV_EXACT", exact), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(ragged_edge_data(), hyp=[0.04, 0.25, 100, 2], var_of_interest=None,
                      temp_agg="mean", method="chan_poon_zhu")
        return flags, model

    def test_default_forces_every_draw(self):
        self.assertFalse(C.CSV_EXACT)
        flags, model = self.flags(False)
        self.assertGreater(len(flags), 1)
        self.assertTrue(all(flags))
        self.assertEqual(model.csv_accept_rate, 1.0)

    def test_exact_mode_forces_only_the_first_draw(self):
        flags, model = self.flags(True)
        self.assertEqual(flags[0], True)
        self.assertFalse(any(flags[1:]))
        self.assertTrue(0.0 < model.csv_accept_rate <= 1.0)


if __name__ == "__main__":
    unittest.main()
