"""
The banded common-stochastic-volatility step (SBFVAR 0.2.6).

``sample_CSV`` used dense T x T copies of its tridiagonal matrices. That
cost O(T^3) per draw for the Cholesky factor and O(T^2) per accept/reject
try. On the weekly CPZ system (T ~ 2,700) the tries run into the hundreds
per draw. 0.2.6 does the same algorithm in banded form. These tests run
both versions from the same random state on a weekly-sized problem with a
pandemic-like volatility spike. They check that the two return the same
path to rounding error, make the same accept decisions, and consume the
same random numbers, so a chain continues identically after the step.
"""
import time
import unittest

import numpy as np

from _real_package import real_sbfvar

real_sbfvar()
from SBFVAR import _cpz_funcs as C  # noqa: E402


def weekly_problem(seed, T=2700, n=7, spike=True):
    rng = np.random.default_rng(seed)
    rho, sigh2 = 0.97, 0.05
    h = np.zeros(T)
    for t in range(1, T):
        h[t] = rho * h[t - 1] + np.sqrt(sigh2) * rng.standard_normal()
    if spike:
        h[T - 300:T - 250] += 4.0            # a pandemic-sized jump in volatility
    s2 = np.exp(h) * rng.chisquare(n, size=T)
    h0 = h + 0.1 * rng.standard_normal(T)     # the chain's current path
    return s2, rho, sigh2, h0, n


class TestBandedMatchesDense(unittest.TestCase):

    def run_both(self, problem, seed, forced):
        np.random.seed(seed)
        hb, ab = C.sample_CSV(*problem, is_forced_accept=forced)
        state_b = np.random.get_state()
        np.random.seed(seed)
        hd, ad = C._sample_CSV_dense(*problem, is_forced_accept=forced)
        state_d = np.random.get_state()
        return hb, ab, state_b, hd, ad, state_d

    def test_same_path_decision_and_random_stream(self):
        for k in range(6):
            problem = weekly_problem(k, spike=(k % 2 == 0))
            for forced in (True, False):
                hb, ab, sb, hd, ad, sd = self.run_both(problem, 100 + k, forced)
                np.testing.assert_allclose(hb, hd, rtol=0, atol=1e-8)
                self.assertEqual(ab, ad)
                self.assertEqual(sb[2], sd[2])                  # stream position
                np.testing.assert_array_equal(sb[1], sd[1])     # Mersenne state

    def test_small_T(self):
        problem = weekly_problem(9, T=40, spike=False)
        hb, ab, sb, hd, ad, sd = self.run_both(problem, 7, False)
        np.testing.assert_allclose(hb, hd, rtol=0, atol=1e-10)
        self.assertEqual(ab, ad)

    def test_it_is_much_faster(self):
        problem = weekly_problem(2)
        np.random.seed(1)
        t = time.perf_counter()
        C.sample_CSV(*problem, is_forced_accept=True)
        banded = time.perf_counter() - t
        np.random.seed(1)
        t = time.perf_counter()
        C._sample_CSV_dense(*problem, is_forced_accept=True)
        dense = time.perf_counter() - t
        self.assertLess(banded, dense / 3)


if __name__ == "__main__":
    unittest.main()
