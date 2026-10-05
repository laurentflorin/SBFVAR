"""
Regression tests for the screened explosive-root check (ported from MBFVAR 0.9.1).

``is_explosive`` must take the same decision as the eigenvalue check
``_is_explosive_eig`` on every input, so that seeded Gibbs runs draw the same
candidates in the same order and produce identical chains.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from _real_package import real_sbfvar

SBFVAR = real_sbfvar()
import SBFVAR._estimation as estimation  # noqa: E402
from SBFVAR.mfbvar_funcs import _is_explosive_eig, is_explosive  # noqa: E402


@contextlib.contextmanager
def silence_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@contextlib.contextmanager
def reference_check():
    original = estimation.is_explosive
    estimation.is_explosive = _is_explosive_eig
    try:
        yield
    finally:
        estimation.is_explosive = original


def near_unit_root_data(seed=7, n_months=96):
    """Random walks in levels on the fixed four-weeks-per-month grid: many
    coefficient draws are explosive by a hair, which is the case the screen is
    built for. The data class expects quarterly, monthly and weekly data."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    weeks = pd.DatetimeIndex([m - pd.Timedelta(days=7 * (3 - k)) for m in months for k in range(4)])
    walk = np.cumsum(rng.normal(scale=0.3, size=(4 * n_months, 3)), axis=0)
    weekly = pd.DataFrame({"w_1": walk[:, 0]}, index=weeks)
    monthly = pd.DataFrame({"m_1": walk[:, 1].reshape(n_months, 4).mean(axis=1)}, index=months)
    quarterly = pd.DataFrame(
        {"q_1": (0.5 * walk[:, 1] + walk[:, 2]).reshape(n_months // 3, 12).mean(axis=1)},
        index=months[2::3])
    return SBFVAR.sbfvar_data([quarterly, monthly, weekly],
                              [np.array([1]), np.array([1]), np.array([1])], ["Q", "M", "W"])


class TestExplosiveCheck(unittest.TestCase):
    def test_matches_eigenvalue_check(self):
        rng = np.random.default_rng(3)
        cases = []
        for n, p in ((4, 3), (10, 12)):
            for scale in (0.0005, 0.002, 0.01, 0.05, 0.3):
                for _ in range(40):
                    phi = rng.normal(0, scale / np.sqrt(p), (n * p + 1, n))
                    phi[:n, :] += np.eye(n)  # unit roots perturbed in both directions
                    cases.append((phi, n, p))
        decisions = [(bool(_is_explosive_eig(c, n, p)), bool(is_explosive(c, n, p))) for c, n, p in cases]
        self.assertTrue(all(a == b for a, b in decisions))
        self.assertTrue(0 < sum(a for a, _ in decisions) < len(decisions))

    def test_complex_and_negative_roots_fall_through_to_eigenvalues(self):
        n, p = 4, 1
        rotation = np.zeros((n * p + 1, n))
        theta, radius = 0.4, 1.001  # complex pair just outside the unit circle
        rotation[:2, :2] = radius * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotation[2, 2] = rotation[3, 3] = 0.5
        negative = np.zeros((n * p + 1, n))
        negative[:n, :] = np.diag([-1.002, 0.3, 0.2, 0.1])  # real root below -1
        stable = np.zeros((n * p + 1, n))
        stable[:n, :] = np.diag([0.999, 0.5, -0.4, 0.2])
        for phi, expected in ((rotation, True), (negative, True), (stable, False)):
            self.assertEqual(bool(is_explosive(phi, n, p)), expected)
            self.assertEqual(bool(_is_explosive_eig(phi, n, p)), expected)

    def test_non_finite_coefficients_raise_as_before(self):
        n, p = 3, 2
        phi = np.zeros((n * p + 1, n))
        phi[1, 1] = np.nan
        with self.assertRaises(ValueError):
            _is_explosive_eig(phi, n, p)
        with self.assertRaises(ValueError):
            is_explosive(phi, n, p)

    def test_seeded_chain_is_unchanged(self):
        data = near_unit_root_data()
        hyp = [0.09, 4.3, 1, 2.7, 4.3]
        draws = {}
        for name in ("screened", "reference"):
            model = SBFVAR.multifrequency_var(12, 0.5, 12, 1, seed=11)
            ctx = reference_check() if name == "reference" else contextlib.nullcontext()
            with ctx, silence_output():
                model.fit(data, hyp, max_it_explosive=10)
            draws[name] = (np.array(model.Phip).copy(), np.array(model.lstate_list[-1]).copy(),
                           model.stability_rejection_share)
        self.assertTrue(np.array_equal(draws["screened"][0], draws["reference"][0]))
        self.assertTrue(np.array_equal(draws["screened"][1], draws["reference"][1]))
        self.assertEqual(draws["screened"][2], draws["reference"][2])


if __name__ == "__main__":
    unittest.main()
