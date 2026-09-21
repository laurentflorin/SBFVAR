"""
A bad hyperparameter draw must not kill the search, and a search that found
nothing must not report a selection.

Both guards were forced by the latent-state study. On its neutral DGP a
substantial share of hyperparameter draws leave the Gibbs sampler stuck in the
explosive region with no usable post-burn-in draw, which fit() now raises on.
The MDD evaluator caught only NameError, so the first such draw propagated
through joblib and killed the whole tuning run. And because the optimiser
still reports a "best" point when every evaluation returned the penalty, a
search in which nothing worked would have written an arbitrary draw from the
search space to disk as though it were a tuned prior.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load():
    root = Path(__file__).resolve().parents[1] / "SBFVAR"
    name = "SBFVAR._hyp_opt"
    if name in sys.modules:
        return sys.modules[name]
    pkg = sys.modules.get("SBFVAR")
    if pkg is None:
        pkg = types.ModuleType("SBFVAR")
        pkg.__path__ = [str(root)]
        sys.modules["SBFVAR"] = pkg
    spec = importlib.util.spec_from_file_location(name, root / "_hyp_opt.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PENALTY = -1e16


class _Fit:
    """Stands in for the model: fit() does whatever the test asks."""

    def __init__(self, behaviour):
        self.nsim = 10
        self._behaviour = behaviour

    def fit(self, *a, **k):
        b = self._behaviour
        if isinstance(b, Exception):
            raise b
        return b


class TestEvaluatorScoresFailuresInsteadOfRaising(unittest.TestCase):
    def setUp(self):
        self.H = _load()

    def _estim(self, behaviour):
        return self.H._estim(_Fit(behaviour), object(), [0.5, 1, 1, 1, 1],
                             nsim=10, var_of_interest=None, temp_agg="mean")

    def test_a_degenerate_fit_is_scored_not_raised(self):
        """The exact failure that killed a tuning run: every post-burn-in
        draw non-finite, which fit() reports as a RuntimeError."""
        exc = RuntimeError("sbfvar: every post-burn-in draw (0) produced "
                           "non-finite output; the fit degenerated")
        self.assertEqual(self._estim(exc), PENALTY)

    def test_the_explosive_restart_case_still_works(self):
        self.assertEqual(self._estim(NameError("No Stable VAR at j=0")), PENALTY)

    def test_other_numerical_failures_are_scored_too(self):
        for exc in (np.linalg.LinAlgError("singular"), IndexError("oob"),
                    ValueError("bad"), ZeroDivisionError()):
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(self._estim(exc), PENALTY)

    def test_nan_and_inf_are_scored(self):
        for v in (np.nan, np.inf, -np.inf):
            with self.subTest(v=v):
                self.assertEqual(self._estim(v), PENALTY)

    def test_a_good_draw_returns_its_mdd_unchanged(self):
        self.assertEqual(self._estim(-1234.5), -1234.5)

    def test_nsim_is_restored_even_when_the_fit_fails(self):
        """The evaluator overrides nsim for the search; leaking the override
        would silently change every later fit."""
        m = _Fit(RuntimeError("boom"))
        m.nsim = 30000
        self.H._estim(m, object(), [0.5, 1, 1, 1, 1], nsim=10,
                      var_of_interest=None, temp_agg="mean")
        self.assertEqual(m.nsim, 30000)


class TestAllFailedSearchIsRefused(unittest.TestCase):
    """Pin the refusal condition itself: the optimiser always returns a best
    point, so the only thing standing between an all-failed search and a
    hyperparameter file is this check."""

    def _would_refuse(self, best_objective):
        return (not np.isfinite(best_objective)) or best_objective <= PENALTY

    def test_an_all_penalty_search_is_refused(self):
        self.assertTrue(self._would_refuse(PENALTY))
        self.assertTrue(self._would_refuse(-1e17))

    def test_a_non_finite_best_is_refused(self):
        for v in (np.nan, -np.inf):
            with self.subTest(v=v):
                self.assertTrue(self._would_refuse(v))

    def test_a_real_selection_is_kept(self):
        for v in (-5000.0, -1e6, 0.0, 12.5):
            with self.subTest(v=v):
                self.assertFalse(self._would_refuse(v))

    def test_both_packages_carry_the_refusal(self):
        root = Path(__file__).resolve().parents[2]
        for rel in ("SBFVAR/SBFVAR/_hyp_opt.py", "MBFVAR/MBFVAR/_hyp_opt.py"):
            src = (root / rel).read_text()
            i = src.index("def update_hyperparameters_mango(")
            body = src[i:src.index("\ndef ", i + 10)]
            with self.subTest(pkg=rel):
                self.assertIn("best_objective", body)
                self.assertIn("No hyperparameters have been saved", body)


if __name__ == "__main__":
    unittest.main()
