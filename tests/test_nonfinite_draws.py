"""
One non-finite draw must not sink a whole fit.

In the latent-state study, 28 SBF-VAR cells (22 of them on the neutral
DGP D3) were recorded as degenerate after full-length runs of 30 000
draws: the output step averaged the nowcast-weekly block with a plain
mean over draws, so a single draw with NaN in it made all 24 entries of
that block NaN, and the study's alignment check rejected the cell. Each
failed attempt cost the full fit time, about 700 core-hours in all, and
the survivors were a selected subsample of the replications.

The sampler now drops draws whose stored output is non-finite and counts
them. These tests pin the selection rule.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_funcs():
    root = Path(__file__).resolve().parents[1] / "SBFVAR"
    if "SBFVAR.mfbvar_funcs" in sys.modules:
        return sys.modules["SBFVAR.mfbvar_funcs"]
    pkg = types.ModuleType("SBFVAR")
    pkg.__path__ = [str(root)]
    sys.modules.setdefault("SBFVAR", pkg)
    spec = importlib.util.spec_from_file_location(
        "SBFVAR.mfbvar_funcs", root / "mfbvar_funcs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["SBFVAR.mfbvar_funcs"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestFiniteDrawMask(unittest.TestCase):
    def setUp(self):
        self.F = _load_funcs()
        rng = np.random.default_rng(0)
        self.Y = rng.standard_normal((50, 13, 8))     # YYactsim-like
        self.L = rng.standard_normal((50, 400, 6))    # lstate-like
        self.P = rng.standard_normal((50, 97, 8))     # Phip-like

    def test_all_finite_keeps_every_draw(self):
        m = self.F.finite_draw_mask(self.Y, self.L, self.P)
        self.assertEqual(m.shape, (50,))
        self.assertTrue(m.all())

    def test_one_poisoned_draw_is_dropped_and_only_that_one(self):
        """The failure mode as observed: one draw with NaN in the nowcast
        block, every other draw clean."""
        self.Y[17, 3, 0] = np.nan
        m = self.F.finite_draw_mask(self.Y, self.L, self.P)
        self.assertFalse(m[17])
        self.assertEqual(int((~m).sum()), 1)

    def test_inf_counts_as_non_finite(self):
        self.L[5, 100, 2] = np.inf
        m = self.F.finite_draw_mask(self.Y, self.L, self.P)
        self.assertFalse(m[5])
        self.assertEqual(int((~m).sum()), 1)

    def test_structural_nan_padding_is_not_a_reason_to_drop(self):
        """YYactsim_list is NaN-padded at its leading rows when the
        ragged-edge tail is shorter than rqw+1. That padding is shared by
        every draw and must not flag them all -- otherwise the filter would
        discard the fit it is meant to save."""
        self.Y[:, :2, :] = np.nan                   # every draw, rows 0-1
        m = self.F.finite_draw_mask(self.Y, self.L, self.P)
        self.assertTrue(m.all())
        self.Y[9, 5, 1] = np.nan                    # plus one real bad draw
        m = self.F.finite_draw_mask(self.Y, self.L, self.P)
        self.assertFalse(m[9])
        self.assertEqual(int((~m).sum()), 1)

    def test_a_position_bad_in_most_draws_is_treated_as_structural(self):
        """The rule is a majority: a position non-finite in more than half
        the draws is padding, not evidence against the minority that has it
        finite."""
        self.Y[:30, 0, 0] = np.nan
        m = self.F.finite_draw_mask(self.Y)
        self.assertTrue(m.all())

    def test_mask_combines_across_arrays(self):
        self.Y[1, 3, 3] = np.nan
        self.P[2, 0, 0] = np.nan
        m = self.F.finite_draw_mask(self.Y, self.L, self.P)
        self.assertEqual(sorted(np.where(~m)[0].tolist()), [1, 2])


if __name__ == "__main__":
    unittest.main()
