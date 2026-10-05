"""
Import the real SBFVAR package in a test module.

Several test files (test_cpz_*, test_hyp_search_guards, test_nonfinite_draws)
load single modules by installing a bare stub package under the name
``SBFVAR`` in ``sys.modules``, some of them at import time. Under
``unittest discover`` that stub is what a later ``import SBFVAR`` returns,
and a test that fits a model fails with "module 'SBFVAR' has no attribute
'multifrequency_var'" although it passes when run on its own. Call
:func:`real_sbfvar` instead of ``import SBFVAR`` in such a test.
"""
import importlib
import sys


def real_sbfvar():
    mod = sys.modules.get("SBFVAR")
    if mod is not None and not hasattr(mod, "multifrequency_var"):
        for name in [k for k in list(sys.modules)
                     if k == "SBFVAR" or k.startswith("SBFVAR.")]:
            del sys.modules[name]
    return importlib.import_module("SBFVAR")
