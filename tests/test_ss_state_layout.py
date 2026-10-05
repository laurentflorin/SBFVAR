"""
The Schorfheide-Song sampler's two state layouts must describe one model.

``_fit_ss`` filters the latent monthly and quarterly variables in a
frequency-blocked state and handles the ragged edge in the VAR's own
lag-interleaved companion state (see ``SBFVAR/_ss_state.py``). Up to 0.2.1
the hand-overs between them were wrong in four places:

* the ragged-edge measurement of every monthly variable pointed at the
  weekly slots of the state, so observed months were imposed, exactly, on
  the wrong series -- in the paper's data CPI inflation on the federal funds
  rate, and a GDP nowcast twice the realised growth rate;
* the initial ragged-edge covariance and the final smoothed draw were copied
  block by block as if both layouts were interleaved;
* the in-sample transition kept the monthly<->quarterly cross-effects at lag
  1 only, so the states were filtered under a truncated VAR;
* the backward smoother used the weekly regressors one week early.

Each test below states a requirement the sampler has to meet -- the same
observation equation in both layouts, the in-sample transition and the
ragged-edge companion both being the VAR, observed data honoured exactly at
the ragged edge -- rather than restating index arithmetic, so the old code
fails them and the new code passes.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from _real_package import real_sbfvar

SBFVAR = real_sbfvar()
from SBFVAR._ss_state import (forecast_measurement, insample_transition,  # noqa: E402
                              latent_position_maps)

DIMS = [(2, 5, 3, 12), (1, 1, 1, 12), (3, 2, 4, 13)]   # (Nw, Nm, Nq, p)
RMW, RQW = 4, 12


@contextlib.contextmanager
def silence_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def insample_measurement(Nm, Nq, p, rmw, rqw):
    """The in-sample aggregation constraints exactly as `_fit_ss` writes them
    (LAMBDAs_m, LAMBDAs_q) -- the independent statement of the
    frequency-blocked layout the maps are checked against."""
    state_size = (Nm + Nq) * (p + 1)
    lam_m = np.hstack((np.tile(np.eye(Nm), rmw),
                       np.zeros((Nm, state_size - rmw * Nm)))) * (1 / rmw)
    lam_q = np.hstack((np.hstack((np.zeros((Nq, (p + 1) * Nm)), np.tile(np.eye(Nq), rqw))),
                       np.zeros((Nq, state_size - (rqw * Nq + (p + 1) * Nm))))) * (1 / rqw)
    return lam_m, lam_q


def companion(Phi, Ntotal, p):
    """The ragged-edge transition exactly as `_fit_ss` writes it (PHIF,
    CONF): the VAR in its own lag-interleaved ordering."""
    kn = Ntotal * (p + 1)
    PHIF = np.zeros((kn, kn))
    for i in range(p):
        PHIF[(i + 1) * Ntotal:(i + 2) * Ntotal, i * Ntotal:(i + 1) * Ntotal] = np.eye(Ntotal)
    PHIF[:Ntotal, :Ntotal * p] = Phi[:-1, :].T
    CONF = np.hstack((Phi[-1, :].T, np.zeros(Ntotal * p)))
    return PHIF, CONF


class TestLatentPositionMaps(unittest.TestCase):
    def test_maps_are_bijections_onto_the_latent_slots(self):
        for Nw, Nm, Nq, p in DIMS:
            ins, fc = latent_position_maps(Nw, Nm, Nq, p)
            Ntotal = Nw + Nm + Nq
            self.assertEqual(sorted(ins), list(range((Nm + Nq) * (p + 1))))
            weekly = {lag * Ntotal + j for lag in range(p + 1) for j in range(Nw)}
            self.assertEqual(set(fc), set(range(Ntotal * (p + 1))) - weekly)

    def test_one_observation_equation_in_both_layouts(self):
        """Moving a state from the in-sample layout to the ragged-edge one
        with the maps must not change what any monthly or quarterly
        aggregate of it is: the in-sample LAMBDAs_m/LAMBDAs_q and the
        ragged-edge measurement must agree element for element."""
        for Nw, Nm, Nq, p in DIMS:
            ins, fc = latent_position_maps(Nw, Nm, Nq, p)
            lam_m, lam_q = insample_measurement(Nm, Nq, p, RMW, RQW)
            ZZ = forecast_measurement(Nw, Nm, Nq, p, RMW, RQW, "mean")
            Z1, Z2 = ZZ[Nw:Nw + Nm], ZZ[Nw + Nm:]
            np.testing.assert_allclose(Z1[:, fc], lam_m[:, ins])
            np.testing.assert_allclose(Z2[:, fc], lam_q[:, ins])
            # and nothing of either aggregate rests on a weekly slot
            self.assertEqual(np.abs(np.delete(Z1, fc, axis=1)).sum(), 0.0)
            self.assertEqual(np.abs(np.delete(Z2, fc, axis=1)).sum(), 0.0)

    def test_covariance_hand_over_keeps_variances_with_their_variables(self):
        """A covariance that is diagonal with a distinct variance for every
        (variable, lag) element must arrive diagonal with each variance on the
        same element -- what the block copy of 0.2.1 did not do."""
        Nw, Nm, Nq, p = 2, 5, 3, 12
        ins, fc = latent_position_maps(Nw, Nm, Nq, p)
        Ntotal, kn = Nw + Nm + Nq, (Nw + Nm + Nq) * (p + 1)
        P = np.diag(np.arange(1.0, (Nm + Nq) * (p + 1) + 1))
        B = np.zeros((kn, kn))
        B[np.ix_(fc, fc)] = P[np.ix_(ins, ins)]
        self.assertEqual(np.count_nonzero(B - np.diag(np.diag(B))), 0)
        Nm_states = Nm * (p + 1)
        for lag in range(p + 1):
            for k in range(Nq):   # quarterly variable k at lag `lag`
                f, i = lag * Ntotal + Nw + Nm + k, Nm_states + lag * Nq + k
                self.assertEqual(B[f, f], P[i, i])


class TestForecastMeasurement(unittest.TestCase):
    def test_aggregates_are_means_of_the_right_series(self):
        """Fill the interleaved state with a known weekly path per variable:
        the monthly row must return the mean of that monthly variable over
        the last 4 weeks, the quarterly row over the last 12."""
        Nw, Nm, Nq, p = 2, 5, 3, 12
        Ntotal = Nw + Nm + Nq
        rng = np.random.default_rng(0)
        path = rng.standard_normal((p + 1, Ntotal))       # path[lag, var]
        x = path.reshape(-1)                                # interleaved by lag
        y = forecast_measurement(Nw, Nm, Nq, p, RMW, RQW, "mean") @ x
        np.testing.assert_allclose(y[:Nw], path[0, :Nw])
        np.testing.assert_allclose(y[Nw:Nw + Nm], path[:RMW, Nw:Nw + Nm].mean(axis=0))
        np.testing.assert_allclose(y[Nw + Nm:], path[:RQW, Nw + Nm:].mean(axis=0))
        ys = forecast_measurement(Nw, Nm, Nq, p, RMW, RQW, "sum") @ x
        np.testing.assert_allclose(ys[Nw:Nw + Nm], path[:RMW, Nw:Nw + Nm].sum(axis=0))

    def test_refuses_a_state_shorter_than_the_aggregation_window(self):
        with self.assertRaises(ValueError):
            forecast_measurement(2, 5, 3, 10, RMW, RQW)


class TestInsampleTransitionIsTheVAR(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def draw(self, Nw, Nm, Nq, p):
        n = Nw + Nm + Nq
        Phi = self.rng.normal(scale=0.2, size=(n * p + 1, n))
        hist = self.rng.standard_normal((p + 1, n))        # hist[l] = y_{t-l}, l = 0..p
        return Phi, hist

    def test_one_step_prediction_matches_the_var(self):
        for Nw, Nm, Nq, p in DIMS:
            Phi, hist = self.draw(Nw, Nm, Nq, p)
            n = Nw + Nm + Nq
            # the VAR's own prediction of y_t from y_{t-1}, ..., y_{t-p}
            x = np.concatenate([hist[l] for l in range(1, p + 1)] + [[1.0]])
            y_var = x @ Phi
            # frequency-blocked state at t-1 and weekly regressors at t
            m, q = slice(Nw, Nw + Nm), slice(Nw + Nm, n)
            a_prev = np.concatenate([hist[l, m] for l in range(1, p + 2) if l <= p]
                                    + [np.zeros(Nm)]
                                    + [hist[l, q] for l in range(1, p + 1)]
                                    + [np.zeros(Nq)])
            z_t = np.concatenate([hist[l, :Nw] for l in range(1, p + 1)])
            M = insample_transition(Phi, Nw, Nm, Nq, p)
            a_pred = M["GAMMAs"] @ a_prev + M["GAMMAz"] @ z_t + M["GAMMAc"][:, 0]
            Nm_states = Nm * (p + 1)
            np.testing.assert_allclose(a_pred[:Nm], y_var[m], atol=1e-12)
            np.testing.assert_allclose(a_pred[Nm_states:Nm_states + Nq], y_var[q], atol=1e-12)
            # lag rows shift: m_{t-1} moves into the first monthly lag slot
            np.testing.assert_allclose(a_pred[Nm:2 * Nm], hist[1, m])
            np.testing.assert_allclose(a_pred[Nm_states + Nq:Nm_states + 2 * Nq], hist[1, q])
            # weekly observation equation: needs the state at t, i.e. with
            # the current latent values in front
            a_t = a_pred
            w_hat = M["LAMBDAs_w"] @ a_t + M["LAMBDAz_w"] @ z_t + M["LAMBDAc_w"][:, 0]
            np.testing.assert_allclose(w_hat, y_var[:Nw], atol=1e-12)

    def test_cross_effects_beyond_lag_one_reach_the_states(self):
        """The 0.2.1 transition dropped them: a VAR whose only link from the
        monthly to the quarterly block runs at lag 2 left the quarterly
        prediction untouched."""
        Nw, Nm, Nq, p = 1, 1, 1, 12
        n = Nw + Nm + Nq
        Phi = np.zeros((n * p + 1, n))
        Phi[1 * n + Nw, Nw + Nm] = 0.7          # m_{t-2} -> q_t
        hist = np.zeros((p + 1, n))
        hist[2, Nw] = 1.0                        # m_{t-2} = 1
        a_prev = np.zeros((Nm + Nq) * (p + 1))
        a_prev[1 * Nm] = 1.0                     # m_{t-2} sits in the second monthly slot
        M = insample_transition(Phi, Nw, Nm, Nq, p)
        a_pred = M["GAMMAs"] @ a_prev
        self.assertAlmostEqual(a_pred[Nm * (p + 1)], 0.7)

    def test_both_layouts_carry_the_same_dynamics(self):
        """Propagate a state one week in each layout -- the in-sample
        transition and the ragged-edge companion -- and compare the latent
        part: with the maps, the two must coincide."""
        for Nw, Nm, Nq, p in DIMS:
            Phi, hist = self.draw(Nw, Nm, Nq, p)
            n = Nw + Nm + Nq
            ins, fc = latent_position_maps(Nw, Nm, Nq, p)
            x_prev = np.concatenate([hist[l] for l in range(1, p + 2) if l <= p]
                                    + [np.zeros(n)])        # interleaved, lags 0..p of t-1
            PHIF, CONF = companion(Phi, n, p)
            x_next = PHIF @ x_prev + CONF
            M = insample_transition(Phi, Nw, Nm, Nq, p)
            a_prev = np.zeros((Nm + Nq) * (p + 1))
            a_prev[ins] = x_prev[fc]
            z_t = np.concatenate([x_prev[l * n:l * n + Nw] for l in range(p)])
            a_next = M["GAMMAs"] @ a_prev + M["GAMMAz"] @ z_t + M["GAMMAc"][:, 0]
            np.testing.assert_allclose(a_next[ins], x_next[fc], atol=1e-12)


def ragged_edge_data(seed=7, n_months=96):
    """The package's near-unit-root fixture with the last quarter's
    quarterly observation withheld, so the sample ends with a quarter of
    weekly and monthly data the ragged-edge filter has to absorb."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    weeks = pd.DatetimeIndex([m - pd.Timedelta(days=7 * (3 - k)) for m in months for k in range(4)])
    walk = np.cumsum(rng.normal(scale=0.3, size=(4 * n_months, 3)), axis=0)
    weekly = pd.DataFrame({"w_1": walk[:, 0]}, index=weeks)
    monthly = pd.DataFrame({"m_1": walk[:, 1].reshape(n_months, 4).mean(axis=1)}, index=months)
    quarterly = pd.DataFrame(
        {"q_1": (0.5 * walk[:, 1] + walk[:, 2]).reshape(n_months // 3, 12).mean(axis=1)},
        index=months[2::3]).iloc[:-1]
    return SBFVAR.sbfvar_data([quarterly, monthly, weekly],
                              [np.array([1]), np.array([1]), np.array([1])], ["Q", "M", "W"])


class TestRaggedEdgeHonoursObservations(unittest.TestCase):
    """End to end on a real fit: at the ragged edge the weekly series are
    observed and the monthly ones are exact 4-week means of their latent
    path. Under 0.2.1 the monthly observations were imposed on the weekly
    slot instead, and neither held."""

    # Measured on this fixture (data s.d. 0.045): the fixed sampler honours
    # the observations to 3e-6 -- numerical noise of the Kalman update --
    # while 0.2.1 missed them by 1.4e-2 (weekly) and 2.7e-2 (monthly means).
    TOL = 1e-4

    @classmethod
    def setUpClass(cls):
        model = SBFVAR.multifrequency_var(20, 0.5, 12, 1, seed=3)
        with silence_output():
            model.fit(ragged_edge_data(), [0.09, 4.3, 1, 2.7, 4.3], max_it_explosive=10)
        cls.model = model

    def test_weekly_observations_and_monthly_means_hold_exactly(self):
        m = self.model
        T0, Nw, Nm = m.nlags, m.Nw, m.Nm
        YW, YM, YQ = m.input_data_W, m.input_data_M, m.input_data_Q
        nobs = min(YM.shape[0] - T0, YQ.shape[0] - T0)
        Tstar = YW.shape[0] - T0
        self.assertGreater(Tstar - nobs, 4, "fixture lost its ragged edge")
        ydata = pd.concat([pd.DataFrame(YW), pd.DataFrame(YM), pd.DataFrame(YQ)], axis=1).values
        checked = 0
        for d in m.valid_draws:
            latent = m.lstate_list[d]                      # rows: weeks T0 .. T0+Tstar-1
            tail = m.YYactsim_list[d]                      # last rqw+1 rows of the VAR data
            for k in range(min(tail.shape[0], Tstar - nobs)):
                r = Tstar - 1 - k                          # ragged-edge row
                np.testing.assert_allclose(tail[-1 - k, :Nw], ydata[T0 + r, :Nw], rtol=0, atol=self.TOL)
                obs_m = ydata[T0 + r, Nw:Nw + Nm]
                if r - (RMW - 1) >= nobs and np.isfinite(obs_m).all():
                    np.testing.assert_allclose(latent[r - RMW + 1:r + 1, :Nm].mean(axis=0),
                                               obs_m, rtol=0, atol=self.TOL)
                    checked += 1
        self.assertGreater(checked, 0, "no monthly observation fell inside the ragged edge")


if __name__ == "__main__":
    unittest.main()
