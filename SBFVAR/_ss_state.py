"""
Index bookkeeping for the Schorfheide-Song sampler's two state layouts.

``_fit_ss`` carries the latent (monthly and quarterly) variables in two
different state vectors:

* the in-sample filter and smoother use a FREQUENCY-BLOCKED layout,
  ``[m_t, m_{t-1}, ..., m_{t-p}, q_t, q_{t-1}, ..., q_{t-p}]`` -- all monthly
  lags first, then all quarterly lags -- with the weekly variables outside
  the state as observed regressors;
* the ragged-edge filter and smoother use a LAG-INTERLEAVED companion
  layout, ``[w_t, m_t, q_t, w_{t-1}, m_{t-1}, q_{t-1}, ...]``, which is the
  VAR's own ordering.

Every hand-over between the two must move each (variable, lag) element to
its counterpart. Before 0.2.2 three of the hand-overs did not: the
ragged-edge measurement of the monthly variables pointed at the weekly
slots, and the initial covariance and the final smoothed draw were copied
block by block as if both layouts were interleaved. All three now go
through :func:`latent_position_maps`, and the in-sample transition is built
from the VAR by :func:`insample_transition` with every cross-lag, not only
the first.
"""
import numpy as np


def latent_position_maps(Nw, Nm, Nq, p):
    """Positions of every latent (variable, lag) element in both layouts.

    Returns ``(insample_idx, forecast_idx)``, two integer arrays of length
    ``(Nm + Nq) * (p + 1)`` such that in-sample state element
    ``insample_idx[k]`` and ragged-edge state element ``forecast_idx[k]``
    hold the same variable at the same lag. Use them as
    ``forecast[forecast_idx] = insample[insample_idx]`` for vectors and with
    ``np.ix_`` on both axes for covariance matrices.
    """
    Ntotal = Nw + Nm + Nq
    Nm_states = Nm * (p + 1)
    insample, forecast = [], []
    for lag in range(p + 1):
        for i in range(Nm):
            insample.append(lag * Nm + i)
            forecast.append(lag * Ntotal + Nw + i)
        for k in range(Nq):
            insample.append(Nm_states + lag * Nq + k)
            forecast.append(lag * Ntotal + Nw + Nm + k)
    return np.asarray(insample, dtype=int), np.asarray(forecast, dtype=int)


def forecast_measurement(Nw, Nm, Nq, p, rmw, rqw, temp_agg="mean"):
    """Measurement matrix of the ragged-edge filter, rows ``[w, m, q]``.

    A weekly variable is observed directly; a monthly (quarterly) variable is
    the mean -- or the sum, with ``temp_agg="sum"`` -- of its latent weekly
    values over the last ``rmw`` (``rqw``) weeks. Columns index the
    lag-interleaved state of length ``(Nw + Nm + Nq) * (p + 1)``.
    """
    if temp_agg not in ("mean", "sum"):
        raise ValueError(f"temp_agg must be 'mean' or 'sum', got {temp_agg!r}")
    if max(rmw, rqw) > p + 1:
        raise ValueError(f"the state holds {p + 1} lags, fewer than the "
                         f"aggregation window ({max(rmw, rqw)} weeks)")
    Ntotal = Nw + Nm + Nq
    kn = Ntotal * (p + 1)
    Z0 = np.zeros((Nw, kn))
    Z0[:, :Nw] = np.eye(Nw)
    Z1 = np.zeros((Nm, kn))
    w_m = 1.0 / rmw if temp_agg == "mean" else 1.0
    for bb in range(Nm):
        for ll in range(rmw):
            Z1[bb, ll * Ntotal + Nw + bb] = w_m
    Z2 = np.zeros((Nq, kn))
    w_q = 1.0 / rqw if temp_agg == "mean" else 1.0
    for bb in range(Nq):
        for ll in range(rqw):
            Z2[bb, ll * Ntotal + Nw + Nm + bb] = w_q
    return np.vstack((Z0, Z1, Z2))


def insample_transition(Phi, Nw, Nm, Nq, p):
    """State-space matrices of the in-sample filter implied by a VAR draw.

    ``Phi`` is the ``(Ntotal * p + 1, Ntotal)`` coefficient matrix with rows
    ordered lag 1 ``[w, m, q]``, lag 2 ``[w, m, q]``, ..., constant, and
    columns ``[w, m, q]``. Returns a dict with

    * ``GAMMAs`` (frequency-blocked state transition), ``GAMMAz`` (loading
      on the weekly regressors ``[w_{t-1}, ..., w_{t-p}]``), ``GAMMAc``
      (constant), so that ``GAMMAs @ a_{t-1} + GAMMAz @ z_t + GAMMAc`` is
      the VAR's one-step prediction of ``[m_t, q_t]`` in the state's first
      rows of each block, with the lag rows shifted down;
    * ``LAMBDAs_w``, ``LAMBDAz_w``, ``LAMBDAc_w``: the weekly observation
      equation, ``LAMBDAs_w @ a_t + LAMBDAz_w @ z_t + LAMBDAc_w`` being the
      VAR's prediction of ``w_t``.

    Before 0.2.2 ``GAMMAs`` carried the monthly->quarterly and
    quarterly->monthly effects at lag 1 only, so the latent states were
    filtered under a different model from the VAR being estimated.
    """
    Ntotal = Nw + Nm + Nq
    Nm_states = Nm * (p + 1)
    state_size = (Nm + Nq) * (p + 1)

    def block(rows, cols):
        """Stack lags 1..p of Phi[rows-of-regressor, cols-of-equation]."""
        return np.vstack([Phi[i * Ntotal + rows.start:i * Ntotal + rows.stop, cols]
                          for i in range(p)])

    w, m, q = slice(0, Nw), slice(Nw, Nw + Nm), slice(Nw + Nm, Ntotal)
    phi_ww, phi_wm, phi_wq = block(w, w), block(m, w), block(q, w)
    phi_mw, phi_mm, phi_mq = block(w, m), block(m, m), block(q, m)
    phi_qw, phi_qm, phi_qq = block(w, q), block(m, q), block(q, q)
    phi_wc = Phi[-1, w, np.newaxis]
    phi_mc = Phi[-1, m, np.newaxis]
    phi_qc = Phi[-1, q, np.newaxis]

    GAMMAs = np.zeros((state_size, state_size))
    # current monthly: own lags, quarterly lags (all p of them), shift rows
    GAMMAs[:Nm, :Nm * p] = phi_mm.T
    GAMMAs[:Nm, Nm_states:Nm_states + Nq * p] = phi_mq.T
    GAMMAs[Nm:Nm_states, :Nm * p] = np.eye(Nm * p)
    # current quarterly: monthly lags, own lags, shift rows
    GAMMAs[Nm_states:Nm_states + Nq, :Nm * p] = phi_qm.T
    GAMMAs[Nm_states:Nm_states + Nq, Nm_states:Nm_states + Nq * p] = phi_qq.T
    GAMMAs[Nm_states + Nq:, Nm_states:Nm_states + Nq * p] = np.eye(Nq * p)

    GAMMAz = np.zeros((state_size, Nw * p))
    GAMMAz[:Nm, :] = phi_mw.T
    GAMMAz[Nm_states:Nm_states + Nq, :] = phi_qw.T

    GAMMAc = np.zeros((state_size, 1))
    GAMMAc[:Nm] = phi_mc
    GAMMAc[Nm_states:Nm_states + Nq] = phi_qc

    LAMBDAs_w = np.hstack((np.zeros((Nw, Nm)), phi_wm.T,
                           np.zeros((Nw, Nq)), phi_wq.T))
    return dict(GAMMAs=GAMMAs, GAMMAz=GAMMAz, GAMMAc=GAMMAc,
                LAMBDAs_w=LAMBDAs_w, LAMBDAz_w=phi_ww.T, LAMBDAc_w=phi_wc)
