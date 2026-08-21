"""Validate the CPZ conditional-MDD formula used by fit_cpz(return_mdd=True).

The closed form implemented in ``_estimation_cpz.fit_cpz``,

    ln p(Y | invSig, h) = -nT/2 ln(2pi) + T/2 ln|invSig| - n/2 sum_t h_t
                          + 1/2 ln|invVbeta| - 1/2 ln|Kbeta|
                          - 1/2 (S - b' Kbeta^{-1} b),

must equal Chib's (1995) basic marginal-likelihood identity evaluated at the
posterior mean beta* = Kbeta^{-1} b:

    ln p(Y | invSig, h) = ln p(Y | beta*) + ln p(beta*) - ln p(beta* | Y).

The test reproduces both sides from scratch on random data and requires
agreement to near machine precision.
"""

import unittest

import numpy as np


class TestConditionalMDDFormula(unittest.TestCase):
    def test_formula_matches_chib_identity(self):
        rng = np.random.default_rng(12345)
        for trial in range(5):
            T = int(rng.integers(20, 60))
            n = int(rng.integers(2, 5))
            k = int(rng.integers(3, 9))
            X = rng.standard_normal((T, k))
            Y = rng.standard_normal((T, n))
            A = rng.standard_normal((n, n))
            invSig = A @ A.T + n * np.eye(n)
            h = rng.standard_normal(T) * 0.3
            ivb = rng.uniform(0.5, 3.0, size=n * k)
            D = np.exp(-h)

            XtDX = (X.T * D) @ X
            Kb = np.kron(invSig, XtDX) + np.diag(ivb)
            b = ((X.T * D) @ Y @ invSig).reshape(n * k, order="F")
            mu = np.linalg.solve(Kb, b)
            S = float(np.sum((Y @ invSig) * Y * D[:, None]))
            ldi = np.linalg.slogdet(invSig)[1]
            ldK = np.linalg.slogdet(Kb)[1]

            formula = (-0.5 * n * T * np.log(2 * np.pi) + 0.5 * T * ldi
                       - 0.5 * n * h.sum() + 0.5 * np.log(ivb).sum()
                       - 0.5 * ldK - 0.5 * (S - b @ mu))

            B = mu.reshape(k, n, order="F")
            E = Y - X @ B
            llik = (-0.5 * n * T * np.log(2 * np.pi) + 0.5 * T * ldi
                    - 0.5 * n * h.sum()
                    - 0.5 * float(np.sum((E @ invSig) * E * D[:, None])))
            lprior = (-0.5 * n * k * np.log(2 * np.pi)
                      + 0.5 * np.log(ivb).sum()
                      - 0.5 * float(mu @ (ivb * mu)))
            lpost = -0.5 * n * k * np.log(2 * np.pi) + 0.5 * ldK
            chib = llik + lprior - lpost

            self.assertAlmostEqual(
                formula, chib, delta=1e-8 * max(1.0, abs(chib)),
                msg=f"trial {trial}: formula {formula} != Chib {chib}")

    def test_dummy_augmented_formula_matches_chib_identity(self):
        """Same identity with sum-of-coefficients/DIO-style dummy rows: the
        prior becomes N(m0, P0^{-1}) with P0 = invVbeta + kron(invSig, Xd'Xd)
        and P0 m0 = vec(Xd' Yd invSig)."""
        rng = np.random.default_rng(777)
        for trial in range(5):
            T = int(rng.integers(20, 50))
            n = int(rng.integers(2, 4))
            k = int(rng.integers(3, 8))
            nd = int(rng.integers(1, n + 2))     # dummy rows
            X = rng.standard_normal((T, k))
            Y = rng.standard_normal((T, n))
            Xd = rng.standard_normal((nd, k)) * 0.5
            Yd = rng.standard_normal((nd, n)) * 0.5
            A = rng.standard_normal((n, n))
            invSig = A @ A.T + n * np.eye(n)
            h = rng.standard_normal(T) * 0.3
            ivb = rng.uniform(0.5, 3.0, size=n * k)
            D = np.exp(-h)

            XtDX = (X.T * D) @ X + Xd.T @ Xd
            Kb = np.kron(invSig, XtDX) + np.diag(ivb)
            b = (((X.T * D) @ Y + Xd.T @ Yd) @ invSig).reshape(n * k, order="F")
            mu = np.linalg.solve(Kb, b)
            S = float(np.sum((Y @ invSig) * Y * D[:, None]))
            ldi = np.linalg.slogdet(invSig)[1]
            ldK = np.linalg.slogdet(Kb)[1]

            P0 = np.diag(ivb) + np.kron(invSig, Xd.T @ Xd)
            b0 = ((Xd.T @ Yd) @ invSig).reshape(n * k, order="F")
            ldP0 = np.linalg.slogdet(P0)[1]
            quad0 = float(b0 @ np.linalg.solve(P0, b0))

            formula = (-0.5 * n * T * np.log(2 * np.pi) + 0.5 * T * ldi
                       - 0.5 * n * h.sum() + 0.5 * ldP0 - 0.5 * ldK
                       - 0.5 * (S + quad0 - b @ mu))

            # Chib identity at beta* = mu with the dummy-augmented prior
            m0 = np.linalg.solve(P0, b0)
            B = mu.reshape(k, n, order="F")
            E = Y - X @ B
            llik = (-0.5 * n * T * np.log(2 * np.pi) + 0.5 * T * ldi
                    - 0.5 * n * h.sum()
                    - 0.5 * float(np.sum((E @ invSig) * E * D[:, None])))
            dmu = mu - m0
            lprior = (-0.5 * n * k * np.log(2 * np.pi) + 0.5 * ldP0
                      - 0.5 * float(dmu @ (P0 @ dmu)))
            lpost = -0.5 * n * k * np.log(2 * np.pi) + 0.5 * ldK
            chib = llik + lprior - lpost

            self.assertAlmostEqual(
                formula, chib, delta=1e-8 * max(1.0, abs(chib)),
                msg=f"trial {trial}: formula {formula} != Chib {chib}")


if __name__ == "__main__":
    unittest.main()
