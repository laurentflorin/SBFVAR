# -*- coding: utf-8 -*-
"""
This file contains functions used in mf_bvar_estim

@author: florinl
"""

#temp


import pandas as pd
import numpy as np
import math
import scipy as sp

from scipy.stats import invwishart
from scipy.special import loggamma
from scipy.special import gamma
from scipy.stats import multivariate_normal
from scipy.linalg import inv
from scipy.linalg.lapack import dpotri
from scipy.linalg import eig

from .pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse

# Penalty MDD value returned when no valid VAR regression rows remain (e.g., at early Gibbs
# sampler iterations with ragged-edge missing data). Signals a poor hyperparameter configuration.
_PENALTY_MDD = -1e16


def prior_mean_vector(prior_mean, nv):
    """Per-variable prior mean of the own first lag, as a float vector.

    ``None`` is the Schorfheide-Song default: every variable centred on a
    random walk (1), which suits levels and year-on-year growth rates. For
    period-on-period growth rates the usual centring is white noise (0), with
    persistent series such as interest rates kept at 1 (Banbura, Giannone and
    Reichlin, 2010).
    """
    if prior_mean is None:
        return np.ones(nv)
    delta = np.asarray(prior_mean, dtype=float).ravel()
    if delta.shape != (nv,):
        raise ValueError(f"prior_mean has {delta.size} entries for {nv} variables")
    if not np.all(np.isfinite(delta)):
        raise ValueError("prior_mean must be finite")
    return delta


def resolve_prior_mean(prior_mean, varlist):
    """Turn ``prior_mean`` into a vector in the model's variable order.

    Accepts ``None`` (the random-walk default, returned as ``None`` so the
    prior is built exactly as before), a mapping from variable name to prior
    mean -- variables it does not name keep 1 -- or a sequence already in the
    model's variable order. A name that is not a model variable is refused,
    since a misspelt name would otherwise leave that series on a random walk.
    """
    if prior_mean is None:
        return None
    names = [str(v) for v in list(varlist)]
    if isinstance(prior_mean, dict):
        unknown = sorted(set(map(str, prior_mean)) - set(names))
        if unknown:
            raise ValueError(f"prior_mean names variables the model does not have: {unknown}")
        return prior_mean_vector([float(prior_mean.get(n, 1.0)) for n in names], len(names))
    return prior_mean_vector(prior_mean, len(names))


def varprior(nv, nlags, nex, hyp, premom, prior_mean=None):
    """
    

    Parameters
    ----------
    nv : TYPE
        numer of variables.
    nlags : TYPE
        number of lags.
    nex : TYPE
        number of exogenous variables inculding intercept.
    hyp : TYPE
        vector of hyperparameters.
    premom : TYPE
        pre-sample moments.

    Returns
    -------
    None.

    """
    lambda1 = hyp[0]
    lambda2 = hyp[1]
    lambda3 = int(hyp[2])
    lambda4 = hyp[3]
    lambda5 = hyp[4]

    # initializations
    dsize = nex + (nlags+lambda3+1)*nv
    breakss = np.zeros((5,1))
    ydu = np.zeros((int(dsize),int(nv)))
    xdu = np.zeros((int(dsize),int(nv*nlags+nex)))
    
    # dummies for the coefficients of the first lag: prior mean delta_i on
    # the own first lag (1 = random walk, the default; 0 = white noise)
    delta = prior_mean_vector(prior_mean, nv)
    sig = np.diag(premom[:,1])
    ydu[range(nv),:] = lambda1*np.diag(premom[:,1]*delta)
    xdu[:nv,:sig.shape[1]] = lambda1*sig
    breakss[0] = nv
    
    
    #dummies for the coefficients of the remaining lags
    if nlags > 1:
        ydu[int(breakss[0,0]):(nv*nlags),:] = np.zeros(((nlags-1)*nv, nv))
        j = 1
        while j <= nlags-1:
            xdu[int(breakss[0,0])+(j-1)*nv:int(breakss[0,0])+j*nv] = np.hstack((np.zeros((nv,j*nv)), lambda1*sig*((j+1)**lambda2), np.zeros((nv,(nlags-1-j)*nv+nex))))
            j=j+1
        breakss[1,0] = breakss[0,0] +(nlags-1)*nv
    else:
        breakss[1,0] = breakss[0,0]
        
    # dummies for the covariance matrix of error terms
    ydu[int(breakss[1,0]):int(breakss[1,0])+lambda3*nv,:] = np.kron(np.ones((lambda3,1)),sig)
    breakss[2,0] = breakss[1,0]+lambda3*nv  
    
    
    # dummies for the coefficents of the constant term
    lammean = lambda4*premom[:,0]
    ydu[int(breakss[2,0]),:] = lammean
    xdu[int(breakss[2,0]),:] = np.hstack((np.squeeze(np.kron(np.ones((1,nlags)),lammean)), lambda4))
    breakss[3] = breakss[2,0]+1
    
    # sum-of-coefficients dummies, scaled by delta on both sides as in
    # Banbura, Giannone and Reichlin (2010): for a variable centred on white
    # noise (delta_i = 0) the unit-root restriction they encode drops out
    mumean = np.diag(lambda5*premom[:,0]*delta)
    ydu[int(breakss[3,0]):int(breakss[3,0])+nv,:] = mumean
    if np.kron(np.ones((1,nlags)),mumean).shape[0] > 1:
        xdu[int(breakss[3,0]):int(breakss[3,0])+nv,:] = np.hstack((np.squeeze(np.kron(np.ones((1,nlags)),mumean)), np.zeros((nv,nex))))
    else: 
        xdu[int(breakss[3,0]):int(breakss[3,0])+nv,:] = np.hstack((np.kron(np.ones((1,nlags)),mumean), np.zeros((nv,nex))))
    breakss[4] = breakss[3,0]+nv 

    return ydu, xdu

    
def prior_init(hyp,YY,spec, prior_mean=None):
    """
    

    Parameters
    ----------
    hyp : TYPE
        DESCRIPTION.
    YY : TYPE
        DESCRIPTION.
    spec : TYPE
        DESCRIPTION.

    Returns
    -------
    Phi_tilde
    
    sigma

    """
    # Data specification and setting
    nlags_  = spec[0]      # number of lags   
    T0      = spec[1]      # size of pre-sample 
    nex_    = spec[2]      # number of exogenous vars 1 means intercept only 
    nv      = spec[3]      # number of variables 
    Nm      = spec[4]      # number of monthly variables
    
    # Dummy observations
    # Obtain mean and standard deviation from expandend pre-sample data
    YY0     =   YY[range(T0),:]  
    ybar    =   np.mean(YY0, axis = 0)
    sbar    =   np.std(YY0, axis = 0, ddof = 1) 
    premom  =   np.column_stack((ybar, sbar))


    #Generate matrices with dummy observations
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom, prior_mean=prior_mean)
    
    inv_x = sp.linalg.pinvh(XXdum.T@XXdum)
    
    
    Phi_tilde = (inv_x) @ XXdum.T @ YYdum
    Sigma = np.transpose(YYdum-XXdum @ Phi_tilde) @ (YYdum-XXdum @ Phi_tilde)
    
    
    # Draws from the density Sigma | Y    
    sigma   = invwishart.rvs(scale = Sigma, df = YYdum.shape[0]-nv*nlags_-1)
    
    return Phi_tilde, sigma    
    
def initialize(GAMMAs,GAMMAz,GAMMAc,GAMMAu,
            LAMBDAs,LAMBDAz,LAMBDAc,LAMBDAu,LAMBDAs_t,LAMBDAz_t,LAMBDAc_t,LAMBDAu_t,
            sig_qq,sig_mm,sig_qm,sig_mq,Zm,YDATA,init_mean,init_var,spec, Nm):
    
    # Specification
    p       = spec[0]      # number of lags   
    T0      = spec[1]      # size of pre-sample 
    nex_    = spec[2]      # number of exogenous vars 1 means intercept only 
    nv      = spec[3]      # number of variables 
    Nm      = spec[4]      # number of monthly variables
    
    # Initialization         
    At   = init_mean[:,np.newaxis] 
    Pt   = init_var
    
    # Kalman Filter Loop
    for t in range(p+1,T0):
    
        if (t+1)%3 == 0:
            At1 = At
            Pt1 = Pt
            
            #Forecasting
            alphahat = GAMMAs @ At1 + GAMMAz @ Zm[t-p-1,:, np.newaxis] + GAMMAc
            Phat = GAMMAs @ Pt1 @ np.transpose(GAMMAs) + GAMMAu @ sig_qq @ np.transpose(GAMMAu)
            Phat = 0.5*(Phat+np.transpose(Phat))
            
            yhat = LAMBDAs @ alphahat + LAMBDAz @ Zm[t-p-1,:, np.newaxis] + LAMBDAc
            
            nut = YDATA[t,:, np.newaxis] - yhat
            
            Ft = (LAMBDAs @ Phat @ LAMBDAs.T + LAMBDAu @ sig_mm @ LAMBDAu.T
                + LAMBDAs @ GAMMAu @ sig_qm @ LAMBDAu.T
                + LAMBDAu @ sig_mq @ GAMMAu.T @ LAMBDAs.T)
            
            Ft = 0.5*(Ft+Ft.T)
            Xit = LAMBDAs @ Phat + LAMBDAu @ sig_mq @ GAMMAu.T
            
            At = alphahat + Xit.T @ sp.linalg.pinvh(Ft) @ nut
            Pt = Phat - Xit.T @ sp.linalg.pinvh(Ft) @ Xit
        
        
        else:
            At1 = At
            Pt1 = Pt
            
            # Forecasting
            alphahat = GAMMAs @ At1 + GAMMAz @ Zm[t-p-1,:, np.newaxis] + GAMMAc
            Phat = GAMMAs @ Pt1 @ np.transpose(GAMMAs) + GAMMAu @ sig_qq @ np.transpose(GAMMAu)
            Phat = 0.5*(Phat+np.transpose(Phat))
            
            yhat = LAMBDAs_t @ alphahat + LAMBDAz_t @ Zm[t-p-1,:, np.newaxis] + LAMBDAc_t
            nut = YDATA[t,:Nm, np.newaxis] - yhat
            
            Ft = (LAMBDAs_t @ Phat @ LAMBDAs_t.T + LAMBDAu_t @ sig_mm @ LAMBDAu_t.T
                + LAMBDAs_t@GAMMAu@sig_qm@LAMBDAu_t.T
                + LAMBDAu_t@sig_mq@GAMMAu.T@LAMBDAs_t.T)
            
            Ft = 0.5*(Ft+Ft.T)
            Xit = LAMBDAs_t @ Phat + LAMBDAu_t @ sig_mq @ GAMMAu.T
            
            At = alphahat + Xit.T @ sp.linalg.pinvh(Ft) @ nut
            Pt = Phat - Xit.T @ sp.linalg.pinvh(Ft) @ Xit
            
            
    At_final = At
    Pt_final = Pt
    
    return(At_final, Pt_final)
            
            
            
            
def _filter_valid_var_rows(YYact, XXact):
    valid = (~np.isnan(YYact).any(axis=1)) & (~np.isnan(XXact).any(axis=1))

    YYact_f = YYact[valid, :]
    XXact_f = XXact[valid, :]

    return YYact_f, XXact_f, valid


def mdd_(hyp, YY, spec, prior_mean=None):
    """

    Parameters
    ----------
    hyp : TYPE
        DESCRIPTION.
    YY : TYPE
        DESCRIPTION.
    spec : TYPE
        DESCRIPTION.
    efficient : TYPE
        DESCRIPTION.

    Returns
    -------
    mdd : float
        Log marginal data density. Returns ``_PENALTY_MDD`` (a large negative value)
        when no valid VAR regression rows remain after filtering ragged-edge NaN rows
        (e.g., at early Gibbs sampler iterations). Callers should treat this as a
        signal to skip the current hyperparameter configuration.
    YYact, YYdum, XXact, XXdum : ndarray
        Actual and dummy observation matrices.

    """
    # Data Specification and setting
            
    nlags_  = int(spec[0])      # number of lags   
    T0      = int(spec[1])      # size of pre-sample 
    nex_    = int(spec[2])      # number of exogenous vars 1 means intercept only 
    nv      = int(spec[3])     # number of variables 
    nobs    = int(spec[4])      # number of observations
            
    # Dummy observations
    
    #Obtain mean and standard deviation from expanded pre-sample data
    
    YY0 = YY[:int(T0+16),:]
    ybar    =   np.mean(YY0, axis = 0)[:,np.newaxis]
    sbar    =   np.std(YY0, axis = 0, ddof = 1)[:,np.newaxis] 
    premom = np.hstack((ybar, sbar))
    
    
    # Create Matrices with dummy observations
    
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom, prior_mean=prior_mean)
    
    # Actual observations
    YYact = YY[T0:T0+nobs, :]
    XXact = np.zeros((nobs, nv*nlags_))
    
    for i in range(nlags_):
        XXact[:, i*nv:(i+1)*nv] = YY[T0-1-i:T0+nobs-(i+1)]
        
    XXact = np.hstack((XXact, np.ones((nobs, 1))))

    # Filter out ragged-edge rows with missing values
    YYact, XXact, _ = _filter_valid_var_rows(YYact, XXact)

    if YYact.shape[0] == 0:
        # No valid rows remain (transient condition at early Gibbs iterations); return penalty MDD
        return _PENALTY_MDD, YYact, YYdum, XXact, XXdum

    nobs_eff = YYact.shape[0]
    #actual: YYact, XXact
    YY = np.transpose(np.hstack((YYdum.T, YYact.T)))
    XX = np.transpose(np.hstack((XXdum.T, XXact.T)))
    
    n_total = np.shape(YY)[0]
    n_dummy = n_total - nobs_eff
    nv = np.shape(YY)[1]
    k = np.shape(XX)[1]
    
    
    #Compute the log marginal data density for the VAR model
    
    #Phi0 = np.linalg.solve((XXdum.T @ XXdum), (XXdum.T @ YYdum))
    #S0 = (YYdum.T @ YYdum) - np.linalg.solve((XXdum.T @ XXdum).T, (YYdum.T @ XXdum).T).T @ XXdum.T @ YYdum
    S0 =  (YYdum.T @ YYdum) - ((YYdum.T @ XXdum) @ calculate_pseudo_inverse((XXdum.T @ XXdum))) @ XXdum.T @ YYdum
    #Phi1 = np.linalg.solve((XX.T @ XX), (XX.T @ YY))
    #S1 = (YY.T @ YY) - np.linalg.solve((XX.T @ XX).T, (YY.T @ XX).T).T @ XX.T @ YY
    S1 = (YY.T @ YY) - ((YY.T @ XX) @ calculate_pseudo_inverse((XX.T @ XX)))  @ XX.T @ YY
    
    # compute constants for integrals
    gam0 = 0
    gam1 = 0
    
    for i in range(nv):
        gam0 = gam0 + loggamma(0.5*(n_dummy-k+1-(i+1)))
        gam1 = gam1 + loggamma(0.5*(n_total-k+1-(i+1)))
    
    #dummy observation
    # Use slogdet instead of log(|det(...)|) to avoid float64 overflow when
    # large lambda values inflate the prior matrices (SBFVAR stacks all nv
    # variables in one system, making determinants scale as lambda^(2*k)).
    _, logdet_XXdum = np.linalg.slogdet(XXdum.T @ XXdum)
    _, logdet_S0    = np.linalg.slogdet(S0)
    lnpY0 = (-nv * (n_dummy-k) * 0.5 * np.log(math.pi) - (nv/2) * logdet_XXdum -
            (n_dummy-k)*0.5*logdet_S0+nv*(nv-1)*0.25*np.log(math.pi)+gam0)
    
    #dummy and actual observation
    _, logdet_XX = np.linalg.slogdet(XX.T @ XX)
    _, logdet_S1 = np.linalg.slogdet(S1)
    lnpY1 = (-nv * (n_total-k) * 0.5 * np.log(math.pi) - (nv/2) * logdet_XX -
            (n_total-k)*0.5*logdet_S1+nv*(nv-1)*0.25*np.log(math.pi)+gam1)
    
    lnpYY = lnpY1 - lnpY0
    
    #marginal data density
    mdd = lnpYY
    
    return mdd, YYact, YYdum, XXact, XXdum
            
def calc_yyact(hyp, YY, spec, prior_mean=None):
    """
    Calculate actual and dummy observations matrices for VAR estimation.
    Handles potential dimension mismatches in the unified approach.

    Parameters
    ----------
    hyp : list
        Hyperparameters for Minnesota prior
    YY : ndarray
        Time series data
    spec : ndarray
        Model specifications [nlags, T0, nex, nv, nobs]

    Returns
    -------
    YYact, YYdum, XXact, XXdum : ndarrays
        Actual and dummy observation matrices
    """
    # Data Specification and setting
    nlags_ = int(spec[0])  # number of lags
    T0 = int(spec[1])  # size of pre-sample
    nex_ = int(spec[2])  # number of exogenous vars (1 means intercept only)
    nv = int(spec[3])  # number of variables
    nobs = int(spec[4])  # number of observations

    # Dummy observations - obtain mean and standard deviation from expanded pre-sample data
    
    YY0 = YY[:int(T0+16),:]
    ybar    =   np.mean(YY0, axis = 0)[:,np.newaxis]
    sbar    =   np.std(YY0, axis = 0, ddof = 1)[:,np.newaxis] 
    premom = np.hstack((ybar, sbar))
    

    # Create matrices with dummy observations
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom, prior_mean=prior_mean)

    # Actual observations - ensure we don't exceed data bounds
    actual_obs = min(nobs, YY.shape[0] - T0)
    if actual_obs < nobs:
        print(f"Warning: Requested {nobs} observations but only {actual_obs} available after lag period")
    
    YYact = YY[T0:T0 + actual_obs, :]
    XXact = np.zeros((actual_obs, nv * nlags_ + nex_))

    # Fill in the lags safely
    for i in range(nlags_):
        for t in range(actual_obs):
            lag_idx = T0 + t - (i + 1)
            if 0 <= lag_idx < YY.shape[0]:
                XXact[t, i * nv:(i + 1) * nv] = YY[lag_idx, :]
            else:
                # If lag goes beyond available data, use the first observation
                XXact[t, i * nv:(i + 1) * nv] = YY[0, :]

    # Add constant term
    XXact[:, -nex_:] = 1.0

    # Filter out ragged-edge rows with missing values
    YYact, XXact, _ = _filter_valid_var_rows(YYact, XXact)
    # If no valid rows remain, return empty arrays (caller handles this)

    return YYact, YYdum, XXact, XXdum
            
            
def prior_pdf(hyp,YY,spec,PHI,SIG, prior_mean=None):
    """
    

    Parameters
    ----------
    hyp : TYPE
        DESCRIPTION.
    YY : TYPE
        DESCRIPTION.
    spec : TYPE
        DESCRIPTION.
    PHI : TYPE
        DESCRIPTION.
    SIG : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """   
    # Data Specification and setting
            
    nlags_  = spec[0]      # number of lags   
    T0      = spec[1]      # size of pre-sample 
    nex_    = spec[2]      # number of exogenous vars 1 means intercept only 
    nv      = spec[3]      # number of variables 
    nobs    = spec[4]      # number of observations
    
    # Dummy Observations
    
    # Obtain mean and standard deviation from expanded pre-sample data
    
    YY0 = YY[:T0,:]
    ybar    =   np.mean(YY0, axis = 0)[:,np.newaxis]
    sbar    =   np.std(YY0, axis = 0, ddof = 1)[:,np.newaxis] 
    premom  =   np.hstack((ybar, sbar))
    
    #generate matrices with dummy observations
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom, prior_mean=prior_mean)
    n = YYdum.shape[1]
    
    
    inv_x = sp.linalg.pinvh(XXdum.T @ XXdum)
    Phi_tilde = inv_x @ XXdum.T @ YYdum
    Sigma = np.transpose(YYdum - (XXdum @ Phi_tilde)) @ (YYdum - (XXdum @ Phi_tilde))
    
    var = logpdf(x = PHI.reshape((n*(n*nlags_+1), 1), order = "F"), mean = np.squeeze(Phi_tilde.reshape((n*(n*nlags_+1), 1), order = "F")), cov= np.kron(SIG, inv_x))
    
    MN_pdf = multivariate_normal.pdf(PHI.reshape((n*(n*nlags_+1), 1), order = "F"), np.squeeze(Phi_tilde.reshape((n*(n*nlags_+1), 1), order = "F")), np.kron(SIG, inv_x) , allow_singular = False)
    MN_logpdf = np.log(MN_pdf)

    # IW_pdf = invwishart.pdf(SIG, len(YYdum-nv*nlags_-1), Sigma)
    IW_logpdf = invwishart.logpdf(SIG, len(YYdum-nv*nlags_-1), Sigma)
    
    return MN_logpdf, IW_logpdf




    def pdf(x, mean, cov):
        return np.exp(logpdf(x, mean, cov))


    def logpdf(x, mean, cov):
        # `eigh` assumes the matrix is Hermitian.
        vals, vecs = np.linalg.eigh(cov)
        sign, logdet     = np.linalg.slogdet(np.kron(SIG, inv_x))
        valsinv    = np.array([1./v for v in vals])
        # `vecs` is R times D while `vals` is a R-vector where R is the matrix 
        # rank. The asterisk performs element-wise multiplication.
        U          = vecs * np.sqrt(valsinv)
        rank       = len(vals)
        dev        = x - mean
        # "maha" for "Mahalanobis distance".
        maha       = np.square(np.dot(dev, U)).sum()
        log2pi     = np.log(2 * np.pi)
        return -0.5 * (rank * log2pi + maha + logdet)

# Points just above one at which the characteristic polynomial is screened.
_EXPLOSIVE_SCREEN_GRID = 1.0 + np.concatenate(([1e-7], np.geomspace(1e-5, 1.0, 40)))


def _companion(Phi, n, p):
    companion_matrix = np.zeros((n * p, n * p))
    companion_matrix[:n, :] = Phi[:n*p, :].T
    if p > 1:
        companion_matrix[n:, :-n] = np.eye(n * (p - 1))
    return companion_matrix


def _is_explosive_eig(Phi, n, p):
    """Reference check: any eigenvalue of the companion matrix outside the unit circle."""
    eigenvalues = eig(_companion(Phi, n, p))[0]
    return np.any(np.abs(eigenvalues) > 1)


def smoother_pinv(P, rcond=1e-10):
    """
    Inverse of the one-step-ahead state covariance in the backward
    (Carter-Kohn) step of the simulation smoothers (SBFVAR 0.2.4).

    That covariance is singular by construction: the state stacks lagged
    copies of the latent series, and the temporal-aggregation constraints are
    observed without error, so the filtered covariance loses rank at every
    low-frequency observation. The backward step therefore needs a
    generalised inverse. With the Moore-Penrose inverse the conditional mean
    and covariance are exact, because the state being conditioned on lies in
    the covariance's range.

    Before 0.2.4 the smoothers called ``invert_matrix``, whose full-pivot LU
    test accepts numerically singular matrices (round-off leaves their zero
    eigenvalues at up to 1e-13 of the largest) and returns inverses with
    entries of up to 1e19. The amplified
    round-off made the smoothed covariance indefinite and, at some parameter
    values, blew the latent paths up to 1e40 and beyond; every coefficient
    draw given such paths is explosive, so the stationarity screen rejected
    them all, the parameters could no longer move, and the chain stayed
    there. The failure was found in the MBF-VAR, whose smoothers share this
    code (MBFVAR 0.9.4); the SBF-VAR's state carries the same lag copies and
    aggregation constraints.

    Eigenvalues below ``rcond`` times the largest are treated as zero. In the
    paper's year-on-year and growth-rate fits the zero eigenvalues come out
    between 1e-20 and 1e-13 of the largest and the genuine ones at 1e-3 and
    above, with none in between, so any ``rcond`` in that gap gives the same
    inverse.

    Parameters
    ----------
    P : ndarray of shape (k, k)
        symmetric positive semi-definite covariance.
    rcond : float
        relative eigenvalue cut-off.

    Returns
    -------
    ndarray of shape (k, k)
        the Moore-Penrose inverse of the symmetrised ``P``.
    """
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    keep = w > rcond * w[-1]
    Vk = V[:, keep]
    return (Vk / w[keep]) @ Vk.T


def is_explosive(Phi, n, p):
    """
    Given Phi checks wether the VAR is explosive, i.e. whether the companion
    matrix has an eigenvalue with modulus above one.

    The Gibbs sampler calls this for every candidate coefficient draw, so the
    check is on the critical path. It first evaluates the characteristic
    polynomial of the companion matrix,
    f(x) = det(x^p I - x^(p-1) A_1 - ... - A_p), on a grid of points just above
    one; f is positive at infinity, so a sign change certifies a real root
    above one and the draw is explosive. Draws without a sign change (stable
    ones, or explosive through complex roots or an even number of real roots)
    get the eigenvalues, without eigenvectors, so the decision is that of the
    eigenvalue check (``_is_explosive_eig``). Ported from MBFVAR 0.9.1.

    Parameters
    ----------
    Phi : ndarray of shape (n*p + nex, n)
        VAR coefficients, lags stacked by lag, exogenous terms last.
    n : int
        number of variables.
    p : int
        number of lags.

    Returns
    -------
    Boolean.
    """
    A = np.asarray(Phi[:n*p, :], dtype=float).reshape(p, n, n).transpose(0, 2, 1)
    if not np.isfinite(A).all():
        # The eigenvalue check raises on non-finite coefficients; keep that
        # rather than let the screen pass them as stable.
        return _is_explosive_eig(Phi, n, p)
    powers =_EXPLOSIVE_SCREEN_GRID[:, None] ** np.arange(p - 1, -1, -1)[None, :]
    M = (_EXPLOSIVE_SCREEN_GRID ** p)[:, None, None] * np.eye(n) - np.einsum("gi,iab->gab", powers, A)
    signs = np.append(np.sign(np.linalg.det(M)), 1.0)
    if np.any(signs[:-1] * signs[1:] < 0):
        return True
    eigenvalues = sp.linalg.eigvals(_companion(Phi, n, p), check_finite=False, overwrite_a=True)
    return np.any(np.abs(eigenvalues) > 1)





"""
def mvnpdf(X, mean, cov):
    
    n, d = np.shape(X)
    
    X0 = X - mean
    
    
def cholcov(SIGMA):
    
    # If Sigma is Positive definite we can use np.chol to compute T such that SIGMA = T'*T.
    # Then T is the square, upper triangular CHolesky factor
    
    n, m = np.shape(SIGMA)
    
    flag = is_pos_def(SIGMA) # test if SIGMA is positive definite
    
    tol = 10*np.spacing(max(abs(np.diagonal(SIGMA))))
    
    if (n == m) and ((np.abs(SIGMA - SIGMA.T) < tol).all()):
        
        if flag == True:
            T = np.linalg.cholesky(SIGMA)
        
        else:
            # Can get factors of the form SIGMA = T' * T using the eigenvalue
            # decomposition of a symmetric matrix, so long as the matrix is
            # positive semi-definite
            U, D = eig((SIGMA + SIGMA.T)/2)
            
            # Pick eigenvector direction so max abs coordinate is positive
            ignore, maxind = np.absolute(U).max(axis=0), np.absolute(U).argmax(axis=0)
            
            negloc = U[maxind - 1 + range(0,m*n,  n)] 
            U[,negloc] = -U[,negloc]

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)  
"""    


def finite_draw_mask(*arrays):
    """Which stored draws are usable: True where every array is finite at
    that draw index, judged only at positions that are finite for most
    draws.

    Each array is indexed by draw along axis 0. A position that is
    non-finite in more than half of the draws is structural (the ragged-edge
    padding in YYactsim_list, for instance) and is ignored; a draw is flagged
    only where it alone is non-finite. This is what lets one poisoned draw be
    dropped without discarding the whole fit, and lets a fit with structural
    NaN padding pass untouched.
    """
    mask = None
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        fin = np.isfinite(a.reshape(a.shape[0], -1))
        common = fin.mean(axis=0) > 0.5
        ok = fin[:, common].all(axis=1)
        mask = ok if mask is None else (mask & ok)
    return mask
