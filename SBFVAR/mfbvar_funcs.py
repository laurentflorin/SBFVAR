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


def varprior(nv,nlags, nex, hyp, premom):
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
    
    # dummies for the coefficients of the first lag
    sig = np.diag(premom[:,1])
    ydu[range(nv),:] = lambda1*sig
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
    
    # dummies for the covariance matrix of coefficients of different lags
    mumean = np.diag(lambda5*premom[:,0])
    ydu[int(breakss[3,0]):int(breakss[3,0])+nv,:] = mumean
    if np.kron(np.ones((1,nlags)),mumean).shape[0] > 1:
        xdu[int(breakss[3,0]):int(breakss[3,0])+nv,:] = np.hstack((np.squeeze(np.kron(np.ones((1,nlags)),mumean)), np.zeros((nv,nex))))
    else: 
        xdu[int(breakss[3,0]):int(breakss[3,0])+nv,:] = np.hstack((np.kron(np.ones((1,nlags)),mumean), np.zeros((nv,nex))))
    breakss[4] = breakss[3,0]+nv 

    return ydu, xdu

    
def prior_init(hyp,YY,spec):
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
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom)
    
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
            
            
            
            
def mdd_(hyp, YY, spec):
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
    None.

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
    
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom)
    
    # Actual observations
    YYact = YY[T0:T0+nobs, :]
    XXact = np.zeros((nobs, nv*nlags_))
    
    for i in range(nlags_):
        XXact[:, i*nv:(i+1)*nv] = YY[T0-1-i:T0+nobs-(i+1)]
        
    XXact = np.hstack((XXact, np.ones((nobs, 1))))
    
    #dummy: YYdum, XXdum
    #actual: YYact, XXact
    YY = np.transpose(np.hstack((YYdum.T, YYact.T)))
    XX = np.transpose(np.hstack((XXdum.T, XXact.T)))
    
    n_total = np.shape(YY)[0]
    n_dummy = n_total - nobs
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
    
    lnpY0 = (-nv * (n_dummy-k) * 0.5 * np.log(math.pi) - (nv/2) * np.log(np.absolute(np.linalg.det(XXdum.T @ XXdum))) -
            (n_dummy-k)*0.5*np.log(np.absolute(np.linalg.det(S0)))+nv*(nv-1)*0.25*np.log(math.pi)+gam0)
    
    #dummy and actual observation
    lnpY1 = (-nv * (n_total-k) * 0.5 * np.log(math.pi) - (nv/2) * np.log(np.absolute(np.linalg.det(XX.T @ XX))) -
            (n_total-k)*0.5*np.log(np.absolute(np.linalg.det(S1)))+nv*(nv-1)*0.25*np.log(math.pi)+gam1)
    
    lnpYY = lnpY1 - lnpY0
    
    #marginal data density
    mdd = lnpYY
    
    return mdd, YYact, YYdum, XXact, XXdum
            
def calc_yyact(hyp, YY, spec):
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
    # Use a safer approach to handle potential size mismatches
    T0_safe = min(T0, YY.shape[0])
    sample_data = YY[:min(T0_safe + 16, YY.shape[0]), :]
    
    ybar = np.mean(sample_data, axis=0)[:, np.newaxis]
    sbar = np.std(sample_data, axis=0, ddof=1)[:, np.newaxis]
    premom = np.hstack((ybar, sbar))

    # Create matrices with dummy observations
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom)

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

    return YYact, YYdum, XXact, XXdum
            
            
def prior_pdf(hyp,YY,spec,PHI,SIG):
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
    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom)
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

def is_explosive(Phi, n, p):
    """
    Given Phi checks wether the VAR is explosive
    Parameters
    ----------
    Phi : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    

    Returns
    -------
    Boolean.
    """
    # Create the companion matrix
    companion_matrix = np.zeros((n * p, n * p))
    companion_matrix[:n, :] = Phi[:n*p, :].T
    if p > 1:
        companion_matrix[n:, :-n] = np.eye(n * (p - 1))
    # Calculate the eigenvalues of the companion matrix
    eigenvalues = eig(companion_matrix)[0]
    # Check if any eigenvalue's absolute value is greater than 1
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