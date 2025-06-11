import os
import sys
import numpy as np
import math
from collections import deque
from scipy.linalg import companion
from scipy.stats import invwishart
from scipy.linalg import block_diag
import pandas as pd
from scipy.stats import multivariate_normal
from datetime import datetime
from pandas.tseries.offsets import Week, MonthBegin, QuarterBegin, Day
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import matplotlib.backends.backend_pdf
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pickle
import copy
from scipy.linalg import block_diag  


from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix
from .mfbvar_funcs import calc_yyact, is_explosive

tqdm = partial(tqdm, position=0, leave=True)
pio.renderers.default = 'browser'



def fit(self, mufbvar_data, hyp, var_of_interest=None, temp_agg='mean', max_it_explosive = 1000, check_explosive = True):
    """
    Fit the mixed-frequency BVAR model using MUFBVAR's approach with
    built-in aggregation relationships in the measurement equation.
    
    Parameters:
    -----------
    mufbvar_data : MUFBVARData object
        Processed data for the BVAR model
    hyp : ndarray
        Hyperparameters for the Minnesota prior
    var_of_interest : str or None
        Variable of interest for forecasting
    temp_agg : str
        Temporal aggregation method ('mean' or 'sum')
    """
    explosive_counter = 0
    valid_draws = []
    
    self.nex = 1
    self.hyp = hyp
    self.temp_agg = temp_agg
    self.var_of_interest = var_of_interest
    
    assert self.temp_agg in ("mean", "sum"), f"Invalid temp_agg: {self.temp_agg}. Choose 'mean' or 'sum'."
    
    # Data from mufbvar_data
    YMX_list = copy.deepcopy(mufbvar_data.YMX_list)
    YM0_list = copy.deepcopy(mufbvar_data.YM0_list)
    select_m_list = copy.deepcopy(mufbvar_data.select_m_list)
    vars_m_list = copy.deepcopy(mufbvar_data.vars_m_list)
    self.YMh_list = copy.deepcopy(mufbvar_data.YMh_list)
    index_list = copy.deepcopy(mufbvar_data.index_list)
    frequencies = copy.deepcopy(mufbvar_data.frequencies)
    self.frequencies = frequencies
    YQX_list = copy.deepcopy(mufbvar_data.YQX_list)
    YQ0_list = copy.deepcopy(mufbvar_data.YQ0_list)
    select_q = copy.deepcopy(mufbvar_data.select_q)
    input_data_Q = copy.deepcopy(mufbvar_data.input_data_Q)
    self.input_data_Q = input_data_Q
    varlist_list = copy.deepcopy(mufbvar_data.varlist_list[-1])
    select_list = copy.deepcopy(mufbvar_data.select_list)
    select_c_list = copy.deepcopy(mufbvar_data.select_c_list)
    Nm_list = copy.deepcopy(mufbvar_data.Nm_list)
    nv_list = copy.deepcopy(mufbvar_data.nv_list)
    Nq_list = copy.deepcopy(mufbvar_data.Nq_list)
    select_list_sep = copy.deepcopy(mufbvar_data.select_list_sep)
    freq_ratio_list = copy.deepcopy(mufbvar_data.freq_ratio_list)
    YQ_list = copy.deepcopy(mufbvar_data.YQ_list)
    Tstar_list = copy.deepcopy(mufbvar_data.Tstar_list)
    T_list = copy.deepcopy(mufbvar_data.T_list)
    YDATA_list = copy.deepcopy(mufbvar_data.YDATA_list)
    YM_list = copy.deepcopy(mufbvar_data.YM_list)
    input_data = copy.deepcopy(mufbvar_data.input_data)
    self.input_data = input_data
    self.varlist = mufbvar_data.varlist_list[-1]
    self.select = mufbvar_data.select_list[-1]
    self.select_w = mufbvar_data.select_m_list[-1]
    self.select_m_q = mufbvar_data.select_q[-1]

    
    nburn = round((self.nburn_perc)*math.ceil(self.nsim/self.thining))
    self.nburn = nburn
    
    nlags = self.nlags  # Number of lags for VAR
    p = self.nlags  # Number of lags for VAR
    
    # Validate frequency ratios
    rmw = freq_ratio_list[1]  # Monthly to weekly ratio (typically 4)
    rqw = freq_ratio_list[0]   # Quarterly to weekly ratio (typically 12)

    # Extract variable counts
    Nq = Nq_list[0]  # Quarterly variables
    Nm = Nm_list[0]  # Monthly variables
    Nw = Nm_list[1]  # Weekly variables (second entry in YM_list)
    Ntotal = Nq + Nm + Nw  # Total number of variables across all frequencies
    
    # Extract data for each frequency
    YQ = copy.deepcopy(YQ_list[0])  # Quarterly data
    YM = copy.deepcopy(YM_list[0])  # Monthly data
    YW = copy.deepcopy(YM_list[1])  # Weekly data (second entry in YM_list)
    
    # Get observation counts
    Tw = YW.shape[0] if hasattr(YW, 'shape') else 0  # Total weekly observations
    Tm = YM.shape[0] if hasattr(YM, 'shape') else 0  # Monthly observations
    Tq = YQ.shape[0] if hasattr(YQ, 'shape') else 0  # Quarterly observations
    
    # Print data dimensions for verification
    print(f"Data dimensions - Weekly: {(Tw, Nw) if Tw > 0 else 'None'}, "
            f"Monthly: {(Tm, Nm) if Tm > 0 else 'None'}, "
            f"Quarterly: {(Tq, Nq) if Tq > 0 else 'None'}")
    
    # Number of observations after burn-in (T0 = initial lag period)
    T0 = int(nlags)  # Initial observations used for lags
    nobs = min(YM.shape[0]-T0, YQ.shape[0]-T0)  # Effective sample size in weeks
    
    # STATE SPACE MODEL STRUCTURE
    # ---------------------------
    
    # Following MUFBVAR approach: state vector contains LF variables and their lags
    var_size = Ntotal * p  # All variables with lags for VAR
    Nstate = Nm + Nq
    state_size = Nstate * (p + 1)

    # Monthly variables with lags
    Nm_states = Nm * (p + 1)  # Monthly variables (current + p lags)

    # Quarterly variables with lags 
    Nq_states = Nq * (p + 1)  # Quarterly variables (current + p lags)

    # Initialize matrices for MCMC sampling
    Sigmap = np.zeros((math.ceil((self.nsim)/self.thining), Ntotal, Ntotal))
    Phip = np.zeros((math.ceil((self.nsim)/self.thining), Ntotal*p+1, Ntotal))
    Cons = np.zeros((math.ceil((self.nsim)/self.thining), Ntotal))
    
    # Initialize state vector and storage for filter/smoothed states
    a_filtered = np.zeros((nobs, state_size))
    P_filtered = np.zeros((nobs, state_size, state_size))
    P_filtered2 = np.zeros((nobs, 1, (Nstate*(p+1))**2))
    a_draws = np.zeros((self.nsim, nobs, state_size))
    
    # 1. STATE TRANSITION DEFINITION (GAMMA MATRICES)
    # ----------------------------------------------
    
    # Initialize VAR coefficients with priors 
    # (replace this with your actual initialization)
    Ntotal = Nw + Nm + Nq
    Phi = np.vstack((0.95 * np.eye(Ntotal), 
                    np.zeros((Ntotal*(p-1), Ntotal)), 
                    np.zeros((1, Ntotal))))  # Last row for constant

    # 1. STATE TRANSITION DEFINITION (GAMMA MATRICES)
    # ----------------------------------------------

    # Initialize GAMMAs as zeros with new dimensions
    GAMMAs = np.zeros((state_size, state_size))

    # Create identity blocks for each variable type 
    Im = np.eye(Nm) 
    Iq = np.eye(Nq) 

    # Set up lag structure (moving variables from lag j to lag j+1)
    for j in range(p):
        # Monthly block

        GAMMAs[(j+1)*Nm:(j+2)*Nm, j*Nm:(j+1)*Nm] = Im

        GAMMAs[(j+1)*Nq+Nm_states:(j+2)*Nq+Nm_states, j*Nq+Nm_states:(j+1)*Nq+Nm_states] = Iq
    
    # Monthly AR coefficients (diagonal of first Nm×Nm block)
    GAMMAs[:Nm, :Nm] = 0.95 * Im
    
    # Quarterly AR coefficients (diagonal of first Nq×Nq block of quarterly section)
    GAMMAs[Nm_states:Nm_states+Nq, Nm_states:Nm_states+Nq] = 0.95 * Iq
    
    # 1.2 GAMMAz - Impact of lagged observables (weekly variables)
    # Now weekly variables need to be treated as exogenous inputs
    
    GAMMAz = np.zeros((state_size, Nw*p))
    Z_t_size = Nw * p  # p lags of Nw weekly variables
    # Set impact of weekly variables on monthly latent states
    # This will be learned during estimation

    # 1.3 GAMMAc - Constant term vector
    GAMMAc = np.zeros((state_size, 1))
    # Constants will be updated during MCMC

    # GAMMAu dimensions: state_size × (Nm + Nq)
    GAMMAu = np.zeros((state_size, Nm + Nq))

    # Monthly innovations affect current monthly variables (first Nm elements)
    if Nm > 0:
        # Set identity block for monthly variables (top-left)
        GAMMAu[:Nm, :Nm] = np.eye(Nm)

    # Quarterly innovations affect current quarterly variables 
    if Nq > 0:
        # Set identity block for quarterly variables
        # Quarterly variables start at position Nm_states in state vector
        # Quarterly innovations start at position Nm in innovation vector
        GAMMAu[Nm_states:Nm_states+Nq, Nm:Nm+Nq] = np.eye(Nq)
        
    # 2. MEASUREMENT EQUATION DEFINITION
    # ---------------------------------

    # We now need three types of measurement equations:
    # 1. Weekly observation equations
    # 2. Monthly temporal aggregation constraints
    # 3. Quarterly temporal aggregation constraints

    # 2.1. Weekly observation equations
    # Weekly variables are now treated as noisy observations of monthly/quarterly data
    LAMBDAs_w = np.zeros((Nw, state_size))
    LAMBDAz_w = np.hstack((np.eye(Nw), np.zeros((Nw, Z_t_size - Nw))))  # No lagged impact in basic setup
    LAMBDAc_w = np.zeros((Nw, 1))


    # 2.2. Monthly temporal aggregation constraints
    LAMBDAs_m = np.hstack((np.tile(np.eye(Nm), rmw), np.zeros((Nm,state_size - (rmw*Nm)))))*1/4
    LAMBDAz_m = np.zeros((Nm, Z_t_size))
    LAMBDAc_m = np.zeros((Nm, 1))


    # 2.3. Quarterly temporal aggregation constraints (if monthly-to-quarterly aggregation needed)
    LAMBDAs_q = np.hstack((np.hstack(((np.zeros((Nq, (p+1) * Nm)), np.tile(np.eye(Nq), rqw)))),np.zeros((Nq,state_size - (rqw*Nq+(p+1)*Nm)))))*1/12
    LAMBDAz_q = np.zeros((Nq, Z_t_size))
    LAMBDAc_q = np.zeros((Nq, 1))

    
    # Initialize with proper structure
    
    LAMBDAu_w = np.eye(Nw)
    LAMBDAu_m = np.vstack(((np.eye(Nw),np.zeros((Nm, Nw)))))
    LAMBDAu_q = np.vstack(((np.eye(Nw), np.zeros((Nstate, Nw)))))
    # 3. COVARIANCE MATRICES
    # ---------------------

    # 3.1 Error covariance structure
    # For a coherent structure with weekly as indicators:

    # Innovation covariance for state variables (monthly and quarterly)
    # Initialize individual variance matrices
    # Full state innovation covariance
    sigma = np.eye((Ntotal)) * (1e-4)
    sigma_ww = sigma[:Nw, :Nw]
    sigma_mm = sigma[Nw:Nm+Nw, Nw:Nm+Nw]
    sigma_qq = sigma[Nw+Nm:, Nw+Nm:]
    sigma_wm = sigma[:Nw, Nw:Nm+Nw]
    sigma_mw = sigma[Nw:Nm+Nw, :Nw]
    sigma_wq = sigma[:Nw, Nw+Nm:]
    sigma_qw = sigma[Nw+Nm:, :Nw]
    sigma_mq =  sigma[Nw:Nm+Nw, Nw+Nm:]
    sigma_qm = sigma[Nw+Nm:, Nw:Nm+Nw]
    # 3.2 Measurement error covariance for weekly variables
    # Weekly variables now have their own measurement noise
    
    # 4. STATE INITIALIZATION
    # ---------------------

    # Initialize state vector and covariance
    a_t = np.zeros(state_size)
    P_t = np.zeros((state_size, state_size))

    
    # Initialize with reasonable values
    # Iterate to find reasonable starting covariance
    for kk in range(state_size):
        P_t = GAMMAs @ P_t @ GAMMAs.T + GAMMAu[:, :Nm] @ sigma_mm @ GAMMAu[:, :Nm].T + GAMMAu[:, Nm:Nm+Nq] @ sigma_qq @ GAMMAu[:, Nm:Nm+Nq].T
        P_t = 0.5 * (P_t + P_t.T)  # Ensure symmetry

    # 5. PREPARE DATA FOR KALMAN FILTER
    # -------------------------------

    # Store lagged high-frequency observations (weekly data) for exogenous inputs
    Z_t = np.zeros((nobs, Z_t_size))

    # Fill with lagged weekly observations
    if Z_t.size > 0 and Nw > 0:
        for j in range(p):
            if T0 > j:  # Ensure we have enough data for lags
                Z_t[:, j*Nw:(j+1)*Nw] = YW[T0-(j+1):T0+nobs-(j+1), :]
    
    
    
    # Prepare observed data vectors with clear frequency markers
    q_obs_periods = np.zeros(Tw, dtype=bool) if Tw > 0 else np.array([])
    m_obs_periods = np.zeros(Tw, dtype=bool) if Tw > 0 else np.array([])
    
    if Tw > 0:
        # Mark which weekly periods have quarterly/monthly observations
        # Quarterly data is available at the end of each quarter
        for t in range(rqw-1, Tw, rqw):
            q_idx = t // rqw
            if q_idx < Tq:
                q_obs_periods[t] = True
        
        # Monthly data is available at the end of each month
        for t in range(rmw-1, Tw, rmw):
            m_idx = t // rmw
            if m_idx < Tm:
                m_obs_periods[t] = True
    # 6. MAIN KALMAN FILTER AND MCMC LOOP
    # ---------------------------------
    print("SBFVAR: Estimation", end = '\n')
    print("Frequencies: ", "Q, M, W", end = "\n")
    print("Total Number of Draws: ",self.nsim)
    tries_at_j0 = 0
    should_restart = True
    while should_restart:
        should_restart = False
        if tries_at_j0 == 100:
            raise NameError('No Stable VAR at j=0')
        for j in tqdm(range(self.nsim)):
            
            restart_j0 = False
            # If not first iteration, use state from previous draw
            if j > 0:
                a_t = At_draw[0,:].copy()# Use .copy() to avoid modifying the stored value
                P_t = Pmean.copy()

            for t in range(nobs):
                # Current week index
                w_idx = T0 + t

                # Check if this is a period with low-frequency observations
                is_quarter_end = q_obs_periods[w_idx] if w_idx < len(q_obs_periods) else False
                is_month_end = m_obs_periods[w_idx] if w_idx < len(m_obs_periods) else False
                
                # PREDICTION STEP
                # ---------------
                a_prev = a_t
                P_prev = P_t
                
                # State prediction
                a_pred = GAMMAs @ a_prev + GAMMAz @ Z_t[t,:] + GAMMAc.flatten()
                P_pred = GAMMAs @ P_prev @ GAMMAs.T + GAMMAu[:, :Nm] @ sigma_mm @ GAMMAu[:, :Nm].T + GAMMAu[:, Nm:Nm+Nq] @ sigma_qq @ GAMMAu[:, Nm:Nm+Nq].T
                P_pred = 0.5 * (P_pred + P_pred.T)  # Ensure symmetry
                
                # MEASUREMENT STEP PREPARATION
                # ---------------------------
                
                # Initialize measurement components
                LAMBDAs_list = []
                LAMBDAc_list = []
                LAMBDAz_list = []
                LAMBDAu_list = []
                y_obs = []
                # Weekly observations (if available)
                if Nw > 0 and w_idx < Tw:
                    LAMBDAs_list.append(LAMBDAs_w)
                    LAMBDAc_list.append(LAMBDAc_w)
                    LAMBDAz_list.append(LAMBDAz_w)
                    #LAMBDAu_list.append(LAMBDAu_w)
                    y_obs.append(YW[w_idx])
                    LAMBDAu = LAMBDAu_w
                    sigma_t = sigma_ww            
                    
                
                # Monthly observations (if at month end)
                if is_month_end and Nm > 0:
                    if m_idx < Tm:
                        # Regular monthly measurement
                        y_obs.append(YM[w_idx])
                        LAMBDAs_list.append(LAMBDAs_m)
                        LAMBDAc_list.append(np.zeros((Nm, 1)))
                        LAMBDAz_list.append(LAMBDAz_m)
                        #LAMBDAu_list.append(LAMBDAu_m)
                        LAMBDAu = LAMBDAu_m
                        sigma_t = np.block([
                            [sigma_ww, sigma_wm],
                            [sigma_mw, sigma_mm]
                        ])


                # Quarterly observations (if at quarter end)
                if is_quarter_end and Nq > 0:
                    if q_idx < Tq:
                        # Regular quarterly measurement
                        y_obs.append(YQ[w_idx])
                        LAMBDAs_list.append(LAMBDAs_q)
                        LAMBDAc_list.append(np.zeros((Nq, 1)))
                        LAMBDAz_list.append(LAMBDAz_q)
                        #LAMBDAu_list.append(LAMBDAu_q )
                        LAMBDAu = LAMBDAu_q
                        sigma_t = np.block([
                            [sigma_ww, sigma_wm, sigma_wq],
                            [sigma_mw, sigma_mm, sigma_mq],
                            [sigma_qw, sigma_qm, sigma_qq]
                        ])          
                
                # Combine measurement components
                if LAMBDAs_list:
                    LAMBDAs = np.vstack(LAMBDAs_list)
                    LAMBDAc = np.vstack(LAMBDAc_list)
                    LAMBDAz = np.vstack(LAMBDAz_list)
                    #LAMBDAu = np.vstack(LAMBDAu_list)
                    y = np.concatenate(y_obs)
                    # MEASUREMENT UPDATE
                    # -----------------
                    
                    # Measurement prediction
                    y_hat = LAMBDAs @ a_pred + LAMBDAz @ Z_t[t,:] + LAMBDAc.flatten()
                    nu = y - y_hat
                    
                    # First term: prediction covariance 
                    S = (LAMBDAs @ P_pred @ LAMBDAs.T +                            # Prediction covariance
                        LAMBDAu @ sigma_ww @ LAMBDAu.T +                          # Weekly measurement error
                        
                        # Monthly cross-terms (both directions)
                        LAMBDAs @ GAMMAu[:, :Nm] @ sigma_mw @ LAMBDAu.T +         # Monthly to measurement
                        LAMBDAu @ sigma_wm @ GAMMAu[:, :Nm].T @ LAMBDAs.T +       # Measurement to monthly
                        
                        # Quarterly cross-terms (both directions)
                        LAMBDAs @ GAMMAu[:, Nm:] @ sigma_qw @ LAMBDAu.T +         # Quarterly to measurement
                        LAMBDAu @ sigma_wq @ GAMMAu[:, Nm:].T @ LAMBDAs.T# +       # Measurement to quarterly
                    )
                    #TODO There are no monthly quarterly cross terms, OK? or Xi and S
                    # Ensure symmetry
                    S = 0.5 * (S + S.T)
                    Xi = LAMBDAs @ P_pred + LAMBDAu @ sigma_wm @ GAMMAu[:, :Nm].T + LAMBDAu @ sigma_wq @ GAMMAu[:, Nm:].T
                    
                    # Calculate Kalman gain
                    K = Xi.T @ invert_matrix(S)

                    # Update state estimate
                    a_t = a_pred + K @ nu

                    # Update state covariance
                    P_t = P_pred - K @ Xi

                    # Ensure symmetry of P_t
                    P_t = 0.5 * (P_t + P_t.T)
                else:
                    # No observations - just prediction
                    a_t = a_pred
                    P_t = P_pred
                
                # Store filtered state and covariance
                a_filtered[t] = a_t.T
                P_filtered[t] = P_t
                P_filtered2[t] = P_t.reshape((1, (Nstate*(p+1))**2), order = "F")
            
            Ptilde = P_filtered[nobs-1].reshape(Nstate*(p+1), Nstate * (p+1), order = "F")
            
            
            
            ########## Unbalanced Dataset with Three Frequencies ###########
                        
            # Define dimensions for companion form
            kn = Ntotal * (p + 1)  # All variables with lags
            Tstar = YW.shape[0] - T0
            # Define forecast horizons consistent with MUFBVAR approach
            Tnew = Tstar - nobs  # Number of periods to forecast
            Tnobs = nobs + Tnew  # Total periods (observed + forecast)
                        
            # Measurement Equation
                        
            # Weekly measurement (direct observation)
            Z0 = np.zeros((Nw, kn))
            Z0[:, :Nw] = np.eye(Nw)  # Current weekly values observed directly

            # Monthly measurement (temporal aggregation from weekly)
            Z1 = np.zeros((Nm, kn))
            # Monthly is aggregate of 4 weeks
            for bb in range(Nm):
                for ll in range(rmw):  # Weekly to monthly ratio (typically 4)
                    if self.temp_agg == "mean":
                        Z1[bb, ll*Ntotal + bb] = 1/rmw
                    if self.temp_agg == "sum":
                        Z1[bb, ll*Ntotal + bb] = 1
                        
            # Quarterly measurement (temporal aggregation from weekly)
            Z2 = np.zeros((Nq, kn))
            # Quarterly is aggregate of 12 weeks
            for bb in range(Nq):
                for ll in range(rqw):  # Weekly to quarterly ratio (typically 12)
                    if self.temp_agg == "mean":
                        Z2[bb, ll*Ntotal + Nw+Nm+bb] = 1/rqw
                    if self.temp_agg == "sum":
                        Z2[bb, ll*Ntotal + Nw+Nm+bb] = 1
                        
            # Combine all measurement equations
            ZZ = np.vstack((Z0, Z1, Z2))
                        
            # Construct full state vector from filtered state and observations

            # We have weekly data for direct variables
            BAt = np.concatenate((
                YW[T0+nobs-1, :],                     # Current weekly obs
                a_filtered[nobs-1, :Nm],              # Current monthly state
                a_filtered[nobs-1, Nm_states:Nm_states+Nq]  # Current quarterly state
            ))
            
            # Add lagged values
            for rr in range(1, p+1):
                if T0+nobs-1-rr >= 0:  # Check if we have enough weekly data
                    BAt = np.concatenate((BAt, np.concatenate((
                        YW[T0+nobs-1-rr, :],  # Lagged weekly obs
                        a_filtered[nobs-1, rr*Nm:(rr+1)*Nm],  # Lagged monthly state
                        a_filtered[nobs-1, Nm_states+rr*Nq:Nm_states+(rr+1)*Nq]  # Lagged quarterly state
                    ))))
                else:
                    BAt = np.concatenate((BAt, np.concatenate((
                        np.zeros(Nw),  # Padding for missing weekly
                        a_filtered[nobs-1, rr*Nm:(rr+1)*Nm],  # Lagged monthly state
                        a_filtered[nobs-1, Nm_states+rr*Nq:Nm_states+(rr+1)*Nq]  # Lagged quarterly state
                    ))))
        
            # Initialize covariance matrix BPt
            BPt = np.zeros((kn, kn))

            # Weekly variables use small initial values
            for rr in range(p+1):
                for vv in range(p+1):
                        BPt[(rr+1)*Nw+rr*Nstate:(rr+1)*(Nw+Nstate), (vv+1)*Nw+vv*Nstate:(vv+1)*(Nw+Nstate)] = np.squeeze(
                            Ptilde[rr*Nstate:(rr+1)*Nstate,vv*Nstate:(vv+1)*Nstate])
            
            # Initialize storage for state and covariance
            BAt_mat = np.zeros((Tnobs, kn))
            BPt_mat = np.zeros((Tnobs, kn**2))
                        
            # Store initial state and covariance
            BAt_mat[nobs-1, :] = BAt
            BPt_mat[nobs-1, :] = BPt.reshape((1, kn**2), order="F")
                        
            # Define companion form matrix PHIF
            PHIF = np.zeros((kn, kn))
            IF = np.eye(Ntotal)
            for i in range(p):
                PHIF[(i+1)*Ntotal:(i+2)*Ntotal, i*Ntotal:(i+1)*Ntotal] = IF
                            
            # Set VAR coefficients
            PHIF[:Ntotal, :Ntotal*p] = Phi[:-1, :].T
                        
            # Define constant term CONF
            CONF = np.hstack((Phi[-1, :].T, np.zeros(Ntotal*p)))
                        
            # Define covariance term SIGF
            SIGF = np.zeros((kn, kn))
            SIGF[:Ntotal, :Ntotal] = sigma
                        
            # Filter Loop
            # --------------------
            for t in range(nobs, Tnobs-T0):
                # Index relative to forecast start
                kkk = t - nobs
                
                # Define new data (ND) and new Z matrix (NZ)
                ND_indices = ~index_NY[:, kkk]
                ND = YDATA_forecast[kkk, ND_indices] if np.any(ND_indices) else np.array([])
                NZ = ZZ[ND_indices, :] if np.any(ND_indices) else np.empty((0, kn))
                
                # Previous state and covariance
                BAt1 = BAt_mat[t-1, :]
                BPt1 = BPt_mat[t-1, :].reshape((kn, kn), order="F")
                
                # Prediction step
                Balphahat = PHIF @ BAt1 + CONF
                BPhat = PHIF @ BPt1 @ PHIF.T + SIGF
                BPhat = 0.5 * (BPhat + BPhat.T)  # Ensure symmetry
                
                # Update step (if we have observations)
                if NZ.shape[0] > 0:
                    Byhat = NZ @ Balphahat
                    Bnut = ND - Byhat
                    
                    BFt = NZ @ BPhat @ NZ.T
                    BFt = 0.5 * (BFt + BFt.T)  # Ensure symmetry
                    
                    # Kalman gain and update
                    sol_1 = (BPhat @ NZ.T) @ invert_matrix(BFt)
                    BAt = Balphahat + sol_1 @ Bnut
                    BPt = BPhat - sol_1 @ (BPhat @ NZ.T).T
                else:
                    # No observations, just prediction
                    BAt = Balphahat
                    BPt = BPhat
                
                # Store filtered state and covariance
                BAt_mat[t, :] = BAt
                BPt_mat[t, :] = BPt.reshape((1, kn**2), order="F")
                        
            # Draw from the posterior at final time point
            AT_draw = np.zeros((Tnew + 1, kn))
                        
            # Draw from multivariate normal
            Pchol = cholcovOrEigendecomp(BPt_mat[Tnobs-1, :].reshape((kn, kn), order="F"))
            AT_draw[-1, :] = BAt_mat[Tnobs-1, :] + np.transpose(Pchol @ np.random.standard_normal(kn))
                        
            # Kalman Smoother
            # -------------------
            for i in range(Tnew):
                # Get filtered state and covariance at time t
                BAtt = BAt_mat[Tnobs-(i+2), :]
                BPtt = BPt_mat[Tnobs-(i+2), :].reshape((kn, kn), order="F")
                
                # Prediction from t to t+1
                BPhat = PHIF @ BPtt @ PHIF.T + SIGF
                BPhat = 0.5 * (BPhat + BPhat.T)
                
                # Inverse of prediction covariance
                inv_BPhat = invert_matrix(BPhat)
                
                # Innovation (difference between sampled state and prediction)
                Bnut = AT_draw[-(i+1), :] - PHIF @ BAtt - CONF
                
                # Smoothed mean and covariance
                Amean = BAtt + (BPtt @ PHIF.T) @ inv_BPhat @ Bnut
                Pmean = BPtt - (BPtt @ PHIF.T) @ inv_BPhat @ np.transpose(BPtt @ PHIF.T)
                
                # Draw from multivariate normal
                Pmchol = cholcovOrEigendecomp(Pmean)
                AT_draw[-2-i, :] = np.transpose(Amean + Pmchol @ np.random.standard_normal(kn))        
                
            
            ########## Balanced Dataset Smoothing ###########
            # Kalman Smoother
            #####################
            
            At_draw = np.zeros((nobs, Nstate * (p+1)))
            for kk in range(p+1):
                At_draw[nobs-1, kk * Nstate:(kk+1)*+Nstate] = AT_draw[0,(kk+1)*Nw + kk*Nstate:(kk+1)*(Nw+Nstate)]
                
            
            for i in range(nobs-1):
                Att = a_filtered[nobs-(i+2),:]#[:, np.newaxis]
                Ptt = P_filtered2[nobs-(i+2),:].reshape(Nstate*(p+1), Nstate*(p+1), order = "F")
                
                
                Phat = GAMMAs @ Ptt @ GAMMAs.T  + GAMMAu[:, :Nm] @ sigma_mm @ GAMMAu[:, :Nm].T + GAMMAu[:, Nm:Nm+Nq] @ sigma_qq @ GAMMAu[:, Nm:Nm+Nq].T
                
                Phat = 0.5*(Phat + Phat.T)
                
                inv_Phat = invert_matrix(Phat)
                
                nut = At_draw[nobs-(i+1), :] - GAMMAs @ Att - GAMMAz @ Z_t[nobs-1-(i+1)] - GAMMAc[:,0]

                
                temp = Ptt @ GAMMAs.T
                Amean = Att + temp @ inv_Phat @ nut
                Pmean = Ptt - temp @ inv_Phat @ np.transpose(temp)
                
                Pmchol = cholcovOrEigendecomp(Pmean)
                At_draw[nobs-1-(i+1), :] = np.transpose(Amean + Pmchol @ np.random.standard_normal(Nstate*(p+1)))
                    
            # Minesota Prior                            
            ########################
            YY = np.vstack((np.hstack((YW[T0:nobs+T0], At_draw[:,:Nm], At_draw[:,Nm_states:Nm_states+Nq])), AT_draw[1:,:(Nw+Nstate)]))

            
            #save latent states
            if (j%self.thining == 0):
                j_temp = int(j/self.thining)
                if j_temp > 0 :
                    lstate_list[j_temp, :] = YY[:, Nw:]
                else:
                    lstate_list = np.zeros((math.ceil((self.nsim)/self.thining), YY.shape[0], Nm + Nq ))
                    lstate_list[j_temp, :] = YY[:, Nw:]

            
            # 9. CALCULATE DUMMY OBSERVATIONS FOR MINNESOTA PRIOR
            # ----------------------------------------------
            
            nobs_ = YY.shape[0] - p  # Adjusted for lags
            spec = np.hstack((p, T0, self.nex, Ntotal, nobs_))
            
            # Calculate dummy observations for the VAR
            YYact, YYdum, XXact, XXdum = calc_yyact(self.hyp, YY, spec)
        
            # Store simulation results
            if (j % self.thining == 0):
                j_temp = int(j/self.thining)
                if j_temp == 0:
                    # Initialize storage
                    YYactsim_list = np.zeros((math.ceil((self.nsim)/self.thining), rqw+1, Ntotal))
                    XXactsim_list = np.zeros((math.ceil((self.nsim)/self.thining), rqw+1, Ntotal*p+1))
                
                # Store the combined data
                YYactsim_list[j_temp, :, :] = YYact[-(rqw+1):, :]
                XXactsim_list[j_temp, :, :] = XXact[-(rqw+1):, :]

            # 10. VAR POSTERIOR SAMPLING
            # ----------------------
            

            # Standard VAR posterior calculations
            Tdummy, n = YYdum.shape
            n = int(n)
            Tdummy = int(Tdummy)
            Tobs, n = YYact.shape
            X = np.vstack((XXact, XXdum))
            Y = np.vstack((YYact, YYdum))
            T = Tobs + Tdummy
            F = np.zeros((int(n*p), int(n*p)))
            I = np.eye(n)
            for i in range(p-1):
                F[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = I        
            # Compute posterior parameters
            vl, d, vr = np.linalg.svd(X, full_matrices=False)
            vr = vr.T
            di = 1/d
            B = vl.T @ Y
            xxi = (vr * np.tile(di.T, (Ntotal*p+1, 1)))
            inv_x = xxi @ xxi.T
            Phi_tilde = xxi @ B
            
            Sigma = (Y - X @ Phi_tilde).T @ (Y - X @ Phi_tilde)
            sigma = invwishart.rvs(scale = Sigma, df = T-n*p-1)
                
                # Draw VAR coefficients and check stability
            if check_explosive:
                attempts = 0
                while attempts < max_it_explosive:
                    sigma_chol = cholcovOrEigendecomp(np.kron(sigma, inv_x))
                    phi_new = np.squeeze(Phi_tilde.reshape(n*(n*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                    Phi = phi_new.reshape(n*p+1, n, order="F")
                    if not is_explosive(Phi, n, p):
                        break
                    attempts += 1
                if attempts == max_it_explosive:
                    explosive_counter += 1
                    print(f"Explosive VAR detected {explosive_counter} times.")
                    if j == 0:
                        restart_j0 = True
                        break
                    else:
                        continue   
            else:
                sigma_chol = cholcovOrEigendecomp(np.kron(sigma, inv_x))
                phi_new = np.squeeze(Phi_tilde.reshape(n*(n*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                Phi = phi_new.reshape(n*p+1, n, order="F") 
                
                
            # Store posterior draws
            if (j % self.thining == 0):
                j_temp = int(j/self.thining)
                Sigmap[j_temp, :, :] = sigma
                Phip[j_temp, :, :] = Phi
                Cons[j_temp, :] = Phi[-1, :]
                valid_draws.append(j_temp)
            
            # Update state space matrices with new VAR parameters
            # ----------------------------------------------
            #Weekly equation coefficients (split by variable type)
            phi_ww = np.zeros((Nw*p, Nw))
            phi_wm = np.zeros((Nm*p, Nw))
            phi_wq = np.zeros((Nq*p, Nw))
            for i in range(p):
                # Weekly variables affecting weekly variables
                phi_ww[Nw*i:Nw*(i+1), :] = Phi[i*Ntotal:i*Ntotal+Nw, :Nw]

                # Monthly variables affecting weekly variables
                phi_wm[Nm*i:Nm*(i+1), :] = Phi[i*Ntotal+Nw:i*Ntotal+Nw+Nm, :Nw]

                # Quarterly variables affecting weekly variables
                phi_wq[Nq*i:Nq*(i+1), :] = Phi[i*Ntotal+Nw+Nm:i*Ntotal+Ntotal, :Nw]

            # Weekly constant term
            phi_wc = Phi[-1, :Nw, np.newaxis]

            # Monthly equation coefficients (split by variable type)
            phi_mw = np.zeros((Nw*p, Nm))
            phi_mm = np.zeros((Nm*p, Nm))
            phi_mq = np.zeros((Nq*p, Nm))
            for i in range(p):
                # Weekly variables affecting monthly variables
                phi_mw[Nw*i:Nw*(i+1), :] = Phi[i*Ntotal:i*Ntotal+Nw, Nw:Nw+Nm]
                
                # Monthly variables affecting monthly variables
                phi_mm[Nm*i:Nm*(i+1), :] = Phi[i*Ntotal+Nw:i*Ntotal+Nw+Nm, Nw:Nw+Nm]
                
                # Quarterly variables affecting monthly variables
                phi_mq[Nq*i:Nq*(i+1), :] = Phi[i*Ntotal+Nw+Nm:i*Ntotal+Ntotal, Nw:Nw+Nm]

            # Monthly constant term
            phi_mc = Phi[-1, Nw:Nw+Nm, np.newaxis]

            # Quarterly equation coefficients (split by variable type)
            phi_qw = np.zeros((Nw*p, Nq))
            phi_qm = np.zeros((Nm*p, Nq))
            phi_qq = np.zeros((Nq*p, Nq))
            for i in range(p):
                # Weekly variables affecting quarterly variables
                phi_qw[Nw*i:Nw*(i+1), :] = Phi[i*Ntotal:i*Ntotal+Nw, Nw+Nm:Ntotal]
                
                # Monthly variables affecting quarterly variables
                phi_qm[Nm*i:Nm*(i+1), :] = Phi[i*Ntotal+Nw:i*Ntotal+Nw+Nm, Nw+Nm:Ntotal]
                
                # Quarterly variables affecting quarterly variables
                phi_qq[Nq*i:Nq*(i+1), :] = Phi[i*Ntotal+Nw+Nm:i*Ntotal+Ntotal, Nw+Nm:Ntotal]

            # Quarterly constant term
            phi_qc = Phi[-1, Nw+Nm:Ntotal, np.newaxis]
            
            # Extract covariance matrix blocks
            # ------------------------------
            # Weekly variances
            sigma_ww = sigma[:Nw, :Nw]

            # Monthly variances
            sigma_mm = sigma[Nw:Nw+Nm, Nw:Nw+Nm]

            # Quarterly variances  
            sigma_qq = sigma[Nw+Nm:, Nw+Nm:]

            
            # Cross-covariances with symmetry enforcement
            # Weekly-Monthly cross-covariance (enforce symmetry)
            sigma_wm = 0.5 * (sigma[:Nw, Nw:Nw+Nm] + sigma[Nw:Nw+Nm, :Nw].T)
            sigma_mw = sigma_wm.T  # Transpose for symmetry
            
            # Weekly-Quarterly cross-covariance (enforce symmetry)
            sigma_wq = 0.5 * (sigma[:Nw, Nw+Nm:] + sigma[Nw+Nm:, :Nw].T)
            sigma_qw = sigma_wq.T  # Transpose for symmetry
            
            # Monthly-Quarterly cross-covariance (enforce symmetry)
            sigma_mq = 0.5 * (sigma[Nw:Nw+Nm, Nw+Nm:] + sigma[Nw+Nm:, Nw:Nw+Nm].T)
            sigma_qm = sigma_mq.T  # Transpose for symmetry
            
            # Update transition matrices
            # -------------------------
            
            # Monthly state transition


            
            GAMMAz_m = np.vstack((
                np.transpose(phi_mw), 
                np.zeros((p*Nm, p*Nw))
            ))
            GAMMAc_m = np.vstack((
                phi_mc, 
                np.zeros((p*Nm, 1))
            ))

            GAMMAz_q = np.vstack((
                np.transpose(phi_qw), 
                np.zeros((p*Nq, p*Nw))
            ))
            GAMMAc_q = np.vstack((
                phi_qc, 
                np.zeros((p*Nq, 1))
            ))
            
            GAMMAz = np.vstack((GAMMAz_m, GAMMAz_q))
            GAMMAc = np.vstack((GAMMAc_m, GAMMAc_q))

            # Combined state transition matrix for all variables
            GAMMAs_m = np.vstack((
                np.hstack((np.transpose(phi_mm), np.zeros((Nm, Nm)))), 
                np.hstack((np.eye(p*Nm), np.zeros((p*Nm, Nm))))
            ))
            # Quarterly state transition
            GAMMAs_q = np.vstack((
                np.hstack((np.transpose(phi_qq), np.zeros((Nq, Nq)))), 
                np.hstack((np.eye(p*Nq), np.zeros((p*Nq, Nq))))
            ))
            GAMMAs = np.zeros((state_size, state_size))
        
            # Monthly block
            GAMMAs[:Nm_states, :Nm_states] = GAMMAs_m

            # Quarterly block
            GAMMAs[Nm_states:, Nm_states:] = GAMMAs_q

            # Cross-influence between monthly and quarterly states
            # Monthly variables affecting quarterly variables
            GAMMAs[Nm_states:Nm_states+Nq, :Nm] = np.transpose(phi_qm[:Nm, :])

            # Quarterly variables affecting monthly variables
            GAMMAs[:Nm, Nm_states:Nm_states+Nq] = np.transpose(phi_mq[:Nq, :])
            
            # 5. Update measurement equation matrices if needed
            # For the unified approach, LAMBDAs matrices might need updating if they depend on VAR coefficients
            # If your temporal aggregation constraints are static, you don't need to update them
            LAMBDAs_w = np.hstack((np.zeros((Nw,Nm)), np.transpose(phi_wm), np.zeros((Nw,Nq)) ,np.transpose(phi_wq)))
            LAMBDAz_w = np.transpose(phi_ww)
            LAMBDAc_w = phi_wc
        
        if restart_j0:
            j = -1  # Will be incremented to 0 at the next iteration of the j loop
            should_restart = True
            continue   


    
    # 12. STORE RESULTS
    # --------------
    
    self.Phip = Phip
    self.Sigmap = Sigmap
    self.nv = Ntotal
    self.Nw = Nw
    self.Nm = Nm
    self.Nq = Nq
    self.freq_ratio = freq_ratio_list
    self.varlist = varlist_list
    self.YYactsim_list = YYactsim_list
    self.XXactsim_list = XXactsim_list
    self.explosive_counter = explosive_counter
    self.valid_draws = [draw for draw in valid_draws if draw >= self.nburn/self.thining]
    self.lstate_list = lstate_list
    # Store state-space model matrices
    self.a_draws = a_draws
    self.GAMMAs = GAMMAs
    self.GAMMAc = GAMMAc
    self.GAMMAu = GAMMAu
    self.LAMBDAs_w = LAMBDAs_w
    self.LAMBDAs_m = LAMBDAs_m
    self.LAMBDAs_q = LAMBDAs_q

    # Store original data
    self.input_data_W = YW
    self.input_data_M = YM
    self.input_data_Q = YQ
    self.input_index_M_W = mufbvar_data.input_data
    self.input_index_Q = mufbvar_data.input_data_Q
    # Store data sizes
    self.w_periods = Tw
    self.m_periods = Tm
    self.q_periods = Tq
    
    # Store index
    self.index_list = index_list
    self.temp_agg = temp_agg
    self.rqw = rqw
    self.rmw = rmw
    
    return None

def forecast(self, H, conditionals = None):
    
    '''
    Method to generate the forecasts in the highest frequency.\n
    
    Parameters
    ----------
    H : int
        Forecast horizon in highest frequnecy
    conditionals : pandas DataFrame or None
        Conditional forecasts\n
        column names must be the variable names\n
        no index needed\n
        either values or np.nan
    
    '''
    self.H = H
    # First we need to extend the index
    # depending on the highest frequencies the approach differs
    
    index = copy.deepcopy(self.index_list[-1])
    
    if self.frequencies[-1] == "D":
        
        index = index.append(pd.date_range(start=index[-1] + Day(), periods = H, freq='D'))

        # Function to check if a month has more than 20 days
        def has_more_than_20_days(month, dti):
            return sum(dti.to_period('M') == month) > 20

        # Function to remove the last day of a month
        def remove_last_day_of_month(month, dti):
            return dti[~((dti.to_period('M') == month) and (dti.day > 20))]

        # Check each month in extended_dti
        for month in index.to_period('M').unique():
            # If a month has more than 20 days
            while has_more_than_20_days(month, index):
                # Remove the last day of that month
                index = remove_last_day_of_month(month, index)
                # Add an additional day at the end
                index = index.append(pd.DatetimeIndex([index[-1] + Day()]))
    
    if self.frequencies[-1] == 'W':
        index = index.append(pd.date_range(start=index[-1] + Week(), periods=H, freq='W'))

        # Function to check if a month has more than 4 weeks
        def has_more_than_4_weeks(month, dti):
            return sum(dti.to_period('M') == month) > 4

        # Function to remove the last week of a month
        def remove_last_week_of_month(month, dti):
            return dti[~((dti.to_period('M') == month) & (dti.day > 28))]

        # Check each month in extended_dti
        for month in index.to_period('M').unique():
            # If a month has more than 4 weeks
            while has_more_than_4_weeks(month, index):
                # Remove the last week of that month
                index = remove_last_week_of_month(month, index)
                # Add an additional week at the end
                index = index.append(pd.DatetimeIndex([index[-1] + Week()]))
                
    if self.frequencies[-1] == 'M':
        index = index.append(pd.date_range(start=index[-1] + MonthBegin(), periods=H, freq='MS'))
    
    if self.frequencies[-1] == 'Q':
        index = index.append(pd.date_range(start=index[-1] + QuarterBegin(), periods=H, freq='QS'))
        
    
    # Now we need to look at the conditional forecasts:
    
    YYcond = pd.DataFrame(np.nan, index =  index[-H:], columns= self.varlist)
    
    if conditionals is not None:
        conditionals.index = YYcond.index[:len(conditionals.index)]
        
        YYcond.update(conditionals)
        YYcond = np.array(YYcond) 
        YYcond[:,(Ntotal == 1)] = YYcond[:,(Ntotal == 1)]/100
        YYcond[:,(Ntotal == 0)] = np.log(YYcond[:,(Ntotal == 0)])
    exc = np.array(~np.isnan(YYcond))
    YYcond = np.array(YYcond)
        
    
    
    H_= int(self.H)
            
    #Prepare index for output
    index = index[index.shape[0]-(self.lstate_list[-1].shape[0]+H_):]

    ###############  
    # Forecasting #
    ###############
    Ntotal = self.Nm+self.Nq+self.Nw
    # store forecasts in monthly frequency
    YYvector_ml  = np.zeros((len(self.valid_draws),H_,Ntotal))     # collects now/forecast      
    
    # store forecasts in quarterly frequency
    YYvector_ql  = np.zeros((len(self.valid_draws),Ntotal))   
    YYvector_qg  = np.zeros((len(self.valid_draws),int(self.H/12),Ntotal))
    
    print(" ", end = '\n')
    print("Multiple Frequency SBFVAR: Forecasting", end = "\n")
    print("Forecast Horizon: ", H_, end = "\n")
    print("Total Draws: ", len(self.valid_draws))
    
    
    for idx, jj in enumerate(tqdm(self.valid_draws)):
        
        YYact = np.squeeze(self.YYactsim_list[jj, -1, :])
        XXact = np.squeeze(self.XXactsim_list[jj, -1, :])
        post_phi = np.squeeze(self.Phip[jj,:,:])
        post_sig = np.squeeze(self.Sigmap[jj,:,:])
        

        # Bayesian Estimation Forecasting 
        ###################################

        
        YYpred = np.zeros((H_+1, Ntotal)) # forecasts from VAR
        YYpred[0,:] = YYact
        XXpred = np.zeros((H_+1, Ntotal*self.nlags+1))
        XXpred[:,-1] = np.full((H_+1), fill_value = 1)
        XXpred[0,:] = XXact
        
        # given posterior draw, draw number (H+1) random sequence

        error_pred = np.zeros((H_+1, Ntotal))
        
        for h in range(H_+1):
            if post_sig.size > 1:
                error_pred[h,:] = np.random.default_rng().multivariate_normal(mean = np.zeros(Ntotal), cov = post_sig, method = "cholesky")
            else:
                error_pred[h,:] = np.random.default_rng().normal(loc = 0, scale = post_sig)
        # given posterior draw, iterate forward to construct forecasts
        
        for h in range(1,H_+1):
            
            XXpred[h,Ntotal:-1] = XXpred[h-1, :-Ntotal-1]
            XXpred[h, :Ntotal] = YYpred[h-1, :]
            #YYpred[h,:] = (XXpred[h,:] @ post_phi + error_pred[h,:])
            YYpred[h,:] = (1-exc[h-1,:]) * (XXpred[h,:] @ post_phi + error_pred[h,:]) + exc[h-1,:] * np.nan_to_num(YYcond[h-1,:])
        
        YYpred1 = copy.deepcopy(YYpred)
        YYpred = YYpred[1:,:]
        

        
        
        # Now-/Forecasts
        # Store in hf
        YYvector_ml[idx,:,:] = YYpred
        
    
    
    YW = copy.deepcopy(self.input_data_W)[self.rqw:-(self.rqw)]
    YW[:, (self.select_w == 1)] = 100 * YW[:, (self.select_w == 1)]
    YW[:, (self.select_w == 0)] = np.exp(YW[:, (self.select_w == 0)])
    #mean
    YYftr_m = np.nanmean(YYvector_ml, axis = 0)
    YYftr_m[:, (self.select == 1)] = 100 * YYftr_m[:, (self.select == 1)]
    YYftr_m[:, (self.select == 0)] = np.exp(YYftr_m[:, (self.select == 0)])
    
    YYnow_m = np.mean(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw], axis = 0) # actual/nowcast weeklies
    if YYnow_m.size:
        YYnow_m[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
        YYnow_m[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
    
    lstate_m = np.mean(self.lstate_list[self.valid_draws,:,:], axis = 0)# hf obs for lf vars
    lstate_m[:, (self.select_m_q == 1)] = 100 * lstate_m[:, (self.select_m_q == 1)]
    lstate_m[:, (self.select_m_q == 0)] = np.exp(lstate_m[:, (self.select_m_q == 0)])
    
    YY_m = np.vstack((np.vstack((np.hstack((YW, lstate_m[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_m[-self.rqw:,:])))), YYftr_m))

    #median
    YYftr_med = np.nanmedian(YYvector_ml, axis = 0)
    YYftr_med[:, (self.select == 1)] = 100 * YYftr_m[:, (self.select == 1)]
    YYftr_med[:, (self.select == 0)] = np.exp(YYftr_m[:, (self.select == 0)])
    
    YYnow_med = np.nanmedian(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw], axis = 0) # actual/nowcast weeklies
    if YYnow_med.size:
        YYnow_med[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
        YYnow_med[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
    
    lstate_med = np.nanmedian(self.lstate_list[self.valid_draws,:,:], axis = 0)# hf obs for lf vars
    lstate_med[:, (self.select_m_q == 1)] = 100 * lstate_med[:, (self.select_m_q == 1)]
    lstate_med[:, (self.select_m_q == 0)] = np.exp(lstate_med[:, (self.select_m_q == 0)])
    
    YY_med = np.vstack((np.vstack((np.hstack((YW, lstate_m[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_med[-self.rqw:,:])))), YYftr_m))

    # 95%
    YYftr_095 = np.nanquantile(YYvector_ml,  q = 0.95, axis = 0)
    YYftr_095[:, (self.select == 1)] = 100 * YYftr_095[:, (self.select == 1)]
    YYftr_095[:, (self.select == 0)] = np.exp(YYftr_095[:, (self.select == 0)])
    
    YYnow_095 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.95, axis = 0) # actual/nowcast weeklies
    if YYnow_095.size:
        YYnow_095[:, (self.select_w== 1)] = 100 * YYnow_095[:, (self.select_w == 1)]
        YYnow_095[:, (self.select_w == 0)] = np.exp(YYnow_095[:,(self.select_w == 0)])
    
    lstate_095 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.95, axis = 0)# hf obs for lf vars
    lstate_095[:, (self.select_m_q == 1)] = 100 * lstate_095[:, (self.select_m_q == 1)]
    lstate_095[:, (self.select_m_q == 0)] = np.exp(lstate_095[:, (self.select_m_q == 0)])
    
    YY_095 = np.vstack((np.vstack((np.hstack((YW, lstate_095[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_095[-self.rqw:,:])))), YYftr_m))

    # 84%
    YYftr_084 = np.nanquantile(YYvector_ml,  q = 0.84, axis = 0)
    YYftr_084[:, (self.select == 1)] = 100 * YYftr_084[:, (self.select == 1)]
    YYftr_084[:, (self.select == 0)] = np.exp(YYftr_084[:, (self.select == 0)])
    
    YYnow_084 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.84, axis = 0) # actual/nowcast weeklies
    if YYnow_084.size:
        YYnow_084[:, (self.select_w== 1)] = 100 * YYnow_084[:, (self.select_w == 1)]
        YYnow_084[:, (self.select_w == 0)] = np.exp(YYnow_084[:,(self.select_w == 0)])
    
    lstate_084 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.84, axis = 0)# hf obs for lf vars
    lstate_084[:, (self.select_m_q == 1)] = 100 * lstate_084[:, (self.select_m_q == 1)]
    lstate_084[:, (self.select_m_q == 0)] = np.exp(lstate_084[:, (self.select_m_q == 0)])
    
    YY_084 = np.vstack((np.vstack((np.hstack((YW, lstate_084[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_084[-self.rqw:,:])))), YYftr_m))


    # 16%
    YYftr_016 = np.nanquantile(YYvector_ml,  q = 0.16, axis = 0)
    YYftr_016[:, (self.select == 1)] = 100 * YYftr_016[:, (self.select == 1)]
    YYftr_016[:, (self.select == 0)] = np.exp(YYftr_016[:, (self.select == 0)])
    
    YYnow_016 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.16, axis = 0) # actual/nowcast weeklies
    if YYnow_016.size:
        YYnow_016[:, (self.select_w== 1)] = 100 * YYnow_016[:, (self.select_w == 1)]
        YYnow_016[:, (self.select_w == 0)] = np.exp(YYnow_016[:,(self.select_w == 0)])
    
    lstate_016 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.16, axis = 0)# hf obs for lf vars
    lstate_016[:, (self.select_m_q == 1)] = 100 * lstate_016[:, (self.select_m_q == 1)]
    lstate_016[:, (self.select_m_q == 0)] = np.exp(lstate_016[:, (self.select_m_q == 0)])
    
    YY_016 = np.vstack((np.vstack((np.hstack((YW, lstate_016[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_016[-self.rqw:,:])))), YYftr_m))
        
    # 5%
    YYftr_005 = np.nanquantile(YYvector_ml,  q = 0.05, axis = 0)
    YYftr_005[:, (self.select == 1)] = 100 * YYftr_005[:, (self.select == 1)]
    YYftr_005[:, (self.select == 0)] = np.exp(YYftr_005[:, (self.select == 0)])
    
    YYnow_005 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.05, axis = 0) # actual/nowcast weeklies
    if YYnow_005.size:
        YYnow_005[:, (self.select_w== 1)] = 100 * YYnow_005[:, (self.select_w == 1)]
        YYnow_005[:, (self.select_w == 0)] = np.exp(YYnow_005[:,(self.select_w == 0)])
    
    lstate_005 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.05, axis = 0)# hf obs for lf vars
    lstate_005[:, (self.select_m_q == 1)] = 100 * lstate_005[:, (self.select_m_q == 1)]
    lstate_005[:, (self.select_m_q == 0)] = np.exp(lstate_005[:, (self.select_m_q == 0)])
    
    YY_005 = np.vstack((np.vstack((np.hstack((YW, lstate_005[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_005[-self.rqw:,:])))), YYftr_m))
    
    YY_mean_pd = pd.DataFrame(YY_m, columns = self.varlist)
    YY_mean_pd.index = index
    
    YY_median_pd = pd.DataFrame(YY_med, columns = self.varlist)
    YY_median_pd.index = index
    
    YY_095_pd = pd.DataFrame(YY_095, columns = self.varlist)
    YY_095_pd.index = index
    
    YY_005_pd = pd.DataFrame(YY_005, columns = self.varlist)
    YY_005_pd.index = index
    
    YY_084_pd = pd.DataFrame(YY_084, columns = self.varlist)
    YY_084_pd.index = index
    
    YY_016_pd = pd.DataFrame(YY_016, columns = self.varlist)
    YY_016_pd.index = index

    self.YY_095 = YY_095
    self.YY_084 = YY_084
    self.YY_016 = YY_016
    self.YY_005 = YY_005
    self.YY_mean = YY_m
    self.YY_median = YY_med
    self.forecast_draws = YYvector_ml
    
    self.YY_mean_pd = YY_mean_pd
    self.YY_median_pd = YY_median_pd
    self.YY_095_pd = YY_095_pd
    self.YY_005_pd = YY_005_pd
    self.YY_084_pd = YY_084_pd
    self.YY_016_pd = YY_016_pd
    
    self.index = index

def aggregate(self, frequency, reset_index = True):
    
    '''
    Aggregates the Mean, Median and quantililes in the highest frequency to the desired frequency. \n
    The Function ensures, that we start at the beginning of a Year or Quarter depending on the chosen frequency \n
    
    Parameters
    ----------
    frequency : str
        The frequency to which the data should be aggregated to
    reset_index : boolean
        Schould index be changed to period Index

    '''
    if self.forecast_draws is None :
            sys.exit("Error: To gaggregate generate forecasts first")
            
    if frequency not in ["Y","Q"] :
            sys.exit("Error: Aggregation currently only implemented for aggregation to yearly and quarterly frequency")
    freq_lf = frequency
    freq_hf = "W"
    
    YY_full_list = deque()
    YY_full_list_agg = deque()
    
    YYnow = copy.copy(self.YYactsim_list)# actual/nowcast
    lstate = copy.copy(self.lstate_list) # hf obs for lf vars
    
    
    YW = copy.deepcopy(self.input_data_W)[self.rqw:-(self.rqw)]
    YW[:, (self.select_w == 1)] = 100 * YW[:, (self.select_w == 1)]
    YW[:, (self.select_w == 0)] = np.exp(YW[:, (self.select_w == 0)])
    

    for i in range(len(self.valid_draws)):
        lstate_temp = lstate[i]
        lstate_temp[:, (self.select_m_q == 1)] = 100 * lstate_temp[:, (self.select_m_q == 1)]
        lstate_temp[:, (self.select_m_q == 0)] = np.exp(lstate_temp[:, (self.select_m_q == 0)])
        YYnow_temp = YYnow[i][1:(self.rqw+1),:self.Nw]
        YYnow_temp[:, (self.select_w == 1)] = 100 * YYnow_temp[:, (self.select_w == 1)]
        YYnow_temp[:, (self.select_w == 0)] = np.exp(YYnow_temp[:, (self.select_w == 0)])
        forecast_draws_temp = copy.copy(self.forecast_draws[i,:,:])
        forecast_draws_temp[:, (self.select == 1)] = 100 * forecast_draws_temp[:, (self.select == 1)]
        forecast_draws_temp[:, (self.select == 0)] = np.exp(forecast_draws_temp[:, (self.select == 0)])

        temp = np.vstack((np.vstack((np.hstack((YW , lstate_temp[:-(self.rqw), :])), np.hstack((YYnow_temp, lstate_temp[-self.rqw:,:])))), forecast_draws_temp))
        temp = pd.DataFrame(temp, columns = self.varlist)
        temp.index = self.index
        YY_full_list.append(temp)

    
    
    def find_first_position(arr, numbers, count):
        for i in range(len(arr) - count + 1):
            if arr[i] in numbers and all(arr[i] == arr[j] for j in range(i+1, i+count)):
                return i
    
    
            
    def agg_helper(freq_lf, freq_hf, df):
        # Set the frequency ratio        
        if freq_hf == "Q" and freq_lf == "Y":
            freq_ratio = 4
        elif freq_hf == "M" and freq_lf == "Y":
            freq_ratio = 12
        elif freq_hf == "W" and freq_lf == "Y":
            freq_ratio = 48
        elif freq_hf == "D" and freq_lf == "Y":
            freq_ratio = 260
        elif freq_hf == "M" and freq_lf == "Q":
            freq_ratio = 3
        elif freq_hf == "W" and freq_lf == "Q":
            freq_ratio = 12
        elif freq_hf == "W" and freq_lf == "M":
            freq_ratio = 4
        elif freq_hf == "D" and freq_lf == "W":
            freq_ratio = 5
        elif freq_hf == "D" and freq_lf == "M":
            freq_ratio = 20
                
        if frequency == 'Q':
            if freq_hf == 'W':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 4 )
            elif freq_hf == 'M':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 1)
            elif freq_hf == 'D':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 20)
        elif frequency == 'Y':
            if freq_hf == 'Q':
                start = find_first_position(df.index.month, [1,3], 1 )
            elif freq_hf == 'M':
                start = find_first_position(df.index.month, [1], 1)
            elif freq_hf == 'W':
                start = find_first_position(df.index.month, [1], 4)
            elif freq_hf == 'D':
                start = find_first_position(df.index.month, [1], 20)
        return freq_ratio, start
    
    
    freq_ratio, start = agg_helper(freq_lf, freq_hf, YY_full_list[0])
    print("Aggregating for each draw")
    for i in tqdm(range(len(self.valid_draws))):
        temp = YY_full_list[i].iloc[start:,].groupby(YY_full_list[i].iloc[start:,].reset_index().index // freq_ratio).filter(lambda x: len(x) == freq_ratio)
        if self.temp_agg == "mean":
            temp = temp.groupby(temp.reset_index().index // freq_ratio).mean()
        if self.temp_agg == "sum":
            temp = temp.groupby(temp.reset_index().index // freq_ratio).sum()
        temp.index = YY_full_list[i].iloc[start:,].index[::freq_ratio][:temp.shape[0]]
        temp.index = temp.index.map(lambda x: x.replace(day=1))
        YY_full_list_agg.append(temp)
        
    self.YY_mean_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).mean()
    self.YY_median_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).median()
    self.YY_095_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.95)
    self.YY_005_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.05)
    self.YY_084_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.84)
    self.YY_016_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.16)
    
    hist = []
    hist.append(self.input_index_Q)
    hist.append(self.input_index_M_W[0])
    hist.append(self.input_index_M_W[0])
    
    i = 0
    for m in self.frequencies:
        if self.frequencies.index(m) == self.frequencies.index(frequency):
            # history ersetzen durch originaldaten 
            # interval durch NA
            idx = self.YY_mean_agg.index.intersection(hist[i].index, sort=False)
            self.YY_mean_agg.loc[idx, hist[i].columns] = hist[i].loc[idx, hist[i].columns]
            self.YY_median_agg.loc[idx, hist[i].columns] = hist[i].loc[idx, hist[i].columns]
            self.YY_095_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_005_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_084_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_016_agg.loc[idx, hist[i].columns] = np.nan
            
        if self.frequencies.index(m) > self.frequencies.index(frequency) & self.frequencies.index(m) < len(self.frequencies)-1:
            freq_ratio_temp, start_temp = agg_helper(frequency, m, hist[i])
            hist_agg = hist[i].iloc[start_temp:,].groupby(hist[i].iloc[start_temp:,].reset_index().index // freq_ratio_temp).filter(lambda x: len(x) == freq_ratio_temp)
            if self.temp_agg == 'mean':
                hist_agg = hist_agg.groupby(hist_agg.reset_index().index // freq_ratio_temp).mean()
            if self.temp_agg == 'sum':
                hist_agg = hist_agg.groupby(hist_agg.reset_index().index // freq_ratio_temp).sum()
            hist_agg.index = hist[i].iloc[start_temp:,].index[::freq_ratio_temp][:hist_agg.shape[0]]
            hist_agg.index = hist_agg.index.map(lambda x: x.replace(day=1))
            
            idx = hist_agg.index.intersection(self.YY_mean_agg.index, sort=False)
            self.YY_mean_agg.loc[idx, hist[i].columns] = hist_agg.loc[idx, hist[i].columns]
            self.YY_median_agg.loc[idx, hist[i].columns] = hist_agg.loc[idx, hist[i].columns]
            self.YY_095_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_005_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_084_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_016_agg.loc[idx, hist[i].columns] = np.nan
        
        i = i+1
    
    if reset_index == True:
        index_new = pd.PeriodIndex(self.YY_mean_agg.index, freq= frequency)
        self.YY_mean_agg.index = index_new
        self.YY_median_agg.index = index_new
        self.YY_095_agg.index = index_new
        self.YY_005_agg.index = index_new
        self.YY_084_agg.index = index_new
        self.YY_016_agg.index = index_new
        
    
        
    if not(self.var_of_interest is None):
        idx_var_of_interest = list(filter(lambda x: self.YY_mean_agg.columns.tolist()[x] in self.YMX_list[-1].columns.tolist() + self.var_of_interest, range(len(self.YY_mean_agg.columns.tolist()))))
        self.YY_mean_agg = self.YY_mean_agg.iloc[:, idx_var_of_interest]
        self.YY_median_agg = self.YY_median_agg.iloc[:, idx_var_of_interest]
        self.YY_095_agg = self.YY_095_agg.iloc[:, idx_var_of_interest]
        self.YY_005_agg = self.YY_005_agg.iloc[:, idx_var_of_interest]
        self.YY_084_agg = self.YY_084_agg.iloc[:, idx_var_of_interest]
        self.YY_016_agg = self.YY_016_agg.iloc[:, idx_var_of_interest]
            
            
    self.agg_freq = frequency