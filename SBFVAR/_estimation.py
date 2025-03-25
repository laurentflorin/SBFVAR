import os
import sys
import numpy as np
import math
from collections import deque
from scipy.linalg import companion
from scipy.stats import invwishart
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
from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix
from .mfbvar_funcs import calc_yyact, is_explosive

tqdm = partial(tqdm, position=0, leave=True)
pio.renderers.default = 'browser'

def fit(self, mufbvar_data, hyp, var_of_interest=None, temp_agg='mean'):
    explosive_counter = 0
    valid_draws = []
    
    self.nex = 1
    self.hyp = hyp
    self.temp_agg = temp_agg
    
    assert self.temp_agg in ("mean", "sum"), f"Invalid temp_agg: {self.temp_agg}. Choose 'mean' or 'sum'."
    
    # Data from mufbvar_data
    YMX_list = copy.deepcopy(mufbvar_data.YMX_list)
    YM0_list = copy.deepcopy(mufbvar_data.YM0_list)
    select_m_list = copy.deepcopy(mufbvar_data.select_m_list)
    vars_m_list = copy.deepcopy(mufbvar_data.vars_m_list)
    YMh_list = copy.deepcopy(mufbvar_data.YMh_list)
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

    nburn = round((self.nburn_perc)*math.ceil(self.nsim/self.thining))
    self.nburn = nburn
    
    nlags = self.nlags
    
    # Validate frequency ratios
    rmw = freq_ratio_list[1]  # Monthly to weekly ratio
    rqw = freq_ratio_list[0] * rmw  # Quarterly to weekly ratio
    assert rmw == 4, "Monthly aggregation requires exactly 4 weeks/month"
    assert rqw == 12, "Quarterly aggregation requires exactly 12 weeks/quarter"
    
    # Extract variable counts
    Nq = Nq_list[0]  # Quarterly variables
    Nm = Nm_list[0]  # Monthly variables
    Nw = Nm_list[1]  # Weekly variables
    Ntotal = Nq + Nm + Nw  # Total number of variables across all frequencies
    
    # Extract data for each frequency
    YQ = copy.deepcopy(YQ_list[0])  # Quarterly data
    YM = copy.deepcopy(YM_list[0])  # Monthly data
    YW = copy.deepcopy(YM_list[1])  # Weekly data (second entry in YM_list)
    
    # Get observation counts
    Tw = YW.shape[0]  # Total weekly observations
    Tm = YM.shape[0]  # Monthly observations
    Tq = YQ.shape[0]  # Quarterly observations
    
    # Number of observations after burn-in (T0 = initial lag period)
    T0 = int(nlags)  # Initial observations used for lags
    nobs = Tw - T0  
    
    # REVISED STATE SPACE STRUCTURE
    # -----------------------------
    
    # The key VAR dynamics are specified at the weekly frequency
    p = self.nlags  # Number of lags for weekly VAR
    
    # State vector structure (modified to properly track latent states)
    # [weekly_vars(t), weekly_vars(t-1), ..., weekly_vars(t-p+1),
    #  latent_monthly_week1, latent_monthly_week2, latent_monthly_week3, latent_monthly_week4,
    #  latent_quarterly_week1, ..., latent_quarterly_week12]
    
    weekly_block_size = Nw * p  # Weekly variables with p lags
    
    # Allocate state blocks for the latent weekly states of monthly and quarterly variables
    monthly_latent_size = Nm * rmw  # Each monthly variable needs 4 weeks of latent states
    quarterly_latent_size = Nq * rqw  # Each quarterly variable needs 12 weeks of latent states
    
    # Total state vector size
    nstate = weekly_block_size + monthly_latent_size + quarterly_latent_size
    
    # Initialize matrices for MCMC sampling
    Sigmap = np.zeros((math.ceil((self.nsim)/self.thining), Ntotal, Ntotal))
    Phip = np.zeros((math.ceil((self.nsim)/self.thining), Nw*p+1, Nw))  # +1 for constant, only for weekly VAR 
    Cons = np.zeros((math.ceil((self.nsim)/self.thining), Nw))  # Constant terms
    
    # Initialize Phi matrix for weekly VAR coefficients
    Phi = np.vstack((0.95 * np.eye(Nw), np.zeros((Nw*(p-1), Nw)), np.zeros((1, Nw))))  # Include constant
    
    # Transition matrix F for the full state-space model
    F = np.zeros((nstate, nstate))
    
    # 1. Weekly VAR dynamics in the top-left block
    # Weekly coefficients section (VAR dynamics)
    F[:Nw, :Nw*p] = Phi[:-1, :].T  # Transpose of VAR coefficients (excluding constant)
    
    # Lag shifting section for weekly vars
    for i in range(p-1):
        F[Nw*(i+1):Nw*(i+2), Nw*i:Nw*(i+1)] = np.eye(Nw)
    
    # 2. Latent state transitions for monthly variables
    monthly_start = weekly_block_size
    
    # Initialize the latent states for monthly variables
    # Weekly shocks affect the first week of each monthly variable's latent states
    F[monthly_start:monthly_start+Nm, :Nw] = np.eye(Nm, Nw)  # Map shocks from weekly vars to monthly latent states
    
    # Shift weeks within each month for the latent states
    for i in range(rmw-1):
        src_pos = monthly_start + i*Nm
        dest_pos = monthly_start + (i+1)*Nm
        F[dest_pos:dest_pos+Nm, src_pos:src_pos+Nm] = np.eye(Nm)
    
    # 3. Latent state transitions for quarterly variables
    quarterly_start = weekly_block_size + monthly_latent_size
    
    # Initialize the latent states for quarterly variables
    # Weekly shocks affect the first week of each quarterly variable's latent states
    F[quarterly_start:quarterly_start+Nq, :Nw] = np.eye(Nq, Nw)  # Map shocks from weekly vars to quarterly latent states
    
    # Shift weeks within each quarter for the latent states
    for i in range(rqw-1):
        src_pos = quarterly_start + i*Nq
        dest_pos = quarterly_start + (i+1)*Nq
        F[dest_pos:dest_pos+Nq, src_pos:src_pos+Nq] = np.eye(Nq)
    
    # Constant term vector
    c = np.zeros((nstate, 1))
    c[:Nw] = np.atleast_2d(Phi[-1, :]).T  # Weekly constant terms
    
    # REVISED MEASUREMENT EQUATIONS
    # ----------------------------
    
    # 1. Measurement matrix for weekly variables (direct observation)
    H_w = np.zeros((Nw, nstate))
    H_w[:, :Nw] = np.eye(Nw)  # Directly observe first Nw state elements
    
    # 2. Measurement matrix for monthly variables (temporal aggregation)
    H_m = np.zeros((Nm, nstate))
    
    # Monthly variables are observed as the temporal aggregation of their latent weekly states
    for m in range(Nm):
        # For each monthly variable, aggregate its 4 weeks of latent states
        for w in range(rmw):
            if self.temp_agg == 'mean':
                H_m[m, monthly_start + w*Nm + m] = 1.0/rmw  # Average of 4 weeks
            else:
                H_m[m, monthly_start + w*Nm + m] = 1.0  # Sum of 4 weeks
    
    # 3. Measurement matrix for quarterly variables (temporal aggregation)
    H_q = np.zeros((Nq, nstate))
    
    # Quarterly variables are observed as the temporal aggregation of their latent weekly states
    for q in range(Nq):
        # For each quarterly variable, aggregate its 12 weeks of latent states
        for w in range(rqw):
            if self.temp_agg == 'mean':
                H_q[q, quarterly_start + w*Nq + q] = 1.0/rqw  # Average of 12 weeks
            else:
                H_q[q, quarterly_start + w*Nq + q] = 1.0  # Sum of 12 weeks
    
    # Initialize state and covariance matrices
    Q = np.zeros((nstate, nstate))
    Q[:Nw, :Nw] = 1e-4 * np.eye(Nw)  # Process noise affects weekly states directly
    
    a_t = np.zeros(nstate)
    P_t = np.eye(nstate)
    
    # Initialize stability by iterating a few times
    for _ in range(5):
        P_t = F @ P_t @ F.T + Q
        P_t = 0.5 * (P_t + P_t.T)  # Ensure symmetry
    
    # Storage for filtered states
    a_filtered = np.zeros((nobs, nstate))
    P_filtered = np.zeros((nobs, nstate, nstate))
    
    # Prepare data for Kalman filtering
    
    # Resample the quarterly and monthly data to weekly frequency
    YQ_weekly = np.zeros((Tw, Nq))
    YM_weekly = np.zeros((Tw, Nm))
    
    # Fill in the values and mark which observations are available
    q_obs_available = np.zeros(Tw, dtype=bool)
    m_obs_available = np.zeros(Tw, dtype=bool)
    
    # Quarterly data is available every rqw weeks (12 weeks)
    for t in range(0, Tw, rqw):
        q_idx = t // rqw
        if q_idx < Tq:
            YQ_weekly[t] = YQ[q_idx]
            q_obs_available[t] = True
    
    # Monthly data is available every rmw weeks (4 weeks)
    for t in range(0, Tw, rmw):
        m_idx = t // rmw
        if m_idx < Tm:
            YM_weekly[t] = YM[m_idx]
            m_obs_available[t] = True
    
    # Prepare lagged data for the VAR
    Z = np.zeros((nobs, Nw * p))
    for i in range(nobs):
        for j in range(p):
            if T0+i-j-1 >= 0 and T0+i-j-1 < Tw:  # Bounds check
                Z[i, j*Nw:(j+1)*Nw] = YW[T0+i-j-1, :]
    
    # Initialize storage for MCMC sampling
    a_draws = np.zeros((self.nsim, nobs, nstate))  # Store all states
    
    print(" ", end = '\n')
    print("Multi Frequency BVAR: Estimation - Unified Three Frequency Approach", end = '\n')
    print("Frequencies: ", self.frequencies, end = "\n")
    print("Total Number of Draws: ", self.nsim)
    
    # Main sampling loop
    for j in tqdm(range(self.nsim)):
        # If it's the first iteration, initialize state and covariance
        if j > 0:
            a_t = a_draws[j-1, -1]
            P_t = P_filtered[-1]
            
        # Kalman filter loop
        for t in range(nobs):
            # Prediction step
            a_pred = F @ a_t + c.flatten()  # Ensure compatible shapes
            P_pred = F @ P_t @ F.T + Q
            
            # Ensure symmetry
            P_pred = 0.5 * (P_pred + P_pred.T)
            
            # Determine which measurements are available
            w_idx = T0 + t  # Weekly index
            m_idx = w_idx // rmw  # Monthly index
            q_idx = w_idx // rqw  # Quarterly index
            
            # Initialize valid measurement tracking
            H_matrices = []
            y_obs = []
            
            # Check for weekly observations
            if w_idx < Tw and w_idx < len(YW) and not np.all(np.isnan(YW[w_idx])):
                try:
                    w_data = np.asarray(YW[w_idx], dtype=float)
                    if len(w_data) == Nw:
                        H_matrices.append(H_w)
                        y_obs.append(w_data)
                except (IndexError, ValueError) as e:
                    if t == 0 and j == 0:
                        print(f"DEBUG: Error processing weekly data at w_idx={w_idx}: {e}")
            
            # Check for monthly observations - only available at the end of month
            if (w_idx+1) % rmw == 0:  # End of month
                m_idx = (w_idx+1) // rmw - 1
                if m_idx < YM.shape[0]:
                    # Create a measurement matrix that enforces the temporal aggregation constraint
                    H_m_constraint = np.zeros((Nm, nstate))
                    for m in range(Nm):
                        for w in range(rmw):
                            if self.temp_agg == 'mean':
                                H_m_constraint[m, monthly_start + w*Nm + m] = 1.0/rmw  # Average
                            else:
                                H_m_constraint[m, monthly_start + w*Nm + m] = 1.0  # Sum
                    H_matrices.append(H_m_constraint)
                    y_obs.append(YM[m_idx])
                    R = np.zeros((len(y_obs[-1]), len(y_obs[-1])))
                    R[-Nm:, -Nm:] = 1e-10 * np.eye(Nm)
                    if len(y_obs[-1]) > Nm:
                        R[:-Nm, :-Nm] = 1e-8 * np.eye(len(y_obs[-1]) - Nm)
            
            # Check for quarterly observations - only available at the end of quarter
            if (w_idx+1) % rqw == 0:  # End of quarter
                q_idx = (w_idx+1) // rqw - 1
                if q_idx < YQ.shape[0]:
                    # Create a measurement matrix that enforces the temporal aggregation constraint
                    H_q_constraint = np.zeros((Nq, nstate))
                    for q in range(Nq):
                        for w in range(rqw):
                            if self.temp_agg == 'mean':
                                H_q_constraint[q, quarterly_start + w*Nq + q] = 1.0/rqw  # Average
                            else:
                                H_q_constraint[q, quarterly_start + w*Nq + q] = 1.0  # Sum
                    H_matrices.append(H_q_constraint)
                    y_obs.append(YQ[q_idx])
                    if len(H_matrices) > 1:
                        R = np.zeros((len(y_obs[-1]), len(y_obs[-1])))
                        R[-Nq:, -Nq:] = 1e-12 * np.eye(Nq)
                        if len(y_obs[-1]) > Nq:
                            if len(y_obs[-1]) == Nq + Nm:
                                R[:-Nq, :-Nq] = 1e-10 * np.eye(Nm)
                            else:
                                R[:-Nq, :-Nq] = 1e-8 * np.eye(len(y_obs[-1]) - Nq)

            # If we have valid measurements, proceed with Kalman update
            if H_matrices and y_obs:
                # Stack measurement matrices and observations
                H = np.vstack(H_matrices)
                y = np.concatenate(y_obs)
                
                # Measurement noise covariance
                R = 1e-8 * np.eye(len(y))  # Small noise for numerical stability
                
                # Update step
                y_hat = H @ a_pred
                nu = y - y_hat  # Innovation
                
                S = H @ P_pred @ H.T + R
                S = 0.5 * (S + S.T)  # Ensure symmetry
                
                try:
                    K = P_pred @ H.T @ invert_matrix(S)  # Kalman gain
                    a_t = a_pred + K @ nu
                    P_t = P_pred - K @ H @ P_pred
                    # Ensure symmetry
                    P_t = 0.5 * (P_t + P_t.T)
                except np.linalg.LinAlgError as e:
                    if t == 0 and j == 0:
                        print(f"DEBUG: Matrix inversion failed: {e} - using prediction only")
                    a_t = a_pred
                    P_t = P_pred
            
            # Store filtered state
            a_filtered[t] = a_t
            P_filtered[t] = P_t
        
        # Kalman Smoother
        a_smooth = np.zeros((nobs, nstate))
        P_smooth = np.zeros((nobs, nstate, nstate))
        
        # Initialize with the last filtered state
        a_smooth[-1] = a_filtered[-1]
        P_smooth[-1] = P_filtered[-1]
        
        try:
            # Draw the last state
            Pchol = cholcovOrEigendecomp(P_smooth[-1])
            a_draw = a_smooth[-1] + Pchol @ np.random.standard_normal(nstate)
            a_draws[j, -1] = a_draw
            
            # Backwards recursion for smoothing and drawing states
            for t in range(nobs-2, -1, -1):
                # Get filtered state and covariance
                a_t = a_filtered[t]
                P_t = P_filtered[t]
                
                # Predict one step ahead
                a_pred = F @ a_t + c.flatten()
                P_pred = F @ P_t @ F.T + Q
                P_pred = 0.5 * (P_pred + P_pred.T)  # Ensure symmetry
                
                try:
                    # Smoothing gain
                    J_t = P_t @ F.T @ invert_matrix(P_pred)
                    
                    # Smoothed mean and covariance
                    a_smooth_t = a_t + J_t @ (a_draw - a_pred)
                    P_smooth_t = P_t - J_t @ (P_pred - P_smooth[t+1]) @ J_t.T
                    P_smooth_t = 0.5 * (P_smooth_t + P_smooth_t.T)  # Ensure symmetry
                    
                    # Draw state
                    Pchol = cholcovOrEigendecomp(P_smooth_t)
                    a_draw = a_smooth_t + Pchol @ np.random.standard_normal(nstate)
                    
                    # Store smoothed state and draw
                    a_smooth[t] = a_smooth_t
                    P_smooth[t] = P_smooth_t
                    a_draws[j, t] = a_draw
                    
                except Exception as e:
                    if j == 0:
                        print(f"DEBUG: Error in smoothing at t={t}: {e}")
                    # Use filtered state as fallback
                    a_smooth[t] = a_filtered[t]
                    P_smooth[t] = P_filtered[t]
                    a_draws[j, t] = a_filtered[t] + np.random.standard_normal(nstate) * 1e-4
            
        except Exception as e:
            if j == 0:
                print(f"DEBUG: Error in smoother initialization: {e}")
            # Use filtered states as fallback
            a_draws[j] = a_filtered
        
        # Extract smoothed time series for each frequency
        # Note the change in indexing to match the revised state space structure
        W_smooth = a_draws[j, :, :Nw]  # Weekly variables - directly observed
        
        # For monthly and quarterly, extract from the latent blocks for VAR estimation
        # This is a simplification - proper handling would adjust based on temporal aggregation
        M_smooth = np.zeros((nobs, Nm))
        Q_smooth = np.zeros((nobs, Nq))
        
        # Extract monthly and quarterly variables - taking temporal aggregation into account
        for t in range(nobs):
            # For monthly vars - average over the latent weekly states 
            for m in range(Nm):
                if self.temp_agg == 'mean':
                    # Average the 4 weeks of latent states for each monthly variable
                    M_smooth[t, m] = np.mean([a_draws[j, t, monthly_start + w*Nm + m] for w in range(rmw)])
                else:
                    # Sum the 4 weeks of latent states
                    M_smooth[t, m] = np.sum([a_draws[j, t, monthly_start + w*Nm + m] for w in range(rmw)])
                    
            # For quarterly vars - average over the latent weekly states
            for q in range(Nq):
                if self.temp_agg == 'mean':
                    # Average the 12 weeks of latent states for each quarterly variable
                    Q_smooth[t, q] = np.mean([a_draws[j, t, quarterly_start + w*Nq + q] for w in range(rqw)])
                else:
                    # Sum the 12 weeks of latent states
                    Q_smooth[t, q] = np.sum([a_draws[j, t, quarterly_start + w*Nq + q] for w in range(rqw)])
        
        # Combine all data for VAR estimation - only use weekly data for VAR dynamics
        # since monthly and quarterly are derived from weekly
        YY = W_smooth
        
        # Compute actual observations for Minnesota prior
        nobs_ = YY.shape[0] - T0  # Adjusted for lags
        spec = np.hstack((p, T0, self.nex, Nw, nobs_))  # Only Nw for VAR, not Ntotal
        
        # Calculate dummy observations - modified to only use weekly variables for VAR
        YYact, YYdum, XXact, XXdum = calc_yyact(self.hyp, YY, spec)
        
        # Store simulation results if needed
        if (j % self.thining == 0):
            if 'YYactsim_list' in locals():
                YYactsim_list[0][int(j/self.thining), :, :] = np.hstack((W_smooth[-rmw:, :], M_smooth[-rmw:, :], Q_smooth[-rmw:, :]))
                XXactsim_list[0][int(j/self.thining), :, :] = XXact[-rmw:, :]
            else:
                # Store the combined data for later use
                YYactsim_list = [np.zeros((math.ceil((self.nsim)/self.thining), rmw, Ntotal))]
                XXactsim_list = [np.zeros((math.ceil((self.nsim)/self.thining), rmw, Nw*p+1))]  # Only for weekly VAR
                YYactsim_list[0][int(j/self.thining), :, :] = np.hstack((W_smooth[-rmw:, :], M_smooth[-rmw:, :], Q_smooth[-rmw:, :]))
                XXactsim_list[0][int(j/self.thining), :, :] = XXact[-rmw:, :]
        
        # Draws from posterior distribution - use only weekly variables for VAR
        try:
            Tdummy, n = YYdum.shape
            n = int(n)
            Tdummy = int(Tdummy)
            Tobs, n = YYact.shape
            X = np.vstack((XXact, XXdum))
            Y = np.vstack((YYact, YYdum))
            T = Tobs + Tdummy
            
            # Compute posterior parameters
            vl, d, vr = np.linalg.svd(X, full_matrices=False)
            vr = vr.T
            di = 1/d
            B = vl.T @ Y
            xxi = (vr * np.tile(di.T, (Nw*p+1, 1)))
            inv_x = xxi @ xxi.T
            Phi_tilde = xxi @ B
            
            Sigma = (Y - X @ Phi_tilde).T @ (Y - X @ Phi_tilde)
            
            # Draw from inverse Wishart for covariance matrix of weekly variables
            sigma_w = invwishart.rvs(scale=Sigma, df=T-Nw*p-1)
            
            # Draw VAR coefficients and check stability
            attempts = 0
            while attempts < 1000:
                sigma_chol = cholcovOrEigendecomp(np.kron(sigma_w, inv_x))
                phi_new = np.squeeze(Phi_tilde.reshape(Nw*(Nw*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                Phi_w = phi_new.reshape(Nw*p+1, Nw, order="F")
                if not is_explosive(Phi_w, Nw, p):
                    break
                attempts += 1
            
            if attempts == 1000:
                explosive_counter += 1
                print(f"Explosive VAR detected {explosive_counter} times.")
                continue
                
            # Construct full covariance matrix for all variables
            # We need to derive the joint covariance for weekly, monthly, and quarterly variables
            
            # For simplicity, assume independence between frequencies initially
            # This is a simplified approach - in a real implementation, these would be estimated jointly
            sigma_m = np.eye(Nm) * 0.01  # Placeholder covariance for monthly vars
            sigma_q = np.eye(Nq) * 0.01  # Placeholder covariance for quarterly vars
            
            # Create a block diagonal covariance matrix
            sigma = np.zeros((Ntotal, Ntotal))
            sigma[:Nw, :Nw] = sigma_w
            sigma[Nw:Nw+Nm, Nw:Nw+Nm] = sigma_m
            sigma[Nw+Nm:, Nw+Nm:] = sigma_q
            
            # Store posterior draws
            if (j % self.thining == 0):
                j_temp = int(j/self.thining)
                Sigmap[j_temp, :, :] = sigma
                Phip[j_temp, :, :] = Phi_w  # Store only weekly VAR coefficients
                Cons[j_temp, :] = Phi_w[-1, :]  # Constants for weekly vars
                valid_draws.append(j_temp)
            
            # Update the transition matrix for the next iteration
            # Update weekly VAR coefficients
            F[:Nw, :Nw*p] = Phi_w[:-1, :].T
            c[:Nw] = np.atleast_2d(Phi_w[-1, :]).T
            
            # Make sure the shifting blocks remain intact for temporal aggregation
            # Monthly block shifting
            for i in range(rmw-1):
                src_pos = monthly_start + i*Nm
                dest_pos = monthly_start + (i+1)*Nm
                F[dest_pos:dest_pos+Nm, src_pos:src_pos+Nm] = np.eye(Nm)
                
            # Quarterly block shifting
            for i in range(rqw-1):
                src_pos = quarterly_start + i*Nq
                dest_pos = quarterly_start + (i+1)*Nq
                F[dest_pos:dest_pos+Nq, src_pos:src_pos+Nq] = np.eye(Nq)
                
            # Weekly innovations enter the first position of monthly and quarterly blocks
            F[monthly_start:monthly_start+Nm, :Nw] = np.eye(Nm, Nw)
            F[quarterly_start:quarterly_start+Nq, :Nw] = np.eye(Nq, Nw)
            
            # Update system covariance for weekly blocks
            Q[:Nw, :Nw] = sigma_w
            
        except Exception as e:
            print(f"DEBUG: Error in VAR posterior sampling: {e}")
            # If VAR estimation fails, continue with next iteration
            continue

    # Save results to self
    self.Phip = Phip  # Weekly VAR coefficients
    self.Sigmap = Sigmap
    self.nv = Ntotal
    self.Nw = Nw
    self.Nm = Nm
    self.Nq = Nq
    self.freq_ratio = freq_ratio_list
    self.select = select_list[0]  # Unified selection vector
    self.varlist = varlist_list  # Unified variable list
    self.YYactsim_list = YYactsim_list
    self.XXactsim_list = XXactsim_list
    self.explosive_counter = explosive_counter
    self.valid_draws = [draw for draw in valid_draws if draw >= self.nburn/self.thining]

    # Store the smoothed states for later use
    self.a_draws = a_draws  # Full state history [n_draws, nobs, nstate]

    # Store original data for reference
    self.input_data_W = YW
    self.input_data_M = YM
    self.input_data_Q = YQ

    # Store indexes
    self.index_list = index_list

    return None




def forecast(self, H, conditionals=None):
    '''
    Method to generate the forecasts in the highest frequency (weekly).
    Includes mean and percentile forecasts (5%, 16%, median, 84%, 95%).
    
    Parameters
    ----------
    H : int
        Forecast horizon in highest frequency (weeks)
    conditionals : pandas DataFrame or None
        Conditional forecasts\n
        column names must be the variable names\n
        no index needed\n
        either values or np.nan
    '''
    self.H = H
    
    # First we need to extend the index
    index = copy.deepcopy(self.index_list[-1])  # Weekly index
    
    # Index extension logic
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
    
    # Get key dimensions from the fitted model
    Nw = self.Nw      # Number of weekly variables
    Nm = self.Nm      # Number of monthly variables
    Nq = self.Nq      # Number of quarterly variables
    Ntotal = Nw + Nm + Nq  # Total variables
    select = self.select  # All variable transformations in unified format
    
    # Check if varlist has the correct length
    if len(self.varlist) != Ntotal:
        print(f"ERROR: Variable list length ({len(self.varlist)}) doesn't match total variables ({Ntotal})")
        print("Regenerating variable list with proper dimensions...")
        
        # Create a properly sized variable list
        varlist_w = [f"Weekly_{i+1}" for i in range(Nw)]
        varlist_m = [f"Monthly_{i+1}" for i in range(Nm)]
        varlist_q = [f"Quarterly_{i+1}" for i in range(Nq)]
        
        # Combine in the correct order
        self.varlist = varlist_w + varlist_m + varlist_q
        
        print(f"New variable list created: {self.varlist}")
    
    # Get frequency ratios
    rmw = self.freq_ratio[1] if len(self.freq_ratio) > 1 else 4  # Monthly to weekly ratio (4)
    rqm = self.freq_ratio[0] if len(self.freq_ratio) > 0 else 3  # Quarterly to monthly ratio (3)
    rqw = rmw * rqm  # Quarterly to weekly ratio (12)
    
    # Initialize conditional forecasts with the right dimensions
    YYcond = pd.DataFrame(np.nan, index=index[-H:], columns=self.varlist)
    
    # Process conditionals if provided
    if conditionals is not None:
        print(f"Conditional forecast requested. Model has {Ntotal} variables.")
        print(f"Model variables: {self.varlist}")
        print(f"Conditional columns: {conditionals.columns.tolist()}")
        
        # Check for any variables in conditionals that aren't in the model
        unknown_vars = [var for var in conditionals.columns if var not in self.varlist]
        if unknown_vars:
            print(f"WARNING: These conditional variables aren't in the model and will be ignored: {unknown_vars}")
        
        # Only use variables that exist in the model
        valid_cond_vars = [var for var in conditionals.columns if var in self.varlist]
        if valid_cond_vars:
            # Map conditional periods to forecast periods
            conditionals.index = YYcond.index[:len(conditionals.index)]
            
            # Update only the valid columns
            YYcond[valid_cond_vars] = conditionals[valid_cond_vars]
            
            # Convert to array for forecasting
            YYcond_array = np.array(YYcond)
            
            # Apply transformations based on variable type
            for i, var in enumerate(self.varlist):
                # Determine which frequency this variable belongs to
                if i < Nw:  # Weekly
                    if i < select.shape[0] and select[i] == 1:  # Growth rate
                        YYcond_array[:, i] = YYcond_array[:, i] / 100 if not np.isnan(YYcond_array[:, i]).all() else YYcond_array[:, i]
                    elif i < select.shape[0]:  # Level
                        YYcond_array[:, i] = np.log(YYcond_array[:, i]) if not np.isnan(YYcond_array[:, i]).all() else YYcond_array[:, i]
                elif i < Nw + Nm:  # Monthly
                    if i < select.shape[0] and select[i] == 1:  # Growth rate
                        YYcond_array[:, i] = YYcond_array[:, i] / 100 if not np.isnan(YYcond_array[:, i]).all() else YYcond_array[:, i]
                    elif i < select.shape[0]:  # Level
                        YYcond_array[:, i] = np.log(YYcond_array[:, i]) if not np.isnan(YYcond_array[:, i]).all() else YYcond_array[:, i]
                else:  # Quarterly
                    if i < select.shape[0] and select[i] == 1:  # Growth rate
                        YYcond_array[:, i] = YYcond_array[:, i] / 100 if not np.isnan(YYcond_array[:, i]).all() else YYcond_array[:, i]
                    elif i < select.shape[0]:  # Level
                        YYcond_array[:, i] = np.log(YYcond_array[:, i]) if not np.isnan(YYcond_array[:, i]).all() else YYcond_array[:, i]
                        
            YYcond = YYcond_array
        else:
            print("WARNING: No valid conditional variables found. Proceeding with unconditional forecast.")
            YYcond = np.full((H, Ntotal), np.nan)
    else:
        # No conditionals provided
        YYcond = np.full((H, Ntotal), np.nan)
    
    # Create the conditional indicator array
    exc = ~np.isnan(YYcond)
    
    # Extract the most recent observed values
    if hasattr(self, 'YYactsim_list') and len(self.YYactsim_list) > 0:
        # Get the last observed state values for forecasting
        YYact = np.squeeze(self.YYactsim_list[0][self.valid_draws[-1], -1, :])
        
        # Get the lagged values for the VAR
        if hasattr(self, 'XXactsim_list') and len(self.XXactsim_list) > 0:
            XXact = np.squeeze(self.XXactsim_list[0][self.valid_draws[-1], -1, :])
        else:
            # Fallback logic if XXactsim_list not available
            print("Warning: XXactsim_list not available")
            XXact = None
    else:
        # If no simulation results available, initialize with zeros
        print("Warning: No simulation history available, using zeros for initial state")
        YYact = np.zeros(Ntotal)
        XXact = None
    
    # Reshape state vector for forecast
    # Now we need to include the original state structure for latent variables
    p = self.nlags  # Number of lags for weekly VAR
    
    # State blocks
    weekly_block_size = Nw * p
    monthly_latent_size = Nm * rmw
    quarterly_latent_size = Nq * rqw
    
    # Total state size
    nstate = weekly_block_size + monthly_latent_size + quarterly_latent_size
    
    # Initialize the full state vector
    state_t = np.zeros(nstate)
    
    # Fill in the weekly VAR parts
    if XXact is not None:
        # Extract weekly variables from XXact (the lagged values)
        for i in range(p):
            if i*Nw < XXact.shape[0] - 1:  # -1 for constant
                state_t[i*Nw:(i+1)*Nw] = XXact[i*Nw:(i+1)*Nw]
    else:
        # Use YYact for the first lag and zeros for the rest
        state_t[:Nw] = YYact[:Nw]
    
    # Initialize monthly and quarterly latent blocks with the most recent values
    # This is a simplification - in a proper implementation, these would be estimated
    # from the smoothed states in the last period
    monthly_start = weekly_block_size
    quarterly_start = monthly_start + monthly_latent_size
    
    # Set equal values across all weeks for initialization
    for i in range(rmw):
        state_t[monthly_start + i*Nm:monthly_start + (i+1)*Nm] = YYact[Nw:Nw+Nm]
        
    for i in range(rqw):
        state_t[quarterly_start + i*Nq:quarterly_start + (i+1)*Nq] = YYact[Nw+Nm:]
    
    # Get VAR coefficients and covariance matrix
    # Get actual dimensions from the VAR coefficient matrix
    coef_dim = self.Phip.shape[1] - 1  # Subtract 1 for the constant term
    p_actual = coef_dim // Nw      # Actual number of lags based on coefficient matrix
    
    print(f"Coefficient matrix suggests {p_actual} lags with {Nw} weekly variables")
    print(f"Total coefficient dimension: {coef_dim + 1} (including constant)")
    
    # Storage for forecasts for all variables
    H_ = int(self.H)
    YYvector_all = np.zeros((len(self.valid_draws), H_, Ntotal))
    
    print(" ", end='\n')
    print("Multiple Frequency BVAR: Forecasting", end="\n")
    print("Forecast Horizon: ", H_, end="\n")
    print("Total Draws: ", len(self.valid_draws))
    
    # Construct transition matrix F and constant c
    # Default to identity transitions for initialization
    F = np.eye(nstate)
    c = np.zeros(nstate)
    
    # Setup the monthly and quarterly aggregation blocks
    # Monthly latent state shifting
    monthly_start = Nw * p
    for i in range(rmw-1):
        src_pos = monthly_start + i*Nm
        dest_pos = monthly_start + (i+1)*Nm
        F[dest_pos:dest_pos+Nm, src_pos:src_pos+Nm] = np.eye(Nm)
    
    # Quarterly latent state shifting
    quarterly_start = monthly_start + monthly_latent_size
    for i in range(rqw-1):
        src_pos = quarterly_start + i*Nq
        dest_pos = quarterly_start + (i+1)*Nq
        F[dest_pos:dest_pos+Nq, src_pos:src_pos+Nq] = np.eye(Nq)
    
    # The first positions get updated from weekly variables
    F[monthly_start:monthly_start+Nm, :Nw] = np.eye(Nm, Nw)
    F[quarterly_start:quarterly_start+Nq, :Nw] = np.eye(Nq, Nw)
    
    # Setup measurement matrices
    H = np.zeros((Ntotal, nstate))
    
    # Weekly variables - direct observation
    H[:Nw, :Nw] = np.eye(Nw)
    
    # Monthly variables - temporal aggregation of latent states
    for m in range(Nm):
        # For each monthly variable, aggregate its 4 weeks of latent states
        for w in range(rmw):
            if self.temp_agg == 'mean':
                H[Nw+m, monthly_start + w*Nm + m] = 1.0/rmw  # Average of 4 weeks
            else:  # 'sum'
                H[Nw+m, monthly_start + w*Nm + m] = 1.0  # Sum of 4 weeks
    
    # Quarterly variables - temporal aggregation of latent states
    for q in range(Nq):
        # For each quarterly variable, aggregate its 12 weeks of latent states
        for w in range(rqw):
            if self.temp_agg == 'mean':
                H[Nw+Nm+q, quarterly_start + w*Nq + q] = 1.0/rqw  # Average of 12 weeks
            else:  # 'sum'
                H[Nw+Nm+q, quarterly_start + w*Nq + q] = 1.0  # Sum of 12 weeks
    
    for idx, jj in enumerate(tqdm(self.valid_draws)):
        # Extract VAR parameters for this draw
        post_phi = np.squeeze(self.Phip[jj, :, :])  # Weekly VAR coefficients
        post_sig = np.squeeze(self.Sigmap[jj, :, :])  # Full covariance matrix
        
        # Update the transition dynamics for weekly variables
        F[:Nw, :Nw*p] = post_phi[:-1, :].T  # VAR coefficients
        c[:Nw] = post_phi[-1, :]  # Constant terms
        
        # Initialize state and forecasts arrays
        state_fcst = np.zeros((H_+1, nstate))  # States including initial
        state_fcst[0, :] = state_t  # Set initial state
        
        YYpred = np.zeros((H_+1, Ntotal))  # Forecasts including initial
        YYpred[0, :] = H @ state_t  # Initial forecast
        
        # Generate forecast innovations
        error_pred = np.zeros((H_+1, Nw))  # Only need innovations for weekly variables
        for h in range(1, H_+1):  # Skip the initial values
            if post_sig.size > 1:
                # Draw from weekly covariance matrix
                error_pred[h, :] = np.random.default_rng().multivariate_normal(
                    mean=np.zeros(Nw), cov=post_sig[:Nw, :Nw], method="cholesky")
            else:
                # Fallback if we don't have a proper covariance matrix
                error_pred[h, :] = np.random.default_rng().normal(
                    loc=0, scale=post_sig, size=Nw)
        
        # Iterate forward to construct forecasts
        for h in range(1, H_+1):
            try:
                # State transition equation
                next_state = F @ state_fcst[h-1, :] + c
                
                # Add innovations to weekly variables only
                next_state[:Nw] += error_pred[h, :]
                
                # Store the state
                state_fcst[h, :] = next_state
                
                # Measurement equation to get observed variables
                model_forecast = H @ next_state
                
                # Apply conditionals if available for this period
                if h-1 < exc.shape[0]:
                    # Apply conditionals only where they exist (exc=True)
                    for i in range(Ntotal):
                        if h-1 < exc.shape[0] and i < exc.shape[1] and exc[h-1, i]:
                            YYpred[h, i] = YYcond[h-1, i]
                        else:
                            YYpred[h, i] = model_forecast[i]
                else:
                    # No conditionals for this period
                    YYpred[h, :] = model_forecast
                        
            except Exception as e:
                print(f"ERROR at h={h}: {e}")
                # Output debug info when an error occurs
                print(f"state_fcst[h-1] shape: {state_fcst[h-1].shape}, F shape: {F.shape}, c shape: {c.shape}")
                print(f"exc[h-1] shape: {exc[h-1].shape if h-1 < len(exc) else 'out of bounds'}")
                print(f"YYcond[h-1] shape: {YYcond[h-1].shape if h-1 < len(YYcond) else 'out of bounds'}")
                print(f"model_forecast shape: {model_forecast.shape if 'model_forecast' in locals() else 'not calculated'}")
                continue
        
        # Store forecasts (excluding the initial values)
        YYvector_all[idx, :, :] = YYpred[1:, :]
    
    # Store all forecast draws
    self.forecast_draws_list = [YYvector_all]
    
    # Calculate statistics from the draws
    # Mean forecast
    YYftr_m = np.nanmean(YYvector_all, axis=0)
    
    # Median forecast
    YYftr_med = np.nanmedian(YYvector_all, axis=0)
    
    # Percentile forecasts
    YYftr_095 = np.nanpercentile(YYvector_all, 95, axis=0)
    YYftr_005 = np.nanpercentile(YYvector_all, 5, axis=0)
    YYftr_084 = np.nanpercentile(YYvector_all, 84, axis=0)
    YYftr_016 = np.nanpercentile(YYvector_all, 16, axis=0)
    
    # Create copies for transformations
    YYftr_m_trans = YYftr_m.copy()
    YYftr_med_trans = YYftr_med.copy()
    YYftr_095_trans = YYftr_095.copy()
    YYftr_005_trans = YYftr_005.copy()
    YYftr_084_trans = YYftr_084.copy()
    YYftr_016_trans = YYftr_016.copy()
    
    # Apply inverse transformations to each statistic
    for i in range(Ntotal):
        if i < select.shape[0]:
            if select[i] == 1:  # Growth rate
                YYftr_m_trans[:, i] = 100 * YYftr_m[:, i]
                YYftr_med_trans[:, i] = 100 * YYftr_med[:, i]
                YYftr_095_trans[:, i] = 100 * YYftr_095[:, i]
                YYftr_005_trans[:, i] = 100 * YYftr_005[:, i]
                YYftr_084_trans[:, i] = 100 * YYftr_084[:, i]
                YYftr_016_trans[:, i] = 100 * YYftr_016[:, i]
            else:  # Level
                YYftr_m_trans[:, i] = np.exp(YYftr_m[:, i])
                YYftr_med_trans[:, i] = np.exp(YYftr_med[:, i])
                YYftr_095_trans[:, i] = np.exp(YYftr_095[:, i])
                YYftr_005_trans[:, i] = np.exp(YYftr_005[:, i])
                YYftr_084_trans[:, i] = np.exp(YYftr_084[:, i])
                YYftr_016_trans[:, i] = np.exp(YYftr_016[:, i])
    
    # Extract historical data for plots
    hist_length = min(24, len(index) - H_)  # Use up to 24 historical periods
    history_data = np.zeros((hist_length, Ntotal))
    
    # Try to populate history data from the smoothed states if available
    if hasattr(self, 'a_draws'):
        try:
            # For weekly variables - direct
            if len(self.a_draws) >= len(self.valid_draws) and self.a_draws.shape[1] >= hist_length:
                # Weekly variables - direct observation
                history_data[:, :Nw] = np.mean(self.a_draws[-len(self.valid_draws):, -hist_length:, :Nw], axis=0)
                
                # Monthly and quarterly variables - use measurement equations
                last_states = np.mean(self.a_draws[-len(self.valid_draws):, -hist_length:, :], axis=0)
                
                # For monthly variables - need to apply temporal aggregation
                for t in range(hist_length):
                    # Monthly variables
                    monthly_start = Nw * p
                    for m in range(Nm):
                        # Average over the 4 weeks of latent states
                        if self.temp_agg == 'mean':
                            history_data[t, Nw+m] = np.mean([last_states[t, monthly_start + w*Nm + m] for w in range(rmw)])
                        else:
                            history_data[t, Nw+m] = np.sum([last_states[t, monthly_start + w*Nm + m] for w in range(rmw)])
                    
                    # Quarterly variables
                    quarterly_start = monthly_start + monthly_latent_size
                    for q in range(Nq):
                        # Average over the 12 weeks of latent states
                        if self.temp_agg == 'mean':
                            history_data[t, Nw+Nm+q] = np.mean([last_states[t, quarterly_start + w*Nq + q] for w in range(rqw)])
                        else:
                            history_data[t, Nw+Nm+q] = np.sum([last_states[t, quarterly_start + w*Nq + q] for w in range(rqw)])
                
                # Transform the history data
                history_trans = history_data.copy()
                
                # Apply transformations to historical data
                for i in range(Ntotal):
                    if i < select.shape[0]:
                        if select[i] == 1:  # Growth rate
                            history_trans[:, i] = 100 * history_data[:, i]
                        else:  # Level
                            history_trans[:, i] = np.exp(history_data[:, i])
                
                history_data = history_trans
        except Exception as e:
            print(f"Error processing history data: {e}")
            # Keep zeros as fallback
    
    # Combine history and forecasts
    combined_mean = np.vstack((history_data, YYftr_m_trans))
    combined_median = np.vstack((history_data, YYftr_med_trans))
    combined_095 = np.vstack((history_data, YYftr_095_trans))
    combined_005 = np.vstack((history_data, YYftr_005_trans))
    combined_084 = np.vstack((history_data, YYftr_084_trans))
    combined_016 = np.vstack((history_data, YYftr_016_trans))
    
    # Create forecast index
    forecast_index = index[-H_-hist_length:]
    
    # Create DataFrames for all statistics
    YY_mean_pd = pd.DataFrame(combined_mean, columns=self.varlist[:combined_mean.shape[1]], index=forecast_index)
    YY_med_pd = pd.DataFrame(combined_median, columns=self.varlist[:combined_median.shape[1]], index=forecast_index)
    YY_095_pd = pd.DataFrame(combined_095, columns=self.varlist[:combined_095.shape[1]], index=forecast_index)
    YY_005_pd = pd.DataFrame(combined_005, columns=self.varlist[:combined_005.shape[1]], index=forecast_index)
    YY_084_pd = pd.DataFrame(combined_084, columns=self.varlist[:combined_084.shape[1]], index=forecast_index)
    YY_016_pd = pd.DataFrame(combined_016, columns=self.varlist[:combined_016.shape[1]], index=forecast_index)
    
    # Store results in self
    self.YY_mean_pd = YY_mean_pd
    self.YY_med_pd = YY_med_pd
    self.YY_095_pd = YY_095_pd
    self.YY_005_pd = YY_005_pd
    self.YY_084_pd = YY_084_pd
    self.YY_016_pd = YY_016_pd
    
    # Store the index
    self.index_list[-1] = index
    
    print("Forecasting complete. Mean, median, and percentile forecasts stored.")
    
    return None
        
def aggregate(self, frequency, reset_index=True):
    '''
    Aggregates the Mean, Median and quantiles in the highest frequency to the desired frequency.
    The Function ensures that we start at the beginning of a Year or Quarter depending on the chosen frequency.
    
    Parameters
    ----------
    frequency : str
        The frequency to which the data should be aggregated to ('Q' or 'Y')
    reset_index : boolean
        Should index be changed to period Index
    '''
    # Check if forecasts exist
    required_attrs = ['YY_mean_pd']
    if not all(hasattr(self, attr) for attr in required_attrs):
        sys.exit("Error: To aggregate, generate forecasts first")
    
    # Validate frequency
    if frequency not in ["Y", "Q"]:
        sys.exit("Error: Aggregation currently only implemented for aggregation to yearly and quarterly frequency")
    
    # Get frequency info
    freq_lf = frequency
    freq_hf = self.frequencies[-1]  # Highest frequency (Weekly)
    
    # Store forecast DataFrames by percentile type
    forecast_dfs = {
        'mean': self.YY_mean_pd,
        'median': self.YY_med_pd if hasattr(self, 'YY_med_pd') else None,
        'p095': self.YY_095_pd if hasattr(self, 'YY_095_pd') else None,
        'p005': self.YY_005_pd if hasattr(self, 'YY_005_pd') else None,
        'p084': self.YY_084_pd if hasattr(self, 'YY_084_pd') else None,
        'p016': self.YY_016_pd if hasattr(self, 'YY_016_pd') else None
    }
    
    # Check which percentiles are available
    available_forecasts = [k for k, v in forecast_dfs.items() if v is not None]
    print(f"Available forecast types: {available_forecasts}")
    
    # Variable counts
    Nw = self.Nw  # Weekly variables
    Nm = self.Nm  # Monthly variables 
    Nq = self.Nq  # Quarterly variables
    
    # Helper functions for finding aggregation starting points
    def find_first_position(arr, numbers, count):
        for i in range(len(arr) - count + 1):
            if arr[i] in numbers and all(arr[i] == arr[j] for j in range(i+1, i+count)):
                return i
        return 0  # Fallback if pattern not found
            
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
                start = find_first_position(df.index.month, [1, 4, 7, 10], 4)
            elif freq_hf == 'M':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 1)
            elif freq_hf == 'D':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 20)
        elif frequency == 'Y':
            if freq_hf == 'Q':
                start = find_first_position(df.index.month, [1, 3], 1)
            elif freq_hf == 'M':
                start = find_first_position(df.index.month, [1], 1)
            elif freq_hf == 'W':
                start = find_first_position(df.index.month, [1], 4)
            elif freq_hf == 'D':
                start = find_first_position(df.index.month, [1], 20)
        return freq_ratio, start
    
    # Calculate frequency ratio and starting point using mean forecast
    freq_ratio, start = agg_helper(freq_lf, freq_hf, forecast_dfs['mean'])
    
    # Dictionary to store aggregated forecasts
    aggregated_dfs = {}
    
    # Aggregate each forecast type
    print("Aggregating forecasts")
    for name, df in forecast_dfs.items():
        if df is None:
            continue
            
        print(f"  Aggregating {name} forecast...")
            
        # Filter to ensure complete periods
        temp = df.iloc[start:].groupby(df.iloc[start:].reset_index().index // freq_ratio).filter(
            lambda x: len(x) == freq_ratio)
        
        # Aggregate according to specified method
        agg_method = self.temp_agg if hasattr(self, 'temp_agg') else "mean"
        
        if agg_method == "mean":
            temp = temp.groupby(temp.reset_index().index // freq_ratio).mean()
        elif agg_method == "sum":
            temp = temp.groupby(temp.reset_index().index // freq_ratio).sum()
        
        # Set the index dates to the first day of the period for consistency
        temp.index = df.iloc[start:].index[::freq_ratio][:temp.shape[0]]
        temp.index = temp.index.map(lambda x: x.replace(day=1))
        
        # Store the aggregated forecast
        aggregated_dfs[name] = temp
    
    # Store aggregated forecasts as attributes
    self.YY_mean_agg = aggregated_dfs.get('mean')
    self.YY_median_agg = aggregated_dfs.get('median')
    self.YY_095_agg = aggregated_dfs.get('p095')
    self.YY_005_agg = aggregated_dfs.get('p005')
    self.YY_084_agg = aggregated_dfs.get('p084')
    self.YY_016_agg = aggregated_dfs.get('p016')
    
    # Also store in a unified dictionary for easier access
    self.aggregated_percentiles = {
        'Mean': self.YY_mean_agg,
        'Median': self.YY_median_agg,
        '95th Percentile': self.YY_095_agg,
        '5th Percentile': self.YY_005_agg,
        '84th Percentile': self.YY_084_agg,
        '16th Percentile': self.YY_016_agg
    }
    
    # Store the main aggregated forecast for backwards compatibility
    self.aggregated_forecast = self.YY_mean_agg
    
    # Replace forecast with historical data where available
    hist = []
    hist_dfs = []  # Store as pandas DataFrames
    
    # Get original data for each frequency and convert to DataFrame if needed
    if hasattr(self, 'input_data_W'):
        hist.append(self.input_data_W)
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(self.input_data_W, np.ndarray):
            print(f"Converting weekly data from numpy array to DataFrame")
            # Create a DataFrame with the right index and columns
            freq_index = self.index_list[-1][-self.input_data_W.shape[0]:] if hasattr(self, 'index_list') else None
            cols = self.varlist[:self.input_data_W.shape[1]] if hasattr(self, 'varlist') else None
            
            if freq_index is not None and cols is not None and len(freq_index) == self.input_data_W.shape[0]:
                hist_df = pd.DataFrame(self.input_data_W, index=freq_index, columns=cols[:self.Nw])
            else:
                # If we can't determine the right index/columns, use default ones
                hist_df = pd.DataFrame(self.input_data_W)
        else:
            # Already a DataFrame
            hist_df = self.input_data_W
        
        hist_dfs.append(hist_df)
    
    if hasattr(self, 'input_data_M'):
        hist.append(self.input_data_M)
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(self.input_data_M, np.ndarray):
            print(f"Converting monthly data from numpy array to DataFrame")
            # Create a DataFrame with appropriate index and columns
            # For monthly data, use a monthly index derived from the weekly index
            weekly_idx = self.index_list[-1] if hasattr(self, 'index_list') else None
            monthly_idx = pd.date_range(
                start=weekly_idx[0] if weekly_idx is not None else '2020-01-01', 
                periods=self.input_data_M.shape[0], 
                freq='MS'
            ) if weekly_idx is not None else None
            
            cols = self.varlist[self.Nw:self.Nw+self.Nm] if hasattr(self, 'varlist') else None
            
            if monthly_idx is not None and cols is not None:
                hist_df = pd.DataFrame(self.input_data_M, index=monthly_idx, columns=cols)
            else:
                # If we can't determine the right index/columns, use default ones
                hist_df = pd.DataFrame(self.input_data_M)
        else:
            # Already a DataFrame
            hist_df = self.input_data_M
        
        hist_dfs.append(hist_df)
    
    if hasattr(self, 'input_data_Q'):
        hist.append(self.input_data_Q)
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(self.input_data_Q, np.ndarray):
            print(f"Converting quarterly data from numpy array to DataFrame")
            # Create a DataFrame with appropriate index and columns
            # For quarterly data, use a quarterly index derived from the weekly index
            weekly_idx = self.index_list[-1] if hasattr(self, 'index_list') else None
            quarterly_idx = pd.date_range(
                start=weekly_idx[0] if weekly_idx is not None else '2020-01-01', 
                periods=self.input_data_Q.shape[0], 
                freq='QS'
            ) if weekly_idx is not None else None
            
            cols = self.varlist[self.Nw+self.Nm:] if hasattr(self, 'varlist') else None
            
            if quarterly_idx is not None and cols is not None:
                hist_df = pd.DataFrame(self.input_data_Q, index=quarterly_idx, columns=cols)
            else:
                # If we can't determine the right index/columns, use default ones
                hist_df = pd.DataFrame(self.input_data_Q)
        else:
            # Already a DataFrame
            hist_df = self.input_data_Q
        
        hist_dfs.append(hist_df)
    
    # Process each frequency using the DataFrame versions
    for i, freq in enumerate(self.frequencies):
        if i >= len(hist_dfs):
            continue
            
        # Use the DataFrame version for operations
        hist_df = hist_dfs[i]
        
        # If this is the target frequency
        if freq == frequency:
            try:
                # Replace forecasts with actual data where available
                idx = self.YY_mean_agg.index.intersection(hist_df.index, sort=False)
                cols = [col for col in hist_df.columns if col in self.YY_mean_agg.columns]
                
                if len(idx) > 0 and len(cols) > 0:
                    self.YY_mean_agg.loc[idx, cols] = hist_df.loc[idx, cols]
                    
                    if self.YY_median_agg is not None:
                        self.YY_median_agg.loc[idx, cols] = hist_df.loc[idx, cols]
                        
                    # Set percentile forecasts to NaN for historical periods
                    for perc_attr in ['YY_095_agg', 'YY_005_agg', 'YY_084_agg', 'YY_016_agg']:
                        if hasattr(self, perc_attr) and getattr(self, perc_attr) is not None:
                            getattr(self, perc_attr).loc[idx, cols] = np.nan
            except Exception as e:
                print(f"Error replacing forecasts with actual {freq} data: {e}")
            
        # For higher frequencies than the target frequency, aggregate the data
        elif self.frequencies.index(freq) > self.frequencies.index(frequency):
            try:
                freq_ratio_temp, start_temp = agg_helper(frequency, freq, hist_df)
                
                if len(hist_df) > start_temp:
                    hist_agg = hist_df.iloc[start_temp:].groupby(hist_df.iloc[start_temp:].reset_index().index // freq_ratio_temp).filter(lambda x: len(x) == freq_ratio_temp)
                    
                    if len(hist_agg) > 0:
                        agg_method = self.temp_agg if hasattr(self, 'temp_agg') else "mean"
                        
                        if agg_method == 'mean':
                            hist_agg = hist_agg.groupby(hist_agg.reset_index().index // freq_ratio_temp).mean()
                        elif agg_method == 'sum':
                            hist_agg = hist_agg.groupby(hist_agg.reset_index().index // freq_ratio_temp).sum()
                        
                        hist_agg.index = hist_df.iloc[start_temp:].index[::freq_ratio_temp][:hist_agg.shape[0]]
                        hist_agg.index = hist_agg.index.map(lambda x: x.replace(day=1))
                        
                        idx = hist_agg.index.intersection(self.YY_mean_agg.index, sort=False)
                        cols = [col for col in hist_agg.columns if col in self.YY_mean_agg.columns]
                        
                        if len(idx) > 0 and len(cols) > 0:
                            self.YY_mean_agg.loc[idx, cols] = hist_agg.loc[idx, cols]
                            
                            if self.YY_median_agg is not None:
                                self.YY_median_agg.loc[idx, cols] = hist_agg.loc[idx, cols]
                                
                            # Set percentile forecasts to NaN for historical periods
                            for perc_attr in ['YY_095_agg', 'YY_005_agg', 'YY_084_agg', 'YY_016_agg']:
                                if hasattr(self, perc_attr) and getattr(self, perc_attr) is not None:
                                    getattr(self, perc_attr).loc[idx, cols] = np.nan
            except Exception as e:
                print(f"Error aggregating {freq} data: {e}")
    
    # Reset index to period format if requested
    if reset_index:
        index_new = pd.PeriodIndex(self.YY_mean_agg.index, freq=frequency)
        self.YY_mean_agg.index = index_new
        
        for attr_name in ['YY_median_agg', 'YY_095_agg', 'YY_005_agg', 'YY_084_agg', 'YY_016_agg']:
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                getattr(self, attr_name).index = index_new
    
    # Filter to variables of interest if specified
    if hasattr(self, 'var_of_interest') and self.var_of_interest is not None:
        try:
            # Adjust this for the unified model structure
            weekly_vars = []
            if hasattr(self, 'input_data_W'):
                if isinstance(self.input_data_W, pd.DataFrame):
                    weekly_vars = list(self.input_data_W.columns)
                else:
                    # If input_data_W is not a DataFrame, try to use varlist
                    weekly_vars = self.varlist[:self.Nw] if hasattr(self, 'varlist') else []
                    
            var_all = list(weekly_vars) + list(self.var_of_interest)
            
            idx_var_of_interest = [i for i, col in enumerate(self.YY_mean_agg.columns) if col in var_all]
            
            if idx_var_of_interest:
                for attr_name in ['YY_mean_agg', 'YY_median_agg', 'YY_095_agg', 'YY_005_agg', 'YY_084_agg', 'YY_016_agg']:
                    if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                        setattr(self, attr_name, getattr(self, attr_name).iloc[:, idx_var_of_interest])
        except Exception as e:
            print(f"Error filtering to variables of interest: {e}")
    
    # Store aggregation frequency
    self.agg_freq = frequency
    self.aggregation_frequency = frequency  # For compatibility
    
    return None