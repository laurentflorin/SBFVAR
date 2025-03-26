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
    self.var_of_interest = var_of_interest
    
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
    
    # Print data dimensions for verification
    print(f"Data dimensions - Weekly: {YW.shape}, Monthly: {YM.shape}, Quarterly: {YQ.shape}")
    
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
    
    # Define starting indices for different components
    monthly_start = weekly_block_size
    quarterly_start = monthly_start + monthly_latent_size
    
    # Print state vector dimensions for debugging
    print(f"State vector dimensions - Weekly: {Nw}, Monthly: {Nm}, Quarterly: {Nq}")
    print(f"State vector structure - Weekly block: {weekly_block_size}, Monthly latent: {monthly_latent_size}, Quarterly latent: {quarterly_latent_size}")
    print(f"State vector indices - Monthly start: {monthly_start}, Quarterly start: {quarterly_start}")
    
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
    
    # Initialize the latent states for monthly variables
    # Weekly shocks affect the first week of each monthly variable's latent states
    F[monthly_start:monthly_start+Nm, :Nw] = np.eye(Nm, Nw)  # Map shocks from weekly vars to monthly latent states
    
    # Shift weeks within each month for the latent states
    for i in range(rmw-1):
        src_pos = monthly_start + i*Nm
        dest_pos = monthly_start + (i+1)*Nm
        F[dest_pos:dest_pos+Nm, src_pos:src_pos+Nm] = np.eye(Nm)
    
    # 3. Latent state transitions for quarterly variables
    
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
    
    # IMPROVED MEASUREMENT EQUATIONS
    # -----------------------------
    
    # 1. Measurement matrix for weekly variables (direct observation)
    H_w = np.zeros((Nw, nstate))
    H_w[:, :Nw] = np.eye(Nw)  # Directly observe first Nw state elements
    
    # 2. Measurement matrix for monthly variables (temporal aggregation)
    H_m = np.zeros((Nm, nstate))
    
    # Monthly variables are observed as the temporal aggregation of their latent weekly states
    # Each month's observation constrains all 4 weeks of latents for that month's variables
    for m in range(Nm):
        for w in range(rmw):
            state_idx = monthly_start + w*Nm + m
            if self.temp_agg == 'mean':
                H_m[m, state_idx] = 1.0/rmw  # Average of 4 weeks
            else:  # 'sum'
                H_m[m, state_idx] = 1.0  # Sum of 4 weeks
    
    # 3. Measurement matrix for quarterly variables (temporal aggregation)
    H_q = np.zeros((Nq, nstate))
    
    # Quarterly variables are observed as the temporal aggregation of their latent weekly states
    # Each quarter's observation constrains all 12 weeks of latents for that quarter's variables
    for q in range(Nq):
        for w in range(rqw):
            state_idx = quarterly_start + w*Nq + q
            if self.temp_agg == 'mean':
                H_q[q, state_idx] = 1.0/rqw  # Average of 12 weeks
            else:  # 'sum'
                H_q[q, state_idx] = 1.0  # Sum of 12 weeks
    
    # Add validation prints for measurement matrices
    for m in range(min(3, Nm)):
        m_indices = np.where(H_m[m, :] != 0)[0]
        print(f"Monthly var {m+1} uses {len(m_indices)} state indices: {m_indices[:5]}...")
    
    for q in range(min(3, Nq)):
        q_indices = np.where(H_q[q, :] != 0)[0]
        print(f"Quarterly var {q+1} uses {len(q_indices)} state indices: {q_indices[:5]}...")
    
    # Initialize state and covariance matrices
    Q = np.zeros((nstate, nstate))
    Q[:Nw, :Nw] = 1e-4 * np.eye(Nw)  # Process noise affects weekly states directly
    
    a_t = np.zeros(nstate)
    P_t = np.eye(nstate)
    
    # IMPROVED INITIALIZATION: Initialize latent states with actual data
    # -----------------------------------------------------------------
    
    # Initialize monthly latent states with available data
    if YM.shape[0] > 0:
        # Use the first available monthly observation to initialize latent states
        m_values = YM[0, :]
        for m in range(Nm):
            for w in range(rmw):
                a_t[monthly_start + w*Nm + m] = m_values[m]
    
    # Initialize quarterly latent states with available data
    if YQ.shape[0] > 0:
        # Use the first available quarterly observation to initialize latent states
        q_values = YQ[0, :]
        for q in range(Nq):
            for w in range(rqw):
                a_t[quarterly_start + w*Nq + q] = q_values[q]
    
    print("Initialized latent states with actual monthly and quarterly data")
    
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
        # If it's the first iteration, use the initialized state and covariance
        # Otherwise, use the last state from the previous draw
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
            
            # Current week index
            w_idx = T0 + t
            
            # IMPROVED OBSERVATION MATCHING: Better identify month and quarter ends
            # Determine which measurements are available
            # We need month and quarter end points - these are when aggregated observations are available
            is_month_end = ((w_idx + 1) % rmw == 0)
            is_quarter_end = ((w_idx + 1) % rqw == 0)
            
            # Get current month and quarter indices
            m_idx = (w_idx + 1) // rmw - 1 if is_month_end else -1
            q_idx = (w_idx + 1) // rqw - 1 if is_quarter_end else -1
            
            # If first iteration of first draw, add debug info
            if j == 0 and t < 5:
                print(f"Week {w_idx}: month_end={is_month_end}, quarter_end={is_quarter_end}, m_idx={m_idx}, q_idx={q_idx}")
            
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
                        if j == 0 and t < 3:
                            print(f"  Adding weekly observation: {w_data}")
                except (IndexError, ValueError) as e:
                    if t == 0 and j == 0:
                        print(f"DEBUG: Error processing weekly data at w_idx={w_idx}: {e}")
            
            # Check for monthly observations - only available at the end of month
            if is_month_end and 0 <= m_idx < YM.shape[0]:
                H_matrices.append(H_m)
                y_obs.append(YM[m_idx])
                if j == 0 and t < 3:
                    print(f"  Adding monthly observation: {YM[m_idx]}")
            
            # Check for quarterly observations - only available at the end of quarter
            if is_quarter_end and 0 <= q_idx < YQ.shape[0]:
                H_matrices.append(H_q)
                y_obs.append(YQ[q_idx])
                if j == 0 and t < 3:
                    print(f"  Adding quarterly observation: {YQ[q_idx]}")
            
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
            else:
                # No valid observations - just use prediction
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
        
        # Add validation of temporal aggregation at end of first draw
        if j == 0:
            # Verify monthly aggregation
            print("\nValidating monthly aggregation:")
            for m in range(min(2, Nm)):  # Check first two monthly variables
                for t_check in range(0, min(24, nobs), rmw):  # Check a few months
                    month_idx = (T0 + t_check) // rmw
                    if month_idx < Tm:
                        # Get the 4 weeks of latent states for this month
                        latent_values = [a_draws[j, t_check + w, monthly_start + w*Nm + m] for w in range(min(rmw, nobs-t_check))]
                        
                        # Calculate aggregated value
                        if self.temp_agg == 'mean':
                            agg_value = np.mean(latent_values)
                        else:
                            agg_value = np.sum(latent_values)
                            
                        # Compare with observed monthly value
                        obs_value = YM[month_idx, m]
                        print(f"Month {month_idx+1}, Var {m+1}: Agg={agg_value:.6f}, Obs={obs_value:.6f}, Error={abs(agg_value-obs_value):.6f}")
            
            # Verify quarterly aggregation
            print("\nValidating quarterly aggregation:")
            for q in range(min(2, Nq)):  # Check first two quarterly variables
                for t_check in range(0, min(24, nobs), rqw):  # Check a few quarters
                    quarter_idx = (T0 + t_check) // rqw
                    if quarter_idx < Tq:
                        # Get the 12 weeks of latent states for this quarter
                        latent_values = [a_draws[j, t_check + w, quarterly_start + w*Nq + q] for w in range(min(rqw, nobs-t_check))]
                        
                        # Calculate aggregated value
                        if self.temp_agg == 'mean':
                            agg_value = np.mean(latent_values)
                        else:
                            agg_value = np.sum(latent_values)
                            
                        # Compare with observed quarterly value
                        obs_value = YQ[quarter_idx, q]
                        print(f"Quarter {quarter_idx+1}, Var {q+1}: Agg={agg_value:.6f}, Obs={obs_value:.6f}, Error={abs(agg_value-obs_value):.6f}")
                        print(f"  Latent values: {[f'{v:.6f}' for v in latent_values[:6]]}...")
        
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
            j_temp = int(j/self.thining)
            if j == 0:
                # Initialize storage on first iteration
                YYactsim_list = [np.zeros((math.ceil((self.nsim)/self.thining), rmw, Ntotal))]
                XXactsim_list = [np.zeros((math.ceil((self.nsim)/self.thining), rmw, Nw*p+1))]  # Only for weekly VAR
            
            # Store the combined data for later use
            YYactsim_list[0][j_temp, :, :] = np.hstack((W_smooth[-rmw:, :], M_smooth[-rmw:, :], Q_smooth[-rmw:, :]))
            XXactsim_list[0][j_temp, :, :] = np.hstack((Z[-rmw:, :], np.ones((rmw, 1))))
        
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

    # Store the state space model components
    self.F = F
    self.c = c
    self.H_w = H_w
    self.H_m = H_m
    self.H_q = H_q
    self.Q = Q
    self.monthly_start = monthly_start
    self.quarterly_start = quarterly_start

    # Store indexes
    self.index_list = index_list

    return None

def check_state_consistency(state, Nw, Nm, Nq, monthly_start, quarterly_start, rmw, rqw):
    """
    Check consistency of latent state variables across frequencies
    
    Parameters
    ----------
    state : ndarray
        State vector
    Nw, Nm, Nq : int
        Number of variables at each frequency
    monthly_start, quarterly_start : int
        Starting indices for monthly and quarterly latent variables
    rmw, rqw : int
        Frequency ratios
    
    Returns
    -------
    bool
        True if state is consistent, False otherwise
    """
    issues = []
    
    # Check if weekly variables are within reasonable bounds
    if np.any(np.abs(state[:Nw]) > 10):
        issues.append(f"Weekly variables have extreme values: {state[:Nw]}")
    
    # Check monthly latent variables
    for m in range(Nm):
        # Get all latent weekly states for this monthly variable
        monthly_latents = [state[monthly_start + w*Nm + m] for w in range(rmw)]
        monthly_mean = np.mean(monthly_latents)
        
        # Check if variance among weekly states is reasonable
        monthly_std = np.std(monthly_latents)
        if monthly_std > 0.5:  # Arbitrary threshold
            issues.append(f"High variance in monthly var {m+1} latent states: {monthly_std:.4f}")
        
        # Check for extreme values
        if np.any(np.abs(monthly_latents) > 10):
            issues.append(f"Extreme values in monthly var {m+1} latent states: {monthly_latents}")
    
    # Check quarterly latent variables
    for q in range(Nq):
        # Get all latent weekly states for this quarterly variable
        quarterly_latents = [state[quarterly_start + w*Nq + q] for w in range(rqw)]
        quarterly_mean = np.mean(quarterly_latents)
        
        # Check if variance among weekly states is reasonable
        quarterly_std = np.std(quarterly_latents)
        if quarterly_std > 0.5:  # Arbitrary threshold
            issues.append(f"High variance in quarterly var {q+1} latent states: {quarterly_std:.4f}")
        
        # Check for extreme values
        if np.any(np.abs(quarterly_latents) > 10):
            issues.append(f"Extreme values in quarterly var {q+1} latent states: {quarterly_latents}")
    
    # Log any issues
    if issues:
        print("State consistency check found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


def forecast(self, H, conditionals=None):
    '''
    Method to generate the forecasts in the highest frequency (weekly).
    Handles proper temporal aggregation across all three frequencies.
    
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
    
    # Check that all required components from fit() are available
    required_attrs = ['F', 'c', 'H_w', 'H_m', 'H_q', 'Q', 'monthly_start', 'quarterly_start', 'a_draws']
    missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
    if missing_attrs:
        raise ValueError(f"Missing required attributes from fit(): {missing_attrs}")
    
    # Extend the index for the forecast period
    index = copy.deepcopy(self.index_list[-1])
    
    # Index extension logic
    if self.frequencies[-1] == 'W':
        index = index.append(pd.date_range(start=index[-1] + Week(), periods=H, freq='W-MON'))

        # Ensure we have the correct number of weeks per month
        def has_more_than_4_weeks(month, dti):
            return sum(dti.to_period('M') == month) > 4

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
    
    # Get key dimensions
    Nw = self.Nw      # Number of weekly variables
    Nm = self.Nm      # Number of monthly variables
    Nq = self.Nq      # Number of quarterly variables
    Ntotal = Nw + Nm + Nq  # Total variables
    
    # Get frequency ratios
    rqm = self.freq_ratio[0]  # Quarterly to monthly ratio (3)
    rmw = self.freq_ratio[1]  # Monthly to weekly ratio (4)
    rqw = rqm * rmw          # Quarterly to weekly ratio (12)
    
    print(f"Forecasting with frequency ratios - Q:M = {rqm}, M:W = {rmw}, Q:W = {rqw}")
    
    # State vector indices
    p = self.nlags  # VAR lags
    weekly_block_size = Nw * p
    monthly_start = self.monthly_start
    quarterly_start = self.quarterly_start
    
    # Initialize conditional forecasts with the right dimensions
    YYcond = pd.DataFrame(np.nan, index=index[-H:], columns=self.varlist)
    
    # Process conditionals if provided
    if conditionals is not None:
        print(f"Processing conditional forecasts for {len(conditionals.columns)} variables")
        
        # Map conditional periods to forecast periods
        conditionals.index = YYcond.index[:len(conditionals.index)]
        
        # Update with conditional values
        for col in conditionals.columns:
            if col in YYcond.columns:
                YYcond[col] = conditionals[col]
            else:
                print(f"Warning: Conditional variable '{col}' not found in model")
        
        # Apply transformations based on variable type
        YYcond_array = np.array(YYcond)
        for i, var in enumerate(self.varlist):
            if i < len(self.select) and not np.isnan(YYcond_array[:, i]).all():
                if self.select[i] == 1:  # Growth rate
                    YYcond_array[:, i] = YYcond_array[:, i] / 100
                else:  # Level
                    YYcond_array[:, i] = np.log(YYcond_array[:, i])
        
        YYcond = YYcond_array
    else:
        # No conditionals provided
        YYcond = np.full((H, Ntotal), np.nan)
    
    # Create the conditional indicator array (True where we have conditionals)
    exc = ~np.isnan(YYcond)
    
    # Get the latest state vector
    if hasattr(self, 'a_draws') and self.a_draws.shape[0] >= len(self.valid_draws):
        # Average the last state across valid draws
        last_states = np.mean(self.a_draws[self.valid_draws, -1, :], axis=0)
    else:
        # Initialize with zeros if no states available
        print("Warning: No valid state draws found, initializing with zeros")
        nstate = weekly_block_size + Nm * rmw + Nq * rqw
        last_states = np.zeros(nstate)
    
    # Get state transition components
    F = self.F  # State transition matrix
    c = self.c  # Constant vector
    Q = self.Q  # System covariance
    
    # Measurement matrices 
    H_w = self.H_w  # Weekly observation matrix
    H_m = self.H_m  # Monthly observation matrix
    H_q = self.H_q  # Quarterly observation matrix
    
    # Full measurement matrix for all variables
    H_full = np.vstack([H_w, H_m, H_q])
    
    # Storage for forecasts for all valid draws
    H_ = int(self.H)
    YYvector_all = np.zeros((len(self.valid_draws), H_, Ntotal))
    
    print(" ", end='\n')
    print("Multi-Frequency BVAR: Forecasting with unified model", end="\n")
    print(f"Forecast Horizon: {H_} weeks", end="\n")
    print(f"Valid Draws: {len(self.valid_draws)}")
    
    # Main forecast loop over all valid draws
    for idx, jj in enumerate(tqdm(self.valid_draws)):
        # Extract VAR parameters for this draw
        post_phi = np.squeeze(self.Phip[jj, :, :])  # Weekly VAR coefficients
        post_sig = np.squeeze(self.Sigmap[jj, :, :])  # Full covariance matrix
        
        # Update state transition matrix with this draw's VAR coefficients
        F_draw = F.copy()
        F_draw[:Nw, :Nw*p] = post_phi[:-1, :].T  # VAR coefficients
        
        # Constant vector for this draw
        # Constant vector for this draw
        c_draw = c.copy()

        # Get the shape of the target array
        target_shape = c_draw[:Nw].shape
        source_data = post_phi[-1, :]

        # Reshape source data to match target shape
        if len(target_shape) > 1:  # If target is multi-dimensional
            if target_shape[1] == 1:  # Column vector
                reshaped_constants = source_data.reshape(-1, 1)
            else:  # Row vector or matrix
                reshaped_constants = source_data.reshape(1, -1)
        else:  # If target is 1D
            reshaped_constants = source_data

        # Assign with proper shape
        c_draw[:Nw] = reshaped_constants
        
        # Initialize state vector for forecasting
        state_fcst = np.zeros((H_+1, len(last_states)))
        state_fcst[0, :] = last_states  # Set initial state
        
        # Initialize observed variables vector
        YYpred = np.zeros((H_+1, Ntotal))
        YYpred[0, :] = H_full @ last_states  # Initial forecast
        
        # Generate forecast innovations for weekly variables
        error_pred = np.zeros((H_+1, Nw))
        for h in range(1, H_+1):  # Skip the initial values
            if post_sig.size > 1:
                # Draw from weekly covariance matrix
                error_pred[h, :] = np.random.default_rng().multivariate_normal(
                    mean=np.zeros(Nw), cov=post_sig[:Nw, :Nw], method="cholesky")
            else:
                # Fallback for scalar covariance
                error_pred[h, :] = np.random.default_rng().normal(
                    loc=0, scale=post_sig, size=Nw)
        
        # Iterate forward to construct forecasts
        for h in range(1, H_+1):
            try:
                # State transition equation
                next_state = F_draw @ state_fcst[h-1, :] + c_draw.flatten()
                
                # Add innovations to weekly variables only
                next_state[:Nw] += error_pred[h, :]
                
                # Store the state
                state_fcst[h, :] = next_state
                
                # Measurement equation to get observed variables
                model_forecast = H_full @ next_state
                
                # Apply conditionals where available
                if h-1 < exc.shape[0]:
                    for i in range(Ntotal):
                        if i < exc.shape[1] and exc[h-1, i]:
                            # Override with conditional value
                            YYpred[h, i] = YYcond[h-1, i]
                        else:
                            # Use model forecast
                            YYpred[h, i] = model_forecast[i]
                else:
                    # No conditionals for this period
                    YYpred[h, :] = model_forecast
                    
            except Exception as e:
                print(f"ERROR at h={h}: {e}")
                # Output debug info
                print(f"state_fcst[h-1] shape: {state_fcst[h-1].shape}, F_draw shape: {F_draw.shape}")
                continue
        
        # Store forecasts (excluding the initial values)
        YYvector_all[idx, :, :] = YYpred[1:, :]
    
    # Store all forecast draws
    self.forecast_draws_list = [YYvector_all]
    
    # Check if we have any valid forecasts
    if len(self.valid_draws) == 0:
        print("ERROR: No valid forecast draws available!")
        return None
    
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
        if i < len(self.select):
            if self.select[i] == 1:  # Growth rate
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
    # Extract historical data for plots - use full history
    hist_length = len(index) - H_  # Use all available historical periods
    history_data = self.extract_historical_data(hist_length, Nw, Nm, Nq)
    
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
    YY_mean_pd = pd.DataFrame(combined_mean, columns=self.varlist, index=forecast_index)
    YY_med_pd = pd.DataFrame(combined_median, columns=self.varlist, index=forecast_index)
    YY_095_pd = pd.DataFrame(combined_095, columns=self.varlist, index=forecast_index)
    YY_005_pd = pd.DataFrame(combined_005, columns=self.varlist, index=forecast_index)
    YY_084_pd = pd.DataFrame(combined_084, columns=self.varlist, index=forecast_index)
    YY_016_pd = pd.DataFrame(combined_016, columns=self.varlist, index=forecast_index)
    
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
    if frequency not in ["Y", "Q", "M"]:
        sys.exit("Error: Aggregation currently only implemented for yearly, quarterly, and monthly frequencies")
    
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
    
    # Improved aggregation helper that handles three frequencies        
    def agg_helper(freq_target, freq_source, df, intermediate_freq=None):
        """
        Improved aggregation helper that can handle intermediate frequencies
        
        Parameters
        ----------
        freq_target : str
            Target frequency (e.g., 'Q')
        freq_source : str
            Source frequency (e.g., 'W')
        df : DataFrame
            Data to aggregate
        intermediate_freq : str, optional
            Intermediate frequency for three-tier aggregation (e.g., 'M')
        
        Returns
        -------
        tuple
            (freq_ratio, start_position)
        """
        # Direct frequency ratios
        freq_ratios = {
            ('Q', 'M'): 3,
            ('Y', 'Q'): 4,
            ('Y', 'M'): 12,
            ('M', 'W'): 4,
            ('Q', 'W'): 12,  # 3*4
            ('Y', 'W'): 48,  # 12*4
        }
        
        # Get direct ratio if available
        freq_ratio = freq_ratios.get((freq_target, freq_source), None)
        
        # If direct ratio not available but intermediate freq provided, use compound ratio
        if freq_ratio is None and intermediate_freq is not None:
            ratio1 = freq_ratios.get((freq_target, intermediate_freq), None)
            ratio2 = freq_ratios.get((intermediate_freq, freq_source), None)
            
            if ratio1 is not None and ratio2 is not None:
                freq_ratio = ratio1 * ratio2
        
        # If still no ratio, raise error
        if freq_ratio is None:
            raise ValueError(f"Cannot determine frequency ratio from {freq_source} to {freq_target}")
            
        # Determine starting position based on target frequency
        if freq_target == 'Q':
            if freq_source == 'W':
                # Find the first occurrence of a 4-week sequence starting in a quarter-start month
                start = find_first_position(df.index.month, [1, 4, 7, 10], 4)
            elif freq_source == 'M':
                # Find the first month that starts a quarter
                start = find_first_position(df.index.month, [1, 4, 7, 10], 1)
        elif freq_target == 'Y':
            if freq_source == 'W':
                # Find the first occurrence of a 4-week sequence starting in January
                start = find_first_position(df.index.month, [1], 4)
            elif freq_source == 'M':
                # Find the first January
                start = find_first_position(df.index.month, [1], 1)
            elif freq_source == 'Q':
                # Find the first Q1
                start = find_first_position(df.index.month, [1], 1)
        elif freq_target == 'M':
            if freq_source == 'W':
                # Find the start of a month
                # This is simplified - in reality need to check day of week and weeks in month
                start = 0
                for i in range(len(df.index) - 4):
                    # Check if this is the start of a month (simplified)
                    if df.index[i].day <= 7 and df.index[i+3].month == df.index[i].month:
                        start = i
                        break
                        
        return freq_ratio, start
    
    # Calculate frequency ratio and starting point using mean forecast
    # Use appropriate intermediate frequency if needed
    intermediate_freq = None
    if freq_lf == 'Q' and freq_hf == 'W':
        intermediate_freq = 'M'
    elif freq_lf == 'Y' and freq_hf == 'W':
        intermediate_freq = 'Q'
    
    freq_ratio, start = agg_helper(freq_lf, freq_hf, forecast_dfs['mean'], intermediate_freq)
    
    print(f"Aggregating from {freq_hf} to {freq_lf} with ratio {freq_ratio}, starting at position {start}")
    
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
    
    # Replace forecast with historical data where available
    self.replace_forecasts_with_history(frequency, intermediate_freq)
    
    # Reset index to period format if requested
    if reset_index:
        index_new = pd.PeriodIndex(self.YY_mean_agg.index, freq=frequency)
        self.YY_mean_agg.index = index_new
        
        for attr_name in ['YY_median_agg', 'YY_095_agg', 'YY_005_agg', 'YY_084_agg', 'YY_016_agg']:
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                getattr(self, attr_name).index = index_new
    
    # Filter to variables of interest if specified
    if hasattr(self, 'var_of_interest') and self.var_of_interest is not None:
        self.filter_to_variables_of_interest()
    
    # Store aggregation frequency
    self.agg_freq = frequency
    
    print(f"Aggregation to {frequency} complete")
    return None

def replace_forecasts_with_history(self, frequency, intermediate_freq=None):
    """
    Replace forecast values with actual historical data where available
    
    Parameters
    ----------
    frequency : str
        Target frequency for aggregation
    intermediate_freq : str, optional
        Intermediate frequency, if using three-tier aggregation
    """
    hist_dfs = []
    
    # Get original data for each frequency
    if hasattr(self, 'input_data_W'):
        hist_dfs.append(self.prepare_history_df(self.input_data_W, 'W', self.Nw, 0))
    
    if hasattr(self, 'input_data_M'):
        hist_dfs.append(self.prepare_history_df(self.input_data_M, 'M', self.Nm, self.Nw))
    
    if hasattr(self, 'input_data_Q'):
        hist_dfs.append(self.prepare_history_df(self.input_data_Q, 'Q', self.Nq, self.Nw + self.Nm))
    
    # Helper function for frequency-to-index mapping
    def freq_to_idx(freq):
        freq_map = {'Q': 0, 'M': 1, 'W': 2}
        return freq_map.get(freq, -1)
    
    # Process each frequency
    for i, hist_df in enumerate(hist_dfs):
        if hist_df is None:
            continue
            
        freq = self.frequencies[i]
        
        # If this is the target frequency, directly use the data
        if freq == frequency:
            self.replace_with_history(hist_df)
        
        # If this frequency is higher than target but lower than highest,
        # aggregate it to target frequency
        elif (freq_to_idx(freq) > freq_to_idx(frequency) and 
              freq_to_idx(freq) < freq_to_idx(self.frequencies[-1])):
            
            # Aggregate to target frequency
            agg_df = self.aggregate_df(hist_df, freq, frequency, intermediate_freq)
            if agg_df is not None:
                self.replace_with_history(agg_df)
    
    return None

def prepare_history_df(self, data, freq, n_vars, start_idx):
    """
    Prepare historical data as DataFrame with proper index and columns
    
    Parameters
    ----------
    data : ndarray or DataFrame
        Historical data
    freq : str
        Frequency of the data
    n_vars : int
        Number of variables at this frequency
    start_idx : int
        Starting index in the overall variable list
    
    Returns
    -------
    DataFrame or None
        Prepared historical data, or None if invalid
    """
    if data is None or (hasattr(data, 'size') and data.size == 0):
        return None
        
    if isinstance(data, np.ndarray):
        # Create index based on frequency
        if freq == 'W':
            date_index = self.index_list[-1][-data.shape[0]:] if hasattr(self, 'index_list') else None
        elif freq == 'M':
            # Create monthly index derived from weekly
            if hasattr(self, 'index_list') and self.index_list:
                weekly_idx = self.index_list[-1] 
                date_index = pd.date_range(start=weekly_idx[0], periods=data.shape[0], freq='MS')
            else:
                date_index = None
        elif freq == 'Q':
            # Create quarterly index
            if hasattr(self, 'index_list') and self.index_list:
                weekly_idx = self.index_list[-1]
                date_index = pd.date_range(start=weekly_idx[0], periods=data.shape[0], freq='QS')
            else:
                date_index = None
        
        # Get variable names if available
        if hasattr(self, 'varlist') and len(self.varlist) >= start_idx + n_vars:
            columns = self.varlist[start_idx:start_idx + n_vars]
        else:
            columns = [f"{freq}_{i+1}" for i in range(n_vars)]
        
        # Create DataFrame
        if date_index is not None and len(date_index) == data.shape[0]:
            return pd.DataFrame(data, index=date_index, columns=columns)
        else:
            return pd.DataFrame(data, columns=columns)
    else:
        # Already a DataFrame, ensure it has proper columns
        if hasattr(self, 'varlist') and len(self.varlist) >= start_idx + n_vars:
            data.columns = self.varlist[start_idx:start_idx + n_vars]
        return data

def replace_with_history(self, hist_df):
    """
    Replace forecast values with actual historical data
    
    Parameters
    ----------
    hist_df : DataFrame
        Historical data to use for replacing forecasts
    """
    if hist_df is None or not hasattr(self, 'YY_mean_agg'):
        return
        
    try:
        # Find common indices and columns
        idx = self.YY_mean_agg.index.intersection(hist_df.index, sort=False)
        cols = [col for col in hist_df.columns if col in self.YY_mean_agg.columns]
        
        if len(idx) > 0 and len(cols) > 0:
            # Replace values in mean and median forecasts
            self.YY_mean_agg.loc[idx, cols] = hist_df.loc[idx, cols]
            
            if hasattr(self, 'YY_median_agg') and self.YY_median_agg is not None:
                self.YY_median_agg.loc[idx, cols] = hist_df.loc[idx, cols]
                
            # Set percentile forecasts to NaN for historical periods
            for perc_attr in ['YY_095_agg', 'YY_005_agg', 'YY_084_agg', 'YY_016_agg']:
                if hasattr(self, perc_attr) and getattr(self, perc_attr) is not None:
                    getattr(self, perc_attr).loc[idx, cols] = np.nan
    except Exception as e:
        print(f"Error replacing forecasts with historical data: {e}")

def aggregate_df(self, df, source_freq, target_freq, intermediate_freq=None):
    """
    Aggregate a DataFrame from source to target frequency
    
    Parameters
    ----------
    df : DataFrame
        Data to aggregate
    source_freq : str
        Source frequency
    target_freq : str
        Target frequency
    intermediate_freq : str, optional
        Intermediate frequency for three-tier aggregation
    
    Returns
    -------
    DataFrame
        Aggregated data
    """
    try:
        # Direct aggregation if possible
        if (source_freq, target_freq) in [('M', 'Q'), ('Q', 'Y')]:
            # Use pandas resample for standard frequency conversion
            if self.temp_agg == 'mean':
                return df.resample(target_freq[0]).mean()
            else:  # 'sum'
                return df.resample(target_freq[0]).sum()
                
        # Two-step aggregation if needed
        elif intermediate_freq is not None:
            # First aggregate to intermediate frequency
            intermediate = self.aggregate_df(df, source_freq, intermediate_freq)
            # Then aggregate to target frequency
            return self.aggregate_df(intermediate, intermediate_freq, target_freq)
            
        # Custom aggregation for other cases
        else:
            # Find aggregation parameters using helper function
            def find_agg_start(df, source_freq, target_freq):
                if target_freq == 'Q' and source_freq == 'W':
                    # Find first week of a quarter
                    for i in range(len(df.index) - 12):
                        if df.index[i].month in [1, 4, 7, 10] and df.index[i].day <= 7:
                            return i
                    return 0
                elif target_freq == 'M' and source_freq == 'W':
                    # Find first week of a month
                    for i in range(len(df.index) - 4):
                        if df.index[i].day <= 7:
                            return i
                    return 0
                else:
                    return 0
            
            # Get frequency ratio and starting point
            if (source_freq, target_freq) == ('W', 'M'):
                ratio = 4
                start = find_agg_start(df, source_freq, target_freq)
            elif (source_freq, target_freq) == ('W', 'Q'):
                ratio = 12
                start = find_agg_start(df, source_freq, target_freq)
            elif (source_freq, target_freq) == ('M', 'Y'):
                ratio = 12
                start = 0  # Simplified - would need to find January
            else:
                raise ValueError(f"Unsupported frequency conversion: {source_freq} to {target_freq}")
            
            # Slice data to ensure complete periods
            filtered_df = df.iloc[start:].groupby(df.iloc[start:].reset_index().index // ratio).filter(
                lambda x: len(x) == ratio)
            
            # Aggregate according to method
            if self.temp_agg == 'mean':
                result = filtered_df.groupby(filtered_df.reset_index().index // ratio).mean()
            else:
                result = filtered_df.groupby(filtered_df.reset_index().index // ratio).sum()
            
            # Set proper index dates
            result.index = df.iloc[start:].index[::ratio][:result.shape[0]]
            result.index = result.index.map(lambda x: x.replace(day=1))
            
            return result
            
    except Exception as e:
        print(f"Error aggregating from {source_freq} to {target_freq}: {e}")
        return None

def filter_to_variables_of_interest(self):
    """
    Filter aggregated forecasts to only include variables of interest
    """
    if not hasattr(self, 'var_of_interest') or self.var_of_interest is None:
        return
        
    try:
        # Identify variables of interest in the aggregated data
        weekly_vars = []
        if hasattr(self, 'input_data_W'):
            if isinstance(self.input_data_W, pd.DataFrame):
                weekly_vars = list(self.input_data_W.columns)
            else:
                weekly_vars = self.varlist[:self.Nw] if hasattr(self, 'varlist') else []
                
        var_all = list(weekly_vars) + list(self.var_of_interest)
        
        # Find columns that match variables of interest
        idx_var_of_interest = [i for i, col in enumerate(self.YY_mean_agg.columns) if col in var_all]
        
        if idx_var_of_interest:
            # Filter each DataFrame to include only variables of interest
            for attr_name in ['YY_mean_agg', 'YY_median_agg', 'YY_095_agg', 'YY_005_agg', 'YY_084_agg', 'YY_016_agg']:
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    setattr(self, attr_name, getattr(self, attr_name).iloc[:, idx_var_of_interest])
                    
            print(f"Filtered to {len(idx_var_of_interest)} variables of interest")
        else:
            print("Warning: No variables of interest found in the aggregated data")
            
    except Exception as e:
        print(f"Error filtering to variables of interest: {e}")
        
def block_diag(*arrs):
    """
    Create a block diagonal matrix from the provided arrays
    
    Parameters
    ----------
    *arrs : list of ndarrays
        Arrays to be placed on the diagonal
        
    Returns
    -------
    ndarray
        Block diagonal matrix
    """
    if len(arrs) == 0:
        return np.array([])
        
    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0))
    
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r+rr, c:c+cc] = arrs[i]
        r += rr
        c += cc
        
    return out

def extract_historical_data(self, hist_length, Nw, Nm, Nq):
    """
    Extract and transform historical data with proper temporal aggregation
    """
    history_data = np.zeros((hist_length, Nw + Nm + Nq))
    
    # Get frequency ratios
    rmw = self.freq_ratio[1]  # Monthly to weekly ratio (4)
    rqw = self.freq_ratio[0] * self.freq_ratio[1]  # Quarterly to weekly ratio (12)
    
    print(f"Extracting {hist_length} periods of historical data")
    
    # PART 1: Use raw original data where available
    # --------------------------------------------
    
    # For weekly data, use direct observations
    if hasattr(self, 'input_data_W') and self.input_data_W is not None:
        w_data = self.input_data_W
        if isinstance(w_data, pd.DataFrame):
            w_data = w_data.values
        
        # Only fill as many periods as we have data
        avail_weeks = min(hist_length, w_data.shape[0])
        if avail_weeks > 0:
            # Map to the end of our history period
            w_start = max(0, w_data.shape[0] - avail_weeks)
            h_start = max(0, hist_length - avail_weeks)
            
            # Fill in weekly data
            history_data[h_start:, :Nw] = w_data[w_start:, :Nw]
            print(f"Filled {avail_weeks} weeks with direct observations")
    
    # For monthly data
    if hasattr(self, 'input_data_M') and self.input_data_M is not None:
        m_data = self.input_data_M
        if isinstance(m_data, pd.DataFrame):
            m_data = m_data.values
        
        # Map months to weeks - each month maps to 4 weeks (rmw)
        m_periods = m_data.shape[0]
        
        for m_idx in range(m_periods):
            # Map this month to the corresponding weeks in history_data
            w_start = max(0, hist_length - (m_periods - m_idx) * rmw)
            w_end = min(hist_length, hist_length - (m_periods - m_idx - 1) * rmw)
            
            # If we have weeks that correspond to this month
            if w_start < w_end:
                # Fill all weeks in this month with same monthly values
                for w in range(w_start, w_end):
                    history_data[w, Nw:Nw+Nm] = m_data[m_idx, :Nm]
        
        print(f"Mapped {m_periods} months to weeks")
    
    # For quarterly data
    if hasattr(self, 'input_data_Q') and self.input_data_Q is not None:
        q_data = self.input_data_Q
        if isinstance(q_data, pd.DataFrame):
            q_data = q_data.values
        
        # Map quarters to weeks - each quarter maps to 12 weeks (rqw)
        q_periods = q_data.shape[0]
        
        for q_idx in range(q_periods):
            # Map this quarter to the corresponding weeks in history_data
            w_start = max(0, hist_length - (q_periods - q_idx) * rqw)
            w_end = min(hist_length, hist_length - (q_periods - q_idx - 1) * rqw)
            
            # If we have weeks that correspond to this quarter
            if w_start < w_end:
                # Fill all weeks in this quarter with same quarterly values
                for w in range(w_start, w_end):
                    history_data[w, Nw+Nm:Nw+Nm+Nq] = q_data[q_idx, :Nq]
        
        print(f"Mapped {q_periods} quarters to weeks")
    
    # PART 2: Apply transformations based on variable types
    # --------------------------------------------------
    
    # Copy data for transformation
    history_trans = history_data.copy()
    
    # Apply transformations based on variable type
    for i in range(Nw + Nm + Nq):
        if i < len(self.select):
            # Skip zeros to avoid transforming empty values
            mask = history_data[:, i] != 0
            
            if np.any(mask):  # Only transform if there are non-zero values
                if self.select[i] == 1:  # Growth rate
                    history_trans[mask, i] = 100 * history_data[mask, i]
                else:  # Level
                    history_trans[mask, i] = np.exp(history_data[mask, i])
    
    # Verify data has been filled correctly
    for var_type, start_idx, count in [("Weekly", 0, Nw), 
                                     ("Monthly", Nw, Nm), 
                                     ("Quarterly", Nw+Nm, Nq)]:
        non_zero = np.count_nonzero(history_trans[:, start_idx:start_idx+count])
        total = hist_length * count
        print(f"{var_type} variables: {non_zero}/{total} values filled ({non_zero/total*100:.1f}%)")
    
    return history_trans