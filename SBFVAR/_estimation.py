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
    """
    Fit the mixed-frequency BVAR self with enhanced measurement equation constraints
    and numerical stability improvements.
    
    Parameters:
    -----------
    mufbvar_data : MUFBVARData object
        Processed data for the BVAR self
    hyp : ndarray
        Hyperparameters for the Minnesota prior
    var_of_interest : str or None
        Variable of interest for forecasting (not currently used)
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
    p = self.nlags  # Number of lags for VAR
    
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
    
    # STATE SPACE self STRUCTURE
    # ---------------------------
    
    # State vector structure includes all variables in VAR
    var_block = Ntotal * p  # All variables with lags for VAR
    monthly_latent_block = Nm * rmw  # Month of weekly latent states for monthly vars
    quarterly_latent_block = Nq * rqw  # Quarter of weekly latent states for quarterly vars
    
    # Define indices within state vector
    monthly_start = var_block
    quarterly_start = monthly_start + monthly_latent_block
    
    # Total state vector size
    nstate = var_block + monthly_latent_block + quarterly_latent_block
    
    # Print state vector structure details
    print(f"State vector structure:")
    print(f"  Variables block (with lags): {var_block} states")
    print(f"  Monthly latent states block: {monthly_latent_block} states (starting at index {monthly_start})")
    print(f"  Quarterly latent states block: {quarterly_latent_block} states (starting at index {quarterly_start})")
    print(f"  Total state vector size: {nstate} states")
    
    # Initialize matrices for MCMC sampling
    Sigmap = np.zeros((math.ceil((self.nsim)/self.thining), Ntotal, Ntotal))
    Phip = np.zeros((math.ceil((self.nsim)/self.thining), Ntotal*p+1, Ntotal))  # All variables
    Cons = np.zeros((math.ceil((self.nsim)/self.thining), Ntotal))  # Constants for all variables
    
    # Initialize transition matrix F
    F = np.zeros((nstate, nstate))
    
    # Initialize VAR coefficients for all variables
    Phi = np.vstack((0.95 * np.eye(Ntotal), np.zeros((Ntotal*(p-1), Ntotal)), np.zeros((1, Ntotal))))  # Last row for constant
    
    # Set VAR dynamics in transition matrix
    F[:Ntotal, :Ntotal*p] = Phi[:-1, :].T  # VAR coefficients (excluding constant)
    
    # Lag-shifting section for all variables
    for i in range(p-1):
        F[Ntotal*(i+1):Ntotal*(i+2), Ntotal*i:Ntotal*(i+1)] = np.eye(Ntotal)
    
    # Monthly latent states: shifting blocks with weekly influence
    # Week 1 is influenced by weekly variables (10%)
    F[monthly_start:monthly_start+Nm, :Nw] = 0.1 * np.eye(Nm, Nw)
    
    # Shift weeks within each month, creating a diagonal structure
    for i in range(rmw-1):
        src_pos = monthly_start + i*Nm
        dest_pos = monthly_start + (i+1)*Nm
        F[dest_pos:dest_pos+Nm, src_pos:src_pos+Nm] = np.eye(Nm)
    
    # Quarterly latent states: shifting blocks with weekly influence
    # Week 1 is influenced by weekly variables (5%)
    F[quarterly_start:quarterly_start+Nq, :Nw] = 0.05 * np.eye(Nq, Nw)
    
    # Shift weeks within each quarter
    for i in range(rqw-1):
        src_pos = quarterly_start + i*Nq
        dest_pos = quarterly_start + (i+1)*Nq
        F[dest_pos:dest_pos+Nq, src_pos:src_pos+Nq] = np.eye(Nq)
    
    # Constant term vector
    c = np.zeros((nstate, 1))
    c[:Ntotal] = np.atleast_2d(Phi[-1, :]).T  # Constants for all variables
    
    # MEASUREMENT EQUATIONS WITH ENHANCED CONSTRAINTS
    # ---------------------------------------------
    
    # Weekly measurement (always available)
    H_w = np.zeros((Nw, nstate))
    H_w[:, :Nw] = np.eye(Nw)  # Directly observe first Nw state elements
    
    # Monthly measurement (available at month-end)
    H_m = np.zeros((Nm, nstate))
    # For monthly variables, aggregate all weekly latent states
    for m in range(Nm):
        for w in range(rmw):
            state_idx = monthly_start + w*Nm + m
            if self.temp_agg == 'mean':
                H_m[m, state_idx] = 1.0/rmw  # Average of weeks in month
            else:  # 'sum'
                H_m[m, state_idx] = 1.0  # Sum of weeks in month
    
    # Quarterly measurement (available at quarter-end)
    H_q = np.zeros((Nq, nstate))
    # For quarterly variables, aggregate all weekly latent states
    for q in range(Nq):
        for w in range(rqw):
            state_idx = quarterly_start + w*Nq + q
            if self.temp_agg == 'mean':
                H_q[q, state_idx] = 1.0/rqw  # Average of weeks in quarter
            else:  # 'sum'
                H_q[q, state_idx] = 1.0  # Sum of weeks in quarter
    
    # Add direct constraints to also enforce VAR block alignment
    # This creates a stronger link between VAR variables and latent states
    H_m_constraint = np.zeros((Nm, nstate))
    H_q_constraint = np.zeros((Nq, nstate))
    
    # Link monthly variables in VAR block with first week of latent states
    for m in range(Nm):
        # VAR block for monthly variable
        H_m_constraint[m, Nw+m] = 1.0
        # First week latent state
        H_m_constraint[m, monthly_start+m] = -1.0
    
    # Link quarterly variables in VAR block with first week of latent states
    for q in range(Nq):
        # VAR block for quarterly variable
        H_q_constraint[q, Nw+Nm+q] = 1.0
        # First week latent state
        H_q_constraint[q, quarterly_start+q] = -1.0
    
    # Print measurement matrix details for verification
    print("\nMeasurement equations:")
    print(f"Weekly measurement: Using {Nw} direct weekly observations")
    for m in range(min(2, Nm)):
        active_indices = np.where(H_m[m] != 0)[0]
        print(f"Monthly var {m+1}: Aggregating {len(active_indices)} weekly latent states with weights {H_m[m, active_indices[0]]:.6f}")
        
    for q in range(min(2, Nq)):
        active_indices = np.where(H_q[q] != 0)[0]
        print(f"Quarterly var {q+1}: Aggregating {len(active_indices)} weekly latent states with weights {H_q[q, active_indices[0]]:.6f}")
    
    print("Added VAR-to-latent state alignment constraints")
    
    # SYSTEM NOISE AND INITIALIZATION
    # ------------------------------
    
    # System noise affects VAR variables differently than latent states
    Q = np.zeros((nstate, nstate))
    
    # VAR block: moderate process noise
    Q[:Ntotal, :Ntotal] = 1e-4 * np.eye(Ntotal)
    
    # Monthly latent states: significantly lower process noise
    for m in range(Nm):
        for w in range(rmw):
            idx = monthly_start + w*Nm + m
            Q[idx, idx] = 1e-6  # Reduced process noise for smoother weekly patterns
    
    # Quarterly latent states: lowest process noise
    for q in range(Nq):
        for w in range(rqw):
            idx = quarterly_start + w*Nq + q
            Q[idx, idx] = 1e-7  # Even lower process noise for quarterly patterns
    
    # Initialize state vector
    a_t = np.zeros(nstate)
    P_t = np.eye(nstate) * 1e-3  # Initialize P_t here first

    # Higher uncertainty for VAR block
    P_t[:Ntotal, :Ntotal] = np.eye(Ntotal) * 1e-2
    # Initialize weekly variables with available data
    if YW.shape[0] > 0:
        a_t[:Nw] = YW[0, :Nw]
        print(f"Weekly vars initialized with first observation")
    
    # Initialize monthly variables in VAR block
    if YM.shape[0] > 0:
        a_t[Nw:Nw+Nm] = YM[0, :]
    
    # Initialize quarterly variables in VAR block
    if YQ.shape[0] > 0:
        a_t[Nw+Nm:Ntotal] = YQ[0, :]
    
    # IMPROVED INITIALIZATION FOR LATENT STATES
    # Initialize monthly latent states with reasonable patterns
    if YM.shape[0] > 0:
        for m in range(Nm):
            m_value = YM[0, m]
            # Create a realistic weekly pattern around the observed value
            for w in range(rmw):
                # Small variations (±5%)
                a_t[monthly_start + w*Nm + m] = m_value * (0.95 + 0.1 * (w / (rmw-1)))
                # Lower initial uncertainty
                P_t[monthly_start + w*Nm + m, monthly_start + w*Nm + m] = 0.01
            print(f"Monthly var {m+1} initialized with more stable weekly pattern")
    
    # Initialize quarterly latent states with reasonable patterns
    if YQ.shape[0] > 0:
        for q in range(Nq):
            q_value = YQ[0, q]
            # Create a realistic weekly pattern for quarterly data
            for w in range(rqw):
                # Gentle seasonal pattern
                a_t[quarterly_start + w*Nq + q] = q_value * (0.9 + 0.2 * np.sin(np.pi * w / rqw))
                # Lower initial uncertainty
                P_t[quarterly_start + w*Nq + q, quarterly_start + w*Nq + q] = 0.01
            print(f"Quarterly var {q+1} initialized with more stable weekly pattern")
    
    # PREPARE KALMAN FILTER DATA
    # -------------------------
    
    # Prepare observed data vectors with clear frequency markers
    q_obs_periods = np.zeros(Tw, dtype=bool)  # Quarters observed
    m_obs_periods = np.zeros(Tw, dtype=bool)  # Months observed
    
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
    
    # Storage for filtered and smoothed states
    a_filtered = np.zeros((nobs, nstate))
    P_filtered = np.zeros((nobs, nstate, nstate))
    a_draws = np.zeros((self.nsim, nobs, nstate))
    
    # MAIN KALMAN FILTER AND SMOOTHING LOOP
    # ------------------------------------
    
    print("\nRunning Kalman filter with mixed-frequency measurements")
    print(f"Total draws: {self.nsim}")
    
    for j in tqdm(range(self.nsim)):
        # If not first iteration, use state from previous draw
        if j > 0:
            a_t = a_draws[j-1, -1]
            P_t = P_filtered[-1]
        
        # Kalman filter loop through all periods
        for t in range(nobs):
            # Current week index
            w_idx = T0 + t
            
            # PREDICTION STEP
            # --------------
            a_pred = F @ a_t + c.flatten()
            P_pred = F @ P_t @ F.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)  # Ensure perfect symmetry
            
            # DETERMINE AVAILABLE OBSERVATIONS
            # -----------------------------
            
            # Check which observations are available at this period
            is_quarter_end = q_obs_periods[w_idx] if w_idx < len(q_obs_periods) else False
            is_month_end = m_obs_periods[w_idx] if w_idx < len(m_obs_periods) else False
            
            # Get indices for the observation vectors
            q_idx = w_idx // rqw if is_quarter_end else -1
            m_idx = w_idx // rmw if is_month_end else -1
            
            # Debug output for first few periods
            if j == 0 and t < 5:
                print(f"Period {t+1}: week_idx={w_idx}, quarter_end={is_quarter_end}, month_end={is_month_end}")
                if is_quarter_end:
                    print(f"  Quarterly observation: {YQ[q_idx]}")
                if is_month_end:
                    print(f"  Monthly observation: {YM[m_idx]}")
            
            # BUILD ENHANCED MEASUREMENT MATRICES AND OBSERVATION VECTOR
            # -----------------------------------------------------
            
            # Start with weekly observations (always available)
            H_matrices = [H_w]
            y_obs = [YW[w_idx]]
            
            # Add monthly observations if available
            monthly_constraints_added = False
            if is_month_end and m_idx >= 0 and m_idx < YM.shape[0]:
                H_matrices.append(H_m)
                y_obs.append(YM[m_idx])
                
                # Always add the constraint that links VAR block to latent states
                H_matrices.append(H_m_constraint)
                y_obs.append(np.zeros(Nm))  # Zero difference means equality constraint
                monthly_constraints_added = True
            
            # Add quarterly observations if available
            quarterly_constraints_added = False
            if is_quarter_end and q_idx >= 0 and q_idx < YQ.shape[0]:
                H_matrices.append(H_q)
                y_obs.append(YQ[q_idx])
                
                # Always add the constraint that links VAR block to latent states
                H_matrices.append(H_q_constraint)
                y_obs.append(np.zeros(Nq))  # Zero difference means equality constraint
                quarterly_constraints_added = True
            
            # Also enforce VAR-latent alignment constraints at every period
            # (not just at month/quarter end)
            if not monthly_constraints_added and t % 5 == 0:  # Every 5 periods to avoid too much constraint
                H_matrices.append(H_m_constraint)
                y_obs.append(np.zeros(Nm))
            
            if not quarterly_constraints_added and t % 10 == 0:  # Less frequent for quarterly
                H_matrices.append(H_q_constraint)
                y_obs.append(np.zeros(Nq))
            
            # Combined measurement matrix and observation vector
            H = np.vstack(H_matrices)
            y = np.concatenate(y_obs)
            
            # DIFFERENT MEASUREMENT NOISE BY FREQUENCY - EXTREMELY PRECISE FOR CONSTRAINTS
            # ---------------------------------------------------------------------
            
            # Create measurement noise matrix with precision scaled by frequency
            R = np.zeros((len(y), len(y)))
            
            # Current position in the observation vector
            obs_pos = 0
            
            # Weekly measurement noise (standard precision)
            R[obs_pos:obs_pos+Nw, obs_pos:obs_pos+Nw] = np.eye(Nw) * 1e-3
            obs_pos += Nw
            
            # Monthly measurement noise (extreme precision to enforce constraint)
            if is_month_end and m_idx >= 0 and m_idx < YM.shape[0]:
                R[obs_pos:obs_pos+Nm, obs_pos:obs_pos+Nm] = np.eye(Nm) * 1e-12
                obs_pos += Nm
                
                # VAR-to-latent constraint noise (extremely low)
                R[obs_pos:obs_pos+Nm, obs_pos:obs_pos+Nm] = np.eye(Nm) * 1e-14
                obs_pos += Nm
            
            # Quarterly measurement noise (even more extreme precision)
            if is_quarter_end and q_idx >= 0 and q_idx < YQ.shape[0]:
                R[obs_pos:obs_pos+Nq, obs_pos:obs_pos+Nq] = np.eye(Nq) * 1e-12
                obs_pos += Nq
                
                # VAR-to-latent constraint noise (extremely low)
                R[obs_pos:obs_pos+Nq, obs_pos:obs_pos+Nq] = np.eye(Nq) * 1e-14
                obs_pos += Nq
            
            # Add VAR-latent alignment constraints outside month/quarter end if included
            if not monthly_constraints_added and t % 5 == 0:
                R[obs_pos:obs_pos+Nm, obs_pos:obs_pos+Nm] = np.eye(Nm) * 1e-12
                obs_pos += Nm
                
            if not quarterly_constraints_added and t % 10 == 0:
                R[obs_pos:obs_pos+Nq, obs_pos:obs_pos+Nq] = np.eye(Nq) * 1e-12
                obs_pos += Nq
            
            # UPDATE STEP WITH NUMERICAL STABILITY
            # ------------------------------
            y_hat = H @ a_pred
            nu = y - y_hat  # Innovation
            
            # Innovation covariance - with careful regularization
            S = H @ P_pred @ H.T + R
            S = 0.5 * (S + S.T)  # Ensure perfect symmetry
            
            # Add regularization more safely
            S_reg = S.copy()
            try:
                # Use eigvalsh for symmetric matrices which returns real eigenvalues
                eig_vals = np.linalg.eigvalsh(S)
                min_eig = np.min(eig_vals)
                if min_eig < 1e-10:
                    S_reg += np.eye(S.shape[0]) * (1e-10 - min_eig)
            except:
                # Fallback to simple regularization if eigvalsh fails
                S_reg += np.eye(S.shape[0]) * 1e-8
                if j == 0 and t < 10:
                    print(f"Warning: Using diagonal regularization for S at t={t}")
            
            try:
                # Kalman gain
                K = P_pred @ H.T @ invert_matrix(S_reg)
                
                # Update state and covariance
                a_t = a_pred + K @ nu
                P_t = P_pred - K @ H @ P_pred
                
                # Force symmetry more carefully
                P_t = 0.5 * (P_t + P_t.T)  # Ensure perfect symmetry
                
                # Force positive definiteness more robustly
                try:
                    # Use eigvalsh for symmetric matrices
                    eig_vals = np.linalg.eigvalsh(P_t)
                    min_eig = np.min(eig_vals)
                    if min_eig < 1e-8:
                        P_t += np.eye(P_t.shape[0]) * (1e-8 - min_eig)
                except:
                    # If eigvalsh fails, use a more brute-force approach
                    P_t += np.eye(P_t.shape[0]) * 1e-6
                    if j == 0 and t < 10:
                        print(f"Warning: Using diagonal regularization for P_t at t={t}")
                
                # ENHANCED MEASUREMENT EQUATION ENFORCEMENT
                # ------------------------------------------------
                
                # Month-end constraint enforcement using the measurement equation
                if is_month_end and m_idx >= 0 and m_idx < YM.shape[0]:
                    for m in range(Nm):
                        # Get observed monthly value
                        m_obs = YM[m_idx, m]
                        var_block_idx = Nw + m
                        
                        # 1. First let's set the VAR block variable to match observation with high precision
                        # This is done through a "phantom" observation using the Kalman update
                        H_phantom = np.zeros((1, nstate))
                        H_phantom[0, var_block_idx] = 1.0
                        y_phantom = np.array([m_obs])
                        R_phantom = np.array([[1e-12]])  # Very low measurement noise
                        
                        # Kalman gain for this phantom observation
                        S_phantom = H_phantom @ P_t @ H_phantom.T + R_phantom
                        K_phantom = P_t @ H_phantom.T @ np.linalg.inv(S_phantom)
                        
                        # Apply update
                        a_t = a_t + K_phantom @ (y_phantom - H_phantom @ a_t)
                        P_t = P_t - K_phantom @ H_phantom @ P_t
                        
                        # 2. Now enforce the temporal aggregation constraint with high precision
                        # Build measurement matrix for aggregation
                        H_agg = np.zeros((1, nstate))
                        for w in range(rmw):
                            state_idx = monthly_start + w*Nm + m
                            if self.temp_agg == 'mean':
                                H_agg[0, state_idx] = 1.0/rmw
                            else:  # 'sum'
                                H_agg[0, state_idx] = 1.0
                        
                        # Use phantom observation for aggregation
                        y_agg = np.array([m_obs])
                        R_agg = np.array([[1e-12]])  # Very low measurement noise
                        
                        # Kalman gain for aggregation
                        S_agg = H_agg @ P_t @ H_agg.T + R_agg
                        K_agg = P_t @ H_agg.T @ np.linalg.inv(S_agg)
                        
                        # Apply update
                        a_t = a_t + K_agg @ (y_agg - H_agg @ a_t)
                        P_t = P_t - K_agg @ H_agg @ P_t
                        
                        # 3. Enforce first-week alignment with VAR block
                        H_align = np.zeros((1, nstate))
                        H_align[0, var_block_idx] = 1.0  # VAR block
                        H_align[0, monthly_start + m] = -1.0  # First week
                        
                        y_align = np.array([0.0])  # Zero difference
                        R_align = np.array([[1e-12]])
                        
                        # Kalman gain for alignment
                        S_align = H_align @ P_t @ H_align.T + R_align
                        K_align = P_t @ H_align.T @ np.linalg.inv(S_align)
                        
                        # Apply update
                        a_t = a_t + K_align @ (y_align - H_align @ a_t)
                        P_t = P_t - K_align @ H_align @ P_t
                        
                        # 4. Apply value clipping afterward to prevent extreme values
                        threshold = 3.0
                        for w in range(rmw):
                            state_idx = monthly_start + w*Nm + m
                            if np.isnan(a_t[state_idx]) or np.isinf(a_t[state_idx]) or abs(a_t[state_idx]) > threshold:
                                # Use a more conservative value but preserve some variability
                                pattern_factor = 0.95 + 0.1 * (w / (rmw-1))
                                a_t[state_idx] = np.sign(a_t[state_idx]) * min(threshold, abs(m_obs) * pattern_factor)
                        
                        # 5. Verify the constraint is reasonably enforced (for debugging)
                        if self.temp_agg == 'mean':
                            agg_value = np.mean([a_t[monthly_start + w*Nm + m] for w in range(rmw)])
                        else:
                            agg_value = np.sum([a_t[monthly_start + w*Nm + m] for w in range(rmw)])
                        
                        # Print verification for first few draws
                        if j == 0 and t < 10:
                            error = abs(agg_value - m_obs)
                            print(f"  Month {m_idx+1}, Var {m+1}: Target={m_obs:.8f}, Achieved={agg_value:.8f}, Error={error:.10f}")
                
                # Quarter-end constraint enforcement using measurement equation - similar approach
                if is_quarter_end and q_idx >= 0 and q_idx < YQ.shape[0]:
                    for q in range(Nq):
                        # Get observed quarterly value
                        q_obs = YQ[q_idx, q]
                        var_block_idx = Nw + Nm + q
                        
                        # 1. Set VAR block variable with high precision
                        H_phantom = np.zeros((1, nstate))
                        H_phantom[0, var_block_idx] = 1.0
                        y_phantom = np.array([q_obs])
                        R_phantom = np.array([[1e-12]])
                        
                        # Kalman gain for phantom observation
                        S_phantom = H_phantom @ P_t @ H_phantom.T + R_phantom
                        K_phantom = P_t @ H_phantom.T @ np.linalg.inv(S_phantom)
                        
                        # Apply update
                        a_t = a_t + K_phantom @ (y_phantom - H_phantom @ a_t)
                        P_t = P_t - K_phantom @ H_phantom @ P_t
                        
                        # 2. Enforce temporal aggregation constraint
                        H_agg = np.zeros((1, nstate))
                        for w in range(rqw):
                            state_idx = quarterly_start + w*Nq + q
                            if self.temp_agg == 'mean':
                                H_agg[0, state_idx] = 1.0/rqw
                            else:  # 'sum'
                                H_agg[0, state_idx] = 1.0
                        
                        # Use phantom observation
                        y_agg = np.array([q_obs])
                        R_agg = np.array([[1e-12]])
                        
                        # Kalman gain for aggregation
                        S_agg = H_agg @ P_t @ H_agg.T + R_agg
                        K_agg = P_t @ H_agg.T @ np.linalg.inv(S_agg)
                        
                        # Apply update
                        a_t = a_t + K_agg @ (y_agg - H_agg @ a_t)
                        P_t = P_t - K_agg @ H_agg @ P_t
                        
                        # 3. Enforce first-week alignment with VAR block
                        H_align = np.zeros((1, nstate))
                        H_align[0, var_block_idx] = 1.0  # VAR block
                        H_align[0, quarterly_start + q] = -1.0  # First week
                        
                        y_align = np.array([0.0])  # Zero difference
                        R_align = np.array([[1e-12]])
                        
                        # Kalman gain for alignment
                        S_align = H_align @ P_t @ H_align.T + R_align
                        K_align = P_t @ H_align.T @ np.linalg.inv(S_align)
                        
                        # Apply update
                        a_t = a_t + K_align @ (y_align - H_align @ a_t)
                        P_t = P_t - K_align @ H_align @ P_t
                        
                        # 4. Apply value clipping to prevent extremes
                        threshold = 3.0
                        for w in range(rqw):
                            state_idx = quarterly_start + w*Nq + q
                            if np.isnan(a_t[state_idx]) or np.isinf(a_t[state_idx]) or abs(a_t[state_idx]) > threshold:
                                # Use conservative value but preserve some variability
                                pattern_factor = 0.95 + 0.1 * np.sin(np.pi * w / rqw)
                                a_t[state_idx] = np.sign(a_t[state_idx]) * min(threshold, abs(q_obs) * pattern_factor)
                        
                        # 5. Verify constraint (for debugging)
                        if self.temp_agg == 'mean':
                            agg_value = np.mean([a_t[quarterly_start + w*Nq + q] for w in range(rqw)])
                        else:
                            agg_value = np.sum([a_t[quarterly_start + w*Nq + q] for w in range(rqw)])
                        
                        # Print verification for first few draws
                        if j == 0 and t < 10:
                            error = abs(agg_value - q_obs)
                            print(f"  Quarter {q_idx+1}, Var {q+1}: Target={q_obs:.8f}, Achieved={agg_value:.8f}, Error={error:.10f}")
                
                # ENHANCED VALUE CLAMPING
                # Apply global state value clipping
                for i in range(nstate):
                    # More aggressive clipping for latent states
                    if i >= monthly_start:
                        if np.isnan(a_t[i]) or np.isinf(a_t[i]):
                            a_t[i] = 0.0  # Reset to zero if NaN or Inf
                        elif abs(a_t[i]) > 3.0:  # Stricter threshold
                            a_t[i] = np.sign(a_t[i]) * 3.0  # Clip to ±3.0
                            
                    # Standard clipping for VAR block
                    else:
                        if np.isnan(a_t[i]) or np.isinf(a_t[i]):
                            a_t[i] = 0.0
                        elif abs(a_t[i]) > 5.0:
                            a_t[i] = np.sign(a_t[i]) * 5.0
                
                # ADDITIONAL GLOBAL STATE MONITORING
                # Periodically scan and reset problematic states
                if t % 10 == 0:
                    # Check for any remaining extreme values or instabilities
                    for i in range(nstate):
                        if abs(a_t[i]) > 2.0 and i >= monthly_start:
                            # For latent states, set to conservative default if still extreme
                            if i < quarterly_start:  # Monthly latent
                                m = (i - monthly_start) % Nm
                                w = (i - monthly_start) // Nm
                                if m_idx >= 0 and m_idx < YM.shape[0]:
                                    # Set to monthly value with small deviation
                                    m_val = YM[m_idx, m]
                                    a_t[i] = m_val * (0.95 + 0.1 * w / rmw)
                                else:
                                    a_t[i] = 0.5  # Conservative default
                            else:  # Quarterly latent
                                q = (i - quarterly_start) % Nq
                                w = (i - quarterly_start) // Nq
                                if q_idx >= 0 and q_idx < YQ.shape[0]:
                                    # Set to quarterly value with small deviation
                                    q_val = YQ[q_idx, q]
                                    a_t[i] = q_val * (0.95 + 0.1 * w / rqw)
                                else:
                                    a_t[i] = 0.5
                                
                            # Reduce uncertainty for reset states
                            P_t[i, i] = 1e-6
                
            except np.linalg.LinAlgError as e:
                if j == 0:
                    print(f"Warning: Matrix inversion failed at t={t}. Using prediction only. Error: {str(e)}")
                a_t = a_pred
                P_t = P_pred
                
                # Apply value clamping even when falling back to prediction
                threshold = 5.0
                for i in range(len(a_t)):
                    if np.isnan(a_t[i]) or np.isinf(a_t[i]) or abs(a_t[i]) > threshold:
                        a_t[i] = threshold * np.sign(a_t[i]) if a_t[i] != 0 else 0.001
            
            # Store filtered state and covariance
            a_filtered[t] = a_t
            P_filtered[t] = P_t
        
        # KALMAN SMOOTHER
        # -------------
        
        # Initialize smoother with last filtered state
        a_smooth = np.zeros((nobs, nstate))
        P_smooth = np.zeros((nobs, nstate, nstate))
        
        # Last state is the same for filtered and smoothed
        a_smooth[-1] = a_filtered[-1]
        P_smooth[-1] = P_filtered[-1]
        
        try:
            # Force symmetry and positive-definiteness of P_smooth[-1] before Cholesky
            P_smooth[-1] = 0.5 * (P_smooth[-1] + P_smooth[-1].T)  # Ensure symmetry
            
            try:
                # Check and fix eigenvalues
                eig_vals = np.linalg.eigvalsh(P_smooth[-1])
                min_eig = np.min(eig_vals)
                if min_eig < 1e-8:
                    P_smooth[-1] += np.eye(P_smooth[-1].shape[0]) * (1e-8 - min_eig)
            except:
                # Fallback to simple regularization
                P_smooth[-1] += np.eye(P_smooth[-1].shape[0]) * 1e-6
                if j == 0:
                    print("Warning: Using diagonal regularization for P_smooth[-1]")
            
            # Draw the last state
            Pchol = cholcovOrEigendecomp(P_smooth[-1])
            a_draw = a_smooth[-1] + Pchol @ np.random.standard_normal(nstate)
            
            # Apply value clamping to the draw - more aggressive for latent states
            threshold_var = 5.0  # VAR block
            threshold_latent = 3.0  # Latent states
            for i in range(len(a_draw)):
                if np.isnan(a_draw[i]) or np.isinf(a_draw[i]):
                    a_draw[i] = a_smooth[-1][i]  # Fall back to smoothed value
                elif i >= monthly_start:  # Latent states
                    if abs(a_draw[i]) > threshold_latent:
                        a_draw[i] = np.sign(a_draw[i]) * threshold_latent
                else:  # VAR block
                    if abs(a_draw[i]) > threshold_var:
                        a_draw[i] = np.sign(a_draw[i]) * threshold_var
            
            # Final check for aggregate constraint on last state
            last_week_idx = T0 + nobs - 1
            last_month_idx = last_week_idx // rmw
            last_quarter_idx = last_week_idx // rqw
            
            # Check if this is month-end
            is_month_end = ((last_week_idx + 1) % rmw == 0)
            if is_month_end and last_month_idx < YM.shape[0]:
                for m in range(Nm):
                    # First enforce VAR block to match observed value
                    var_block_idx = Nw + m
                    m_obs = YM[last_month_idx, m]
                    a_draw[var_block_idx] = m_obs
                    
                    # Now enforce latent state aggregation using Kalman update approach
                    H_agg = np.zeros((1, nstate))
                    for w in range(rmw):
                        state_idx = monthly_start + w*Nm + m
                        if self.temp_agg == 'mean':
                            H_agg[0, state_idx] = 1.0/rmw
                        else:  # 'sum'
                            H_agg[0, state_idx] = 1.0
                    
                    # Check current aggregation
                    m_values = np.array([a_draw[monthly_start + w*Nm + m] for w in range(rmw)])
                    if self.temp_agg == 'mean':
                        agg_value = np.mean(m_values)
                    else:
                        agg_value = np.sum(m_values)
                    
                    # If aggregation is off by more than a small tolerance, adjust the pattern
                    if abs(agg_value - m_obs) > 1e-6:
                        if self.temp_agg == 'mean':
                            # Shift pattern to match target mean while preserving shape
                            adjustment = m_obs - agg_value
                            for w in range(rmw):
                                a_draw[monthly_start + w*Nm + m] += adjustment
                        else:  # 'sum'
                            if abs(agg_value) > 1e-10:  # Avoid division by zero
                                # Scale pattern to match target sum while preserving shape
                                scale = m_obs / agg_value
                                scale = np.clip(scale, 0.5, 2.0)  # Limit scale factor
                                for w in range(rmw):
                                    a_draw[monthly_start + w*Nm + m] *= scale
                            else:
                                # If near zero, distribute evenly
                                for w in range(rmw):
                                    a_draw[monthly_start + w*Nm + m] = m_obs / rmw
            
            # Check if this is quarter-end
            is_quarter_end = ((last_week_idx + 1) % rqw == 0)
            if is_quarter_end and last_quarter_idx < YQ.shape[0]:
                for q in range(Nq):
                    # First enforce VAR block to match observed value
                    var_block_idx = Nw + Nm + q
                    q_obs = YQ[last_quarter_idx, q]
                    a_draw[var_block_idx] = q_obs
                    
                    # Now enforce latent state aggregation using Kalman update approach
                    H_agg = np.zeros((1, nstate))
                    for w in range(rqw):
                        state_idx = quarterly_start + w*Nq + q
                        if self.temp_agg == 'mean':
                            H_agg[0, state_idx] = 1.0/rqw
                        else:  # 'sum'
                            H_agg[0, state_idx] = 1.0
                    
                    # Check current aggregation
                    q_values = np.array([a_draw[quarterly_start + w*Nq + q] for w in range(rqw)])
                    if self.temp_agg == 'mean':
                        agg_value = np.mean(q_values)
                    else:
                        agg_value = np.sum(q_values)
                    
                    # If aggregation is off by more than a small tolerance, adjust the pattern
                    if abs(agg_value - q_obs) > 1e-6:
                        if self.temp_agg == 'mean':
                            # Shift pattern to match target mean while preserving shape
                            adjustment = q_obs - agg_value
                            for w in range(rqw):
                                a_draw[quarterly_start + w*Nq + q] += adjustment
                        else:  # 'sum'
                            if abs(agg_value) > 1e-10:  # Avoid division by zero
                                # Scale pattern to match target sum while preserving shape
                                scale = q_obs / agg_value
                                scale = np.clip(scale, 0.75, 1.5)  # Tighter limits for quarterly
                                for w in range(rqw):
                                    a_draw[quarterly_start + w*Nq + q] *= scale
                            else:
                                # If near zero, distribute evenly
                                for w in range(rqw):
                                    a_draw[quarterly_start + w*Nq + q] = q_obs / rqw
            
            # Store the draw
            a_draws[j, -1] = a_draw
            
            # Backward recursion
            for t in range(nobs-2, -1, -1):
                # Get filtered state and covariance
                a_filt = a_filtered[t]
                P_filt = P_filtered[t]
                
                # Predict one step ahead
                a_pred = F @ a_filt + c.flatten()
                P_pred = F @ P_filt @ F.T + Q
                P_pred = 0.5 * (P_pred + P_pred.T)  # Ensure symmetry
                
                try:
                    # Add regularization to P_pred before inversion if needed
                    P_pred_reg = P_pred.copy()
                    try:
                        eig_vals = np.linalg.eigvalsh(P_pred)
                        min_eig = np.min(eig_vals)
                        if min_eig < 1e-8:
                            P_pred_reg += np.eye(P_pred.shape[0]) * (1e-8 - min_eig)
                    except:
                        # Fallback to simple regularization
                        P_pred_reg += np.eye(P_pred.shape[0]) * 1e-6
                    
                    # Smoothing gain
                    J_t = P_filt @ F.T @ invert_matrix(P_pred_reg)
                    
                    # Smoothed mean and covariance
                    a_smooth_t = a_filt + J_t @ (a_draw - a_pred)
                    P_smooth_t = P_filt - J_t @ (P_pred - P_smooth[t+1]) @ J_t.T
                    P_smooth_t = 0.5 * (P_smooth_t + P_smooth_t.T)  # Ensure symmetry
                    
                    # Force positive definiteness
                    try:
                        eig_vals = np.linalg.eigvalsh(P_smooth_t)
                        min_eig = np.min(eig_vals)
                        if min_eig < 1e-8:
                            P_smooth_t += np.eye(P_smooth_t.shape[0]) * (1e-8 - min_eig)
                    except:
                        # Fallback
                        P_smooth_t += np.eye(P_smooth_t.shape[0]) * 1e-6
                    
                    # Store smoothed state and covariance
                    a_smooth[t] = a_smooth_t
                    P_smooth[t] = P_smooth_t
                    
                    # Draw state
                    Pchol = cholcovOrEigendecomp(P_smooth_t)
                    a_draw = a_smooth_t + Pchol @ np.random.standard_normal(nstate)
                    
                    # Apply value clamping to the draw - more aggressive for latent states
                    threshold_var = 5.0  # VAR block
                    threshold_latent = 3.0  # Latent states
                    for i in range(len(a_draw)):
                        if np.isnan(a_draw[i]) or np.isinf(a_draw[i]):
                            a_draw[i] = a_smooth_t[i]  # Fall back to smoothed value
                        elif i >= monthly_start:  # Latent states
                            if abs(a_draw[i]) > threshold_latent:
                                a_draw[i] = np.sign(a_draw[i]) * threshold_latent
                        else:  # VAR block
                            if abs(a_draw[i]) > threshold_var:
                                a_draw[i] = np.sign(a_draw[i]) * threshold_var
                    
                    # Check and enforce constraints for this period
                    w_idx = T0 + t
                    m_idx = w_idx // rmw
                    q_idx = w_idx // rqw
                    
                    # Check if this is month-end
                    is_month_end = ((w_idx + 1) % rmw == 0)
                    if is_month_end and m_idx < YM.shape[0]:
                        for m in range(Nm):
                            # First enforce VAR block to match observed value
                            var_block_idx = Nw + m
                            m_obs = YM[m_idx, m]
                            a_draw[var_block_idx] = m_obs
                            
                            # Now enforce latent state aggregation using Kalman update approach
                            # Check current aggregation
                            m_values = np.array([a_draw[monthly_start + w*Nm + m] for w in range(rmw)])
                            if self.temp_agg == 'mean':
                                agg_value = np.mean(m_values)
                            else:
                                agg_value = np.sum(m_values)
                            
                            # If aggregation is off by more than a small tolerance, adjust the pattern
                            if abs(agg_value - m_obs) > 1e-6:
                                if self.temp_agg == 'mean':
                                    # Shift pattern to match target mean while preserving shape
                                    adjustment = m_obs - agg_value
                                    for w in range(rmw):
                                        a_draw[monthly_start + w*Nm + m] += adjustment
                                else:  # 'sum'
                                    if abs(agg_value) > 1e-10:  # Avoid division by zero
                                        # Scale pattern to match target sum while preserving shape
                                        scale = m_obs / agg_value
                                        scale = np.clip(scale, 0.5, 2.0)  # Limit scale factor
                                        for w in range(rmw):
                                            a_draw[monthly_start + w*Nm + m] *= scale
                                    else:
                                        # If near zero, distribute evenly
                                        for w in range(rmw):
                                            a_draw[monthly_start + w*Nm + m] = m_obs / rmw
                    
                    # Check if this is quarter-end
                    is_quarter_end = ((w_idx + 1) % rqw == 0)
                    if is_quarter_end and q_idx < YQ.shape[0]:
                        for q in range(Nq):
                            # First enforce VAR block to match observed value
                            var_block_idx = Nw + Nm + q
                            q_obs = YQ[q_idx, q]
                            a_draw[var_block_idx] = q_obs
                            
                            # Now enforce latent state aggregation using Kalman update approach
                            # Check current aggregation
                            q_values = np.array([a_draw[quarterly_start + w*Nq + q] for w in range(rqw)])
                            if self.temp_agg == 'mean':
                                agg_value = np.mean(q_values)
                            else:
                                agg_value = np.sum(q_values)
                            
                            # If aggregation is off by more than a small tolerance, adjust the pattern
                            if abs(agg_value - q_obs) > 1e-6:
                                if self.temp_agg == 'mean':
                                    # Shift pattern to match target mean while preserving shape
                                    adjustment = q_obs - agg_value
                                    for w in range(rqw):
                                        a_draw[quarterly_start + w*Nq + q] += adjustment
                                else:  # 'sum'
                                    if abs(agg_value) > 1e-10:  # Avoid division by zero
                                        # Scale pattern to match target sum while preserving shape
                                        scale = q_obs / agg_value
                                        scale = np.clip(scale, 0.75, 1.5)  # Tighter limits for quarterly
                                        for w in range(rqw):
                                            a_draw[quarterly_start + w*Nq + q] *= scale
                                    else:
                                        # If near zero, distribute evenly
                                        for w in range(rqw):
                                            a_draw[quarterly_start + w*Nq + q] = q_obs / rqw
                    
                    # Also enforce VAR-to-latent alignment for first week of each period
                    if (w_idx % rmw == 0) and (w_idx // rmw) < YM.shape[0]:
                        for m in range(Nm):
                            # Force first week to match VAR block
                            a_draw[monthly_start + m] = a_draw[Nw + m]
                    
                    if (w_idx % rqw == 0) and (w_idx // rqw) < YQ.shape[0]:
                        for q in range(Nq):
                            # Force first week to match VAR block
                            a_draw[quarterly_start + q] = a_draw[Nw + Nm + q]
                    
                    # Store the draw
                    a_draws[j, t] = a_draw
                    
                except Exception as e:
                    # Fallback for smoother error
                    if j == 0:
                        print(f"Smoother error at t={t}: {str(e)}")
                    a_smooth[t] = a_filtered[t]
                    P_smooth[t] = P_filtered[t]
                    
                    # Use filtered state with small noise but apply value clamping
                    a_draws[j, t] = a_filtered[t] + np.random.standard_normal(nstate) * 1e-4
                    
                    # Apply value clamping
                    threshold = 5.0
                    for i in range(len(a_draws[j, t])):
                        if np.isnan(a_draws[j, t, i]) or np.isinf(a_draws[j, t, i]) or abs(a_draws[j, t, i]) > threshold:
                            a_draws[j, t, i] = threshold * np.sign(a_draws[j, t, i]) if a_draws[j, t, i] != 0 else 0.001
            
        except Exception as e:
            # Fallback for smoother initialization error
            if j == 0:
                print(f"Smoother initialization error: {str(e)}")
            
            # Copy filtered states but apply value clamping
            for t in range(nobs):
                a_draws[j, t] = a_filtered[t]
                # Apply value clamping
                threshold = 5.0
                for i in range(len(a_draws[j, t])):
                    if np.isnan(a_draws[j, t, i]) or np.isinf(a_draws[j, t, i]) or abs(a_draws[j, t, i]) > threshold:
                        a_draws[j, t, i] = threshold * np.sign(a_draws[j, t, i]) if a_draws[j, t, i] != 0 else 0.001
            
        # FINAL VERIFICATION FOR THIS DRAW
        # ------------------------------
        
        # Verify key constraints for debugging
        if j == 0 or j == self.nsim - 1:
            constraint_errors = 0
            
            # Check all month-ends
            for t in range(nobs):
                w_idx = T0 + t
                # Check if this is month-end
                if (w_idx + 1) % rmw == 0:
                    m_idx = w_idx // rmw
                    if m_idx < YM.shape[0]:
                        for m in range(Nm):
                            # Get latent states
                            m_values = np.zeros(rmw)
                            for w in range(rmw):
                                m_values[w] = a_draws[j, t, monthly_start + w*Nm + m]
                            
                            # Check aggregation
                            if self.temp_agg == 'mean':
                                agg_value = np.mean(m_values)
                            else:
                                agg_value = np.sum(m_values)
                            
                            error = abs(agg_value - YM[m_idx, m])
                            if error > 1e-6:
                                constraint_errors += 1
                                if constraint_errors <= 5:  # Limit output
                                    print(f"Draw {j}, Month {m_idx+1}, Var {m+1}: Agg={agg_value:.8f}, "
                                          f"Target={YM[m_idx, m]:.8f}, Error={error:.8f}")
            
            # Check all quarter-ends
            for t in range(nobs):
                w_idx = T0 + t
                # Check if this is quarter-end
                if (w_idx + 1) % rqw == 0:
                    q_idx = w_idx // rqw
                    if q_idx < YQ.shape[0]:
                        for q in range(Nq):
                            # Get latent states
                            q_values = np.zeros(rqw)
                            for w in range(rqw):
                                q_values[w] = a_draws[j, t, quarterly_start + w*Nq + q]
                            
                            # Check aggregation
                            if self.temp_agg == 'mean':
                                agg_value = np.mean(q_values)
                            else:
                                agg_value = np.sum(q_values)
                            
                            error = abs(agg_value - YQ[q_idx, q])
                            if error > 1e-6:
                                constraint_errors += 1
                                if constraint_errors <= 5:  # Limit output
                                    print(f"Draw {j}, Quarter {q_idx+1}, Var {q+1}: Agg={agg_value:.8f}, "
                                          f"Target={YQ[q_idx, q]:.8f}, Error={error:.8f}")
            
            if constraint_errors > 0:
                print(f"Draw {j}: Found {constraint_errors} constraint violations")
            else:
                print(f"Draw {j}: All aggregation constraints satisfied")
        
        # CALCULATE VAR POSTERIOR USING ALL VARIABLES
        # ---------------------------------------
        
        # Create a combined matrix of all variables for VAR estimation
        combined_smooth = np.zeros((nobs, Ntotal))
        
        # Weekly variables (direct)
        combined_smooth[:, :Nw] = a_draws[j, :, :Nw]
        
        # Monthly variables (from state vector)
        combined_smooth[:, Nw:Nw+Nm] = a_draws[j, :, Nw:Nw+Nm]
        
        # Quarterly variables (from state vector)
        combined_smooth[:, Nw+Nm:Ntotal] = a_draws[j, :, Nw+Nm:Ntotal]
        
        # Use the combined matrix for VAR estimation
        YY = combined_smooth
        
        # Prepare lagged data for the VAR
        Z = np.zeros((nobs, Ntotal * p))
        for i in range(nobs):
            for lag in range(p):
                if i - lag >= 0:
                    Z[i, lag*Ntotal:(lag+1)*Ntotal] = YY[i-lag]
        
        # Compute actual observations for Minnesota prior
        nobs_ = YY.shape[0] - p  # Adjusted for lags
        spec = np.hstack((p, p, self.nex, Ntotal, nobs_))  # Now using all variables (Ntotal)
        
        # Calculate dummy observations for the full VAR
        YYact, YYdum, XXact, XXdum = calc_yyact(self.hyp, YY, spec)
        
        # Store simulation results for all variables
        if (j % self.thining == 0):
            j_temp = int(j/self.thining)
            if j == 0:
                # Initialize storage for all variables
                YYactsim_list = [np.zeros((math.ceil((self.nsim)/self.thining), rmw, Ntotal))]
                XXactsim_list = [np.zeros((math.ceil((self.nsim)/self.thining), rmw, Ntotal*p+1))]  # For all variables
            
            # Store the combined data
            YYactsim_list[0][j_temp, :, :] = combined_smooth[-rmw:, :]
            
            # Create lagged data matrix for all variables
            X_combined = np.zeros((rmw, Ntotal*p+1))
            for i in range(rmw):
                for lag in range(p):
                    t_idx = nobs - rmw + i - lag
                    if t_idx >= 0:
                        X_combined[i, lag*Ntotal:(lag+1)*Ntotal] = combined_smooth[t_idx, :]
            X_combined[:, -1] = 1.0  # Add constant
            
            XXactsim_list[0][j_temp, :, :] = X_combined
        
        # VAR POSTERIOR SAMPLING
        # -------------------
        
        try:
            # Standard VAR posterior calculations
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
            xxi = (vr * np.tile(di.T, (Ntotal*p+1, 1)))
            inv_x = xxi @ xxi.T
            Phi_tilde = xxi @ B
            
            Sigma = (Y - X @ Phi_tilde).T @ (Y - X @ Phi_tilde)
            
            # Ensure Sigma is symmetric positive definite before inverse Wishart draw
            Sigma = 0.5 * (Sigma + Sigma.T)
            
            try:
                # Check and fix eigenvalues if needed
                eig_vals = np.linalg.eigvalsh(Sigma)
                min_eig = np.min(eig_vals)
                if min_eig < 1e-8:
                    Sigma += np.eye(Sigma.shape[0]) * (1e-8 - min_eig)
            except:
                # Fallback to simple regularization
                Sigma += np.eye(Sigma.shape[0]) * 1e-6
                if j == 0:
                    print("Warning: Using diagonal regularization for Sigma")
            
            # Draw from inverse Wishart for covariance matrix
            sigma_w = invwishart.rvs(scale=Sigma, df=T-Ntotal*p-1)
            
            # Draw VAR coefficients and check stability
            attempts = 0
            while attempts < 1000:
                try:
                    # Compute Cholesky safely
                    kron_matrix = np.kron(sigma_w, inv_x)
                    kron_matrix = 0.5 * (kron_matrix + kron_matrix.T)  # Ensure symmetry
                    
                    try:
                        # Check eigenvalues
                        eig_vals = np.linalg.eigvalsh(kron_matrix)
                        min_eig = np.min(eig_vals)
                        if min_eig < 1e-8:
                            kron_matrix += np.eye(kron_matrix.shape[0]) * (1e-8 - min_eig)
                    except:
                        # Fallback
                        kron_matrix += np.eye(kron_matrix.shape[0]) * 1e-6
                    
                    sigma_chol = cholcovOrEigendecomp(kron_matrix)
                    phi_new = np.squeeze(Phi_tilde.reshape(Ntotal*(Ntotal*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                    Phi_w = phi_new.reshape(Ntotal*p+1, Ntotal, order="F")
                    
                    # NEW: Apply regularization to prevent explosive estimates
                    # Shrink coefficients for lagged variables toward zero
                    for lag in range(1, p+1):
                        lag_idx_start = (lag-1) * Ntotal
                        lag_idx_end = lag * Ntotal
                        shrinkage_factor = 0.95 ** lag  # More shrinkage for more distant lags
                        Phi_w[lag_idx_start:lag_idx_end, :] *= shrinkage_factor

                    # Constrain diagonal elements of first lag to reasonable range
                    for i in range(Ntotal):
                        # First lag diagonal elements shouldn't be too extreme
                        if abs(Phi_w[i, i]) > 0.9:
                            Phi_w[i, i] = 0.9 * np.sign(Phi_w[i, i])
                    
                    if not is_explosive(Phi_w, Ntotal, p):
                        break
                except Exception as e:
                    if j == 0 and attempts == 0:
                        print(f"VAR coefficient sampling error: {str(e)}, retrying...")
                
                attempts += 1
            
            if attempts == 1000:
                explosive_counter += 1
                print(f"Explosive VAR detected {explosive_counter} times.")
                continue
            
            # Store posterior draws
            if (j % self.thining == 0):
                j_temp = int(j/self.thining)
                Sigmap[j_temp, :, :] = sigma_w
                Phip[j_temp, :, :] = Phi_w
                Cons[j_temp, :] = Phi_w[-1, :]
                valid_draws.append(j_temp)
            
            # Update transition matrix for next iteration - now for all variables
            F[:Ntotal, :Ntotal*p] = Phi_w[:-1, :].T  # All VAR coefficients (excluding constant)
            c[:Ntotal] = np.atleast_2d(Phi_w[-1, :]).T  # Constants for all variables
            
            # Remember to keep the block structure intact for variable lags
            for i in range(p-1):
                F[Ntotal*(i+1):Ntotal*(i+2), Ntotal*i:Ntotal*(i+1)] = np.eye(Ntotal)
            
            # Monthly block shifting for latent states
            for i in range(rmw-1):
                src_pos = monthly_start + i*Nm
                dest_pos = monthly_start + (i+1)*Nm
                F[dest_pos:dest_pos+Nm, src_pos:src_pos+Nm] = np.eye(Nm)
            
            # Quarterly block shifting for latent states
            for i in range(rqw-1):
                src_pos = quarterly_start + i*Nq
                dest_pos = quarterly_start + (i+1)*Nq
                F[dest_pos:dest_pos+Nq, src_pos:src_pos+Nq] = np.eye(Nq)
            
            # Weekly influence for latent states
            F[monthly_start:monthly_start+Nm, :Nw] = 0.1 * np.eye(Nm, Nw)
            F[quarterly_start:quarterly_start+Nq, :Nw] = 0.05 * np.eye(Nq, Nw)
            
            # Update system covariance
            Q[:Ntotal, :Ntotal] = sigma_w
            
        except Exception as e:
            print(f"VAR posterior sampling error: {str(e)}")
            continue
    
    # Store results
    self.Phip = Phip
    self.Sigmap = Sigmap
    self.nv = Ntotal
    self.Nw = Nw
    self.Nm = Nm
    self.Nq = Nq
    self.freq_ratio = freq_ratio_list
    self.select = select_list[0]
    self.varlist = varlist_list
    self.YYactsim_list = YYactsim_list
    self.XXactsim_list = XXactsim_list
    self.explosive_counter = explosive_counter
    self.valid_draws = [draw for draw in valid_draws if draw >= self.nburn/self.thining]

    # Store smoothed states and self components
    self.a_draws = a_draws
    self.F = F
    self.c = c
    self.H_w = H_w
    self.H_m = H_m
    self.H_q = H_q
    self.Q = Q
    
    # Store indices for data extraction
    self.monthly_start = monthly_start
    self.quarterly_start = quarterly_start
    self.rqw = rqw
    self.rmw = rmw
    
    # Store original data
    self.input_data_W = YW
    self.input_data_M = YM
    self.input_data_Q = YQ
    
    # Store data sizes
    self.w_periods = YW.shape[0] if hasattr(YW, 'shape') else 0
    self.m_periods = YM.shape[0] if hasattr(YM, 'shape') else 0
    self.q_periods = YQ.shape[0] if hasattr(YQ, 'shape') else 0
    
    # Store index
    self.index_list = index_list
    
    print(f"Model fitting complete - {len(self.valid_draws)} valid draws")
    print(f"Data periods: Weekly={self.w_periods}, Monthly={self.m_periods}, Quarterly={self.q_periods}")
    
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
    """
    Generate forecasts using the full weekly VAR self that includes all variables.
    
    Parameters:
    -----------
    H : int
        Forecast horizon in highest frequency (weeks)
    conditionals : pandas DataFrame or None
        Conditional forecasts (not fully implemented yet)
    """
    # Store forecast horizon
    self.H = H
    
    # Extract variable counts and self parameters
    Nw = self.Nw  # Weekly variables
    Nm = self.Nm  # Monthly variables
    Nq = self.Nq  # Quarterly variables
    Ntotal = Nw + Nm + Nq  # Total number of variables
    rmw = self.freq_ratio[1]  # Weekly to monthly ratio (4)
    rqw = self.freq_ratio[0] * rmw  # Weekly to quarterly ratio (12)
    
    # Create forecast date index - extending from the last historical date
    self._create_forecast_dates(H)
    
    # Extract variable transformation settings
    select = self.select  # 1 for growth rates, 0 for levels
    
    # Initialize arrays for storing forecasts across draws
    n_valid_draws = len(self.valid_draws)
    forecast_draws = np.zeros((n_valid_draws, H, Ntotal))
    
    print(f"Generating forecasts for {H} periods ({n_valid_draws} valid draws)")
    
    # Process conditionals if provided
    conditional_mask = None
    conditional_values = None
    if conditionals is not None:
        # Convert conditionals to correct format and create mask
        print("Conditional forecasting enabled")
        conditional_mask = ~np.isnan(conditionals.values)
        conditional_values = conditionals.values.copy()
        # Apply transformations to conditional values
        conditional_values[:, select == 1] /= 100  # Convert percentages to decimals
        conditional_values[:, select == 0] = np.log(conditional_values[:, select == 0])  # Log levels
    
    # Loop through all valid MCMC draws
    for i, draw_idx in enumerate(tqdm(self.valid_draws)):
        # Get VAR coefficients and covariance matrix for this draw
        Phi = self.Phip[draw_idx, :, :]  # All VAR coefficients (including constant)
        Sigma = self.Sigmap[draw_idx, :, :]  # Covariance matrix
        
        # Get current state and lagged values from the end of the historical period
        if hasattr(self, 'YYactsim_list') and len(self.YYactsim_list) > 0:
            # Extract the last observed state vector (latest week)
            YYact = self.YYactsim_list[0][draw_idx, -1, :]
            # Extract the lagged values
            XXact = self.XXactsim_list[0][draw_idx, -1, :]
        else:
            # If YYactsim_list doesn't exist, extract from a_draws directly
            # Get the last state from the draw
            last_state = self.a_draws[draw_idx, -1, :]
            
            # Weekly variables (direct)
            YYact = np.zeros(Ntotal)
            YYact[:Nw] = last_state[:Nw]
            
            # Monthly variables (from VAR block)
            YYact[Nw:Nw+Nm] = last_state[Nw:Nw+Nm]
            
            # Quarterly variables (from VAR block)
            YYact[Nw+Nm:Ntotal] = last_state[Nw+Nm:Ntotal]
            
            # Create lagged values based on prior observed values
            nlags = self.nlags
            XXact = np.zeros(Ntotal * nlags + 1)
            for lag in range(nlags):
                if lag < len(self.a_draws[draw_idx]):
                    state_idx = -lag-1
                    # Weekly variables
                    XXact[lag*Ntotal:lag*Ntotal+Nw] = self.a_draws[draw_idx, state_idx, :Nw]
                    # Monthly/quarterly variables from VAR block
                    XXact[lag*Ntotal+Nw:lag*Ntotal+Ntotal] = self.a_draws[draw_idx, state_idx, Nw:Ntotal]
            XXact[-1] = 1.0  # Constant term
        
        # Iteratively generate forecasts for H periods
        forecast_this_draw = np.zeros((H, Ntotal))
        
        # Storage for current prediction and its lagged values
        YYpred = np.zeros((H+1, Ntotal))  # Includes current state (t=0) and H forecasts
        YYpred[0, :] = YYact  # Current state
        
        XXpred = np.zeros((H+1, Ntotal*self.nlags+1))  # Includes current lags and constant
        XXpred[0, :] = XXact  # Current lags
        
        # Generate random errors for all forecast periods
        error_pred = np.zeros((H, Ntotal))
        for h in range(H):
            try:
                # Draw random errors from multivariate normal with covariance Sigma
                error_pred[h, :] =  np.random.default_rng().multivariate_normal(
                    mean=np.zeros(Ntotal), 
                    cov=Sigma,
                    method="cholesky"
                )
            except np.linalg.LinAlgError:
                # Fallback if Cholesky decomposition fails
                print(f"Warning: Cholesky decomposition failed for draw {draw_idx}, using diagonal covariance")
                error_pred[h, :] = np.random.normal(0, np.sqrt(np.diag(Sigma)))
        
        # Generate forecasts period-by-period
        for h in range(1, H+1):
            # Update lags - shift previous values
            XXpred[h, Ntotal:Ntotal*self.nlags] = XXpred[h-1, :Ntotal*(self.nlags-1)]
            # Add newest observation to first lag
            XXpred[h, :Ntotal] = YYpred[h-1, :]
            # Add constant term
            XXpred[h, -1] = 1.0
            
            # Generate forecast using VAR equation
            if conditional_mask is not None and h-1 < conditional_mask.shape[0]:
                # Apply conditionals where specified (where mask is True)
                # First generate the unconstrained forecast
                unconstrained_forecast = XXpred[h, :] @ Phi + error_pred[h-1, :]
                
                # Then override with conditional values where specified
                mask = conditional_mask[h-1, :]
                forecast_values = np.where(
                    mask,
                    conditional_values[h-1, :],  
                    unconstrained_forecast
                )
                YYpred[h, :] = forecast_values
            else:
                # Standard forecast - VAR equation with random error
                YYpred[h, :] = XXpred[h, :] @ Phi + error_pred[h-1, :]
        
        # Store forecasts for this draw (excluding the initial state)
        forecast_this_draw = YYpred[1:, :]
        forecast_draws[i, :, :] = forecast_this_draw
    
    # Calculate statistics across draws
    print("Calculating forecast statistics across draws")
    
    # Mean forecast
    YYftr_m = np.nanmean(forecast_draws, axis=0)
    
    # Median forecast
    YYftr_med = np.nanmedian(forecast_draws, axis=0)
    
    # Quantiles for uncertainty bands
    YYftr_095 = np.nanquantile(forecast_draws, q=0.95, axis=0)
    YYftr_005 = np.nanquantile(forecast_draws, q=0.05, axis=0)
    YYftr_084 = np.nanquantile(forecast_draws, q=0.84, axis=0)
    YYftr_016 = np.nanquantile(forecast_draws, q=0.16, axis=0)
    
    # Apply transformations based on variable type
    # Growth rates: multiply by 100
    # Levels: apply exponential
    
    # Initialize transformed arrays
    YYftr_m_trans = YYftr_m.copy()
    YYftr_med_trans = YYftr_med.copy()
    YYftr_095_trans = YYftr_095.copy()
    YYftr_005_trans = YYftr_005.copy()
    YYftr_084_trans = YYftr_084.copy()
    YYftr_016_trans = YYftr_016.copy()
    
    # Apply transformations
    for i in range(Ntotal):
        if i < len(select):
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
    
    # Store the transformed forecasts
    self.YYftr_m = YYftr_m
    self.YYftr_med = YYftr_med
    self.YYftr_095 = YYftr_095
    self.YYftr_005 = YYftr_005
    self.YYftr_084 = YYftr_084
    self.YYftr_016 = YYftr_016
    
    self.YYftr_m_trans = YYftr_m_trans
    self.YYftr_med_trans = YYftr_med_trans
    self.YYftr_095_trans = YYftr_095_trans
    self.YYftr_005_trans = YYftr_005_trans
    self.YYftr_084_trans = YYftr_084_trans
    self.YYftr_016_trans = YYftr_016_trans
    
    # Store raw forecast draws
    self.forecast_draws = forecast_draws
    
    # EXTRACT HISTORICAL DATA FOR PLOTS
    # ------------------------------
    # Calculate full history length for extraction
    hist_length = self.input_data_W.shape[0] - self.nlags
    
    # Set index for all periods (history + forecast)
    self._create_forecast_dates(H, hist_length)
    
    # Extract historical data 
    print(f"Extracting historical data for {hist_length} periods")
    history_data = self.extract_historical_data(hist_length, Nw, Nm, Nq)
    
    # Combine with forecasts
    combined_mean = np.vstack((history_data['mean'], YYftr_m_trans))
    combined_median = np.vstack((history_data['median'], YYftr_med_trans))
    combined_095 = np.vstack((history_data['095'], YYftr_095_trans)) 
    combined_005 = np.vstack((history_data['005'], YYftr_005_trans))
    combined_084 = np.vstack((history_data['084'], YYftr_084_trans))
    combined_016 = np.vstack((history_data['016'], YYftr_016_trans))
    
    
    # Create DataFrames with dates and variable names
    self.YY_mean_pd = pd.DataFrame(combined_mean, index=self.fcast_dates, columns=self.varlist)
    self.YY_median_pd = pd.DataFrame(combined_median, index=self.fcast_dates, columns=self.varlist)
    self.YY_095_pd = pd.DataFrame(combined_095, index=self.fcast_dates, columns=self.varlist)
    self.YY_005_pd = pd.DataFrame(combined_005, index=self.fcast_dates, columns=self.varlist)
    self.YY_084_pd = pd.DataFrame(combined_084, index=self.fcast_dates, columns=self.varlist)
    self.YY_016_pd = pd.DataFrame(combined_016, index=self.fcast_dates, columns=self.varlist)
    
    
    print(f"Forecast generation complete. Results saved in YY_*_pd DataFrames.")
    return



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
    Extract historical data with correct alignment to original observations.
    
    This revised version ensures proper extraction from latent states and
    direct mapping to original observations when available.
    """
    # Total variables
    Ntotal = Nw + Nm + Nq
    
    # Initialize arrays for different statistics
    history_mean = np.zeros((hist_length, Ntotal))
    history_median = np.zeros((hist_length, Ntotal))
    history_095 = np.zeros((hist_length, Ntotal))
    history_005 = np.zeros((hist_length, Ntotal))
    history_084 = np.zeros((hist_length, Ntotal))
    history_016 = np.zeros((hist_length, Ntotal))
    
    # Get frequency ratios
    rmw = self.freq_ratio[1]  # Monthly to weekly ratio (4)
    rqw = self.freq_ratio[0] * self.freq_ratio[1]  # Quarterly to weekly ratio (12)
    
    # Get variable indices in the state vector
    monthly_start = self.monthly_start
    quarterly_start = self.quarterly_start
    
    print(f"Extracting {hist_length} periods of historical data")
    
    # PART 1: WEEKLY DATA - Use raw input values
    # -----------------------------------------
    if hasattr(self, 'input_data_W') and self.input_data_W is not None:
        w_data = self.input_data_W
        if isinstance(w_data, pd.DataFrame):
            w_data = w_data.values

        history_mean[:, :Nw] = w_data[self.nlags:, :Nw]
        history_median[:, :Nw] = w_data[self.nlags:, :Nw]
        history_095[:, :Nw] = np.nan
        history_005[:, :Nw] = np.nan
        history_084[:, :Nw] = np.nan
        history_016[:, :Nw] = np.nan
    
    # PART 2: MONTHLY & QUARTERLY DATA - PRIORITIZE ORIGINAL OBSERVATIONS
    # -----------------------------------------------------------------
    
    if hasattr(self, 'a_draws') and hasattr(self, 'valid_draws') and len(self.valid_draws) > 0:
        # Get array of valid draws
        valid_draws = self.valid_draws
        n_draws = len(valid_draws)
        
        
        weekly_quarterly_latent = self.get_weekly_quarterly_latent(draw_indices = valid_draws)
        weekly_monthly_latent = self.get_weekly_monthly_latent(draw_indices = valid_draws)
        
        history_mean[:, Nw:Nw+Nm] = np.nanmean(weekly_monthly_latent, axis = 0)
        history_median[:, Nw:Nw+Nm] = np.nanquantile(weekly_monthly_latent, axis = 0, q = 0.5)
        history_095[:, Nw:Nw+Nm] = np.nanquantile(weekly_monthly_latent, axis = 0, q = 0.95)
        history_005[:, Nw:Nw+Nm] = np.nanquantile(weekly_monthly_latent, axis = 0, q = 0.05)
        history_084[:, Nw:Nw+Nm] = np.nanquantile(weekly_monthly_latent, axis = 0, q = 0.84)
        history_016[:, Nw:Nw+Nm] = np.nanquantile(weekly_monthly_latent, axis = 0, q = 0.16)
        
        history_mean[:, Nw+Nm:] = np.nanmean(weekly_quarterly_latent, axis = 0)
        history_median[:, Nw+Nm:] = np.nanquantile(weekly_quarterly_latent, axis = 0, q = 0.5)
        history_095[:, Nw+Nm:] = np.nanquantile(weekly_quarterly_latent, axis = 0, q = 0.95)
        history_005[:, Nw+Nm:] = np.nanquantile(weekly_quarterly_latent, axis = 0, q = 0.05)
        history_084[:, Nw+Nm:] = np.nanquantile(weekly_quarterly_latent, axis = 0, q = 0.84)
        history_016[:, Nw+Nm:] = np.nanquantile(weekly_quarterly_latent, axis = 0, q = 0.16)     
        
    # PART 4: APPLY TRANSFORMATIONS
    # ---------------------------
    print("Applying variable transformations")
    
    # Make copies for transformed data
    history_mean_trans = history_mean.copy()
    history_median_trans = history_median.copy()
    history_095_trans = history_095.copy()
    history_005_trans = history_005.copy()
    history_084_trans = history_084.copy()
    history_016_trans = history_016.copy()
    
    # Apply transformations based on variable type
    for i in range(Ntotal):
        if i < len(self.select):
            # Only transform non-zero values
            mask = history_mean[:, i] != 0
            
            if np.any(mask):
                if self.select[i] == 1:  # Growth rate
                    history_mean_trans[mask, i] = 100 * history_mean[mask, i]
                    history_median_trans[mask, i] = 100 * history_median[mask, i]
                    history_095_trans[mask, i] = 100 * history_095[mask, i]
                    history_005_trans[mask, i] = 100 * history_005[mask, i]
                    history_084_trans[mask, i] = 100 * history_084[mask, i]
                    history_016_trans[mask, i] = 100 * history_016[mask, i]
                    var_name = self.varlist[i] if i < len(self.varlist) else f"var_{i+1}"
                    print(f"Variable {i+1} ({var_name}): Applied growth rate transformation (×100)")
                else:  # Level
                    history_mean_trans[mask, i] = np.exp(history_mean[mask, i])
                    history_median_trans[mask, i] = np.exp(history_median[mask, i])
                    history_095_trans[mask, i] = np.exp(history_095[mask, i])
                    history_005_trans[mask, i] = np.exp(history_005[mask, i])
                    history_084_trans[mask, i] = np.exp(history_084[mask, i])
                    history_016_trans[mask, i] = np.exp(history_016[mask, i])
                    var_name = self.varlist[i] if i < len(self.varlist) else f"var_{i+1}"
                    print(f"Variable {i+1} ({var_name}): Applied level transformation (exp)")
    
    # Return as dictionary for easy access
    history_data = {
        'mean': history_mean_trans,
        'median': history_median_trans,
        '095': history_095_trans,
        '005': history_005_trans,
        '084': history_084_trans,
        '016': history_016_trans,
        'raw_mean': history_mean,  # Also store untransformed data
        'raw_median': history_median
    }
    
    # Check data completeness
    non_zero_count = np.count_nonzero(history_mean_trans)
    total_values = history_mean_trans.size
    print(f"Historical data extraction complete: {non_zero_count}/{total_values} values filled ({non_zero_count/total_values*100:.1f}%)")
    
    return history_data

def debug_quarterly_extraction(self):
    """
    Diagnose quarterly data extraction issues by comparing original vs. extracted values.
    """
    if not hasattr(self, 'input_data_Q') or not hasattr(self, 'YY_mean_pd'):
        print("Missing quarterly data or extracted data")
        return
    
    print("\n== QUARTERLY DATA EXTRACTION FROM LATENT STATES ==")
    
    q_data = self.input_data_Q
    if isinstance(q_data, pd.DataFrame):
        q_data = q_data.values
    
    # Original quarterly data
    print(f"Original quarterly data shape: {q_data.shape}")
    
    # Extracted data
    hist_data = self.YY_mean_pd
    print(f"Historical data shape: {hist_data.shape}")
    
    # Check quarterly aggregation from latent states
    print("Verifying latent state extraction:")
    
    # Get frequency ratio
    rqw = self.freq_ratio[0] * self.freq_ratio[1]  # Quarterly to weekly ratio (12)
    Nm = self.Nm  # Number of monthly variables
    Nq = self.Nq  # Number of quarterly variables
    
    # Examine first few quarters in the latent states
    for t in range(5):
        # Determine quarter index
        q_idx = -78  # Far back in time
        
        # Print quarter number and verification results
        print(f"\nPeriod {t} (quarter {q_idx}):")
        
        # For each quarterly variable, compare original vs. latent
        for q in range(Nq):
            # Extract original value from quarterly data
            orig_val = 0
            try:
                if q_idx >= 0 and q_idx < q_data.shape[0]:
                    orig_val = q_data[q_idx, q]
            except:
                pass
            
            # Extract aggregated value from latent states
            latent_val = 0
            
            # Check condition (whether aggregation matches original)
            error = abs(latent_val - orig_val)
            match_status = "MATCH" if error < 1e-4 else "MISMATCH"
            
            # Print results
            print(f"  Q var {q+1}: Original={orig_val:.6f}, Latent agg={latent_val:.6f} - {match_status} (error={error:.6f})")
    
    # Now check some actual quarters in the extracted data
    print("\nExtracted quarterly values in historical data:")
    
    # Get index dates for reference
    date_index = self.YY_mean_pd.index
    
    # Check last 5 quarters
    for q_offset in range(1, 6):
        # Calculate quarter start and end in the historical data
        q_end_idx = len(date_index) - 96 - (q_offset-1)*rqw - 1  # 96 = forecast horizon
        q_start_idx = q_end_idx - rqw + 1
        
        if q_start_idx < 0 or q_end_idx >= len(date_index):
            continue
        
        # Get quarter start and end dates
        q_start_date = date_index[q_start_idx]
        q_end_date = date_index[q_end_idx]
        
        # Calculate corresponding quarter index in original data
        q_idx = -q_offset
        
        print(f"\nQuarter {q_idx} ({q_start_date} to {q_end_date}):")
        
        # Original values from quarterly data
        orig_values = q_data[q_idx, :Nq] if q_idx >= -q_data.shape[0] else np.zeros(Nq)
        print(f"  Original values: {orig_values}")
        
        # Apply transformations to original values for proper comparison
        orig_transformed = orig_values.copy()
        for q in range(Nq):
            if q < len(self.select):
                if self.select[Nm+q] == 1:  # Growth rate
                    orig_transformed[q] = 100 * orig_values[q]
                else:  # Level
                    orig_transformed[q] = np.exp(orig_values[q])
        print(f"  Transformed original: {orig_transformed}")
        
        # Get extracted values for the first week of this quarter
        extracted_values = self.YY_mean_pd.iloc[q_start_idx, -Nq-1:-1].values
        print(f"  First week extract: {extracted_values}")
        
        # Check consistency
        is_consistent = True
        for q in range(min(Nq, len(extracted_values))):
            if abs(orig_transformed[q] - extracted_values[q]) > 1e-4:
                is_consistent = False
                break
                
        print(f"  Values are consistent: {is_consistent}")
        
        # Show detailed comparison for each variable
        for q in range(min(Nq, len(extracted_values))):
            error = abs(orig_transformed[q] - extracted_values[q])
            match_status = "MATCH" if error < 1e-4 else "MISMATCH"
            print(f"  Q var {q+1}: Orig={orig_transformed[q]:.6f}, Extracted={extracted_values[q]:.6f} - {match_status} (error={error:.6f})")
            
def _create_forecast_dates(self, H, hist_length=None):
    """
    Create date indices for both historical and forecast periods.
    
    Parameters:
    -----------
    H : int
        Forecast horizon in highest frequency (weeks)
    hist_length : int, optional
        Historical data length, will be calculated if not provided
    """
    
    # Check if we have dates from the self
    dates = self.index_list[-1][self.nlags:]
    last_date = dates[-1]
    start_date = last_date + pd.Timedelta(days=7)
    # Get the weekday name (lowercase)
    weekday_name = start_date.day_name()[:3].upper()
    # Create custom weekly frequency with the specific weekday
    custom_weekly = f'W-{weekday_name}'
    extra_dates = pd.date_range(start=start_date, periods=self.H, freq=custom_weekly)
    
    # Combine historical and forecast dates
    self.fcast_dates = dates.union(extra_dates)
    
    return None

def debug_latent_states(self):
    """
    Debug method to check latent state values
    """
    if not hasattr(self, 'a_draws') or len(self.valid_draws) == 0:
        print("No valid latent states available")
        return
    
    # Get first valid draw
    j = self.valid_draws[0]
    
    # Get dimensions
    Nw = self.Nw
    Nm = self.Nm
    Nq = self.Nq
    nobs = self.a_draws.shape[1]
    
    print(f"Examining latent states for draw {j}")
    print(f"State dimensions: {self.a_draws.shape}")
    
    # Check first few periods
    for t in range(min(5, nobs)):
        print(f"\nPeriod {t}:")
        
        # Print weekly variables
        print("Weekly variables:")
        for w in range(Nw):
            print(f"  W{w+1}: {self.a_draws[j, t, w]:.8f}")
        
        # Print monthly variables from VAR block
        print("Monthly variables (VAR block):")
        for m in range(Nm):
            print(f"  M{m+1}: {self.a_draws[j, t, Nw+m]:.8f}")
        
        # Print quarterly variables from VAR block
        print("Quarterly variables (VAR block):")
        for q in range(Nq):
            print(f"  Q{q+1}: {self.a_draws[j, t, Nw+Nm+q]:.8f}")
    
    # Check if latent states have been populated
    if hasattr(self, 'monthly_start') and hasattr(self, 'quarterly_start'):
        m_start = self.monthly_start
        q_start = self.quarterly_start
        
        # Check first month's latent states
        print("\nFirst month latent states (first period):")
        for m in range(Nm):
            states = [self.a_draws[j, 0, m_start + w*Nm + m] for w in range(self.freq_ratio[1])]
            print(f"  M{m+1}: {states}")
        
        # Check first quarter's latent states
        print("\nFirst quarter latent states (first period):")
        for q in range(Nq):
            states = [self.a_draws[j, 0, q_start + w*Nq + q] for w in range(self.freq_ratio[0]*self.freq_ratio[1])]
            print(f"  Q{q+1}: {states[:4]}... (showing first 4 of {len(states)})")
            
            
def get_weekly_quarterly_latent(self, draw_indices=None):
    """
    Extract quarterly latent values as a weekly time series.
    
    Parameters:
    -----------
    self : SBFVAR self object
        The fitted self
    draw_indices : list or None
        Specific draw indices to extract (default: None, uses all draws)
    
    Returns:
    --------
    weekly_quarterly_latent : ndarray
        Array with shape [n_draws, n_weekly_times, n_quarterly_vars]
        Weekly latent values for quarterly variables
    """
    # Get dimensions
    Nq = self.Nq  # Number of quarterly variables
    rqw = self.rqw  # Weeks per quarter (typically 12)
    quarterly_start = self.quarterly_start  # Starting index for quarterly latent states
    
    # Number of time periods (quarters)
    T = self.a_draws.shape[1]
    
    # Use all draws if not specified
    if draw_indices is None:
        if hasattr(self, 'valid_draws') and len(self.valid_draws) > 0:
            draw_indices = self.valid_draws
        else:
            draw_indices = range(self.a_draws.shape[0])
    
    n_draws = len(draw_indices)
    
    # Initialize array for weekly time series
    # Shape: [draws, weekly_time, quarterly_vars]
    weekly_quarterly_latent = np.zeros((n_draws, T, Nq))
    
    # Extract and organize quarterly latent values as weekly time series
    for d_idx, d in enumerate(draw_indices):
        for t in range(T):
            for q in range(Nq):
                # For each quarter and variable, get the corresponding week in the quarter
                # This assumes the state is organized with weeks cycle within each quarter
                week_in_quarter = t % rqw
                state_idx = quarterly_start + week_in_quarter * Nq + q
                weekly_quarterly_latent[d_idx, t, q] = self.a_draws[d, t, state_idx]
                
    return weekly_quarterly_latent

def get_weekly_monthly_latent(self, draw_indices=None):
    """
    Extract monthly latent values as a weekly time series.
    
    Parameters:
    -----------
    self : SBFVAR self object
        The fitted self
    draw_indices : list or None
        Specific draw indices to extract (default: None, uses all draws)
    
    Returns:
    --------
    weekly_monthly_latent : ndarray
        Array with shape [n_draws, n_weekly_times, n_monthly_vars]
        Weekly latent values for monthly variables
    """
    # Get dimensions
    Nm = self.Nm  # Number of monthly variables
    rmw = self.rmw  # Weeks per month (typically 4)
    monthly_start = self.monthly_start  # Starting index for monthly latent states
    
    # Number of time periods (weeks)
    T = self.a_draws.shape[1]
    
    # Use all draws if not specified
    if draw_indices is None:
        if hasattr(self, 'valid_draws') and len(self.valid_draws) > 0:
            draw_indices = self.valid_draws
        else:
            draw_indices = range(self.a_draws.shape[0])
    
    n_draws = len(draw_indices)
    
    # Initialize array for weekly time series
    # Shape: [draws, weekly_time, monthly_vars]
    weekly_monthly_latent = np.zeros((n_draws, T, Nm))
    
    # Extract and organize monthly latent values as weekly time series
    for d_idx, d in enumerate(draw_indices):
        for t in range(T):
            for m in range(Nm):
                # For each month and variable, get the corresponding week in the month
                # This assumes the state is organized with weeks cycling within each month
                week_in_month = t % rmw
                state_idx = monthly_start + week_in_month * Nm + m
                weekly_monthly_latent[d_idx, t, m] = self.a_draws[d, t, state_idx]
    

    return weekly_monthly_latent