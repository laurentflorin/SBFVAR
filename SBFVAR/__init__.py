import os
import sys

import pandas as pd
import numpy as np

from collections import deque
import copy
import itertools

class multifrequency_var:
    
    '''
    MUFBVAR class
    
    Parameters
    ----------
    nsim : int
        Number of simulations
    nburn_perc : numeric
        Between 0 and 1, proportion of simulations to throw away as burn in.
    nlags : int
        Number of lags in the highest frequency
    thining : int
        To save only every nth draw


    '''
    
    def __init__(self, nsim, nburn_perc, nlags, thining):
        
        self.nsim = nsim
        self.nburn_perc = nburn_perc
        self.nlags = nlags
        self.thining = thining
        
    # Imported methods
    from ._estimation import fit, forecast, aggregate
    from ._save import to_excel, save

    
class mufbvar_data:
    
    """
    Class to prepare the data that will be used in the MUFBVAR
    ...

    Parameters
    ----------
    data : list of pandas DataFrames
        Data of each frequency stored in a pandas DataFrame, all stored in one list
    trans : list of numpy arrays
        A separate numpy array for each frequency all stored in a list. /n
        0: log is taken 
        1: divided by 100
    frequencies : List of the frequencies of the data, in order lowest to highest 
        "Y", "Q", "M", "W", "D" 
    """
    
    def __init__(self, data, trans, frequencies):
        
        
        
        # Creating lists of highfrequency data
        
        YMX_list = deque()
        YM0_list = deque()
        select_m_list = deque()
        vars_m_list = deque()
        YMh_list = deque()
        exc_list = deque()
        index_list = deque()
        
        
        for i in range(1, len(frequencies)):
            YMX_list.append(data[i])
            YM0_list.append(data[i].to_numpy())
            select_m_list.append(trans[i])
            vars_m_list.append(data[i].columns[:])
            YMh_list.append(data[i].to_numpy())
            index_list.append(data[i].index)
        
        if not isinstance(index_list[-1], pd.DatetimeIndex):
            try:
                index_list[-1] = pd.to_datetime(index_list[-1])
            except:
                print("Index must be of the form 'YYYY-MM-DD'")
        
        
        input_data = copy.deepcopy(YMX_list)
        
        # Creating list of low frequency data
        # Here only the first entry with the lowest frequency data is generated
        # The other entries are generated during the estimation
        
        YQX_list = deque()
        YQ0_list = deque()
        select_q = deque()
        
        YQX_list.append(data[0])
        YQ0_list.append(YQX_list[-1].to_numpy())
        select_q.append(trans[0])
        vars_q = YQX_list[0].columns[:]
        
        input_data_Q =  copy.deepcopy(YQX_list[0])
        
        
        varlist_list = deque()
        select_list = deque()
        select_c_list = deque()
        
        Nm_list = deque()
        nv_list = deque()
        Nq_list = deque()
        
        Nq_list.append(YQX_list[0].shape[1])
        
        for i in range(len(YMX_list)-1):
            Nq_list.append(YQX_list[0].shape[1] + YMX_list[i].shape[1]) 
            if len(select_m_list[i]) == 0:
                select_q.append(select_q[0])
            else:
                select_q.append(np.hstack((select_m_list[i], select_q[0])))
        
        # FIX: Improve variable list creation to properly include all frequencies
        for i in range(len(YMX_list)):
            if i > 0:
                # For weekly level (i=1), combine weekly + monthly + quarterly variables
                # Collect all higher frequency variable lists up to current level
                new_list = [item for item in list(itertools.islice(select_m_list, 0, i+1)) if len(item) > 0]
                select_list.append(np.hstack((np.hstack(list(reversed(new_list))), select_q[0])))
                
                # Collect all variable names from higher frequencies (weekly, monthly)
                rev_vars_m = list(itertools.islice(vars_m_list, 0, i+1))
                rev_vars_m.reverse()
                
                # Create combined variable list: weekly + monthly + quarterly
                try:
                    combined_vars = np.hstack(rev_vars_m)
                    # Ensure it's a flat array
                    if isinstance(combined_vars, np.ndarray) and combined_vars.ndim > 1:
                        combined_vars = np.squeeze(combined_vars)
                    varlist_list.append(np.hstack((combined_vars, vars_q)))
                except ValueError as e:
                    print(f"WARNING: Error combining variable lists: {e}")
                    # Fallback to manually concatenating variable names
                    flat_vars = []
                    for var_list in rev_vars_m:
                        flat_vars.extend(var_list)
                    flat_vars.extend(vars_q)
                    varlist_list.append(np.array(flat_vars))
                    
            else:
                # For monthly level (i=0), combine monthly + quarterly variables
                if select_m_list[i].size:
                    select_list.append(np.hstack((select_m_list[i], select_q[0])))
                    varlist_list.append(np.hstack((vars_m_list[i], vars_q)))
                else:
                    select_list.append(select_q[0])
                    if hasattr(vars_q, 'shape') and len(vars_q.shape) > 1:
                        varlist_list.append(np.squeeze(vars_q))
                    else:
                        varlist_list.append(vars_q)
            
            Nm_list.append(int(np.shape(YM0_list[i])[1]))
            nv_list.append(int(Nm_list[i] + Nq_list[i]))
        
        # Print debug info to verify the variable lists
        print(f"DEBUG INFO - Variable counts by frequency:")
        print(f"  Quarterly variables: {YQX_list[0].shape[1]}")
        for i, freq in enumerate(frequencies[1:]):
            if i < len(YMX_list):
                print(f"  {freq} variables: {YMX_list[i].shape[1]}")
        
        print(f"DEBUG INFO - Combined variable lists:")
        for i, varlist in enumerate(varlist_list):
            print(f"  varlist_list[{i}]: {len(varlist)} variables - {varlist}")
        
        select_list_sep = list(select_q)
        select_list_sep.extend(select_m_list)
        
        
        # Calculate the frequency ratios
        
        freq_ratio_list = deque()
        
        for freq in range(1,len(frequencies)):
            freq_lf = frequencies[freq-1]
            freq_hf = frequencies[freq]
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
                
            else:
                print("Higher frequency: ", freq_hf, " Lower Frequency: ", freq_lf, end ='\n')
                freq_ratio = input("Please enter frequency ratio")
            
            freq_ratio_list.append(freq_ratio)
        
        
        #performe data transformations
        
        for i in range(len(YM0_list)):
            if select_m_list[i].size:
                YM0_list[i][:,(select_m_list[i] == 1)] = YM0_list[i][:,(select_m_list[i] == 1)]/100
                YM0_list[i][:,(select_m_list[i] == 0)] = np.log(YM0_list[i][:,(select_m_list[i] == 0)])
        
        YQ0_list[0][:,(select_q[0] == 1)] = YQ0_list[0][:,(select_q[0] == 1)]/100
        YQ0_list[0][:,(select_q[0] == 0)] = np.log(YQ0_list[0][:,(select_q[0] == 0)])
        
        
        YM_list = copy.deepcopy(YM0_list)
        
        # Low frequency data in higher frequency
        YQ_list = deque()
        YQ_list.append(np.kron(YQ0_list[0], np.ones((freq_ratio_list[0],1))))
        
        Tstar_list = deque()
        T_list = deque()
        YDATA_list = deque()
        
        if YM_list[0].size:
            Tstar_list.append(YM_list[0].shape[0])
            YDATA_list.append(np.full((Tstar_list[0],nv_list[0]), np.nan))
            YDATA_list[0][:,:Nm_list[0]] = YM_list[0]
        else:
            Tstar_list.append(YQ_list[0].shape[0])
            YDATA_list.append(np.full((0,nv_list[0]), np.nan))
            
        T_list.append(YQ_list[0].shape[0])
        #for freq in range(1,len(self.frequencies)-1):
        #    T_list.append(np.kron(YMX_list[freq-1], np.ones((freq_ratio_list[freq],1))).shape[0])
        
        
        if YDATA_list[0].size:     
            YDATA_list[0][:T_list[0],Nm_list[0]:] = YQ_list[0]   
        else:
            YDATA_list[0] = YQ_list[0] 
            
        
        #attach variables to instance
        
        self.YMX_list = YMX_list
        self.YM0_list = YM0_list
        self.select_m_list = select_m_list
        self.vars_m_list = vars_m_list
        self.YMh_list = YMh_list
        self.exc_list = exc_list
        self.index_list = index_list
        self.frequencies = frequencies
        self.YQX_list = YQX_list
        self.YQ0_list = YQ0_list
        self.select_q = select_q
        self.input_data_Q =  input_data_Q
        self.varlist_list = varlist_list
        self.select_list = select_list
        self.select_c_list = select_c_list
        self.Nm_list = Nm_list
        self.nv_list = nv_list
        self.Nq_list = Nq_list
        self.select_list_sep = select_list_sep
        self.freq_ratio_list = freq_ratio_list
        self.YQ_list = YQ_list
        self.Tstar_list = Tstar_list
        self.T_list = T_list
        self.YDATA_list = YDATA_list
        self.YM_list = YM_list
        self.input_data = input_data