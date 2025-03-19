import os
import sys

import numpy as np
import math

from collections import deque

from scipy.stats import invwishart
import pandas as pd
from scipy.stats import multivariate_normal
from datetime import datetime
from pandas.tseries.offsets import Week , MonthBegin, QuarterBegin, Day

import itertools

import matplotlib.pyplot as plt

#for progressbar
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position = 0, leave=True) # this line does the magic

# for plots

import matplotlib.backends.backend_pdf

import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default='browser'
import plotly.express as px

#to save objects
import pickle
import copy

def save(self, filename="mufbvar_model"):
    '''
    Saves the MFBVAR Object
    
    Parameters
    ----------
    filename : str
        Path where to save the object. End must be .pkl
    '''
    with open(filename + ".pkl", 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        
def to_excel(self, filename, agg=False, include_metadata=True):
    '''
    Writes the results to an excel file, handling both regular and aggregated forecasts.
    
    Parameters
    ----------
    filename : str
        File path for the Excel output
    agg : bool, default False
        Whether to use the aggregated series instead of standard forecasts
    include_metadata : bool, default True
        Whether to include model metadata in a separate sheet
    '''
    # Check if we have forecast data available
    has_forecasts = hasattr(self, 'forecast_draws_list') or hasattr(self, 'YY_mean_pd')
    has_aggregated = hasattr(self, 'YY_mean_agg') or hasattr(self, 'aggregated_forecast')
    
    # Model metadata for Excel
    if include_metadata:
        metadata = {
            'Parameter': [
                'Model Type', 'Date Created', 'Variables Count', 
                'Weekly Variables', 'Monthly Variables', 'Quarterly Variables',
                'Forecast Horizon', 'Aggregation Frequency'
            ],
            'Value': [
                'Multi-Frequency BVAR',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                self.nv if hasattr(self, 'nv') else 'N/A',
                self.Nw, self.Nm, self.Nq,
                self.H if hasattr(self, 'H') else 'N/A',
                self.agg_freq if hasattr(self, 'agg_freq') and agg else 'None'
            ]
        }
        metadata_df = pd.DataFrame(metadata)
    
    # Get variable counts
    Nw = self.Nw  # Weekly variables
    Nm = self.Nm  # Monthly variables
    Nq = self.Nq  # Quarterly variables
    Ntotal = Nw + Nm + Nq  # Total variables
    
    # Get variable list
    varlist = self.varlist if hasattr(self, 'varlist') else None
    
    if not has_forecasts:
        # For models where we only have MCMC output without forecasts
        print("No forecasts available. Creating Excel file from smoothed states...")
        
        # Get selection vector for transformations
        select = self.select if hasattr(self, 'select') else np.zeros(Ntotal)
        
        # Check if we have smoothed states
        if hasattr(self, 'a_draws'):
            # Get valid draws (after burn-in)
            valid_draws = self.valid_draws if hasattr(self, 'valid_draws') else \
                          [i for i in range(self.nsim) if i >= self.nburn]
            
            # Get index for the output
            if len(self.index_list[-1]) == self.a_draws.shape[1]:
                index = copy.deepcopy(self.index_list[-1])
            else:
                # Create a numeric index if date index not available
                index = pd.RangeIndex(self.a_draws.shape[1])
            
            # Process smoothed states - SAFELY without boolean indexing
            
            # Extract mean of smoothed states
            mean_states = np.mean(self.a_draws[valid_draws], axis=0)
            median_states = np.median(self.a_draws[valid_draws], axis=0)
            p095_states = np.quantile(self.a_draws[valid_draws], 0.95, axis=0)
            p005_states = np.quantile(self.a_draws[valid_draws], 0.05, axis=0)
            p084_states = np.quantile(self.a_draws[valid_draws], 0.84, axis=0)
            p016_states = np.quantile(self.a_draws[valid_draws], 0.16, axis=0)
            
            # Apply transformations safely using loops
            
            # For mean states
            transformed_mean = mean_states.copy()
            transformed_median = median_states.copy()
            transformed_p095 = p095_states.copy()
            transformed_p005 = p005_states.copy()
            transformed_p084 = p084_states.copy()
            transformed_p016 = p016_states.copy()
            
            # Weekly variables
            for i in range(min(Nw, transformed_mean.shape[1])):
                if i < select.shape[0]:
                    if select[i] == 1:  # Growth rate
                        transformed_mean[:, i] = 100 * mean_states[:, i]
                        transformed_median[:, i] = 100 * median_states[:, i]
                        transformed_p095[:, i] = 100 * p095_states[:, i]
                        transformed_p005[:, i] = 100 * p005_states[:, i]
                        transformed_p084[:, i] = 100 * p084_states[:, i]
                        transformed_p016[:, i] = 100 * p016_states[:, i]
                    else:  # Level
                        transformed_mean[:, i] = np.exp(mean_states[:, i])
                        transformed_median[:, i] = np.exp(median_states[:, i])
                        transformed_p095[:, i] = np.exp(p095_states[:, i])
                        transformed_p005[:, i] = np.exp(p005_states[:, i])
                        transformed_p084[:, i] = np.exp(p084_states[:, i])
                        transformed_p016[:, i] = np.exp(p016_states[:, i])
            
            # Monthly variables
            for i in range(min(Nm, transformed_mean.shape[1] - Nw)):
                idx = Nw + i
                if idx < select.shape[0]:
                    if select[idx] == 1:  # Growth rate
                        transformed_mean[:, idx] = 100 * mean_states[:, idx]
                        transformed_median[:, idx] = 100 * median_states[:, idx]
                        transformed_p095[:, idx] = 100 * p095_states[:, idx]
                        transformed_p005[:, idx] = 100 * p005_states[:, idx]
                        transformed_p084[:, idx] = 100 * p084_states[:, idx]
                        transformed_p016[:, idx] = 100 * p016_states[:, idx]
                    else:  # Level
                        transformed_mean[:, idx] = np.exp(mean_states[:, idx])
                        transformed_median[:, idx] = np.exp(median_states[:, idx])
                        transformed_p095[:, idx] = np.exp(p095_states[:, idx])
                        transformed_p005[:, idx] = np.exp(p005_states[:, idx])
                        transformed_p084[:, idx] = np.exp(p084_states[:, idx])
                        transformed_p016[:, idx] = np.exp(p016_states[:, idx])
            
            # Quarterly variables
            for i in range(min(Nq, transformed_mean.shape[1] - Nw - Nm)):
                idx = Nw + Nm + i
                if idx < select.shape[0]:
                    if select[idx] == 1:  # Growth rate
                        transformed_mean[:, idx] = 100 * mean_states[:, idx]
                        transformed_median[:, idx] = 100 * median_states[:, idx]
                        transformed_p095[:, idx] = 100 * p095_states[:, idx]
                        transformed_p005[:, idx] = 100 * p005_states[:, idx]
                        transformed_p084[:, idx] = 100 * p084_states[:, idx]
                        transformed_p016[:, idx] = 100 * p016_states[:, idx]
                    else:  # Level
                        transformed_mean[:, idx] = np.exp(mean_states[:, idx])
                        transformed_median[:, idx] = np.exp(median_states[:, idx])
                        transformed_p095[:, idx] = np.exp(p095_states[:, idx])
                        transformed_p005[:, idx] = np.exp(p005_states[:, idx])
                        transformed_p084[:, idx] = np.exp(p084_states[:, idx])
                        transformed_p016[:, idx] = np.exp(p016_states[:, idx])
            
            # Create DataFrames
            df_mean = pd.DataFrame(transformed_mean, index=index, columns=varlist)
            df_median = pd.DataFrame(transformed_median, index=index, columns=varlist)
            df_p095 = pd.DataFrame(transformed_p095, index=index, columns=varlist)
            df_p005 = pd.DataFrame(transformed_p005, index=index, columns=varlist)
            df_p084 = pd.DataFrame(transformed_p084, index=index, columns=varlist)
            df_p016 = pd.DataFrame(transformed_p016, index=index, columns=varlist)
            
            # Write to Excel
            with pd.ExcelWriter(filename, engine="xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
                df_mean.to_excel(writer, sheet_name="mean")
                df_median.to_excel(writer, sheet_name="median")
                df_p095.to_excel(writer, sheet_name="95_quantile")
                df_p005.to_excel(writer, sheet_name="5_quantile")
                df_p084.to_excel(writer, sheet_name="84_quantile")
                df_p016.to_excel(writer, sheet_name="16_quantile")
                
                if include_metadata:
                    metadata_df.to_excel(writer, sheet_name="Model_Info", index=False)
                
            print(f"Smoothed states exported to {filename}")
            
        else:
            print("No smoothed states available. Cannot create Excel file.")
            return None
    
    else:
        # For models with forecasts
        if agg:
            # Check if aggregations are available
            if not has_aggregated:
                print("No aggregated forecasts available. Run model.aggregate() first.")
                return None
            
            print(f"Exporting aggregated forecasts to {filename}...")
            
            # Check which aggregation format we're using (old or new)
            if hasattr(self, 'YY_mean_agg'):
                # Old-style aggregation
                
                # Handle different index types
                if isinstance(self.YY_mean_agg.index, pd.DatetimeIndex):
                    with pd.ExcelWriter(filename, engine="xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
                        self.YY_mean_agg.to_excel(writer, sheet_name="mean")
                        
                        # Export other percentiles if available
                        for name, attr in [
                            ('median', 'YY_median_agg'),
                            ('95_quantile', 'YY_095_agg'),
                            ('5_quantile', 'YY_005_agg'),
                            ('84_quantile', 'YY_084_agg'),
                            ('16_quantile', 'YY_016_agg')
                        ]:
                            if hasattr(self, attr) and getattr(self, attr) is not None:
                                getattr(self, attr).to_excel(writer, sheet_name=name)
                        
                        if include_metadata:
                            metadata_df.to_excel(writer, sheet_name="Model_Info", index=False)
                
                else:
                    # For period indices, format them appropriately
                    if hasattr(self, 'agg_freq') and self.agg_freq == "Q":
                        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                            self.YY_mean_agg.assign(Index=self.YY_mean_agg.index.strftime('%YQ%q')).set_index('Index').to_excel(writer, sheet_name="mean")
                            
                            # Export other percentiles if available
                            for name, attr in [
                                ('median', 'YY_median_agg'),
                                ('95_quantile', 'YY_095_agg'),
                                ('5_quantile', 'YY_005_agg'),
                                ('84_quantile', 'YY_084_agg'),
                                ('16_quantile', 'YY_016_agg')
                            ]:
                                if hasattr(self, attr) and getattr(self, attr) is not None:
                                    getattr(self, attr).assign(Index=getattr(self, attr).index.strftime('%YQ%q')).set_index('Index').to_excel(writer, sheet_name=name)
                            
                            if include_metadata:
                                metadata_df.to_excel(writer, sheet_name="Model_Info", index=False)
                    
                    elif hasattr(self, 'agg_freq') and self.agg_freq == "Y":
                        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                            self.YY_mean_agg.assign(Index=self.YY_mean_agg.index.strftime('%Y')).set_index('Index').to_excel(writer, sheet_name="mean")
                            
                            # Export other percentiles if available
                            for name, attr in [
                                ('median', 'YY_median_agg'),
                                ('95_quantile', 'YY_095_agg'),
                                ('5_quantile', 'YY_005_agg'),
                                ('84_quantile', 'YY_084_agg'),
                                ('16_quantile', 'YY_016_agg')
                            ]:
                                if hasattr(self, attr) and getattr(self, attr) is not None:
                                    getattr(self, attr).assign(Index=getattr(self, attr).index.strftime('%Y')).set_index('Index').to_excel(writer, sheet_name=name)
                            
                            if include_metadata:
                                metadata_df.to_excel(writer, sheet_name="Model_Info", index=False)
            
            else:
                # New-style aggregation (uses aggregated_forecast and aggregated_percentiles)
                with pd.ExcelWriter(filename, engine="xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
                    # Write mean forecast
                    if hasattr(self, 'aggregated_forecast') and self.aggregated_forecast is not None:
                        self.aggregated_forecast.to_excel(writer, sheet_name="mean")
                    
                    # Write percentile forecasts if available
                    if hasattr(self, 'aggregated_percentiles'):
                        percentile_map = {
                            'Median': 'median',
                            '95th Percentile': '95_quantile',
                            '5th Percentile': '5_quantile',
                            '84th Percentile': '84_quantile',
                            '16th Percentile': '16_quantile'
                        }
                        
                        for label, sheet_name in percentile_map.items():
                            if label in self.aggregated_percentiles and self.aggregated_percentiles[label] is not None:
                                self.aggregated_percentiles[label].to_excel(writer, sheet_name=sheet_name)
                    
                    if include_metadata:
                        metadata_df.to_excel(writer, sheet_name="Model_Info", index=False)
        
        else:
            # Use non-aggregated forecasts
            print(f"Exporting regular forecasts to {filename}...")
            
            # Check for all required forecast attributes
            all_attrs = ['YY_mean_pd', 'YY_med_pd', 'YY_095_pd', 'YY_005_pd', 'YY_084_pd', 'YY_016_pd']
            present_attrs = [attr for attr in all_attrs if hasattr(self, attr) and getattr(self, attr) is not None]
            
            if not present_attrs:
                print("No forecast data found. Please run model.forecast() first.")
                return None
            
            # Export available forecasts
            with pd.ExcelWriter(filename, engine="xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
                # Map of attribute names to sheet names
                attr_to_sheet = {
                    'YY_mean_pd': 'mean',
                    'YY_med_pd': 'median',
                    'YY_095_pd': '95_quantile',
                    'YY_005_pd': '5_quantile',
                    'YY_084_pd': '84_quantile',
                    'YY_016_pd': '16_quantile'
                }
                
                for attr, sheet_name in attr_to_sheet.items():
                    if hasattr(self, attr) and getattr(self, attr) is not None:
                        getattr(self, attr).to_excel(writer, sheet_name=sheet_name)
                
                if include_metadata:
                    metadata_df.to_excel(writer, sheet_name="Model_Info", index=False)
        
        print(f"Successfully exported forecasts to {filename}")
    
    return None