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

def save(self, filename="sbfvar_model"):
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
    has_forecasts = hasattr(self, 'forecast_draws') or hasattr(self, 'YY_mean_pd')
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
                'SBFVAR',
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
        
        if len(self.index_list[-1][self.nlags:]) == self.lstate_list[0].shape[0]:
        
            index = copy.deepcopy(self.index_list[-1][self.nlags:])
        
        else:
            index = range(self.lstate_list[0,:,:].shape[0])
        
        #mean
        YYnow_m = np.mean(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw], axis = 0) # actual/nowcast weeklies
        if YYnow_m.size:
            YYnow_m[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
            YYnow_m[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
        
        lstate_m = np.mean(self.lstate_list[self.valid_draws,:,:], axis = 0)# hf obs for lf vars
        lstate_m[:, (self.select_m_q == 1)] = 100 * lstate_m[:, (self.select_m_q == 1)]
        lstate_m[:, (self.select_m_q == 0)] = np.exp(lstate_m[:, (self.select_m_q == 0)])
        
        YW = copy.deepcopy(self.input_data_W)[self.rqw:-(self.rqw)]
        YW[:, (self.select_w == 1)] = 100 * YW[:, (self.select_w == 1)]
        YW[:, (self.select_w == 0)] = np.exp(YW[:, (self.select_w == 0)])

        YY_m = np.vstack((np.vstack((np.hstack((YW, lstate_m[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_m[-self.rqw:,:]))))))
        #median
        
        YYnow_med = np.nanmedian(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw], axis = 0) # actual/nowcast weeklies
        if YYnow_med.size:
            YYnow_med[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
            YYnow_med[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
        
        lstate_med = np.nanmedian(self.lstate_list[self.valid_draws,:,:], axis = 0)# hf obs for lf vars
        lstate_med[:, (self.select_m_q == 1)] = 100 * lstate_med[:, (self.select_m_q == 1)]
        lstate_med[:, (self.select_m_q == 0)] = np.exp(lstate_med[:, (self.select_m_q == 0)])
        
        YY_med = np.vstack((np.vstack((np.hstack((YW, lstate_med[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_med[-self.rqw:,:]))))))
        
        YYnow_095 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.95, axis = 0) # actual/nowcast weeklies
        if YYnow_095.size:
            YYnow_095[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
            YYnow_095[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
        
        lstate_095 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.95, axis = 0)# hf obs for lf vars
        lstate_095[:, (self.select_m_q == 1)] = 100 * lstate_095[:, (self.select_m_q == 1)]
        lstate_095[:, (self.select_m_q == 0)] = np.exp(lstate_095[:, (self.select_m_q == 0)])
        
        YY_095 = np.vstack((np.vstack((np.hstack((YW, lstate_095[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_095[-self.rqw:,:]))))))

        # 84%
        
        YYnow_084 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.84, axis = 0) # actual/nowcast weeklies
        if YYnow_084.size:
            YYnow_084[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
            YYnow_084[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
        
        lstate_084 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.84, axis = 0)# hf obs for lf vars
        lstate_084[:, (self.select_m_q == 1)] = 100 * lstate_084[:, (self.select_m_q == 1)]
        lstate_084[:, (self.select_m_q == 0)] = np.exp(lstate_084[:, (self.select_m_q == 0)])
        
        YY_084 = np.vstack((np.vstack((np.hstack((YW, lstate_084[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_084[-self.rqw:,:]))))))


        # 16%
        
        YYnow_016 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.16, axis = 0) # actual/nowcast weeklies
        if YYnow_016.size:
            YYnow_016[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
            YYnow_016[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
        
        lstate_016 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.16, axis = 0)# hf obs for lf vars
        lstate_016[:, (self.select_m_q == 1)] = 100 * lstate_016[:, (self.select_m_q == 1)]
        lstate_016[:, (self.select_m_q == 0)] = np.exp(lstate_016[:, (self.select_m_q == 0)])
        
        YY_016 = np.vstack((np.vstack((np.hstack((YW, lstate_016[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_016[-self.rqw:,:]))))))
            
        # 5%
        
        YYnow_005 = np.nanquantile(self.YYactsim_list[self.valid_draws,1:(self.rqw+1),:self.Nw],  q = 0.05, axis = 0) # actual/nowcast weeklies
        if YYnow_005.size:
            YYnow_005[:, (self.select_w== 1)] = 100 * YYnow_m[:, (self.select_w == 1)]
            YYnow_005[:, (self.select_w == 0)] = np.exp(YYnow_m[:,(self.select_w == 0)])
        
        lstate_005 = np.nanquantile(self.lstate_list[self.valid_draws,:,:],  q = 0.05, axis = 0)# hf obs for lf vars
        lstate_005[:, (self.select_m_q == 1)] = 100 * lstate_005[:, (self.select_m_q == 1)]
        lstate_005[:, (self.select_m_q == 0)] = np.exp(lstate_005[:, (self.select_m_q == 0)])
        
        YY_005 = np.vstack((np.vstack((np.hstack((YW, lstate_005[:-(self.rqw),:])), np.hstack((YYnow_m, lstate_005[-self.rqw:,:]))))))
        
        index_start = index[0]
        
        index = pd.date_range(start = index_start, periods = YY_m.shape[0], freq = self.frequencies[-1])
        

        
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
        
            
        with pd.ExcelWriter(filename, engine = "xlsxwriter", datetime_format='yyyy-mm-dd') as writer:
            #writer = pd.ExcelWriter("sim_data.xlsx", engine="xlsxwriter")
                YY_mean_pd.to_excel(writer, sheet_name = "mean")
                YY_median_pd.to_excel(writer, sheet_name = "median")
                YY_095_pd.to_excel(writer, sheet_name = "95_quantile")
                YY_005_pd.to_excel(writer, sheet_name = "5_quantile")
                YY_084_pd.to_excel(writer, sheet_name = "84_quantile")
                YY_016_pd.to_excel(writer, sheet_name = "16_quantile")
                
                if include_metadata:
                    metadata_df.to_excel(writer, sheet_name="Model_Info", index=False)    
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