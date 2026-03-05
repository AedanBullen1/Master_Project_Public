"""
This file will be used to sort data into different stability, wind, and snow conditions.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
import calendar
import glob
from datetime import datetime

sys.path.append('/Users/aedanbullen/Desktop/Projet de Master/Projet_de_master/src')
from ec.func_read_data import load_fastdata, read_eddypro_data

class FastDataLoader:

    def __init__(self,
                 instrument,
                 start,
                 end,
                 sampling_freq=10):
        """
        initialises inputs and sets up folder path based on instrument name
     """
      
        self.base_path = (#ENTER BASE PATH)
        self.instrument = instrument
        self.folder_path = f"{self.base_path}/{instrument}"
        self.start = start
        self.end = end
        self.sampling_freq = sampling_freq
        self.df = None
        self.processed = None
        self.eddypro_data = None


    def set_window(self, start, end):
        """
        sets the time window for loading data"""
        self.start = start
        self.end = end
        self.df = None
        self.processed = None
        self.eddypro_data = None

    def load_data(self):
        """loads data for the specified instrument and time window
        """
        print(f"\nLoading {self.instrument}")
        print(f"Window: {self.start} → {self.end}")

        df = load_fastdata(
            folder = self.folder_path,
            start=self.start,
            end=self.end
        )

        if df.empty:
            raise RuntimeError("No data loaded — check the date range or folder path.")
        
        self.df = df
        
        print(f"Loaded {len(df):,} samples")

    def load_eddypro_data(self):
        """
        Loads eddypro data for a month, but will need to filter by time window later.
        """
        print(f"\nLoading eddypro data for {self.instrument}")
        
        try:
            eddypro_data = read_eddypro_data(
                folder = self.base_path,
                sensor=self.instrument
            )

            if eddypro_data.empty:
                raise RuntimeError("No eddypro data loaded — check the folder path.")
            
            start_dt = datetime.strptime(self.start, "%Y-%m-%d_%H:%M:%S")
            end_dt = datetime.strptime(self.end, "%Y-%m-%d_%H:%M:%S")

            eddypro_filtered = eddypro_data.loc[start_dt: end_dt]
            self.eddypro_data = eddypro_filtered
            print(f"Loaded {len(eddypro_filtered):,} eddypro samples for the specified time window.")
            return eddypro_filtered
        
        except Exception as e:  
            print(f"Error loading eddypro data: {e}")
            return None
        
    def clean(self):
        """
        Keeps only core vars.
        """
        if self.df is None:
            raise RuntimeError("No data loaded — cannot clean empty DataFrame.")
        
        vars = [c for c in ['Ux', 'Uy', 'Uz', 'Ts'] if c in self.df.columns]
        if not vars:
            raise RuntimeError("None of the core turbulence variables (Ux, Uy, Uz, T) are available in the data.")
        
        df_clean = self.df[vars].copy() # created copy of data frame with only the core turbulence variables
        self.processed = df_clean
       
    def classify_wind(self):
        """
        Classifies wind conditions based on the processed data.
        Katabatic : 100-180 degrees
        Synoptic : 45-100 degrees
        Other : outside of those ranges"""


        if self.processed is None:
            raise RuntimeError("Data not cleaned — cannot classify wind conditions.")
        
        if self.eddypro_data is None:
            raise RuntimeError("No eddypro data loaded — cannot classify wind conditions.")

        if 'wind_dir' not in self.eddypro_data.columns:
            raise RuntimeError("Wind direction data not available — cannot classify wind conditions.")

        def classify(wind_dir):
            """Classify wind direction into categories."""
            if pd.isna(wind_dir):
                return 'Unknown'
            if 100 <= wind_dir <= 180:
                return 'Katabatic'
            elif 45 <= wind_dir < 100:
                return 'Synoptic'
            else:
                return 'Other'
            
        df = self.processed.copy()
            
        # For each fast data point, find the nearest EddyPro timestamp and use its wind direction
        df['wind_condition'] = df.index.map(
            lambda t: classify(
                self.eddypro_data['wind_dir'].iloc[
                    np.abs(self.eddypro_data.index - t).to_series().abs().argmin()
                ]
            )
        )
        
        self.processed = df 
        print("Wind conditions classified based on EddyPro wind direction.")
        return df
    
    def classify_stability(self):
        """
        Classifies stability based on (z-d)/L.
        
        Stable: (z-d)/L > 0.05 
        Neutral: -0.05 <= (z-d)/L <= 0.05  
        Unstable: (z-d)/L < -0.05
        """

        if self.processed is None:
            raise RuntimeError("Data not cleaned — cannot classify stability.")
        
        if self.eddypro_data is None:
            raise RuntimeError("No eddypro data loaded — cannot classify stability.")   

        if '(z-d)/L' not in self.eddypro_data.columns:
            raise RuntimeError("Stability parameter (z-d)/L not available — cannot classify stability.")
    
        def classify(val):
            """Classify stability based on (z-d)/L value."""
            if pd.isna(val):
                return 'Unknown'
            if val > 0.05:
                return 'Stable'
            elif -0.05 <= val <= 0.05:
                return 'Neutral'
            else:
                return 'Unstable'
        
        df = self.processed.copy()
        
        # For each fast data point, find the nearest EddyPro timestamp and use its stability
        df['stability'] = df.index.map(
            lambda t: classify(
                self.eddypro_data['(z-d)/L'].iloc[
                    (self.eddypro_data.index - t).to_series().abs().argmin()
                ]
            )
        )
        
        
        self.processed = df
        print("Stability classified based on EddyPro (z-d)/L.")
        return df
