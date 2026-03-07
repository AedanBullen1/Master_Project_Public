""" This module will produce power spectra for given dates and instruments.

This will use Welch's method, indentifying the inertial subrange and comparing to the -5/3 slope.
Some of Rainette's functions are used."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import sys
from scipy.stats import linregress
plt.rcParams['font.family'] = 'Helvetica'
import seaborn as sns
palette = sns.color_palette('Set2', 4)

sys.path.append('REPLACE')
from data_sorter import FastDataLoader

class SpectralAnalyser:
    """Class to perform spectral analysis on turbulence data."""

    def __init__(self, sampling_freq: float=20, nperseg=1024):
        """
        Initializes the SpectralAnalyser with the data and sampling frequency.

        Parameters:
        df (pd.DataFrame): DataFrame containing the turbulence data.
        sampling_freq (float): Sampling frequency in Hz.
        """
        self.sampling_freq = sampling_freq
        self.nperseg = nperseg
        self.data = None
        self.loader = None

    def data_for_spectrum(self, instrument, start, end):
        """Prepares data for spectrum, using FastDataLoader to load"""

        print(f"Preparing data for spectral analysis: {instrument} from {start} to {end}")

        self.loader = FastDataLoader(
            instrument=instrument,
            start=start, end=end, 
            sampling_freq=self.sampling_freq)
        
        self.loader.load_data()
        print("Data loaded successfully.")
        self.loader.clean()
        self.data = self.loader.processed
        print(f"Data cleaned and ready for spectral analysis. Shape: {self.data.shape}")


    def compute_spectrum(self, velocity: np.ndarray):
        """
        Computes the power spectral density using Welch's method.

        detrend (str): Type of detrending to apply ('linear', 'constant', or None).
        """

        data_clean = velocity[~np.isnan(velocity)] # Remove NaNs for spectral analysis

        if len(velocity) == 0:
            print("No data for period — cannot compute spectrum.")
            return None, None

        if len(data_clean) < self.nperseg:
            print(f"Warning: Not enough data points ({len(data_clean)}) for nperseg={self.nperseg}.") # Shannon-Nyquist warning

        #power spectra:

        frequencies, psd = welch(
            data_clean,
            fs=self.sampling_freq,
            nperseg=self.nperseg,
            noverlap=self.nperseg//2,
            detrend='linear'
        )

        return frequencies, psd
    
    def fit_inertial_subrange(self, frequencies, psd, f_min=.01, f_max=2): # to adjust
        """Identifies the inertial subrange."""

        #mask for inertial subrange
        mask = (frequencies >= f_min) & (frequencies <= f_max) & (psd > 0) # intersect. 

        if np.sum(mask) < 2: # masks are Boolean arrays, so it's a stencil over the data to keep True outputs that satisfy condition.
            return {
                'slope': np.nan,
                'intercept': np.nan,
                'r_squared': np.nan,
                'std_err': np.nan,
                'f_inertial': None
            }
        
        log_f = np.log10(frequencies[mask])
        log_psd = np.log10(psd[mask])

        slope, intercept, r_val, p_val, std_err = linregress(log_f, log_psd) # log(PSD) = slope * log(f) + intercept

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_val**2,
            'std_err': std_err,
            'f_inertial': frequencies[mask]
            }
        
    def plot_spectrum(self, frequencies, f_inertial, Suu, Svv, Sww, instrument, start, end, ax=None):
        """Plots the power spectral density with the inertial subrange and -5/3 slope."""
        
        #TKE
        S_TKE = 0.5 * (Suu + Svv + Sww)

        f = frequencies

        # Plot spectra
        colors = {'TKE': palette[0], 'u': palette[1], 'v': palette[2], 'w': palette[3]}
        labels = {'TKE': '$TKE$', 'u': "$u'^2$", 'v': "$v'^2$", 'w': "$w'^2$"}

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            standalone = True
        else:
             standalone = False

        ax.loglog(f, S_TKE, '-', linewidth=2.5, label=labels['TKE'], color=colors['TKE'], alpha=0.9)
        ax.loglog(f, Suu, '-', linewidth=2, label=labels['u'], color=colors['u'], alpha=0.8)
        ax.loglog(f, Svv, '-', linewidth=2, label=labels['v'], color=colors['v'], alpha=0.8)
        ax.loglog(f, Sww, '-', linewidth=2, label=labels['w'], color=colors['w'], alpha=0.7)

        #Komolhorov -5/3 reference slope
        f_ref = .5 * S_TKE[np.argmin(np.abs(f - 0.1))] / ((.1)**(-5/3))

        if f_inertial is not None:
            S_komolgorov = f_ref * f_inertial**(-5/3)
            ax.loglog(f_inertial, S_komolgorov, 'k--', linewidth=1.5,
                      label='Komolgorov', alpha=0.7)

        # Formatting
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (m²/s²/Hz)', fontsize=12)
        ax.set_title(f'Power Spectral Density for {instrument} from {start} to {end}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

        if standalone:
            plt.tight_layout()
            plt.savefig(f'spectrum_{instrument}_{start}_{end}.png', dpi=300)
            plt.show()

    def analyse_spectrum(self, instrument, start, end, ax=None):
        """Combining everythig together for a full analysis."""

        self.data_for_spectrum(instrument, start, end)

        if self.data is None:
            print("No data available for spectral analysis.")
            return

        if 'Ux' in self.data.columns:
            U_mean = self.data['Ux'].mean() # remember it's double rotated
            print(f"Mean horizontal wind speed (Ux): {U_mean:.2f} m/s")

        print("Computing spectra for u, v, w...")

        f, Suu = self.compute_spectrum(self.data['Ux'].values)
        _, Svv = self.compute_spectrum(self.data['Uy'].values)
        _, Sww = self.compute_spectrum(self.data['Uz'].values)  

        if f is None:
            print("Spectrum computation failed — cannot proceed with analysis.")
            return None    
        
        print("Fitting inertial subrange...")
        fit_results = self.fit_inertial_subrange(f, Suu)

        if not np.isnan(fit_results['slope']):
            print(f"✓ Slope: {fit_results['slope']:.3f}")
            print(f"  R²: {fit_results['r_squared']:.4f}")            
            print(f"  Std Err: {fit_results['std_err']:.4f}")

        print("Plotting spectrum...")

        self.plot_spectrum(f, fit_results['f_inertial'], Suu, Svv, Sww, instrument, start, end, ax=ax)

        return {
            'frequencies': f,
            'Suu': Suu,
            'Svv': Svv,
            'Sww': Sww,
            'S_TKE': 0.5 * (Suu + Svv + Sww),
            'fit_results': fit_results
        }

class CompareHeights(SpectralAnalyser):
    """Class to compare spectra over multiple instrument heights"""

    def compare_spectra(self, instruments, start, end):
        """Compares spectra across multiple instruments."""
        results = {}
        if len(instruments) < 2:
            print("At least two instruments are required for comparison.")
            return None
        
        
        print("Plotting comparison as subplots...")

        fig, axes = plt.subplots(len(instruments), 1, figsize=(10, 5 * len(instruments)))
        if len(instruments) == 1:
            axes = [axes]  # Ensure axes is iterable for a single instrument

        for ax, instrument in zip(axes, instruments):
            res = self.analyse_spectrum(instrument, start, end, ax=ax)
            if res is not None: 
                results[instrument] = res

        plt.tight_layout()
        plt.savefig(f'spectral_comparison_{start}_{end}.png', dpi=300)
        plt.show()

        return results

    


"""This code employs as e.g.: 
    instrument = 'CSAT_26m_DR'
    start = '2024-05-14_00:00:00'
    end = '2024-05-14_23:00:00'
    sampling_freq = 10  # Hz

    analyser = SpectralAnalyser(sampling_freq=sampling_freq, nperseg=1024)
    results = analyser.analyse_spectrum(instrument=instrument, start=start, end=end)

"""
