import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle # from astropy.stats import LombScargle
from astropy.coordinates import SkyCoord
from astropy import units as u

from scipy.signal import find_peaks


class MyPeriodogram:
    """
    A class used to compute, plot, and analize astropy Lomb Scargle periodograms
    """
    
    def __init__(self, xarray, xarray_unit, yarray, yarray_err = None, 
                 obswindow_flag = False,
                 obs_cadence = None, 
                 xaxis_unit = None,
                 samples_per_peak = 5, 
                 minimum_x = None, 
                 maximum_x = None,
                 prob_levels = [0.1, 0.01], 
                 bootstrap_flag = False, 
                 n_bootstrap = 1000,
                 
                 user_freqs = False):   
        """
        Constructor method for creating a MyPeriodogram object.

        
        Parameters
        ----------
        xarray : numpy array, float
            The array of time measurements
        xarray_unit : astropy unit
            Unit of the time measurements 
        yarray : numpy array, float
            Fluxes associated with the time measurements 
        yarray_err : numpy array, float, optional
            Errors in the fluxes    
        obswindow_flag : bool, optional
            To compute periodogram of the window function      
        obs_cadence : astropy quantity, optional 
            Cadence of the data
        xaxis_unit : astropy unit, optional
            Desired unit for the x-axis of the periodogram 
        samples_per_peak : float, optional
            Frequency resolution of the periodogram
        minimum_x : float, optional
            Minimum value for the x-axis of the periodogram in units of 'xaxis_unit'
        maximum_x : float, optional
            Maximum value for the x-axis of the periodogram in units of 'xaxis_unit'
        prob_levels : list of floats, optional
            False Alarm Levels to compute
        bootstrap_flag = bool, optional
            If include False Alarm Probabilities calculated by the bootstrap resampling
        n_bootstrap = int, optional
            If bootstrap_flag, number of bootstrap samples
            
        """
        
        self.xarray, self.yarray, self.yarray_err = xarray, yarray, yarray_err
        self.remove_nans() # Remove NaN values from the data:
                
        self.xarray_unit = xarray_unit
        # Units for the periodogram's x-axis: 
        self.xaxis_unit = self.xarray_unit if xaxis_unit is None else xaxis_unit
        
        self.obswindow_flag = obswindow_flag
        self.compute_obswindow() # Compute observing window error and switch to observing function if requested

        self.obs_cadence = obs_cadence.to(self.xaxis_unit) # Convert cadence to desired periodogram x-axis units
        self.nyquist = 2*self.obs_cadence # Compute Nyquist limit

        self.minimum_x = minimum_x 
        self.maximum_x = maximum_x 
        self.find_freqlims() # Convert periodogram x-axis limits to limiting frequencies
        
        self.samples_per_peak = samples_per_peak         
        self.prob_levels = prob_levels
        self.bootstrap_flag = bootstrap_flag
        self.n_bootstrap = n_bootstrap

        self.user_freqs = user_freqs

        self.compute_periodogram() # Compute periodogram, FAPs and FALs

        
    def remove_nans(self):
        """
        Remove NaN values from the input data 
        """
        
        if np.isnan(self.yarray).any() or (hasattr(self.yarray, 'unmasked') and np.isnan(self.yarray.unmasked).any()):
            mask = ~np.isnan(self.yarray)
            self.xarray = self.xarray[mask]
            self.yarray = self.yarray[mask]
            self.yarray_err = self.yarray_err[mask] if self.yarray_err is not None else None

    
    def compute_obswindow(self):
        """
        Calculate the length of the observing window and the peak width 
        Convert the data to the window function if requested
        """
        self.windowlength = ((self.xarray[-1] - self.xarray[0]) * self.xarray_unit).to(self.xaxis_unit) # Compute observing window length
        self.windowlength_err = 1 / self.windowlength # Peak width in frequency space from the observing window length

        if self.obswindow_flag: 
            self.yarray = np.ones(len(self.xarray)) # Convert data to the window function 
            self.yarray_err = None


    def find_freqlims(self):  
        """
        Convert periodogram x-axis limits to limiting frequencies
        """

        # Set maximum frequency:
        if self.minimum_x != None:
            self.maximum_frequency = (1 / self.minimum_x) * (1/self.xaxis_unit).unit
        else:
            self.maximum_frequency = None 
            
        # Set minimum frequency:
        if self.maximum_x != None:
            self.minimum_frequency = (1 / self.maximum_x) * (1/self.xaxis_unit).unit
        else: 
            self.minimum_frequency = (1 / self.windowlength) # One exact oscillation cycle of the window length     

    
    def compute_periodogram(self):
        """
        Compute the Lomb Scargle periodogram, False Alarm Probabilities, and False Alarm Levels
        """

        ls = LombScargle(self.xarray*self.xarray_unit, self.yarray, self.yarray_err) # Initialize class

        if self.user_freqs is False: # Periodogram if frequency grid is not provided 
            x_freqs, y_power = ls.autopower(samples_per_peak = self.samples_per_peak, 
                                            minimum_frequency = self.minimum_frequency, 
                                            maximum_frequency = self.maximum_frequency)
        else: # Periodogram if frequency grid is provided
            x_freqs = self.user_freqs
            y_power = ls.power(x_freqs)

        x_pers = 1 / x_freqs # Periodogram x-axis from frequency to period space
        x_pers = x_pers.to(self.xaxis_unit) # Period space in the desired unit
        pdgm_xvalues = x_pers # Final periodogram x-axis
           
        FAPs = ls.false_alarm_probability(y_power, method = 'baluev') # Compute False Alarm Probabilities through Baluev method
        
        if self.bootstrap_flag: # FAPs and FALs through bootstrap resampling
            FAPs_bootstrap = ls.false_alarm_probability(y_power, samples_per_peak = self.samples_per_peak, 
                                                        method = 'bootstrap', method_kwds = {'n_bootstraps': self.n_bootstrap})
            FALs = ls.false_alarm_level(self.prob_levels, samples_per_peak = self.samples_per_peak, method = 'bootstrap', method_kwds = {'n_bootstraps': self.n_bootstrap})
        else: # FALs through Baluev method 
            FAPs_bootstrap = None
            FALs = ls.false_alarm_level(self.prob_levels, samples_per_peak = self.samples_per_peak, method = 'baluev')

        # Save variables:
        self.ls = ls
        self.x_freqs = x_freqs
        self.x_pers = x_pers
        self.y_power = y_power
        self.pdgm_xvalues = pdgm_xvalues
        self.FAPs = FAPs
        self.FAPs_bootstrap = FAPs_bootstrap
        self.FALs = FALs
        self.peak_type = None
        

    def find_peaks(self, print_flag = True, maxpeak_flag = False, regionpeak_flag = False, regionpeak_lims = [None], FAPcutpeak_flag = False, FAPcut = 0.1):
        """
        Provides different ways to find the peaks in the periodogram

        Parameters
        ----------
        print_flag : boolean
            Display the information of the found peaks 
        maxpeak_flag : boolean
            Find the peak with the maximum power 
        regionpeak_flag : boolean
            Find the peak with the maximum power in a specified region 
        regionpeak_lims : list of two floats
            Left and right limit of the interest region if 'regionpeak_flag', in units of 'xaxis_unit'
        FAPcutpeak_flag : boolean
            Find the peaks below a FAP threshold
        FAPcut : float
            FAP threshold if 'FAPcutpeak_flag'
        """
    
        if maxpeak_flag: # Find the peak with the max power of the periodogram       
            self.peak_x, self.peak_y, self.peak_FAP, self.peak_FAP_bootstrap = self.find_maxpeak(self.pdgm_xvalues, self.y_power, self.FAPs, self.FAPs_bootstrap)
            self.peak_owerror = self.find_owerror(self.peak_x, self.windowlength_err, self.xaxis_unit) # Peak width from the inverse of the observing window
            self.peak_hwerror = None
            self.peak_type = 'MaxPeak' # Save used method
        
        elif regionpeak_flag: # Find the peak with the max power in an user defined region of the periodogram  
            self.peak_x, self.peak_y, self.peak_FAP, self.peak_FAP_bootstrap = self.find_regionpeak(self.pdgm_xvalues, self.y_power, self.FAPs, self.FAPs_bootstrap, regionpeak_lims[0], regionpeak_lims[1])
            self.peak_hwerror = self.find_hwerror(self.pdgm_xvalues, self.y_power, regionpeak_lims[0], regionpeak_lims[1], self.peak_y) # Measure peak width directly
            self.peak_owerror = self.find_owerror(self.peak_x, self.windowlength_err, self.xaxis_unit) # Peak width from the inverse of the observing window
            self.regionpeak_lims = regionpeak_lims
            self.peak_type = 'RegionPeak' # Save used method

        elif FAPcutpeak_flag: # Find the peaks below a FAP threshold
            self.peak_x, self.peak_y, self.peak_FAP, self.peak_FAP_bootstrap = self.find_FAPcutpeaks(FAPcut)
            self.FAPcut = FAPcut # Save FAP threshold
            # Compute peak width to each peak from the inverse of the observing window
            owerror = [None]*len(self.peak_x)
            for i,peak in enumerate(self.peak_x):
                owerror[i] = self.find_owerror(peak, self.windowlength_err, self.xaxis_unit)
            self.peak_owerror = owerror
            self.peak_hwerror = [None]*len(self.peak_x)
            self.peak_type = 'FAPcutPeak' # Save used method
            
        else:
            self.peak_x = None
            self.peak_y = None
            self.peak_FAP = None
            self.peak_FAP_bootstrap = None
            self.peak_hwerror = None
            self.peak_owerror = None
            self.peak_type = None

        # Store information of the peak in a data frame:
        results = pd.DataFrame()
        results['Peak x'] = [self.peak_x] if not FAPcutpeak_flag else [x for x in self.peak_x] 
        results['Peak y'] = [self.peak_y] if not FAPcutpeak_flag else self.peak_y
        results['Peak FAP - Baluev'] = [self.peak_FAP] if not FAPcutpeak_flag else self.peak_FAP
        results['Peak FAP - Bootstrap'] = [self.peak_FAP_bootstrap] if not FAPcutpeak_flag else self.peak_FAP_bootstrap
        results['Peak Error - Obs Wind'] = [self.peak_owerror] if not FAPcutpeak_flag else self.peak_owerror
        results['Peak Error - FWHM'] = [self.peak_hwerror] if not FAPcutpeak_flag else self.peak_hwerror
        self.results = results
    
        if print_flag: # Print peak information
            print(self.results)

    
    @staticmethod
    def cut_region(xvalues, tocut_list, xleft = None, xright = None):
        """
        Cut the region of interest based on given x-limits
        """
        
        # If lims are not provided:
        if xleft is None: xleft = xvalues.value.min()
        if xright is None: xright = xvalues.value.max()
        
        mask = (xvalues.value >= xleft) & (xvalues.value <= xright)
        region_x = xvalues[mask] # Cut the x region
        
        cut_regions = [None]*len(tocut_list) # List with the cut for the other arrays
        for i,tocut in enumerate(tocut_list): # Cut the other arrays 
            cut_regions[i] = tocut[mask] if tocut is not None else None
            
        return region_x, cut_regions


    @staticmethod
    def find_maxpeak(xvalues, yvalues, FAPs = None, FAPs_bootstrap = None):
        """
        Find the peak with the maximum power
        """
        peak_arg = np.argmax(yvalues) # Index of the peak element with the maximum power
        peak_x = xvalues[peak_arg]
        peak_y = yvalues[peak_arg]
        peak_FAP = FAPs[peak_arg] 
        peak_FAP_bootstrap = FAPs_bootstrap[peak_arg] if FAPs_bootstrap is not None else None

        return peak_x, peak_y, peak_FAP, peak_FAP_bootstrap

    @staticmethod
    def find_regionpeak(xvalues, yvalues, FAPs, FAPs_bootstrap, xleft, xright):   
        """
        Find the peak with the maximum power within an user specified region
        """
        # Isolate the region:
        region_x, [region_y, region_FAP, region_FAP_bootstrap] = MyPeriodogram.cut_region(xvalues, [yvalues, FAPs, FAPs_bootstrap], xleft, xright)  
        # Find the peak with the maximum power:
        peak_x, peak_y, peak_FAP, peak_FAP_bootstrap = MyPeriodogram.find_maxpeak(region_x, region_y, region_FAP, region_FAP_bootstrap)
        
        return peak_x, peak_y, peak_FAP, peak_FAP_bootstrap

        
    def find_FAPcutpeaks(self, FAPcut):
        """
        Find the peaks below an user specified false alarm probability treshold
        """
        # If bootstrap FAPs are calculated evaluate the threshold with them:
        FAPs_evaluate = self.FAPs_bootstrap if self.bootstrap_flag else self.FAPs
        
        peaks_arg, _ = find_peaks(FAPs_evaluate*(-1), height = FAPcut*(-1)) # Convert minimums in maximums and find the peaks
        peaks_x = self.pdgm_xvalues[peaks_arg]
        peaks_y = self.y_power[peaks_arg]
        peaks_FAP = self.FAPs[peaks_arg]
        peaks_FAP_bootstrap = self.FAPs_bootstrap[peak_arg] if self.bootstrap_flag else [None]*len(peaks_x)
        
        return peaks_x, peaks_y, peaks_FAP, peaks_FAP_bootstrap

        
    @staticmethod
    def find_hwerror(xvalues, yvalues, xleft, xright, peak_y):
        """
        Compute peak error from half width at half maximum
        """
        # Only used for small regions:
        region_x, [region_y] = MyPeriodogram.cut_region(xvalues, [yvalues], xleft, xright) 
        mask = region_y >= peak_y/2 # Part of the peak above half maximum
        half_width = region_x[mask]
        uncertainty = (half_width[0] - half_width[-1]) / 2 # Uncertainty is half the width peak at half maximum 
        
        return uncertainty    


    @staticmethod        
    def find_owerror(peak_x, windowlength_err, xaxis_unit):
        """
        Compute peak error from inverse of the observing window
        """
        peak_centr = peak_x # Peak in period
        peak_centr_inv = 1 / peak_centr # Peak in frequency
        peak_left = 1 / (peak_centr_inv + windowlength_err/2) # Left limit of the peak width in frequency
        peak_right = 1 / (peak_centr_inv - windowlength_err/2) # Right limit of the peak width in frequency
        # Uncertainty is the mean of both half peak widths
        uncertainty = np.mean([peak_right.value-peak_centr.value, peak_centr.value-peak_left.value]) * xaxis_unit
        
        return uncertainty

        
    def plot_periodogram(self, xlims = [None, None], ylims = [None, None], pdgm_style = ':.', fap_style = ':.', FAPs_alpha = 1, styles_levels = ['-.',':'], xscale = None, legend_flag = True, savefig = None, figwidth = 10, figheight = 3):
        """
        Plot computed periodogram and peak-related information

        Parameters
        ----------
        xlims : list of two float values
            Left and right limits (x-axis) of the region of the periodogram to display  
        ylims : list of two float values, or list of two strings (with 'Min', 'Max' and 'Peak' as possible keywords)
            Lower and upper limits (y-axis) of the region of the periodogram to display  
        pdgm_style : string
            linestyle of the periodogram              
        fap_style : strin
            linestyle of the false alarm probabilities
        FAPs_alpha : float
            Transparency of the false alarm probabilites
        styles_levels : list of strings
            linestyles for the false alarm levels
        xscale : string
            xscale of the periodogram, either 'linear', 'log' or by default both with 'None'
        legend_flag : boolean
            Display plot legends
        savefig : False or string
            Save the figure with the specified string as the name
        figwidth : float
            Width of the figure 
        figheight : float
            Height of the figure 
        """

        # Select the limits for visualization:
        if xlims[0] == None:
            xlims[0] = self.pdgm_xvalues.value[-1]
        if xlims[1] == None:
            xlims[1] = self.pdgm_xvalues.value[0]            
        if ylims[0] == 'Min':
            ylims[0] = self.y_power.value.min()
        if ylims[1] == 'Max':
            ylims[1] = self.y_power.value.max()
        if ylims[1] == 'Peak':
            ylims[0] = 0
            ylims[1] = self.peak_y

        # To plot visualization region only: 
        plot_pdgm_xvalues, [plot_y_power, plot_FAPs] = self.cut_region(self.pdgm_xvalues, [self.y_power, self.FAPs], xlims[0], xlims[1])                

        # Initialize figures depending on the scale of the  periodogram:
        if xscale is None:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(figwidth, figheight))
            axes_list = [ax1, ax2]
            xscale_list = ['linear', 'log']
        else:
            fig, ax = plt.subplots(1, figsize=(figwidth/2, figheight))
            axes_list = [ax]
            xscale_list = [xscale]


        for axes,xscale in zip(axes_list,xscale_list):
            axes.plot(plot_pdgm_xvalues, plot_y_power, pdgm_style, color = 'k', label = 'Periodogram') # Plot periodogram
            axes.set_xlabel(f'Period [{self.pdgm_xvalues.unit}]', fontsize = 14)
            axes.set_ylabel('Lomb-Scargle power', fontsize = 14)
            
            axes.plot(plot_pdgm_xvalues, plot_FAPs, fap_style, color = 'r', alpha = FAPs_alpha, label = 'FAPs'); # Plot Baluev false alarm probabilities
    
            for n, lsty in enumerate(styles_levels):
                axes.axhline(y = self.FALs[n], color = 'b', linestyle = lsty, label=f'FAL ({self.prob_levels[n]*100}% FAP)') # Plot false alarm levels
    
            if self.nyquist != None:
                axes.axvline(self.nyquist.value, color ='gold', linestyle = ':', linewidth = 2, label = 'Nyquist') # Identify Nyquist limit
                axes.axvline(self.obs_cadence.value, color ='gold', linestyle = '--', linewidth = 2, label = 'Cadence harmonics') # Identify cadence peak
                axes.axvline(self.obs_cadence.value/2, color ='gold', linestyle = '--', linewidth = 2) # Identify first harmonic of the cadence

            if self.peak_type == 'MaxPeak': 
                # Mark peak:
                axes.plot(self.peak_x, self.peak_y, 'x', color = 'g', markersize = 10, label = 'Peak'); 
                axes.axvline(self.peak_x.value, ymin = 0.9, ymax = 1, color = 'g', linestyle = '-', alpha = 1)
                axes.axvline(self.peak_x.value, ymin = 0, ymax = 0.1, color = 'g', linestyle = '-', alpha = 1)
                
            elif self.peak_type == 'RegionPeak':
                # Mark peak:
                axes.plot(self.peak_x, self.peak_y, 'x', color = 'g', markersize = 10, label = 'Peak'); 
                axes.axvline(self.peak_x.value, ymin = 0.9, ymax = 1, color = 'g', linestyle = '-', alpha = 1)
                axes.axvline(self.peak_x.value, ymin = 0, ymax = 0.1, color = 'g', linestyle = '-', alpha = 1)
                # Mark user-specified region:
                axes.axvline(self.regionpeak_lims[0], color = 'g', linestyle = ':')              
                axes.axvline(self.regionpeak_lims[1], color = 'g', linestyle = ':')
                # Mark peak width
                errorlinex = np.linspace(self.peak_x.value - self.peak_hwerror.value, self.peak_x.value + self.peak_hwerror.value, 100)
                errorliney = np.ones(100) * self.peak_y/2
                axes.plot(errorlinex, errorliney, '-', color = 'silver')

            elif self.peak_type == 'FAPcutPeak':
                # Mark peaks:
                axes.plot(self.peak_x, self.peak_y, 'x', color = 'g', markersize = 10, label = 'Peaks');
                for p in self.peak_x.value:
                    axes.axvline(p, ymin = 0.9, ymax = 1,color = 'g', linestyle = '-', alpha = 1)
                    axes.axvline(p, ymin = 0, ymax = 0.1,color = 'g', linestyle = '-', alpha = 1)
                # Mark false alarm probability treshold
                axes.axhline(y = self.FAPcut, color = 'g', linestyle = '-', alpha = 0.5, label = 'FAP cut')
                
            axes.set_xlim(xlims)
            axes.set_ylim(ylims) 
            axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            if legend_flag: axes.legend(fontsize = 8, loc = 'best') # Plot legends
    
            axes.set_xscale(xscale)
            
        plt.tight_layout()
        plt.show()

        if savefig is not None: # Save figure
            fig.savefig(savefig, format = 'png', dpi = 300, bbox_inches = "tight")   


class MySpectrogram:
    """
    A class used to compute, plot, and analize running astropy Lomb Scargle periodograms
    """
    
    def __init__(self, 
                 xarray, 
                 yarray, 
                 yarray_err,
                 obs_cadence,
                 window,
                 overlap = 0.2,
                 samples_per_peak = 5,
                 minimum_x = None, 
                 maximum_x = None,
                 bootstrap_flag = False, 
                 n_bootstrap = 1000):
        """
        Constructor method for creating a MySpectrogram object.

        
        Parameters
        ----------
        xarray : numpy array, float
            The array of time measurements in units of days
        yarray : numpy array, float
            Fluxes associated with the time measurements 
        yarray_err : numpy array, float, optional
            Errors in the fluxes      
        obs_cadence : astropy quantity
            Cadence of the data
        window : astropy quantity
            Length of the running window to compute spectrogram
        overlap : float, optional
            Overlap fraction [0, 1) between succesice running windows
        samples_per_peak : float, optional
            Frequency resolution of the periodogram
        minimum_x : float, optional
            Minimum value for the y-axis of the spectrogram in units of days
        maximum_x : float, optional
            Maximum value for the y-axis of the spectrogram in units of days
        bootstrap_flag = bool, optional
            If include False Alarm Probabilities calculated by the bootstrap resampling
        n_bootstrap = int, optional
            If bootstrap_flag, number of bootstrap samples
            
        """
        
        self.xarray = xarray
        self.yarray = yarray
        self.yarray_err = yarray_err
        self.obs_cadence = obs_cadence
        self.window = window.to(u.d) 
        self.overlap = overlap 
        self.samples_per_peak = samples_per_peak
        self.minimum_x = minimum_x 
        self.maximum_x = maximum_x
        self.bootstrap_flag = bootstrap_flag
        self.n_bootstrap = n_bootstrap

        # Find shift between successive windows depending on defined overlap:
        self.shift = self.window.value if self.overlap == 0 else self.window.value*(1-self.overlap)

        # Compute spectrogram:
        self.calculate_spectrogram()
        

    def calculate_spectrogram(self, user_freqs = False):
        """
        Compute the spectrogram, and
        """
        
        self.total_xi = self.xarray[0] # First point of time array
        self.total_xf = self.xarray[-1] # Last point of time array
        
        # Initial times for the window's left and right points:
        xi0 = self.total_xi
        xf0 = xi0 + self.window.value
        
        i = 0 # Initialize counter
        user_freqs = user_freqs  
        
        while True: 
            # Recalculate window's left and right points
            xi = xi0 + self.shift*i  
            xf = xf0 + self.shift*i

            # The cycle terminates if the right point of the window exceeds the last point of the time array
            if xf >= self.total_xf: break
            
            # Light curve of the window: 
            xarray_cut, [yarray_cut, yarray_err_cut] = MyPeriodogram.cut_region(self.xarray*u.d, [self.yarray, self.yarray_err], xi, xf)                
            
            # Periodogram of the window's light curve:
            prdrg_temp = MyPeriodogram(xarray_cut.value, u.d, yarray_cut, yarray_err_cut, 
                                       obs_cadence = self.obs_cadence, xaxis_unit = u.min, 
                                       samples_per_peak = self.samples_per_peak, minimum_x = self.minimum_x, maximum_x = self.maximum_x, obswindow_flag = False,
                                       bootstrap_flag = self.bootstrap_flag, n_bootstrap = self.n_bootstrap, user_freqs = user_freqs)
            
            if i == 0: # First iteration
                ctimes = [(xf-xi)/2 + xi] # Initialize central time of the window
                prdrgsy = [prdrg_temp.y_power.value] # Powers of the periodogram of the window
                user_freqs = prdrg_temp.x_freqs # Frequency grid of the periodogram of the window which will be used in posterior iterations
                prdrgsx = prdrg_temp.pdgm_xvalues.value # Period space of the periodogram of the window
                self.windowlength_inv = prdrg_temp.windowlength_err # Save half the peak width 
                
            else: # Rest of iterations
                ctimes.extend([(xf-xi)/2 + xi]) # Add the central time of the shifted window
                prdrgsy.append(prdrg_temp.y_power.value) # Add the powers of the periodogram of the shifted window
            
            i = i + 1 # Increase counter 
    
        self.its = i # Number of iterations (number of windows)
        self.ctimes = np.array(ctimes) # Number of central times
        self.prdrgsx = prdrgsx # Save frequency grid
        self.prdrgsy = prdrgsy # Save the array of periodogram powers
        
        numelem = len(prdrgsx) # Number of frequency grid points
        
        matrix = np.zeros((self.its, numelem)) # Initialize matrix with periodogram power for each window
        for j in range(self.its):
            matrix[j] = prdrgsy[j] # Insert powers of each window            
        self.matrix = np.transpose(matrix)


    def plots(self, vis0 = None, vis1 = None, x0 = None, x1 = None, hlines = None, vlines = None, cmap = 'rainbow', yscale = 'linear', figwidth = 6, figheight = 4, figheight2 = 2, plotsflag = False, plotsflag2 = False, title = None, savefig1 = None, savefig2 = None, savefig3 = None):
        # 'rainbow', 'jet', 'viridis'
        
        """
        Plot the computed spectrogram, the comparation between integrated and peak power across windows, and the peak period across windows 

        Parameters
        ----------
        vis0 : float
            Lower limit (y-axis) of the region of the spectrogram to display (minimum period), in units of minutes 
        vis1 : float
            Upper limit (y-axis) of the region of the spectrogram to display (maximum period), in units of minutes
        x0 : float
            Left limit (x-axis) of the region of the spectrogram to display (minimum window central time), in units of days 
        x1 : float
            Right limit (x-axis) of the region of the spectrogram to display (maximum window central time), in units of days
        hlines: list of floats
            List with the position of reference lines to plot in the y-axis, in units of minutes
        vlines: list of floats
            List with the position of reference lines to plot in the x-axis, in units of days
        cmap : string
            Color map of the mesh plot
        yscale : string
            yscale of the spectrogram, either 'linear' or 'log'
        figwidth : float
            Width of the spectrogram, integrated and peak power, and the peak period figures 
        figheight : float
            Height of the spectrogram figure 
        figheight2 : float
            Height of the integrated and peak power figure 
        plotsflag : boolean
            Flag to plot the comparation between integrated and peak power across windows 
        plotsflag : boolean
            Flag to plot the peak period across windows
        title : None or string
            Title of the spectrogram figure 
        savefig1 : None or string
            Save the spectrogram figure with the specified string as the name
        savefig2 : None or string
            Save power comparison figure with the specified string as the name  
        savefig3 : None or string
            Save the peak period figure with the specified string as the name
        """

        # Define the visualization limits of the spectrogram y-axis
        if vis0 == None: vis0 = self.minimum_x
        if vis1 == None: vis1 = self.maximum_x
            
        self.new_matrix(vis0, vis1, x0, x1) # Extract the variables between the visualization limits ranges    
        self.calcs_newmatrix() # Integrate power and find power and period of the peak from the new variables 
        
        fig, ax = plt.subplots(1, 1, figsize = (figwidth, figheight))

        mesh = ax.pcolormesh(self.newctimes, self.newprdrgsx, self.newmatrix, cmap = cmap) # Create the pcolormesh plot:
       
        # Add a colorbar to the plot:
        cbar = plt.colorbar(mesh)
        cbar.set_label('Power', rotation = 90, fontsize = 12)
        cbar.ax.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0, 0))

        # Plot lines to reference start and end times of the complete data set:
        ax.axvline(x = self.total_xi, color = 'r', linestyle = '--')        
        ax.axvline(x = self.total_xf, color = 'r', linestyle = '--')        
        
        # Plot horizontal and vertical reference lines
        if hlines is not None:
            for h in hlines:
                ax.axhline(y = h, color = 'white', linestyle = ':')
        if vlines is not None:
            for v in vlines:
                ax.axvline(x = v, color = 'black', linestyle = '--')   
        
        if x0 is not None and x1 is not None: 
            ax.set_xlim([x0, x1])
        ax.set_ylim([vis0, vis1])
            
        ax.set_xlabel('Time [d]', fontsize = 12)
        ax.set_ylabel('Period [min]', fontsize = 12)
        if title is not None: ax.set_title(title, fontsize = 10)        
        
        ax.set_yscale(yscale)

    #     ax.set_xticklabels(ctimes, rotation = 'vertical') 
        plt.show() 
        
        if savefig1 is not None:
            fig.savefig(savefig1, format = 'png', dpi = 300, bbox_inches = "tight")    
        
        if plotsflag: # Plot the power comparison figure 
            
            fig, ax2 = plt.subplots(1, 1, figsize = (figwidth*0.8, figheight2))
            
            # Plot the integrated power across windows:
            ax2.plot(self.newctimes, self.inpowers, '--', color = 'tab:orange', alpha = 0.5) 
            ax2.plot(self.newctimes, self.inpowers, '.', color = 'tab:orange', label = 'Integrated power')
            # Plot the peak power across windows:
            ax2.plot(self.newctimes, self.maxperiod_y, ':', color = 'gray', alpha = 0.5)
            ax2.plot(self.newctimes, self.maxperiod_y, 'x', color = 'gray', label = 'Max peak power')   
            # Plot lines to reference start and end times of the complete data set:
            ax2.axvline(x = self.total_xi, color = 'r', linestyle = '--')        
            ax2.axvline(x = self.total_xf, color = 'r', linestyle = '--') 

            # Plot vertical reference lines
            if vlines is not None:
                for v in vlines:
                    ax2.axvline(x = v, color = 'k', linestyle = '--') 
                
            if x0 is not None and x1 is not None: 
                ax2.set_xlim([x0,x1])
            ax2.set_ylim([0,np.max([np.max(self.inpowers), np.max(self.maxperiod_y)])*1.1])                

            ax2.set_xlabel('Time [d]', fontsize = 12)
            ax2.set_ylabel('Scaled power', fontsize = 12)
            if title is not None: ax.set_title(title, fontsize = 10) 
            ax2.legend(fontsize = 10)

            plt.show()
            
            if savefig2 is not None: 
                fig.savefig(savefig2, format = 'png', dpi = 300, bbox_inches = "tight") 

            
        if plotsflag2: # Plot the peak period figure   
            
            fig, ax3 = plt.subplots(1, 1, figsize = (figwidth*0.8, figheight2))

            # Plot peak period across windows:
            ax3.plot(self.newctimes, self.maxperiod_x, '-', color = 'tab:brown', alpha = 0.5) 
            ax3.plot(self.newctimes, self.maxperiod_x, '.', color = 'tab:brown')
            # Plot lines to reference start and end times of the complete data set                    
            ax3.axvline(x = self.total_xi, color = 'r', linestyle = '--')        
            ax3.axvline(x = self.total_xf, color = 'r', linestyle = '--')  

            # Plot horizontal reference lines
            if hlines is not None:
                for h in hlines:
                    ax3.axhline(y = h, color='k', linestyle=':')

            ax3.set_xlabel('Time [d]', fontsize = 12)
            ax3.set_ylabel('Max Period [min]', fontsize = 12)
            if title is not None: ax.set_title(title, fontsize = 10)
            
            if x0 is not None and x1 is not None: 
                ax3.set_xlim([x0, x1])
                
            if savefig3 is not None:
                fig.savefig(savefig3, format='png', dpi=300, bbox_inches="tight") 

    
    def new_matrix(self, vis0, vis1, x0 = None, x1 = None):
        """
        New variables (matrix with periodogram powers, windows central times, and frequency grid) within the x-axis (time) and y-axis (period space) visualization limits
        """

        # Indexes of the frequency grid points closest to the defined y-axis visualization limits:
        index_vis0 = (np.abs(self.prdrgsx - vis0)).argmin()
        index_vis1 = (np.abs(self.prdrgsx - vis1)).argmin()
        
        if x0 is None and x1 is None: # If x-axis visualization limits were not defined
            self.newmatrix = self.matrix[index_vis1:index_vis0, :] # New matrix with the periodogram power data 
            self.newprdrgsx = self.prdrgsx[index_vis1:index_vis0] # New frequency grid
            self.newctimes = self.ctimes # Windows central times
            self.newits = self.its # Algorithm iterations
            
        else: # If x-axis visualization limits were defined
            # Indexes of the windows central times closest to the defined x-axis visualization limits:
            index_x0 = (np.abs(self.ctimes - x0)).argmin()
            index_x1 = (np.abs(self.ctimes - x1)).argmin()
            
            self.newmatrix = self.matrix[index_vis1:index_vis0, index_x0:index_x1] # New matrix with the periodogram power data 
            self.newprdrgsx = self.prdrgsx[index_vis1:index_vis0] # New frequency grid
            self.newctimes = self.ctimes[index_x0:index_x1] # New windows central times
            self.newits = len(self.newctimes) # Algorithm iterations
    

    def calcs_newmatrix(self):
        """
        Calculate from the new matrix, and for each window central time, the integrated power and the power and period of the peak
        """

        # Initialize variables:
        inpowers = np.zeros(self.newits) # Integrated power for each window central time
        maxperiod_x = np.zeros(self.newits) # Period of the max peak for each window central time
        maxperiod_y = np.zeros(self.newits) # Power of the max peak for each window central time

        for j in range(self.newits): # Iterate over each window periodogram
            nm = self.newmatrix[:,j] # Extract powers of the given window periodogram
            inpowers[j] = np.sum(nm) # Sum all the extracted powers 
            peak_arg = np.argmax(nm) # Index of the peak
            maxperiod_x[j] = self.newprdrgsx[peak_arg] # Period of peak
            maxperiod_y[j] = nm[peak_arg] # Power of the peak 

        # Save vectors
        self.inpowers = inpowers / np.median(inpowers) + np.median(inpowers) # Normalize integrated powers with a vertical offset
        self.maxperiod_x = maxperiod_x # Periods of the peaks
        self.maxperiod_y = maxperiod_y / np.median(maxperiod_y) + np.median(inpowers) # Normalized powers of the peaks with a vertical offset    

    