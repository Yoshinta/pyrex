# Copyright (C) 2020 Yoshinta Setyawati <yoshintaes@gmail.com>
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

"""
  Computes waveform's amplitude, phase, omega, time sample, and strain components and align them.
"""

__author__ = 'Yoshinta Setyawati'

from numpy import *
import os
import glob
import h5py
from pyrex.decor import *
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks, savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import warnings
warnings.filterwarnings('ignore')


def get_components(data_path):
     """
        Get amplitude, phase, and strain of the l=2, m=2 mode from a numerical simulation.

        Parameters
        ----------
        data_path : {str}
		  The directory of an NR file that contains 'rhOverM_Asymptotic_GeometricUnits_CoM.h5'.

        Returns
        ------
        times     : []
		          Array of the sample time.
        amp22	  : []
		          Array of the amplitude of the l=2, m=2 mode.
	    phase22   : []
		          Array of the phase of the l=e, m=e mode.
	    h22       : []
		          Array of the l=2, m=2 strain.
    """
     with h5py.File(os.path.join(data_path, "rhOverM_Asymptotic_GeometricUnits_CoM.h5" ), 'r') as f:
         h22 = f["OutermostExtraction.dir/Y_l2_m2.dat"][:]
         h2m2 = f["OutermostExtraction.dir/Y_l2_m-2.dat"][:]

     times = h22[:,0]
     for t1, t2 in zip(times, h2m2[:,0]):
         assert t1 == t2

     h22 = h22[:,1] + 1.j * h22[:,2]
     h2m2 = h2m2[:,1] + 1.j * h2m2[:,2]
     amp22=abs(h22)
     phase22 = unwrap(angle(h22))
     return times,amp22,phase22,h22

def t_align(names,data_path,t_peak,dt=0.4,t_chopped=-50):
    """
        Align waveform such that the peak amplitude is at t=0 and chopped -50M before merger (max t).
        Modify the delta t of every waveform with the same number.

        Parameters
        ----------
        names       : {str}
		           A set of waveform simulation name such as 'SXS_BBH_1081'.
        t_peak     : {float}
                   Merger time before alignment (in positive t/M).
        dt          : {float}
                   delta t of the new time samples. Default 0.4.
        t_chopped   : {float}
                   t final before binary circularizes. Default -50M.

        Returns
        ------
        time_window : []
		          Array of the new sample time.
        amp_window	: []
		          Array of the new amplitude.
	    phase_window: []
		          Array of the aligned phase.
	    h22_window  : []
		          Array of the aligned strain.
    """

    amp_window=[]
    phase_window=[]
    h22_window=[]
    new_time=[]
    time_window=[]

    for i in range(len(names)):
        temp_time,amps,phas,h22c=get_components(data_path+names[i])
        tim_r=temp_time-temp_time[argmax(amps)]
        new_time=(arange(-t_peak[i],tim_r[::-1][0],dt))

        amp_inter=spline(tim_r,amps)
        phase_inter=spline(tim_r,phas)
        h22r_inter=spline(tim_r,h22c.real)
        h22i_inter=spline(tim_r,h22c.imag)

        amp=amp_inter(new_time)
        phase=phase_inter(new_time)
        h22r=h22r_inter(new_time)
        h22i=h22i_inter(new_time)

        array_late_inspiral=int(argmax(amp)+t_chopped/dt)
        time_window.append(new_time[:array_late_inspiral])
        amp_window.append(amp[:array_late_inspiral])
        phase_window.append(phase[:array_late_inspiral])
        h22_window.append(h22r[:array_late_inspiral]+1j*h22i[:array_late_inspiral])
    return asarray(time_window), asarray(amp_window), asarray(phase_window), asarray(h22_window)

def compute_omega(time_sample,h22):
    """
        Computes omega from time sample and h22.
        Omega=d/dt (arg h22) [Husa 2008]

        Parameters
        ----------
        time_sample : []
		          1 dimensional array of time sample of the strain data.
        h22         : []
                  1 dimensional array of strain with l=2, m=2 mode.

        Returns
        ------
        omega     : []
		          1 dimensional array of omega.

    """
    omega=gradient(-unwrap(angle(h22)),time_sample)
    return omega

def interp_omega(time_circular,time_eccentric,omega_circular):
    """
    Interpolate omega circular to the time points of the eccentric waveforms.
    Parameters
    ----------
    time_circular   : []
                    1 dimensional array of time sample in the circular data.
    time_eccentric  : []
                    1 dimensional array of time sample in the eccentric data.
    omega_circular  : []
                    1 dimensional array of the original omega circular data.

    Returns
    ------
    omega_interp    : []
                    1 dimensional array of the interpolated omega circular following time sample of the eccentric data.
    """

    interpol=spline(time_circular,omega_circular)
    omega_interp=interpol(time_eccentric)
    return omega_interp

def f_sin(time_sample, freq, amplitude, phase, offset):
    """
        Computes sinusoidal function given the input parameters.

        Parameters
        ----------
        time_sample   : []
                    1 dimensional array of time sample.
        freq          : {float}
                    Frequency parameter.
        amplitude     : {float}
                    Amplitude parameter.
        phase         : {float}
                    Phase parameter.
        offset        : {float}
                    Offset parameter (for the fitting).

        Returns
        ------
        sin_func      : []
                    1 dimensional array of a sinusoidal function.

    """
    sin_func=amplitude*sin(time_sample * freq + phase)+ offset
    return sin_func

def fit_sin(time_sample, data):
    """
        Computes the optimize curve fitting for a sinusoidal function with sqrt(time_sample).

        Parameters
        ----------
        time_sample   : []
                    1 dimensional array of time sample.
        data          : []
                    1 dimensional array of data to be fitted to a sinusoidal function.

        Returns
        ------
        popt        : []
                    1 dimensional array of the fitting parameters (frequency, amplitude, phase, and offset).
        fit_result  : []
                    1 dimensional array of the fitted data.
    """

    popt,pcov = curve_fit(f_sin, sqrt(time_sample), data)
    fit_result=f_sin(sqrt(time_sample),*popt)
    return popt,fit_result

def find_locals(data,local_min=True,sfilter=True):
    """
        Find local minima/maxima of a given data.

        Parameters
        ----------
        data        : []
                    1 dimensional array of data.
        local_min   : bool
                    If True, find local minima, otherwise local maxima. Default=True.
        sfilter      : bool
                    If True, filter to remove noise will be applied (smooth curve) with savgol filter. Default=True.
        Returns
        ------
        local_array  : []
                    1 dimensional array of local minima/maxima from a given function.

    """
    if filter:
        new_data = savgol_filter(data, 501, 2)
    else:
        new_data=data

    if local_min:
        local=argrelextrema(new_data,less)
    else:
        local=argrelextrema(new_data,greater)
    local_array=asarray(local).reshape(len(local[0]))
    return local_array

def find_roots(x,y):
    """
        Find the values of the x data from a given y data.

        Parameters
        ----------
        x        : []
                1 dimensional array of the x data.
        y        : []
                1 dimensional array of the y data.

        Returns
        ------
        roots_data: []
                1 dimensional array of x values from a given y data.
    """

    s = abs(diff(sign(y))).astype(bool)
    roots_data=x[:-1][s] + diff(x)[s]/(abs(y[1:][s]/y[:-1][s])+1)
    return roots_data

def find_intercept(x,y,y_to_find):
    """
        Find the values of the x data from a given positive/negative values of the y data.

        Parameters
        ----------
        x        : []
                1 dimensional array of the x data.
        y        : []
                1 dimensional array of the y data.
        y_to_find: {float}
                The y value that intercepts y (always positive value).

        Returns
        ------
        roots_pos: []
                1 dimensional array of the x values from a given +y_to_find data.
        roots_neg: []
                1 dimensional array of the x values from a given -y_to_find data.

    """
    if y_to_find<0:
        error("y_to_find is always positive.")
    else:
        roots_pos = find_roots(x, y-y_to_find)
        roots_neg = find_roots(x, y+y_to_find)
    return roots_pos,roots_neg

def compute_residual(time_sample,component,deg=4):
    """
        Computes the residual of sqrt polynomial fits of a given data.

        Parameters
        ----------
        time_sample : []
                    1 dimensional positive array of time sample.
        component   : []
                    1 dimensional array of the component to be fitted.
        deg         : {int}
                    degree of the polynomial function to fit the data. Default=4.

        Returns
        ------
        res         : []
                    1 dimensional array of the residual of the fitted function.
        B_sec       : []
                    1 dimensional array of a polynomial function fitted to the data.
    """
    time_sample=abs(time_sample)
    B_t=component
    p=poly1d(polyfit(sqrt(time_sample),B_t,deg=deg))
    B_sec=p(sqrt(time_sample))
    res=B_t-B_sec

    return res, B_sec

def find_e_amp(ampli,ampli_circular,time,time_circular):
    """
        Computes eccentricity from a set of amplitude data.
        
        Parameters
        ----------
        ampli            : []
        1 dimensional array of ampli data.
        ampli_circular   : []
        1 dimensional array of ampli for zero eccentricity data.
        time            : []
        1 dimensional array of time samples of the corresponding ampli data.
        time_circular   : []
        1 dimensional array of time samples of the corresponding ampli_circular data.
        
        
        Returns
        ------
        e_ampli           : []
        1 dimensional array of e_ampli.
        
        """
    
    ampli_c_interp=interp_omega(time_circular,time,ampli_circular)
    e_ampli=(ampli-ampli_c_interp)/(2*ampli_c_interp)
    return e_ampli


def find_e_omega(omega,omega_circular,time,time_circular):
    """
        Computes eccentricity from omega (e_omega) from a set of omega data.

        Parameters
        ----------
        omega            : []
                        1 dimensional array of omega data.
        omega_circular   : []
                        1 dimensional array of omega for zero eccentricity data.
        time            : []
                        1 dimensional array of time samples of the corresponding omega data.
        time_circular   : []
                        1 dimensional array of time samples of the corresponding omega_circular data.


        Returns
        ------
        e_omg           : []
                        1 dimensional array of e_omega.

    """

    omega_c_interp=interp_omega(time_circular,time,omega_circular)
    e_omg=(omega-omega_c_interp)/(2*(omega_c_interp))
    return e_omg

def measure_e_amp(time,ampli,time_circular,amp_circular):
    """
        Computes the eccentricity from amplitude.
        
        Parameters
        ----------
        time             : []
        Arrays of time samples.
        ampli            : []
        Arrays of the amplitude.
        time_circular    : []
        1 dimensional array to of time samples in circular eccentricity.
        amp_circular   : []
        1 dimensional array to of amp in circular eccentricity.
        
        
        Returns
        ------
        e_omg   : []
        Array of eccenntricity omega as time function.
        
        """
    
    e_amp=[]
    for i in range(len(time)):        
        e_amp.append(find_e_amp(ampli,amp_circular,time_circular,time[i]))
    return e_amp

def measure_e_omega(time,h22,time_circular,h22_circular):
    """
        Computes the eccentricity from omega (see Husa08).

        Parameters
        ----------
        time             : []
                         Arrays of time samples.
        h22              : []
                         Arrays of the strain.
        time_circular    : []
                         1 dimensional array to of time samples in circular eccentricity.
        h22_circular   : []
                         1 dimensional array to of h22 in circular eccentricity.


        Returns
        ------
        e_omg   : []
                Array of eccenntricity omega as time function.

    """

    e_omg=[]
    for i in range(len(time)):
        omega_circular=compute_omega(time_circular,h22_circular)
        omega_high_e=compute_omega(time[i],h22[i])
        e_omg.append(find_e_omega(omega_high_e,omega_circular,time_circular,time[i]))
    return e_omg

def time_window_greater(time,time_point,data):
    """
        Windows data in time series greater than a point in time.
        This function cuts early signal in time.

        Parameters
        ----------
        time            : []
                        1 dimensional array of time samples.
        time_point      : {float}
                        Minimum time in the data after the window.
        data            : []
                        Data to be put in the window.


        Returns
        ------
        new_data       : []
                        Data in the window.

    """
    window=where(time>time_point)
    new_data=data[window]
    return new_data

def noisy_peaks(data,prominence=0.1):
    """
        Finds local maxima in a noisy data.

        Parameters
        ----------
        data        : []
                    1 dimensional array to find its peak.
        prominence  : {float}
                    The minimum height necessary to descend to get from the summit to any higher terrain.
                    Default=0.1.

        Returns
        ------
        peaks       : []
                    1 dimensional array that contains array numbers of the local maxima (peaks) in the noisy data.

    """

    peaks,_ = find_peaks(data,prominence=0.1)
    return peaks
