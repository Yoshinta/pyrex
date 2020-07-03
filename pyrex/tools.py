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
import lalsimulation as ls
import lal
import h5py
from pyrex.decor import *
from pyrex.basics import *
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks, savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import integrate
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
     phase22 = -unwrap(angle(h22))
     return times,amp22,phase22,h22

def t_align(names,data_path,dt=0.4,t_junk=250.,t_circ=-50):
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
        temp_time,temp_amp,temp_phase,temp_h22=get_components(data_path+names[i])
        timeshift=temp_time-temp_time[argmax(temp_amp)]
        shifted_time=arange(timeshift[0],timeshift[::-1][0],dt)

        amp_inter=spline(timeshift,temp_amp)
        phase_inter=spline(timeshift,temp_phase)
        h22r_inter=spline(timeshift,temp_h22.real)
        h22i_inter=spline(timeshift,temp_h22.imag)

        amp=amp_inter(shifted_time)
        phase=phase_inter(shifted_time)
        h22r=h22r_inter(shifted_time)
        h22i=h22i_inter(shifted_time)

        array_early_inspiral=int(t_junk/dt)                   #remove the junk radiation
        array_late_inspiral=int(argmax(amp)+t_circ/dt)   #due to circularization for low q & low e binaries, remove some t before merger.

        time_window.append(shifted_time[array_early_inspiral:array_late_inspiral])
        amp_window.append(amp[array_early_inspiral:array_late_inspiral])
        h22_window.append(h22r[array_early_inspiral:array_late_inspiral]+1j*h22i[array_early_inspiral:array_late_inspiral])
        phase_window.append(-unwrap(angle(h22_window[i])))
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

def f_sin(xdata, amplitude, B, freq, phase):
    #TODO: fix the function description.
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
    sin_func=amplitude*exp(B*xdata)*sin(xdata*freq/(2*pi)+phase)
    #sin_func=amplitude*sin(time_sample * freq + phase)+ offset
    return sin_func

def fit_sin(xdata, ydata):
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

    popt,pcov = curve_fit(f_sin, xdata, ydata)
    fit_result=f_sin(xdata,*popt)
    return popt,fit_result

def fitting_eccentric_function(pwr,e_amp_phase,interpol_circ):
    x=(interpol_circ)**pwr-(interpol_circ[0])**pwr
    y=e_amp_phase
    par,fsn=fit_sin(x,y)
    return par,fsn

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
    #TODO: delete this function
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
    #TODO: delete this function
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

def find_Y22(iota,coa_phi):
    '''
        Compute Y22 of spherical harmonics waveform.
        Source: https://arxiv.org/abs/0709.0093.
        Parameters
        ----------
        iota: {float}
                Inclination angle (rad).
        phi : {float}
                Phase of coalescence (rad).

        Returns
        ------
        Y22 : Spherical harmonics of the l=2, m=2 mode.
    '''
    Y22=sqrt(5./(64*pi))*((1+cos(iota))**2)*exp(2*coa_phi*1j)
    return Y22

def find_x(old_time,omega,new_time):
    '''
        Compute x at the beginning of new time array.
    '''
    interp_omega=spline(old_time,omega)
    x=interp_omega(new_time[0])**(2./3)
    return x

def lal_waves(q,total_mass,approximant,f_lower,distance,inclination,coa_phase,**kwargs):
    m1,m2=masses_from_q(q,total_mass)
    spin1x=spin1y=spin1z=spin2x=spin2y=spin2z=0.
    long_asc_nodes=0.
    eccentricity=0.
    mean_per_ano=0.
    f_ref=f_lower
    f_max=2048.
    lal_pars=None
    aprox=eval('ls.'+str(approximant))
    if 'delta_t' and 'delta_f' in kwargs:
        error("Please provide delta_t or delta_f.")
    elif 'delta_t' in kwargs and 'delta_f' not in kwargs:
        hp1,hc1=ls.SimInspiralChooseTDWaveform(lal.MSUN_SI*m1,
               lal.MSUN_SI*m2,
               spin1x, spin1y, spin1z,
               spin2x, spin2y, spin2z,
               lal.PC_SI*distance*1e6,
               inclination, coa_phase,
               long_asc_nodes, eccentricity, mean_per_ano,
               kwargs['delta_t'], f_lower, f_ref,
               lal_pars,aprox)

        hp = hp1.data.data
        hc = hc1.data.data
        Ttot = hp1.data.length * hp1.deltaT
        t1 = arange(hp1.data.length, dtype=float) * hp1.deltaT
        t1 = t1+hp1.epoch
        x=t1-t1[argmax(abs(hp+hc*1j))]

    elif 'delta_f' in kwargs and 'delta_t' not in kwargs:
        hp1,hc1=ls.SimInspiralChooseFDWaveform(lal.MSUN_SI*m1,
               lal.MSUN_SI*m2,
               spin1x, spin1y, spin1z,
               spin2x, spin2y, spin2z,
               lal.PC_SI*distance*1e6,
               inclination, coa_phase,
               long_asc_nodes, eccentricity, mean_per_ano,
               kwargs['delta_f'], f_lower, f_max, f_ref,
               lal_pars,aprox)
        hp = hp1.data.data
        hc = hc1.data.data
        Ftot = hp1.data.length * hp1.deltaF
        t1 = arange(hp1.data.length, dtype=float) * hp1.deltaF
        t1 = t1+hp1.epoch
        x=t1
    else:
        error("Please provide delta_t or delta_f.")
    return x,hp,hc

def lalwaves_to_nr_scale(q,total_mass,approximant,f_low,distance,iota,coa_phi,sample_rate):
    dt=1./sample_rate
    #Beware: numerical error 1e-30 when return the scale back! More obvious on phase.
    amp_scale=total_mass*lal.MTSUN_SI*lal.C_SI/(1e6*distance*lal.PC_SI)
    sample_times,hp,hc=lal_waves(q,total_mass,approximant,f_low,distance,iota,coa_phi,delta_t=dt)
    h2=hp+hc*1j
    Y22=find_Y22(iota,coa_phi)
    hs=h2/(amp_scale*Y22)

    time=sample_times/(total_mass*lal.MTSUN_SI)
    amp=abs(hs)
    phase=unwrap(angle(hs))
    omega=compute_omega(time,hs)
    return time,amp,phase,omega

def eccentric_from_circular(par_omega,par_amp,new_time,time,amp,phase,omega,phase_pwr=-59./24,amp_pwr=-83./24):

    interp_omega=spline(time,omega)
    interp_amp=spline(time,amp)
    omega_circ=-interp_omega(new_time)
    amp_circ=interp_amp(new_time)

    x_omega=(omega_circ)**phase_pwr-((omega_circ[0])**phase_pwr)
    x_amp=(amp_circ)**amp_pwr-(amp_circ[0])**amp_pwr

    fit_ex_omega=f_sin(x_omega,par_omega[0],par_omega[1],par_omega[2],par_omega[3])
    fit_ex_amp=f_sin(x_amp,par_amp[0],par_amp[1],par_amp[2],par_amp[3])
    omega_rec=fit_ex_omega*2*omega_circ+omega_circ
    phase_rec=integrate.cumtrapz(omega_rec,new_time,initial=0)
    amp_rec=fit_ex_amp*2*amp_circ+amp_circ
    return amp_rec,phase_rec

#def compute_match_waves(new_time, test_time, test_h22, amp_recon,phase_recon,f_lower,sample_rate,psd='aLIGO'):

#    delta_t=1./sample_rate
#    realh=spline(test_time,np.real(test_h22))
#    imagh=spline(test_time,np.imag(test_h22))
#    h_reckon=amp_recon*exp(phase_recon*1j)

#    analytic_real_tser=TimeSeries(real(h_reckon),delta_t=delta_t)
#    nr_real_tser=TimeSeries(realh(new_time),delta_t=delta_t)

#    if psd=='aLIGO':
#        tlen=len(h2o_tser)
#        delta_f = 1.0 / nr_real_tser.duration
#        flen = tlen//2 + 1
#        psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
#    else:
#        psd =None

#    match22=match(analytic_real_tser,nr_real_tser,psd=psd,low_frequency_cutoff=f_lower)[0]
#    overlap_amp=overlap(TimeSeries(amp_recon,delta_t=delta_t),TimeSeries(abs(realh(new_time)+imagh(new_time)*1j),delta_t=delta_t),psd=psd,low_frequency_cutoff=f_lower)
#    overlap_phase=overlap(TimeSeries(phase_recon,delta_t=delta_t),TimeSeries(-np.unwrap(np.angle(realh(new_time)+imagh(new_time)*1j)),delta_t=delta_t),psd=psd,low_frequency_cutoff=f_lower)
#    return match22,overlap_amp,overlap_phase
