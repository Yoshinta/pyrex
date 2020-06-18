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
Read numerical waveforms and measure their eccentricity.
"""

__author__ = 'Yoshinta Setyawati'

from numpy import *
from pyrex.decor import *
from pyrex.tools import *
from pyrex.basics import *
from scipy.signal import savgol_filter

class Glassware:
   """
   	A class to measure the eccentricity of a given NR waveform.
   """
   def __init__(self,q,chi,data_path,names,e_ref,nr=True):
        """
            Initiates Glassware class for non-spinning, low eccentricity, and mass ratio<=3 binaries.

            Parameters
            ----------
            q           : []
            		  Mass ratio.
            chi         : {float}
                      Dimensionless spin parameters.
            data_path   : {str}
                      Directory of the NR simulations.
            names       : []
                      Simulation names.
            e_ref       : []
                      e at the reference frequency ('e_comm').
            nr          : {}
                      Data type. Default nr=True.

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

        if (abs(chi)==0.0):
            #if q>=1 and q<=3:
            self.q=q
            self.chi=chi
            self.data_path=data_path
            self.names=names
            self.e_ref=e_ref
            #else:
            #    error("Please correct your mass ratio, only for q<=3.")
        else:
            error("Please correct your spin, only for the non-spinning binaries, s1x=s1y=s1z=s2x=s2y=s2z=0.")

   def components(self):
        """
            Computes and align the amplitude, phase, strain of the l=2, m=2 mode of NR waveforms.

            Parameters
            ----------
            time_peak   : {float}
                        The maximum amplitude before alignment.
       """

        time,amp,phase,h22=t_align(self.names,self.data_path)
        self.time=time
        self.amp=amp
        self.phase=phase
        self.h22=h22

        omega=[]
        for i in range(len(self.time)):
            omega.append(compute_omega(self.time[i],self.h22[i]))
        self.omega=asarray(omega)


   def check_double_circ(self):
       circ_q=[]
       circ_names=[]
       circ_amp=[]
       circ_phase=[]
       circ_omega=[]
       circ_time=[]
       for i in range(len(self.names)):
           if self.e_ref[i]==0:
               circ_q.append(self.q[i])
               circ_names.append(self.names[i])
               circ_amp.append(self.amp[i])
               circ_phase.append(self.phase[i])
               circ_omega.append(self.omega[i])
               circ_time.append(self.time[i])
       if checkIfDuplicates(circ_q):
           error("Please check duplicates of mass ratio and eccentricity in the provided circular waveforms.")
       else:
           for j in range(len(self.q)):
               if self.q[j] not in circ_q:
                   error('"Simulation name {} has no circular waveform with the same mass ratio"'.format(self.names[j]))
               else:
                   pass
       return circ_names,circ_q,circ_time,circ_amp,circ_phase,circ_omega

   @staticmethod
   def get_eX(self,circ_q,circ_time,circ,component,new_time):

       eX=[]
       for i in range(len(circ_q)):
           circs=spline(circ_time[i],circ[i])
           for j in range(len(self.q)):
               ecc=spline(self.time[j],component[j])
               if self.q[j]==circ_q[i]:
                   eX_filter=(ecc(new_time)-circs(new_time))/(2.*circs(new_time))
                   if self.e_ref[j]!=0:
                       eX_filter=savgol_filter(eX_filter, 501, 2)
                   eX.append(eX_filter)
       return eX

   def compute_e_estimator(self):
       """
           Computes eccentricity from omega asca function in time (see Husa).

           Parameters
           ----------
           time_circular    : []
                            1 dimensional array to of time samples in circular eccentricity.
           omega_circular   : []
                            1 dimensional array to of omega in circular eccentricity.
           h22              : []
                            1 dimensional array to of h22 in circular eccentricity.

       """
       begin_tm=-1500.
       end_tm=-50.4
       len_tm=15000
       new_time=linspace(begin_tm,end_tm,len_tm)

       circ_names,circ_q,circ_time,circ_amp,circ_phase,circ_omega=Glassware.check_double_circ(self)

       eX_omega=Glassware.get_eX(self,circ_q,circ_time,circ_omega,self.omega,new_time)
       eX_amp=Glassware.get_eX(self,circ_q,circ_time,circ_amp,self.amp,new_time)
       self.eX_omega=eX_omega
       self.eX_amp=eX_amp
       self.new_time=new_time

   def fit_model(self):
       phase_params=zeros((len(self.names),4))
       amp_params=zeros((len(self.names),4))
       fit_phase=[]
       fit_amp=[]

       circ_names,circ_q,circ_time,circ_amp,circ_phase,circ_omega=Glassware.check_double_circ(self)

       for i in range(len(circ_omega)):
           interp_omega_c=spline(circ_time[i],circ_omega[i])
           interp_amp_c=spline(circ_time[i],circ_amp[i])
           for j in range(len(self.names)):
               if self.q[j]==circ_q[i]:
                   phase_params[j],fit_phaser=fitting_eccentric_function(-59./24,self.eX_omega[j],interp_omega_c(self.new_time))
                   amp_params[j],fit_ampr=fitting_eccentric_function(-83./24,self.eX_amp[j],interp_amp_c(self.new_time))

                   fit_phase.append(fit_phaser)
                   fit_amp.append(fit_ampr)

       self.A_omega=phase_params.T[0]
       self.B_omega=phase_params.T[1]
       self.freq_omega=phase_params.T[2]
       self.phi_omega=phase_params.T[3]
       self.fit_omega=fit_phase

       self.A_amp=amp_params.T[0]
       self.B_amp=amp_params.T[1]
       self.freq_amp=amp_params.T[2]
       self.phi_amp=amp_params.T[3]
       self.fit_amp=fit_amp


   def compute_e_from_amplitude(self,time_circular,amplitude_circular):
          #"""
        #               Computes eccentricity from omega asa function in time (see Husa).

        #               Parameters
        #               ----------
        #               time_circular    : []
        #               1 dimensional array to of time samples in amplitude eccentricity.
        #               amplitude_circular   : []
        #               1 dimensional array to of amplitude in circular eccentricity.
        # """

      e_amplitude=[]
      try:
          for i in range(len(self.time)):
              e_amplitude.append(find_e_amp(self.amp[i],amplitude_circular,self.time[i],time_circular))
      except:
          e_amplitude=measure_e_amp(self.time,self.amp,time_circular,amplitude_circular)
      self.e_amp=asarray(e_amplitude)
