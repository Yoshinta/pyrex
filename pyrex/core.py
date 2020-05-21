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

class Glassware:
   """
   	A class to measure the eccentricity of a given NR waveform.
   """
   def __init__(self,q,chi,data_path,names):
        """
            Initiates Glassware class for non-spinning, low eccentricity, and mass ratio<=3 binaries.

            Parameters
            ----------
            q           : {float}
            		  Mass ratio.
            chi         : {float}
                      Dimensionless spin parameters.
            data_path   : {str}
                      Directory of the NR simulations.
            names       : {str}
                      Simulation names

            Returns
            ------
            times     : []
            		          Array of the sample time.
            amp22	  : []
            		          Array of the amplitude of the l=2, m=2 mode.
            phase22   : []
            		          Array of the phase of the l=e, m=e mode.
            h22       : []            		          Array of the l=2, m=2 strain.
        """

        if (abs(chi)==0.0):
            if q>=1 and q<=3:
                self.q=q
                self.chi=chi
                self.data_path=data_path
                self.names=names
            else:
                error("Please correct your mass ratio, only for q<=3.")
        else:
            error("Please correct your spin, only for the non-spinning binaries, s1x=s1y=s1z=s2x=s2y=s2z=0.")

   def components(self,time_peak):
        """
            Computes and align the amplitude, phase, strain of the l=2, m=2 mode of NR waveforms.

            Parameters
            ----------
            time_peak   : {float}
                        The maximum amplitude before alignment.
       """

        time,amp,phase,h22=t_align(self.names,self.data_path,time_peak)
        self.time=time
        self.amp=amp
        self.phase=phase
        self.h22=h22

        omega=[]
        for i in range(len(self.time)):
            omega.append(compute_omega(self.time[i],self.h22[i]))
        self.omega=asarray(omega)

   def compute_e_from_omega(self,time_circular,omega_circular,h22_circular):
       """
           Computes eccentricity from omega asa function in time (see Husa).

           Parameters
           ----------
           time_circular    : []
                            1 dimensional array to of time samples in circular eccentricity.
           omega_circular   : []
                            1 dimensional array to of omega in circular eccentricity.
           h22              : []
                            1 dimensional array to of h22 in circular eccentricity.

       """
       e_omega=[]
       try:
           for i in range(len(self.time)):
               e_omega.append(find_e_omega(self.omega[i],omega_circular,self.time[i],time_circular))
       except:
           e_omega=measure_e_omega(self.time,self.h22,time_circular,h22_circular)
       self.e_omega=asarray(e_omega)

   def compute_e_from_amplitude(self,time_circular,amplitude_circular):
          #"""
        #               Computes eccentricity from omega asa function in time (see Husa).

        #               Parameters
        #               ----------
        #               time_circular    : []
        #               1 dimensional array to of time samples in amplitude eccentricity.
        #               amplitude_circular   : []
        #               1 dimensional array to of amplitude in circular eccentricity.
         #"""

      e_amplitude=[]
      try:
          for i in range(len(self.time)):
              e_amplitude.append(find_e_amp(self.amp[i],amplitude_circular,self.time[i],time_circular))
      except:
          e_amplitude=measure_e_amp(self.time,self.amp,time_circular,amplitude_circular)
      self.e_amp=asarray(e_amplitude)
