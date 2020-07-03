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
Twist analytic circular waveforms into eccentric.
"""

__author__ = 'Yoshinta Setyawati'

from numpy import *
from pyrex.decor import *
from pyrex.tools import *
from pyrex.basics import *


class Cookware:
   """
   	A class to twist any analytic circular waveforms into eccentric model.
   """
   def __init__(self,approximant,mass1,mass2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,eccentricity,x,inclination,distance,coa_phase,sample_rate=4096.):
        """
            Initiates Cookware class for non-spinning, low eccentricity, and mass ratio<=3 binaries.

            Parameters
            ----------
            mass1         : {float}
            		  Mass of the hevaiest object (MSun).
            mass2         : {float}
                      Dimensionless spin parameters.
            approximant   : {str}
                      Waveform approximant of analytic waves.

            chi           : {float}
                      Spin of the system.
            distance      : {float}
                      Distance of the two bodies (Mpc).
            inclination   : {float}
                      Inclination angle (rad).
            coa_phase     : {float}
                      Coalescence phase (rad).

            Returns
            ------
            times         : []
            		 Time sample array.
            h22	          : []
            		 Complex numbers of the eccentric l=2, m=2 mode.
        """
        #TODO: run database.py
        self.approximant=approximant
        self.mass1=mass1
        self.mass2=mass2
        self.spin1x=spin1x
        self.spin1y=spin1y
        self.spin1z=spin1z
        self.spin2x=spin2x
        self.spin2y=spin2y
        self.spin2z=spin2z
        self.eccentricity=eccentricity
        self.inclination=inclination
        self.distance=distance
        self.coa_phase=coa_phase
        self.x=x

        total_mass=50.
        f_low=25.
        q=mass1/mass2

#check requirements
        self.checkParBoundaris()
        training=checkIfFilesExist(message="training data found: ")
#generate analytic waveform
        laltime,lalamp,lalphase,lalomega=lalwaves_to_nr_scale(q,total_mass,self.approximant,f_low,self.distance,self.inclination,self.coa_phase,sample_rate)
#read training file
        training_dict=read_pkl(training)
#TODO: perform the twis
        self.get_key_quant(training_dict)
        amp_rec,phase_rec=eccentric_from_circular(self.omega_keys,self.amp_keys,training_dict['new_time'],laltime,lalamp,lalphase,lalomega)
#TODO: define after mass scaling
        self.amp=amp_rec
        self.phase=phase_rec
        self.time=training_dict['new_time']
        self.h22=amp_rec*exp(phase_rec*1j)
        #TODO: add the circular close to merger
        #TODO: rescale with total mass

   @staticmethod
   def checkEccentricInp(mass1,mass2,eccentricity):
       ori_total_mass=mass1+mass2
       if eccentricity<0. or eccentricity>0.2:
           error("This version has only been calibrated up to eccentricity<0.2.")
       else:
           pass

   def checkParBoundaris(self):
       chi1=sqrt(self.spin1x**2+self.spin1y**2+self.spin1z**2)
       chi2=sqrt(self.spin2x**2+self.spin2y**2+self.spin2z**2)
       if (abs(chi1)==0.0 and abs(chi2)==0.0):
           self.q=self.mass1/self.mass2
           if self.q>=1 and self.q<=3:
               Cookware.checkEccentricInp(self.mass1,self.mass2,self.eccentricity)
           elif self.q>3:
               warning("This version has only been calibrated up to q<=3.")
               Cookware.checkEccentricInp(self.mass1,self.mass2,self.eccentricity)
           else:
               error("Please correct your mass ratio, only for q>=1.")
       else:
           error("This version has only been calibrated to non-spinning binaries.")

   @staticmethod
   def interpol_key_quant(training_quant,training_keys,test_quant):
        '''
            Interpolate key quantities.
        '''
        A=interp1D(training_quant[1],training_keys[0],test_quant[1])
        B=log((interp1D(training_quant[1],training_keys[0]*exp(training_keys[1]),test_quant[1]))/asarray(A))
        freq=sqrt(interp1D(asarray(training_quant[0]),asarray(training_keys[2])**2,test_quant[0]))
        phi=interp1D(asarray(training_quant[2]),asarray(training_keys[3]),asarray(test_quant[2]))
        return A,B,freq,phi

   def get_key_quant(self,training_dict):
        #TODO: check/remove circular data training
        #TODO: compute x
        q=self.mass1/self.mass2
        training_quant=[training_dict['q'],training_dict['e_ref'],training_dict['x']]
        test_quant=[q,self.eccentricity,self.x]
        par_omega=[training_dict['A_omega'],training_dict['B_omega'],training_dict['freq_omega'],training_dict['phi_omega']]
        par_amp=[training_dict['A_amp'],training_dict['B_amp'],training_dict['freq_amp'],training_dict['phi_amp']]
        A_omega,B_omega,freq_omega,phi_omega=Cookware.interpol_key_quant(training_quant,par_omega,test_quant)
        A_amp,B_amp,freq_amp,phi_amp=Cookware.interpol_key_quant(training_quant,par_amp,test_quant)
        self.omega_keys=[A_omega,B_omega,freq_omega,phi_omega]
        self.amp_keys=[A_amp,B_amp,freq_amp,phi_amp]
