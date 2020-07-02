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
import os

class Cookware:
   """
   	A class to twist any analytic circular waveforms into eccentric model.
   """
   def __init__(self,approximant,sample_rate,mass1,mass2,chi,eccentricity,inclination,distance,coa_phase):
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
            		 Complex number of eccentric h22 mode.
        """

        try:
            file = read_HDF5(database.h5)
        except IOError:
            #TODO: run database.py

        if (abs(chi)==0.0):
            #if q>=1 and q<=3:
            q=mass1/mass2
            ori_total_mass=mass1+mass2
            total_mass=50.
            f_low=25.
            if q>3.:
                error("Please correct your mass ratio, only for q<=3.")
            if eccentricity<0 and eccentricity>0.2:
                error("This version has only been calibrated to eccentricity<0.2.")
            #TODO: generate analytic waveform
            laltime,lalamp,lalphase,lalomega=lalwaves_to_nr_scale(q,total_mass,approximant,f_low,distance,inclination,coa_phase,sample_rate)
            #TODO: read the hdf5 file
            #TODO: perform the twis
            #TODO: rescale with total mass

        else:
            error("Please correct your spin, only for the non-spinning binaries, s1x=s1y=s1z=s2x=s2y=s2z=0.")
