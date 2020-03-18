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
   def __init__(self,q,chi):
        """
            Initiates Glassware class for non-spinning, low eccentricity, and mass ratio<=3 binaries.
        """
        if chi<0.0001:
            if q>=1 and q<=3:
                self.q=q
                self.chi=chi
            else:
                error("Please correct your mass ratio, only for q<=3.")
        else:
            error("Please correct your spin, only for the non-spinning binaries, s1x=s1y=s1z=s2x=s2y=s2z=0.")
