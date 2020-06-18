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
  Utilities functions for basic computations.
"""

__author__ = 'Yoshinta Setyawati'

from numpy import *
import h5py


def read_HDF5(file_dir):
    """
        Read a HDF5 file.
        Parameters
        ----------
        The directory of the file.

        Returns
        ------
        The read file with keys.
    """
    f=h5py.File(file_dir,'r')
    return f

def masses_from_eta(eta,total_mass=50.,**unused):
    """
        Computes mass1 and mass2 from eta and total mass.

        Parameters
        ----------
        eta     : {float}
                Eta = (m1m2)/(m1+m2)**2 of a given system.
        total_m : {float}
                Total mass m1+m2 of the system in MSun.

        Returns
        ------
        mass1 and mass2 in MSun.
    """
    m1m2_from_eta=eta*total_mass**2
    mass2_from_eta = -(-total_mass+sqrt((total_mass**2)-(4.*m1m2_from_eta)))/2
    mass1_from_eta = -(-total_mass-sqrt((total_mass**2)-(4.*m1m2_from_eta)))/2
    return mass1_from_eta,mass2_from_eta

def check_total_spin(spinx,spiny,spinz):
    """
        Check if the spins are reasonable.
        The magnitude of the spin in each body should less than 1.
        sqrt(spin_ix**2+spin_iy**2+spin_iz**2)<1.
        If over than one, will be normalized.

        Parameters
        ----------
        spinx: {float}
                Dimensionless spin in x direction.
        spiny: {float}
                Dimensionless spin in y direction.
        spinz: {float}
                Dimensionless spin in z direction.

        Returns
        ------
        spinx: Normalized dimensionless spin in x direction
        spiny: Normalized dimensionless spin in y direction
        spinz: Normalized dimensionless spin in z direction
    """
    tspin=sqrt(spinx**2+spiny**2+spinz**2)
    if (tspin>1):
        spinx=spinx/(tspin+0.0001)
        spiny=spiny/(tspin+0.0001)
        spinz=spinz/(tspin+0.0001)
    return spinx,spiny,spinz

def filter_dicts(alldata,key,val,target):
    '''
        Obtain the value of target if the value of key is val in alldata.
    '''
    seen=set()
    return [d[target] for d in alldata if d[key]==val]

def checkIfDuplicates(listofElems):
    '''
        Check if given list contains any duplicates.
    '''
    for elem in listofElems:
        if listofElems.count(elem)>1:
            return True
    return False
