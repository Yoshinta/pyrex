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
import pickle
import glob
import os
from scipy import interpolate
from pyrex.decor import *
import statistics

def read_HDF5(file_dir):
    """
        Read an HDF5 file.
        Parameters
        ----------
        file_dir    : The directory of the file.

        Returns
        ------
        f           : The read file with keys.
    """
    f=h5py.File(file_dir,'r')
    return f

def write_HDF5(outfname,data_dict):
    """
        Write an HDF5 file.
        Parameters
        ----------
        outfname  : The directory of the output file.
        data_dict : The data variables to be written.

        Returns
        ------
        The written file with keys in outfname.
    """
    fh5 = h5py.File(outfname, 'w')
    for var in data_dict.keys():
        print(var)
        try:
            print(dtype(data_dict[var][0]),var)
            fh5.create_dataset(var, data = asarray(data_dict[var]))
        except:
            print(type(data_dict[var]),var)
            if type(data_dict[var])!=list:
                if type(data_dict[var])!=str:
                    fh5.create_dataset(var, data = asarray(data_dict[var]))
                else:
                    asciiList = [n.encode("ascii", "ignore") for n in data_dict[var]]
                    fh5.create_dataset(var, (len(asciiList),1),'S10', asciiList)
            elif type(data_dict[var][0])==list:
                fh5.create_dataset(var, data = asarray(data_dict[var]))
            else:
                asciiList = [n.encode("ascii", "ignore") for n in data_dict[var]]
                fh5.create_dataset(var, (len(asciiList),1),'S10', asciiList)
    fh5.close()

def read_pkl(file_dir):
    """
        Read a pickle file.
        Parameters
        ----------
        file_dir    : The directory of the file.

        Returns
        ------
        f           : The read file with keys.
    """
    with open(file_dir, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(outfname,data_dict):
    """
        Write a pickle file.
        Parameters
        ----------
        outfname  : The directory of the output file.
        data_dict : The data variables to be written.

        Returns
        ------
        The written file with keys in outfname.
    """

    f = open(outfname,"wb")
    pickle.dump(data_dict,f)
    f.close()

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

def masses_from_q(q,total_mass=50.):
    """
        Computes mass1 and mass2 from eta and total mass.

        Parameters
        ----------
        q       : {float}
                q = m1/m2, where m1>m2.
        total_m : {float}
                Total mass m1+m2 of the system in MSun.

        Returns
        ------
        mass1 and mass2 in MSun.
    """

    mass1 = q/(1+q)*total_mass
    mass2 = total_mass-mass1
    return mass1,mass2

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
        spinx: {float}
            Normalized dimensionless spin in x direction
        spiny: {float}
            Normalized dimensionless spin in y direction
        spinz: {float}
            Normalized dimensionless spin in z direction
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
        Check if the given list contains any duplicates.
    '''
    for elem in listofElems:
        if listofElems.count(elem)>1:
            return True
    return False

def checkIfFilesExist(message,dirfile="../data/"):
    '''
        Check if pickle files exist.
    '''
    r=0
    os.chdir(dirfile)
    print(message)
    for file in glob.glob("*.pkl"):
        print(file)
        r=r+1
    if r<1:
        error("No *pkl files found in " + str(dirfile) + " . Please run 'example/traindata.py' to produce the train data.")
    if r>1:
        error("Found " + str(r) + "*pkl files in " + dirfile + " . Please remove other *pkl files than the training data.")
    else:
        dfs=dirfile+str(file)
    return dfs

def interp1D(trainkey,trainval,testkey):
    '''
        Perform 1D interpolation.

        Parameters
        ----------
        trainkey: []
                Array of the x interpolation values.
        trainval: []
                Array of the y interpolation values.
        testkey: []
                The position of the new x for interpolation.

        Returns
        ------
        result: []
                The interpolated value in 1 dimension.

    '''
    newkey,newval=check_duplicate_training(trainkey,trainval)

    if testkey<min(trainkey) or testkey>max(trainkey):
        interp=interpolate.interp1d(newkey,newval, fill_value='extrapolate')
        result=interp(testkey)
    else:
        interp=interpolate.interp1d(trainkey,trainval)
        result=interp(testkey)
    return result

def check_duplicate_training(trainkey,trainval):
    '''
        Check if the training keys have duplicate numbers.
        If so, get its average values before performing interpolation.
        Parameters
        ----------
        trainkey: []
                Array of the x interpolation values.
        trainval: {float}
                Array of the y interpolation values.

        Returns
        ------
        newkey: []
                The new x interpolation values (no duplicates).
        newval: []
                The new y interpolation values, average of the old trainval with duplicate trainkey.

    '''
    d = {}
    newkey=[]
    newval=[]

    for a, b in zip(list(trainkey), list(trainval)):
        d.setdefault(a, []).append(b)

    for key in d:
        newkey.append(key)
        newval.append(statistics.median(d[key]))
    return newkey,newval

__all__ = ["read_HDF5", "write_HDF5", "read_pkl",
           "write_pkl",
           "masses_from_eta", "masses_from_q",
           "check_total_spin", "filter_dicts",
           "checkIfDuplicates", "checkIfFilesExist",
           "interp1D","check_duplicate_training"]
