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
  Download a set of SXS data from given simulation numbers.
"""

__author__ = 'Yoshinta Setyawati'

#TODO: test how it works
#

import sxs
from sxs import zenodo as zen
# For interacting with the data
import h5py
from numpy import *
import json

#import eccentric simulations as used PRD 98, 044015(2018)
chosen_ids=['0180','1355','1356','1357','1358','1359','1360','1361','1362','1363','0184','1364','1365','1366','1367','1368','1369','1370','0183','1371','1372','1373','1374']
def download_data(ids=chosen_ids):
    """
        Download given simulations of the SXS catalog.

        Parameters
        ----------
        ids : []
            1 dimensional array of the SXS simulation IDs such as '0180', '1356' etc.
            Default=chosen_ids.


    """
    for ids in chosen_ids:
        name_ids='SXS:BBH:'+ids
        zen.download.matching("common-metadata.txt", "metadata.txt", "metadata.json", \
                      "rhOverM_Asymptotic_GeometricUnits_CoM.h5", \
                      "Horizons.h5", \
                      sxs_ids = name_ids, \
                      dry_run = False, \
                      highest_lev_only = True)
