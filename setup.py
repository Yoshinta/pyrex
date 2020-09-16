#!/usr/bin/env python

#  Copyright (C) 2020, Yoshinta Setyawati.
#
#  This file is part of pyrex.
#
#  pyrex is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  pyrex is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with pyrex.  If not, see <http://www.gnu.org/licenses/>.
#


# Import useful things
from distutils.core import setup
from setuptools import find_packages

# Load README for long description
with open('README.md', 'r') as f:
    readme = f.read()

setup(name='pyrex',
      version='0.0.4',
      description='Python package for transforming circular gravitational waveforms to low-exentric waveforms from numerical simulations.',
      long_description=readme,
      author='Yoshinta Setyawati',
      author_email='yoshintaes@gmail.com',
      packages=find_packages(),
      include_package_data=True,
#     package_dir={'pyrex': 'codes'},
      url='https://github.com/Yoshinta/pyrex',
      download_url='https://github.com/Yoshinta/pyrex/archive/master.zip',
      keywords=['numerical relativity', 'gravitational waves', 'waveform', 'eccentric', 'compact binary'],
      install_requires=[
        'jsonschema>=3.0.2',
	'pickleshare>=0.7.5',
        'scipy>=1.3.1',
	'lalsuite>=6.62',
        'h5py>=2.10',
	'sxs==2019.9.9.23.27.50'],
      classifiers=(
                   'Programming Language :: Python :: 3.6',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering'
                   ),
      license='MIT')
