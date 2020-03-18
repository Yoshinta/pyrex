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
  Decoration and features for package.
"""

__author__ = 'Yoshinta Setyawati'

#derived from London's package.

import os
import inspect
import subprocess

def linenum():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# Class for basic print manipulation
class print_format:
    magenta = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    grey = gray = '\033[1;30m'
    ul = '\033[4m'
    end = '\033[0m'
    underline = '\033[4m'

# Function that uses the print_format class to make tag text for bold printing
def bold(string):
    return print_format.bold + string + print_format.end
def red(string):
    return print_format.red + string + print_format.end
def green(string):
    return print_format.green + string + print_format.end
def magenta(string):
    return print_format.magenta + string + print_format.end
def blue(string):
    return print_format.blue + string + print_format.end
def grey(string):
    return print_format.grey + string + print_format.end
def yellow(string):
    return print_format.yellow + string + print_format.end
def cyan(string):
    return print_format.cyan + string + print_format.end
def darkcyan(string):
    return print_format.darkcyan + string + print_format.end
def textul(string):
    return print_format.underline + string + print_format.end

def alert(msg, fname=None):
    if fname is None:
      fname = 'note'
    print('('+cyan(fname)+')>> '+msg)

def warning(msg, fname=None):
    if fname is None:
      fname = 'warning'
    print('('+yellow(fname)+')>> '+msg)

def error(msg, fname=None):
    if fname is None:
      fname = 'error'
    raise ValueError( '('+red(fname)+')!! '+msg )

def exception(msg, fname=None):
    if fname is None:
      fname = 'exception'
    raise Exception( '('+blue(fname)+')!! '+msg )
