################################################################################
#
# TPRF: a Two-Particle Response Function Toolbox
#
# Copyright (C) 2018 by The Simons Foundation
# Author: H. U.R. Strand
# modified for triqs_spectrometer by A. Hampel & S. Beck
#
# TPRF is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TPRF is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TPRF. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy as np
from io import StringIO
from collections import OrderedDict

# ----------------------------------------------------------------------

#import ase.units as units

class Units(object):
    Bohr = 0.5291772105638411
    Angstrom = 1.0

units = Units()

# ----------------------------------------------------------------------
def parse_hopping_from_wannier90_hr(file):

    r""" Wannier90 real space hopping parser of filestream
         different from function in triqs.lattice.utils
         because it cannot read from file!

    Returns a dictionary where the keys are the real-space hopping vectors,
    in terms of multiples of the lattice vectors, and the values are
    ``num_wann * num_wann`` numpy ndarrays with the hopping integrals.

    Parameters
    ----------

    file : string
        Wannier90 hr file as string to parse

    Returns
    -------

    hopp_dict : dict
        Dictionary of real space hoppings.
    num_wann : int
        Total number of Wannier functions per unit-cell.

    """

    lines = file.splitlines()

    lines.pop(0) # pop time header

    num_wann = int(lines.pop(0))
    nrpts = int(lines.pop(0))

    # print( 'num_wann =', num_wann)
    # print( 'nrpts =', nrpts)

    nlines = int(np.ceil(float(nrpts / 15.)))

    deg = np.array([])
    for line in lines[:nlines]:
        deg = np.concatenate((deg, np.loadtxt(StringIO(line), dtype=int, ndmin=1)))

    assert( deg.shape == (nrpts,) )

    hopp = "\n".join(lines[nlines:])
    hopp = np.loadtxt(StringIO(hopp))
    assert( hopp.shape == (num_wann**2 * nrpts, 7) )

    R = np.array(hopp[:, :3], dtype=int) # Lattice coordinates in multiples of lattice vectors
    nm = np.array(hopp[:, 3:5], dtype=int) - 1 # orbital index pairs, wannier90 counts from 1, fix by remove 1

    t_re = hopp[:, 5]
    t_im = hopp[:, 6]
    t = t_re + 1.j * t_im # complex hopping amplitudes for each R, mn (H(R)_{mn})

    # -- Dict with hopping matrices

    r_dict = OrderedDict()
    hopp_dict = {}
    for idx in range(R.shape[0]):
        r = tuple(R[idx])

        if r not in r_dict:
            r_dict[r] = 1
        else:
            r_dict[r] += 1

        if r not in hopp_dict:
            hopp_dict[r] = np.zeros((num_wann, num_wann), dtype=complex)

        n, m = nm[idx]
        hopp_dict[r][n, m] = t[idx]

    # -- Account for degeneracy of the Wigner-Seitz points

    for r, weight in zip(list(r_dict.keys()), deg):
        hopp_dict[r] /= weight

    return hopp_dict, num_wann

# ----------------------------------------------------------------------
def parse_lattice_vectors_from_wannier90_wout(file):

    r""" Wannier90 real space lattice vectors parser of filestream
         different from function in triqs.lattice.utils
         because it cannot read from file!

    Parameters
    ----------

    file : string
        Wannier90 ``*.wout`` file to parse.

    Returns
    -------

    vectors : list of three three-tuples of floats
        Lattice vectors.

    """

    lines = file.splitlines()

    # -- Find start of data in text file

    for idx, line in enumerate(lines):
        if 'Lattice Vectors' in line:
            if '(Ang)' in line:
                unit = units.Angstrom
            elif '(Bohr)' in line:
                unit = units.Bohr
            else:
                raise NotImplementedError
            break

    if not 'Lattice Vectors' in line:
        raise IOError

    lines = "\n".join(lines[idx+1:idx+4])
    array = np.loadtxt(StringIO(lines), usecols=(1, 2, 3))

    array *= unit

    # -- convert 3x3 array to list of tuples

    vectors = []
    for idx in range(array.shape[0]):
        v = tuple(array[idx])
        vectors.append(v)

    return vectors

# ----------------------------------------------------------------------
def parse_reciprocal_lattice_vectors_from_wannier90_wout(filename):

    r""" Wannier90 reciprocal space lattice vectors parser of ``*.wout`` files.

    Parameters
    ----------

    filename : str
        Wannier90 ``*.wout`` file to parse.

    Returns
    -------

    vectors : list of three three-tuples of floats
        Reciprocal lattice vectors.

    """

    with open(filename, 'r') as fd:
        lines = fd.readlines()

    # -- Find start of data in text file

    for idx, line in enumerate(lines):
        if 'Reciprocal-Space Vectors (Ang^-1)' in line:
            break

    lines = "".join(lines[idx+1:idx+4])
    array = np.loadtxt(StringIO(lines), usecols=(1, 2, 3))

    # -- convert 3x3 array to list of tuples

    vectors = []
    for idx in range(array.shape[0]):
        v = tuple(array[idx])
        vectors.append(v)

    return vectors

