#!/bin/python3

"""
Reads DMFT_ouput observables such as real-frequency Sigma and a Wannier90
TB Hamiltonian to compute spectral properties. It runs in two modes,
either calculating the bandstructure or Fermi slice.

Written by Sophie Beck, 2021
"""

import numpy as np
from numpy import dtype
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib import cm, colors
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import itertools
import matplotlib.pyplot as plt

# triqs
from triqs.sumk import SumkDiscreteFromLattice
from h5 import HDFArchive
from triqs.gf import BlockGf
from triqs.gf import GfReFreq, MeshReFreq
from triqs.utility.dichotomy import dichotomy
import tools.tools as tools
from tools.TB_functions import *

def _convert_kpath(k_mesh):
    k_path = k_mesh['k_path']
    k_path = [list(map(lambda item: (k[item]), k.keys())) for k in k_path] # turn dict into list
    k_point_labels = [k.pop(0) for k in k_path] # remove first item, which is label
    # make sure all kpts are floats
    k_path = [list(map(float,k)) for k in k_path]
    k_path = [(np.array(k), np.array(k_path[ct+1])) for ct, k in enumerate(k_path) if ct+1 < len(k_path)] # turn into tuples

    return k_path, k_point_labels

def get_tb_bands(e_mat, mu):
    """
    Compute band eigenvalues and eigenvectors from matrix per k-point
    """

    e_val = np.zeros((e_mat.shape[0], e_mat.shape[2]), dtype=complex)
    e_vec = np.zeros(np.shape(e_mat), dtype=complex)
    for ik in range(np.shape(e_mat)[2]):
        e_val[:,ik], e_vec[:,:,ik] = np.linalg.eigh(e_mat[:,:,ik])

    return e_val.real-mu, e_vec

def get_tb_kslice(tb, k_mesh, dft_mu):
    """
    Compute band eigenvalues and eigenvectors...
    """

    prim_to_cart = [[0,1,1],
                    [1,0,1],
                    [1,1,0]]
    cart_to_prim = np.linalg.inv(prim_to_cart)
    k_path, _ = _convert_kpath(k_mesh)
    final_x, final_y = k_path[1]
    Z = np.array(k_mesh['Z'])

    fermi = dft_mu
    e_val, e_vec = get_kx_ky_FS(final_x, final_y, Z, tb, k_trans_back=cart_to_prim, N_kxy=k_mesh['n_k'], kz=k_mesh['kz'], fermi=fermi)

    return e_val, e_vec

def calc_tb_bands(data, add_spin, add_local, k_mesh, fermi_slice, projected_orbs, band_basis = False ):
    """
    calculate tight-binding bands based on a W90 Hamiltonian
    """
    # set up Wannier Hamiltonian
    n_orb_rescale = 2 * data['n_wf'] if add_spin else data['n_wf']
    H_add_loc = np.zeros((n_orb_rescale, n_orb_rescale), dtype=complex)
    if add_spin: H_add_loc += tools.lambda_matrix_w90_t2g(add_local)

    hopping = {eval(key): np.array(value, dtype=complex) for key, value in data['hopping'].items()}
    tb = tools.get_TBL(hopping, data['units'], data['n_wf'], extend_to_spin=add_spin, add_local=H_add_loc)
    # print local H(R)
    h_of_r = tb.hoppings[(0,0,0)][2:5,2:5] if add_spin else tb.hoppings[(0,0,0)]
    tools.print_matrix(h_of_r, data['n_wf'], 'H(R=0)')

    # bands info
    k_path, k_point_labels = _convert_kpath(k_mesh)

    # calculate tight-binding eigenvalues
    if not fermi_slice:
        k_disc, k_points, e_mat = energy_matrix_on_bz_paths(k_path, tb, n_pts=k_mesh['n_k'])
        if add_spin: e_mat = e_mat[2:5,2:5]
        if band_basis:
            e_vecs = np.zeros(e_mat.shape, dtype=complex)
            total_proj = np.zeros(np.shape(e_vecs[0]))
            for ik in range(np.shape(e_mat)[2]):
                evals, e_vecs[:,:,ik] = np.linalg.eigh(e_mat[:,:,ik])
                e_mat[:,:,ik] = np.zeros(e_mat[:,:,ik].shape)
                np.fill_diagonal(e_mat[:,:,ik],evals)
            for band in range(data['n_wf']):
                for orb in projected_orbs:
                    total_proj[band] += np.real(e_vecs[orb,band] * e_vecs[orb,band].conjugate())
            #print(total_proj)
        else:
            e_vecs = np.array([None])
            total_proj = np.array([None])
    else:
        e_mat = np.zeros((n_orb_rescale, n_orb_rescale, k_mesh['n_k'], k_mesh['n_k']), dtype=complex)
        e_vecs = np.array([None])
        total_proj = np.array([None])
        final_x, final_y = k_path[1]
        Z = np.array(k_mesh['Z'])
        for ik_y in range(k_mesh['n_k']):
            path_along_x = [(final_y / (k_mesh['n_k'] - 1) * ik_y + k_mesh['kz'] * Z, final_x + final_y / (k_mesh['n_k'] - 1) * ik_y + k_mesh['kz'] * Z)]
            _, _, e_mat[:,:,:,ik_y] = energy_matrix_on_bz_paths(path_along_x, tb, n_pts=k_mesh['n_k'])
        k_disc = k_points = np.array([0,1])
        if add_spin: e_mat = e_mat[2:5,2:5]

    k_mesh = {'k_disc': k_disc.tolist(), 'k_points': k_points.tolist(), 'k_point_labels': k_point_labels, 'k_points_dash': k_mesh['k_path']}
    
    return k_mesh, e_mat, e_vecs, tb, total_proj

