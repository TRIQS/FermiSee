#!/bin/python3
"""
Reads DMFT_ouput observables such as real-frequency Sigma and a Wannier90
TB Hamiltonian to compute spectral properties. It runs in two modes,
either calculating the bandstructure or Fermi slice.

Written by Sophie Beck, 2021
"""

import numpy as np
from scipy.optimize import brentq, root_scalar
from scipy.interpolate import interp1d
import itertools

# triqs
from triqs.sumk import SumkDiscreteFromLattice
from tools.TB_functions import *
from triqs.gf import Gf, MeshReFreq
from triqs.utility.dichotomy import dichotomy
import tools.tools as tools

upscale = lambda quantity, n_orb: quantity * np.identity(n_orb)


def calc_alatt(tb_data, sigma_data, akw_data, solve=False, band_basis=False):

    # read data
    n_orb = tb_data['n_wf']
    eta = upscale(1j * akw_data['eta'], n_orb)
    w_dict = sigma_data['w_dict']
    w_vec = np.array(w_dict['w_mesh'])[:, None, None] * np.eye(n_orb)
    e_mat = np.array(tb_data['e_mat_re']) + 1j * np.array(tb_data['e_mat_im'])
    n_k = e_mat.shape[2]
    # sigma
    sigma = np.array(
        sigma_data['sigma_re']) + 1j * np.array(sigma_data['sigma_im'])
    sigma_rot = sigma.copy()
    if band_basis:
        e_vecs = np.array(
            tb_data['evecs_re']) + 1j * np.array(tb_data['evecs_im'])

    # TODO add local
    add_local = [0.] * tb_data['n_wf']
    triqs_mesh = MeshReFreq(omega_min=w_dict['window'][0],
                            omega_max=w_dict['window'][1],
                            n_max=w_dict['n_w'])
    Sigma_triqs = Gf(mesh=triqs_mesh, target_shape=[n_orb, n_orb])
    Sigma_triqs.data[:, :, :] = sigma.transpose((2, 0, 1)) 
    if 'sigma_mu_re' in sigma_data.keys():
        sigma_mu = np.array(
            sigma_data['sigma_mu_re']) + 1j * np.array(sigma_data['sigma_mu_im'])
        Sigma_mu_triqs = Gf(mesh=triqs_mesh, target_shape=[n_orb, n_orb])
        Sigma_mu_triqs.data[:, :, :] = sigma_mu.transpose((2, 0, 1))
        new_mu, _ = calc_mu(tb_data,
                         tb_data['n_elect'],
                         tb_data['add_spin'],
                         add_local,
                         current_mu=akw_data['dmft_mu'],
                         Sigma=Sigma_mu_triqs,
                         eta=akw_data['eta'])
    else:
        new_mu, _ = calc_mu(tb_data,
                         tb_data['n_elect'],
                         tb_data['add_spin'],
                         add_local,
                         current_mu=akw_data['dmft_mu'],
                         Sigma=Sigma_triqs,
                         eta=akw_data['eta'])

    mu = upscale(new_mu, n_orb)

    Aw = calc_Aw(tb_data,
                 new_mu,
                 add_spin=tb_data['add_spin'],
                 add_local=add_local,
                 Sigma=Sigma_triqs,
                 eta=akw_data['eta'],
                 n_k=10)

    if not solve:

        def invert_and_trace(w, eta, mu, e_mat, sigma):
            # inversion is automatically vectorized over first axis of 3D array (omega first index now)
            Glatt = np.linalg.inv(w + eta[None, ...] + mu[None, ...] -
                                  e_mat[None, ...] - sigma.transpose(2, 0, 1))
            return -1.0 / np.pi * np.trace(Glatt, axis1=1, axis2=2).imag

        alatt_k_w = np.zeros((n_k, w_dict['n_w']))
        for ik in range(n_k):
            # if evecs are given transform sigma into band basis
            if band_basis:
                sigma_rot = np.einsum(
                    'ij,jkw->ikw', e_vecs[:, :, ik].conjugate().transpose(),
                    np.einsum('ijw,jk->ikw', sigma, e_vecs[:, :, ik]))
            alatt_k_w[ik, :] = invert_and_trace(w_vec, eta, mu,
                                                e_mat[:, :, ik], sigma_rot)

    else:
        alatt_k_w = np.zeros((n_k, n_orb))
        kslice = np.zeros((w_dict['n_w'], n_orb))
        kslice_interp = lambda orb: interp1d(w_dict['w_mesh'], kslice[:, orb])

        for ik in range(n_k):
            for iw, w in enumerate(w_dict['w_mesh']):
                np.fill_diagonal(sigma[:, :, iw],
                                 np.diag(sigma[:, :, iw]).real)
                kslice[iw], _ = np.linalg.eigh(
                    upscale(w, n_orb) + eta + mu - e_mat[:, :, ik] -
                    sigma[:, :, iw])

            for orb in range(n_orb):
                w_min, w_max = w_dict['window']
                try:
                    x0 = brentq(kslice_interp(orb), w_min, w_max)
                    w_bin = int(
                        (x0 - w_min) / ((w_max - w_min) / w_dict['n_w']))
                    alatt_k_w[ik, orb] = w_dict['w_mesh'][w_bin]
                except ValueError:
                    pass

    return alatt_k_w, Aw, new_mu


def calc_kslice(tb_data, sigma_data, akw_data, solve=False, band_basis=False):

    # read data
    n_orb = tb_data['n_wf']
    eta = upscale(1j * akw_data['eta'], n_orb)
    w_dict = sigma_data['w_dict']
    w_vec = np.array(w_dict['w_mesh'])[:, None, None] * np.eye(n_orb)
    e_mat = np.array(tb_data['e_mat_re']) + 1j * np.array(tb_data['e_mat_im'])
    n_kx, n_ky = e_mat.shape[2:4]

    # sigma
    sigma = np.array(
        sigma_data['sigma_re']) + 1j * np.array(sigma_data['sigma_im'])
    sigma_rot = sigma.copy()
    if band_basis:
        e_vecs = np.array(
            tb_data['evecs_re']) + 1j * np.array(tb_data['evecs_im'])
    iw0 = np.where(np.sign(w_dict['w_mesh']) == True)[0][0] - 1
    tools.print_matrix(sigma[:, :, iw0], n_orb, 'Zero-frequency Sigma')

    # TODO add local
    add_local = [0.] * tb_data['n_wf']
    triqs_mesh = MeshReFreq(omega_min=w_dict['window'][0],
                            omega_max=w_dict['window'][1],
                            n_max=w_dict['n_w'])
    Sigma_triqs = Gf(mesh=triqs_mesh, target_shape=[n_orb, n_orb])
    Sigma_triqs.data[:, :, :] = sigma.transpose((2, 0, 1))

    mu = upscale(float(akw_data['dmft_mu']), n_orb)

    if not solve:
        alatt_k_w = np.zeros((n_kx, n_ky))
        invert_and_trace = lambda w, eta, mu, e_mat, sigma: -1.0 / np.pi * np.trace(
            np.linalg.inv(w + eta + mu - e_mat - sigma).imag)

        for ikx, iky in itertools.product(range(n_kx), range(n_ky)):
            alatt_k_w[ikx, iky] = invert_and_trace(
                upscale(w_dict['w_mesh'][iw0], n_orb), eta, mu,
                e_mat[:, :, ikx, iky], sigma[:, :, iw0])
    else:
        assert n_kx == n_ky, 'Not implemented for N_kx != N_ky'
        alatt_k_w = np.zeros((n_kx, n_ky, n_orb))
        for it in range(2):
            kslice = np.zeros((n_kx, n_ky, n_orb))
            if it == 0:
                kslice_interp = lambda ik, orb: interp1d(
                    range(n_kx), kslice[:, ik, orb])
            else:
                kslice_interp = lambda ik, orb: interp1d(
                    range(n_kx), kslice[ik, :, orb])

            for ik1 in range(n_kx):
                e_temp = e_mat[:, :, :, ik1] if it == 0 else e_mat[:, :,
                                                                   ik1, :]
                for ik2 in range(n_kx):
                    e_val, _ = np.linalg.eigh(eta + mu - e_temp[:, :, ik2] -
                                              sigma[:, :, iw0])
                    k1, k2 = [ik2, ik1] if it == 0 else [ik1, ik2]
                    kslice[k1, k2] = e_val

                for orb in range(n_orb):
                    try:
                        x0 = brentq(kslice_interp(ik1, orb), 0, n_kx - 1)
                        k1, k2 = [int(np.floor(x0)), ik1] if it == 0 else [
                            ik1, int(np.floor(x0))
                        ]
                        alatt_k_w[k1, k2, orb] += 1
                    except ValueError:
                        pass

        alatt_k_w[np.where(alatt_k_w > 1)] = 1
        alatt_k_w[np.where(alatt_k_w < 1)] = None
        for ik1 in range(n_ky):
            for orb in range(n_orb):
                try:
                    cross = np.where(alatt_k_w[:, ik1, orb] == 1)[0]
                    for idx in cross:
                        alatt_k_w[idx, ik1, orb] = np.linspace(0, 1,
                                                               n_ky)[ik1 + 1]
                except (IndexError):
                    pass

    return alatt_k_w, mu


def sumk(mu, Sigma, bz_weight, eps_nuk, w_mat, eta=0.0):
    '''
    calc Gloc
    '''
    Gloc = Sigma.copy()
    Gloc << 0.00+0.00j

    mu_mat = mu * np.eye(Gloc.target_shape[0])
    eta_mat = 1j * eta * np.eye(Gloc.target_shape[0])
   
    #check if the eps_nuk is a matrix otherwise it must be converted to a
    #diagonal matrix
    if len(eps_nuk.shape) == 2:
        #Loop on k points
        for eps_nu in eps_nuk:
            Gloc.data[:, :, :] += bz_weight * np.linalg.inv(w_mat[:] + mu_mat - np.diag(eps_nu) - Sigma.data[:, :, :] + eta_mat)
    else:
        for eps_nu in eps_nuk:
            Gloc.data[:, :, :] += bz_weight * np.linalg.inv(w_mat[:] + mu_mat - eps_nu - Sigma.data[:, :, :] + eta_mat)
    return Gloc


def calc_mu(tb_data,
            n_elect,
            add_spin,
            add_local,
            current_mu = 0.0,
            Sigma=None,
            eta=0.0,
            w_spacing=0.005,
            n_k=10):
    """
    This function determines the chemical potential based on tb_data, an optional sigma and a number of electrons.
    """
    def dens(mu,n_elect):
        # 2 times for spin degeneracy

        dens = sp_factor * sumk(mu=mu,
                                Sigma=Sigma,
                                bz_weight=bz_weight,
                                eps_nuk=eps_nuk,
                                w_mat=w_mat,
                                eta=eta).total_density()
        print(f"dens: {dens.real:.4f} n_elect: {n_elect:.2f}  mu: {mu:.4f}")
        return dens.real-n_elect

    # set up Wannier Hamiltonian
    n_orb_rescale = 2 * tb_data['n_wf'] if add_spin else tb_data['n_wf']
    sp_factor = 1 if add_spin else 2
    H_add_loc = np.zeros((n_orb_rescale, n_orb_rescale), dtype=complex)
    if add_spin: H_add_loc += tools.lambda_matrix_w90_t2g(add_local)

    hopping = {
        eval(key): np.array(value, dtype=complex)
        for key, value in tb_data['hopping'].items()
    }
    tb = tools.get_TBL(hopping,
                       tb_data['units'],
                       tb_data['n_wf'],
                       extend_to_spin=add_spin,
                       add_local=H_add_loc)

    k_spacing = np.linspace(0, 1, n_k, endpoint=False)
    k_array = np.array(np.meshgrid(k_spacing, k_spacing, k_spacing)).T.reshape(-1, 3)
    bz_weight = 1/(n_k**3)
    if not Sigma:
        eps_nuk = tb.dispersion(k_array)
        eps_max = np.max(eps_nuk)
        eps_min = np.min(eps_nuk)
        bandwidth = np.abs(eps_max - eps_min)
        n_w = int((bandwidth+0.6)/w_spacing)
        Sigma = Gf(mesh=MeshReFreq(window=[-bandwidth-0.5, 0.1], n_w=n_w),
                   target_shape=[tb_data['n_wf'], tb_data['n_wf']])
    else:
        eps_min, eps_max = tb_data['eps_min_max']
        eps_nuk = tb.fourier(k_array)
        #TODO: there could be an edge case that is beyond the boundaries
        sigma_hartree = Sigma(0.0).real
        sigma_eig, _ = np.linalg.eigh(sigma_hartree)
        if sigma_eig[-1] > 0: eps_max+=sigma_eig[-1]
        if sigma_eig[0] <  0: eps_min+=sigma_eig[0]

    w_mat = np.array([w.value * np.eye(tb_data['n_wf']) for w in Sigma.mesh])
    
    #Check if stored mu is the correct mu to avoid recalculation
    #if mu is correct, dens-n_elect == 0
    if np.isclose(0.0,dens(current_mu, n_elect),atol=1e-3):
        return current_mu, (eps_min, eps_max)
    mu = brentq(dens,eps_max,eps_min,(n_elect),xtol=1e-4)
    return mu, (eps_min, eps_max)


def calc_Aw(tb_data, mu, add_spin, add_local, Sigma=None, eta=0.0, n_k=10):

    n_orb_rescale = 2 * tb_data['n_wf'] if add_spin else tb_data['n_wf']
    sp_factor = 1 if add_spin else 2
    H_add_loc = np.zeros((n_orb_rescale, n_orb_rescale), dtype=complex)
    if add_spin: H_add_loc += tools.lambda_matrix_w90_t2g(add_local)

    if not Sigma:
        w_min = tb_data['bnd_low'] - 0.2*abs(tb_data['bnd_low'])
        w_max = tb_data['bnd_high'] + 0.2*abs(tb_data['bnd_high'])
        w_spacing = 0.005
        n_w = int(np.abs(w_max-w_min)/w_spacing)
        Sigma = Gf(mesh=MeshReFreq(window=[w_min, w_max], n_max=n_w),
                   target_shape=[n_orb_rescale, n_orb_rescale])

    hopping = {
        eval(key): np.array(value, dtype=complex)
        for key, value in tb_data['hopping'].items()
    }
    tb = tools.get_TBL(hopping,
                       tb_data['units'],
                       tb_data['n_wf'],
                       extend_to_spin=add_spin,
                       add_local=H_add_loc)

    k_spacing = np.linspace(0, 1, n_k, endpoint=False)
    k_array = np.array(np.meshgrid(k_spacing, k_spacing, k_spacing)).T.reshape(-1, 3)
    bz_weight = 1/(n_k**3)
    eps_nuk = tb.dispersion(k_array)
    w_mat = np.array([w.value * np.eye(tb_data['n_wf']) for w in Sigma.mesh])
    
    Gloc = sp_factor * sumk(mu=mu,
                            Sigma=Sigma,
                            bz_weight=bz_weight,
                            eps_nuk=eps_nuk,
                            w_mat=w_mat,
                            eta=eta)

    Aw = -1.0 / np.pi * np.trace(Gloc.data, axis1=1, axis2=2).imag

    return Aw

def sigma_from_dmft(n_orb,
                    orbital_order,
                    sigma,
                    spin,
                    block,
                    dc,
                    w_dict,
                    linearize=False):
    """
    Takes a sigma obtained from DMFT and interpolates on a given mesh
    """

    block_spin = spin + '_' + str(block)  # if with_sigma == 'calc' else spin
    SOC = True if spin == 'ud' else False
    w_mesh_dmft = [x.real for x in sigma[block_spin].mesh]
    w_mesh = w_dict['w_mesh']
    sigma_mat = {
        block_spin:
        sigma[block_spin].data.real - np.eye(n_orb) * dc +
        1j * sigma[block_spin].data.imag
    }

    # rotate sigma from orbital_order_dmft to orbital_order, where 0,1,2 is the basis given by the Wannier Ham
    change_of_basis = tools.change_basis(n_orb, orbital_order,
                                         tuple(range(n_orb)))
    sigma_mat[block_spin] = np.einsum(
        'ij, kjl -> kil', np.linalg.inv(change_of_basis),
        np.einsum('ijk, kl -> ijl', sigma_mat[block_spin], change_of_basis))

    sigma_interpolated = np.zeros((n_orb, n_orb, w_dict['n_w']), dtype=complex)

    if linearize:
        print('Linearizing Sigma at zero frequency:')
        eta = eta * 1j
        iw0 = np.where(np.sign(w_mesh_dmft) == True)[0][0] - 1
        if SOC:
            sigma_interpolated += np.expand_dims(
                sigma_mat[block_spin][iw0, :, :], axis=-1)
        # linearize diagonal elements of sigma
        for ct in range(n_orb):
            _, _, fit_params = _linefit(w_mesh_dmft,
                                        sigma_mat[block_spin][:, ct, ct],
                                        specs['linearize']['window'])
            zeroth_order, first_order = fit_params[::-1].real
            print('Zeroth and first order fit parameters: [{0:.4f}, {1:.4f}]'.
                  format(zeroth_order, first_order))
            sigma_interpolated[
                ct, ct] = zeroth_order + w_dict['w_mesh'] * first_order

    else:
        eta = 0 * 1j
        # interpolate sigma
        interpolate_sigma = lambda w_mesh, w_mesh_dmft, orb1, orb2: np.interp(
            w_mesh, w_mesh_dmft, sigma_mat[block_spin][:, orb1, orb2])

        for ct1, ct2 in itertools.product(range(n_orb), range(n_orb)):
            if ct1 != ct2 and not SOC: continue
            sigma_interpolated[ct1, ct2] = interpolate_sigma(
                w_mesh, w_mesh_dmft, ct1, ct2)

    return sigma_interpolated
