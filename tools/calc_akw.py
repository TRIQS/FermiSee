#!/bin/python3

"""
Reads DMFT_ouput observables such as real-frequency Sigma and a Wannier90
TB Hamiltonian to compute spectral properties. It runs in two modes,
either calculating the bandstructure or Fermi slice.

Written by Sophie Beck, 2021
"""

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
from tools.TB_functions import *
from h5 import HDFArchive
from triqs.gf import BlockGf
from triqs.gf import GfReFreq, MeshReFreq
from triqs.utility.dichotomy import dichotomy
import tools.tools as tools

upscale = lambda quantity, n_orb: quantity * np.identity(n_orb)

def calc_alatt(tb_data, sigma_data, akw_data, solve=False, band_basis=False):

    # read data
    n_orb = tb_data['n_wf']
    eta = upscale(1j * akw_data['eta'], n_orb)
    w_dict = sigma_data['w_dict']
    w_vec = np.array(w_dict['w_mesh'])[:,None,None] * np.eye(n_orb)
    e_mat = np.array(tb_data['e_mat_re']) + 1j * np.array(tb_data['e_mat_im'])
    n_k = e_mat.shape[2]
    # sigma
    sigma = np.array(sigma_data['sigma_re']) + 1j * np.array(sigma_data['sigma_im'])
    sigma_rot = sigma.copy()
    if band_basis:
        e_vecs = np.array(tb_data['evecs_re']) + 1j * np.array(tb_data['evecs_im'])

    # TODO add local
    add_local = [0.] * tb_data['n_wf']
    triqs_mesh = MeshReFreq(omega_min=w_dict['window'][0], omega_max=w_dict['window'][1],n_max=w_dict['n_w'])
    Sigma_triqs = GfReFreq(mesh=triqs_mesh , target_shape = [n_orb,n_orb])
    Sigma_triqs.data[:,:,:] = sigma.transpose((2,0,1))

    new_mu = calc_mu(tb_data, tb_data['n_elect'],  tb_data['add_spin'], add_local,
                     mu_guess= akw_data['dmft_mu'], Sigma=Sigma_triqs, eta=akw_data['eta'])

    mu = upscale(new_mu, n_orb)

    if not solve:

        def invert_and_trace(w, eta, mu, e_mat, sigma):
            # inversion is automatically vectorized over first axis of 3D array (omega first index now)
            Glatt =  np.linalg.inv(w + eta[None,...] + mu[None,...] - e_mat[None,...] - sigma.transpose(2,0,1) )
            return -1.0/np.pi * np.trace( Glatt, axis1=1, axis2=2).imag

        alatt_k_w = np.zeros((n_k, w_dict['n_w']))
        for ik in range(n_k):
            # if evecs are given transform sigma into band basis
            if band_basis:
                sigma_rot = np.einsum('ij,jkw->ikw',
                                      e_vecs[:,:,ik].conjugate().transpose(),
                                      np.einsum('ijw,jk->ikw', sigma, e_vecs[:,:,ik]))
            alatt_k_w[ik, :] = invert_and_trace(w_vec, eta, mu, e_mat[:,:,ik], sigma_rot)
            
    else:
        alatt_k_w = np.zeros((n_k, n_orb))
        kslice = np.zeros((w_dict['n_w'], n_orb))
        kslice_interp = lambda orb: interp1d(w_dict['w_mesh'], kslice[:, orb])

        for ik in range(n_k):
            for iw, w in enumerate(w_dict['w_mesh']):
                np.fill_diagonal(sigma[:,:,iw], np.diag(sigma[:,:,iw]).real)
                kslice[iw], _ = np.linalg.eigh( upscale(w, n_orb) + eta + mu - e_mat[:,:,ik] - sigma[:,:,iw])
                
            for orb in range(n_orb):
                w_min, w_max = w_dict['window']
                try:
                    x0 = brentq( kslice_interp(orb), w_min, w_max)
                    w_bin = int( (x0 - w_min) / ((w_max - w_min)/ w_dict['n_w']) )
                    alatt_k_w[ik, orb] = w_dict['w_mesh'][w_bin]
                except ValueError:
                    pass

    return alatt_k_w, new_mu

def calc_kslice(tb_data, sigma_data, akw_data, solve=False, band_basis=False):

    # read data
    n_orb = tb_data['n_wf']
    eta = upscale(1j * akw_data['eta'], n_orb)
    w_dict = sigma_data['w_dict']
    w_vec = np.array(w_dict['w_mesh'])[:,None,None] * np.eye(n_orb)
    e_mat = np.array(tb_data['e_mat_re']) + 1j * np.array(tb_data['e_mat_im'])
    n_kx, n_ky = e_mat.shape[2:4]

    # sigma
    sigma = np.array(sigma_data['sigma_re']) + 1j * np.array(sigma_data['sigma_im'])
    sigma_rot = sigma.copy()
    if band_basis:
        e_vecs = np.array(tb_data['evecs_re']) + 1j * np.array(tb_data['evecs_im'])
    iw0 = np.where(np.sign(w_dict['w_mesh']) == True)[0][0]-1
    tools.print_matrix(sigma[:,:,iw0], n_orb, 'Zero-frequency Sigma')

    # TODO add local
    add_local = [0.] * tb_data['n_wf']
    triqs_mesh = MeshReFreq(omega_min=w_dict['window'][0], omega_max=w_dict['window'][1],n_max=w_dict['n_w'])
    Sigma_triqs = GfReFreq(mesh=triqs_mesh , target_shape = [n_orb,n_orb])
    Sigma_triqs.data[:,:,:] = sigma.transpose((2,0,1))

    mu = upscale(float(akw_data['dmft_mu']), n_orb)

    if not solve:
        alatt_k_w = np.zeros((n_kx, n_ky))
        invert_and_trace = lambda w, eta, mu, e_mat, sigma: -1.0/np.pi * np.trace( np.linalg.inv( w + eta + mu - e_mat - sigma ).imag )

        for ikx, iky in itertools.product(range(n_kx), range(n_ky)):
            alatt_k_w[ikx, iky] = invert_and_trace(upscale(w_dict['w_mesh'][iw0], n_orb), eta, mu, e_mat[:,:,ikx,iky], sigma[:,:,iw0])
    else:
        assert n_kx == n_ky, 'Not implemented for N_kx != N_ky'
        alatt_k_w = np.zeros((n_kx, n_ky, n_orb))
        for it in range(2):
            kslice = np.zeros((n_kx, n_ky, n_orb))
            if it == 0:
                kslice_interp = lambda ik, orb: interp1d(range(n_kx), kslice[:, ik, orb])
            else:
                kslice_interp = lambda ik, orb: interp1d(range(n_kx), kslice[ik, :, orb])

            for ik1 in range(n_kx):
                e_temp = e_mat[:,:,:,ik1] if it == 0 else e_mat[:,:,ik1,:]
                for ik2 in range(n_kx):
                    e_val, _ = np.linalg.eigh( eta + mu - e_temp[:,:,ik2] - sigma[:,:,iw0])
                    k1, k2 = [ik2, ik1] if it == 0 else [ik1, ik2]
                    kslice[k1, k2] = e_val

                for orb in range(n_orb):
                    try:
                        x0 = brentq( kslice_interp(ik1, orb), 0, n_kx - 1)
                        k1, k2 = [int(np.floor(x0)), ik1] if it == 0 else [ik1, int(np.floor(x0))]
                        alatt_k_w[k1, k2, orb] += 1
                    except ValueError:
                        pass

        alatt_k_w[np.where(alatt_k_w > 1)] = 1
        alatt_k_w[np.where(alatt_k_w < 1)] = None
        for ik1 in range(n_ky):
            for orb in range(n_orb):
                try:
                    cross = np.where(alatt_k_w[:,ik1,orb] == 1)[0]
                    for idx in cross:
                        alatt_k_w[idx, ik1, orb] = np.linspace(0, 1, n_ky)[ik1 + 1]
                except(IndexError):
                    pass

    return alatt_k_w, mu

def sumk(mu, Sigma, bz_weights, hopping, eta=0.0):
    '''
    calc Gloc
    '''
    Gloc = Sigma.copy()
    Gloc << 0.0+0.0j

    n_orb = Gloc.target_shape[0]

    w_mat = np.array([w.value * np.eye(n_orb) for w in Gloc.mesh])
    mu_mat = mu * np.eye(n_orb)
    eta_mat = 1j*eta * np.eye(n_orb)

    #Loop on k points
    for wk, eps_k in zip(bz_weights, hopping):
        Gloc.data[:,:,:] += wk*np.linalg.inv(w_mat[:] + mu_mat - eps_k - Sigma.data[:,:,:] + eta_mat)

    return Gloc

def calc_mu(tb_data, n_elect, add_spin, add_local, mu_guess= 0.0, Sigma=None, eta=0.0):
    """
    This function determines the chemical potential based on tb_data, an optional sigma and a number of electrons.
    """

    def dens(mu):
        # 2 times for spin degeneracy
        sp_factor = 1 if add_spin else 2
        dens = sp_factor*sumk(mu = mu, Sigma = Sigma, bz_weights=SK.bz_weights, hopping=SK.hopping, eta=eta).total_density()
        return dens.real

    # set up Wannier Hamiltonian
    n_k = 10
    n_orb_rescale = 2 * tb_data['n_wf'] if add_spin else tb_data['n_wf']
    H_add_loc = np.zeros((n_orb_rescale, n_orb_rescale), dtype=complex)
    if add_spin: H_add_loc += tools.lambda_matrix_w90_t2g(add_local)

    if not Sigma:
        Sigma = GfReFreq(mesh=MeshReFreq(omega_min=-5, omega_max=1, n_max=1001) , target_shape = [tb_data['n_wf'],tb_data['n_wf']])

    hopping = {eval(key): np.array(value, dtype=complex) for key, value in tb_data['hopping'].items()}
    tb = tools.get_TBL(hopping, tb_data['units'], tb_data['n_wf'], extend_to_spin=add_spin, add_local=H_add_loc)

    SK = SumkDiscreteFromLattice(lattice=tb, n_points=n_k)

    mu, density = dichotomy(dens, mu_guess, n_elect, 1e-3, 0.5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)

    return mu

def sigma_from_dmft(n_orb, orbital_order, sigma, spin, block, dc, w_dict, linearize= False):
    """
    Takes a sigma obtained from DMFT and interpolates on a given mesh
    """

    block_spin = spin + '_' + str(block) # if with_sigma == 'calc' else spin
    SOC = True if spin == 'ud' else False
    w_mesh_dmft = [x.real for x in sigma[block_spin].mesh]
    w_mesh = w_dict['w_mesh']
    sigma_mat = {block_spin: sigma[block_spin].data.real - np.eye(n_orb) * dc + 1j * sigma[block_spin].data.imag}

    # rotate sigma from orbital_order_dmft to orbital_order, where 0,1,2 is the basis given by the Wannier Ham
    change_of_basis = tools.change_basis(n_orb, orbital_order,  tuple(range(n_orb)))
    sigma_mat[block_spin] = np.einsum('ij, kjl -> kil', np.linalg.inv(change_of_basis), np.einsum('ijk, kl -> ijl', sigma_mat[block_spin], change_of_basis))

    sigma_interpolated = np.zeros((n_orb, n_orb, w_dict['n_w']), dtype=complex)

    if linearize:
        print('Linearizing Sigma at zero frequency:')
        eta = eta * 1j
        iw0 = np.where(np.sign(w_mesh_dmft) == True)[0][0]-1
        if SOC: sigma_interpolated += np.expand_dims(sigma_mat[block_spin][iw0,:,:], axis=-1)
        # linearize diagonal elements of sigma
        for ct in range(n_orb):
            _, _, fit_params = _linefit(w_mesh_dmft, sigma_mat[block_spin][:,ct,ct], specs['linearize']['window'])
            zeroth_order, first_order = fit_params[::-1].real
            print('Zeroth and first order fit parameters: [{0:.4f}, {1:.4f}]'.format(zeroth_order,first_order))
            sigma_interpolated[ct,ct] = zeroth_order + w_dict['w_mesh'] * first_order

    else:
        eta = 0 * 1j
        # interpolate sigma
        interpolate_sigma = lambda w_mesh, w_mesh_dmft, orb1, orb2: np.interp(w_mesh, w_mesh_dmft, sigma_mat[block_spin][:, orb1, orb2])

        for ct1, ct2 in itertools.product(range(n_orb), range(n_orb)):
            if ct1 != ct2 and not SOC: continue
            sigma_interpolated[ct1,ct2] = interpolate_sigma(w_mesh, w_mesh_dmft, ct1, ct2)

    return sigma_interpolated
