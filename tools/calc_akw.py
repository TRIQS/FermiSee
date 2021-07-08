#!/bin/python3

"""
Reads DMFT_ouput observables such as real-frequency Sigma and a Wannier90
TB Hamiltonian to compute spectral properties. It runs in two modes,
either calculating the bandstructure or Fermi slice.

Written by Sophie Beck, 2021
"""

from tools.TB_functions import *
from h5 import HDFArchive
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib import cm, colors
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import itertools
from triqs.gf import BlockGf
import matplotlib.pyplot as plt

def _linefit(x, y, interval, spacing=50, addspace=0.0):

    calc_Z = lambda slope: 1/(1-slope)
    
    x = np.array(x)
    lim_l, lim_r = interval
    indices = np.where(np.logical_and(x>=lim_l, x<=lim_r))
    fit = np.polyfit(x[indices], y[indices], 1)
    slope = fit[1]
    Z = calc_Z(slope)
    f_x = np.poly1d(fit)
    x_cont = np.linspace(x[indices][0] - addspace, x[indices][-1] + addspace, spacing)
    
    return x_cont, f_x(x_cont), fit

def _lambda_matrix_w90_t2g(add_lambda):
    """
    Add local SOC term to H(R) for t2g shell
    """

    lambda_x, lambda_y, lambda_z = add_lambda

    lambda_matrix = np.zeros((6,6), dtype=complex)
    lambda_matrix[0,1] = -1j*lambda_z/2.0
    lambda_matrix[0,5] =  1j*lambda_x/2.0
    lambda_matrix[1,5] =    -lambda_y/2.0
    lambda_matrix[2,3] = -1j*lambda_x/2.0
    lambda_matrix[2,4] =     lambda_y/2.0
    lambda_matrix[3,4] =  1j*lambda_z/2.0
    lambda_matrix += np.transpose(np.conjugate(lambda_matrix))

    return lambda_matrix

def _change_basis(n_orb, orbital_order_to, orbital_order_from):
    """
    Rotation between orbital basis
    """
    
    change_of_basis = np.eye(n_orb)
    for ct, orb in enumerate(orbital_order_to):
        orb_idx = orbital_order_from.index(orb)
        change_of_basis[orb_idx,:] = np.roll(np.eye(n_orb,1),ct)[:,0]

    return change_of_basis

def _print_matrix(matrix, n_orb, text):
    """
    Pre-determined print command for matrices
    """

    print('{}:'.format(text))
    fmt = '{:16.4f}' * n_orb
    for row in matrix:
        print((' '*4 + fmt).format(*row))

def sigma_from_dmft(n_orb, orbital_order, sigma, spin, block, orbital_order_dmft, dc, w_dict, linearize= False):
    """
    Takes a sigma obtained from DMFT and interpolates on a given mesh
    """

    block_spin = spin + '_' + str(block) # if with_sigma == 'calc' else spin
    SOC = True if spin == 'ud' else False
    w_mesh_dmft = [x.real for x in sigma[block_spin].mesh]
    w_mesh = w_dict['w_mesh']
    sigma_mat = {block_spin: sigma[block_spin].data.real - np.eye(n_orb) * dc + 1j * sigma[block_spin].data.imag}

    # rotate sigma from orbital_order_dmft to orbital_order
    change_of_basis = _change_basis(n_orb, orbital_order, orbital_order_dmft)
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

def _sigma_from_model(n_orb, orbital_order, zeroth_order, first_order, efermi, eta=0.0, **w):
    
    print('Setting model Sigma')

    mu = dft_mu = efermi
    eta = eta * 1j
    
    # set up mesh
    w_dict = w['w_mesh']
    w_mesh = np.linspace(*w_dict['window'], w_dict['n_w'])
    w_dict.update({'w_mesh': w_mesh})

    # interpolate sigma
    sigma_interpolated = np.zeros((n_orb, n_orb, w_dict['n_w']), dtype=complex)
    approximate_sigma = lambda zeroth_order, first_order, orb: zeroth_order[orb] + w_dict['w_mesh'] * first_order[orb]
    for ct, orb in enumerate(orbital_order):
        sigma_interpolated[ct,ct] = approximate_sigma(zeroth_order, first_order, ct)
    
    return sigma_interpolated, mu, dft_mu, eta, w_dict

def calc_alatt(tb_data, sigma_data, akw_data, solve=False):

    # adjust to system size
    upscale = lambda quantity, n_orb: quantity * np.identity(n_orb)
    n_orb = tb_data['n_wf']
    mu = upscale(akw_data['dmft_mu'], n_orb)
    eta = upscale(1j * akw_data['eta'], n_orb)
    sigma = np.array(sigma_data['sigma_re']) + 1j * np.array(sigma_data['sigma_im'])
    e_mat = np.array(tb_data['e_mat'])
    w_dict = sigma_data['w_dict']

    n_k = e_mat.shape[2]
    
    if not solve:
        alatt_k_w = np.zeros((n_k, w_dict['n_w']))
        invert_and_trace = lambda w, eta, mu, e_mat, sigma: -1.0/np.pi * np.trace( np.linalg.inv( w + eta + mu - e_mat - sigma ).imag )

        for iw, ik in itertools.product(range(w_dict['n_w']), range(n_k)):
            alatt_k_w[ik, iw] = invert_and_trace(upscale(w_dict['w_mesh'][iw], n_orb), eta, mu, e_mat[:,:,ik], sigma[:,:,iw])
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

    return alatt_k_w

def _calc_kslice(n_orb, mu, eta, e_mat, sigma, solve, **w_dict):

    # adjust to system size
    upscale = lambda quantity, n_orb: quantity * np.identity(n_orb)
    mu = upscale(mu, n_orb)
    eta = upscale(eta, n_orb)

    iw0 = np.where(np.sign(w_dict['w_mesh']) == True)[0][0]-1
    _print_matrix(sigma[:,:,iw0], n_orb, 'Zero-frequency Sigma')

    n_kx, n_ky = e_mat.shape[2:4]

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

    return alatt_k_w

def get_tb_bands(e_mat):
    """
    Compute band eigenvalues and eigenvectors from matrix per k-point
    """

    e_val = np.zeros((e_mat.shape[0], e_mat.shape[2]), dtype=complex)
    e_vec = np.zeros(np.shape(e_mat), dtype=complex)
    for ik in range(np.shape(e_mat)[2]):
        e_val[:,ik], e_vec[:,:,ik] = np.linalg.eigh(e_mat[:,:,ik])

    return e_val.real, e_vec

def _get_tb_kslice(tb, dft_mu, **specs):

    prim_to_cart = [[0,1,1],
                    [1,0,1],
                    [1,1,0]]
    cart_to_prim = np.linalg.inv(prim_to_cart)
    w90_paths = list(map(lambda section: (np.array(specs[section[0]]), np.array(specs[section[1]])), specs['bands_path']))
    final_x, final_y = w90_paths[1]
    Z = np.array(specs['Z'])

    e_val, e_vec = get_kx_ky_FS(final_x, final_y, Z, tb, k_trans_back=cart_to_prim, N_kxy=specs['n_k'], kz=specs['kz'], fermi=dft_mu)

    return e_val, e_vec

def plot_bands(fig, ax, alatt_k_w, tb_data, w_dict, n_orb, dft_mu, tb=True, alatt=False, **plot_dict):
    
    if alatt:
        if alatt_k_w is None: raise ValueError('A(k,w) unknown. Specify "with_sigma = True"')
        kw_x, kw_y = np.meshgrid(tb_data['k_mesh'], w_dict['w_mesh'])
        if len(alatt_k_w.shape) > 2:
            for orb in range(n_orb):
                ax.contour(kw_x, kw_y, alatt_k_w[:,:,orb].T, colors=np.array([eval('cm.'+plot_dict['colorscheme_solve'])(0.7)]), levels=1, zorder=2.)
        else:
            graph = ax.pcolormesh(kw_x, kw_y, alatt_k_w.T, cmap=plot_dict['colorscheme_bands'], norm=LogNorm(vmin=plot_dict['vmin'], vmax=np.max(alatt_k_w)))
            colorbar = plt.colorbar(graph)
            colorbar.set_label(r'$A(k, \omega)$')

    if tb:
        #eps_nuk, evec_nuk = _get_tb_bands(**tb_data)
        for band in range(n_orb):
            orbital_projected = [0] if not plot_dict['proj_on_orb'] else np.real(evec_nuk[plot_dict['proj_on_orb'], band] * evec_nuk[plot_dict['proj_on_orb'], band].conjugate())
            ax.scatter(tb_data['k_mesh'], eps_nuk[band].real - dft_mu, c=cm.plasma(orbital_projected), s=1, label=r'tight-binding', zorder=1.)

    _setup_plot_bands(ax, tb_data['k_points'], tb_data['k_points_labels'], w_dict)
    
def plot_kslice(fig, ax, alatt_k_w, tb_data, w_dict, n_orb, tb_dict, tb=True, alatt=False, quarter=0, dft_mu=0.0, **plot_dict):
    
    sign = [1,-1]
    quarters = np.array([sign,sign])
    
    if alatt:
        if alatt_k_w is None: raise ValueError('A(k,w) unknown. Specify "with_sigma = True"')
        n_kx, n_ky = tb_data['e_mat'].shape[2:4]
        kx, ky = np.meshgrid(range(n_kx), range(n_ky))
        for qrt in list(itertools.product(*quarters))[quarter:quarter+1]:
            if len(alatt_k_w.shape) > 2:
                for orb in range(n_orb):
                    ax.contour(qrt[0] * kx/(n_kx-1), qrt[1] * ky/(n_ky-1), alatt_k_w[:,:,orb].T, colors=np.array([eval('cm.'+plot_dict['colorscheme_solve'])(0.7)]), levels=1, zorder=2)
            else:
                graph = ax.pcolormesh(qrt[0] * kx/(n_kx-1), qrt[1] * ky/(n_ky-1), alatt_k_w.T, cmap=plot_dict['colorscheme_kslice'],
                                      norm=LogNorm(vmin=plot_dict['vmin'], vmax=np.max(alatt_k_w)))
                #colorbar = plt.colorbar(graph)
                #colorbar.set_label(r'$A(k, 0$)')

    if tb:
        quarters *= 2
        eps_nuk, evec_nuk = _get_tb_kslice(tb_data['tb'], dft_mu, **tb_dict)
        for qrt in list(itertools.product(*quarters))[quarter:quarter+1]:
            for band in range(len(eps_nuk)):
                for segment in range(eps_nuk[band].shape[0]):
                    #orbital_projected = evec_nuk[band][segment][plot_dict['proj_on_orb']]
                    ax.plot(qrt[0] * eps_nuk[band][segment:segment+2,0], qrt[1] * eps_nuk[band][segment:segment+2,1], '-',
                            solid_capstyle='round', c=eval('cm.'+plot_dict['colorscheme_kslice'])(1.0), lw=1., zorder=1.)

    _setup_plot_kslice(ax, tb_data['k_points'], w_dict)

    return ax

def _get_TBL(hopping, units, n_wf, extend_to_spin=False, add_local=None, add_field=None, renormalize=None):
    """
    get triqs tight-binding object from hoppings + units
    """

    if extend_to_spin:
    	hopping, n_wf = extend_wannier90_to_spin(hopping, n_wf)
    if add_local is not None:
        hopping[(0,0,0)] += add_local
    if renormalize is not None:
        assert len(np.shape(renormalize)) == 1, 'Give Z as a vector'
        assert len(renormalize) == n_wf, 'Give Z as a vector of size n_orb (times two if SOC)'
        
        Z_mat = np.diag(np.sqrt(renormalize))
        for R in hopping:
            hopping[R] = np.dot(np.dot(Z_mat, hopping[R]), Z_mat)

    if add_field is not None:
        hopping[(0,0,0)] += add_field

    TBL = TBLattice(units = units, hopping = hopping, orbital_positions = [(0,0,0)]*n_wf,
                    orbital_names = [str(i) for i in range(n_wf)])
    return TBL

def calc_tb_bands(data, add_spin, mu, add_local, orbital_order, k_mesh, fermi_slice):
    """
    calculate tight-binding bands based on a W90 Hamiltonian 
    """

    # set up Wannier Hamiltonian
    n_orb_rescale = 2 * data['n_wf'] if add_spin else data['n_wf']
    change_of_basis = _change_basis(data['n_wf'], orbital_order, orbital_order)
    H_add_loc = np.zeros((n_orb_rescale, n_orb_rescale), dtype=complex)
    H_add_loc += np.diag([-mu]*n_orb_rescale)
    if add_spin: H_add_loc += _lambda_matrix_w90_t2g(add_local)

    hopping = {eval(key): np.array(value, dtype=complex) for key, value in data['hopping'].items()}
    tb = _get_TBL(hopping, data['units'], data['n_wf'], extend_to_spin=add_spin, add_local=H_add_loc)
    # print local H(R)
    h_of_r = tb.hopping_dict()[(0,0,0)][2:5,2:5] if add_spin else tb.hopping_dict()[(0,0,0)]
    h_of_r = np.einsum('ij, jk -> ik', np.linalg.inv(change_of_basis), np.einsum('ij, jk -> ik', h_of_r, change_of_basis))
    _print_matrix(h_of_r, data['n_wf'], 'H(R=0)')

    # bands info
    k_path = k_mesh['k_path']
    k_path = [list(map(lambda item: (k[item]), k.keys())) for k in k_path] # turn dict into list
    k_point_labels = [k.pop(0) for k in k_path] # remove first time, which is label
    k_path = [(np.array(k), np.array(k_path[ct+1])) for ct, k in enumerate(k_path) if ct+1 < len(k_path)] # turn into tuples
    # this is no longer necessary since we fixed this in the energy_matrix_on_bz_paths function
    # if not fermi_slice: k_path.append((k_path[-1][-1], k_path[-1][-1])) # add last k-point

    # calculate tight-binding eigenvalues
    if not fermi_slice:
        k_disc, k_points, e_mat = energy_matrix_on_bz_paths(k_path, tb, n_pts=k_mesh['n_k'])
        if add_spin: e_mat = e_mat[2:5,2:5]
        e_mat = np.einsum('ij, jkl -> ikl', np.linalg.inv(change_of_basis), np.einsum('ijk, jm -> imk', e_mat, change_of_basis))
    else:
        e_mat = np.zeros((n_orb_rescale, n_orb_rescale, k_mesh['n_k'], k_mesh['n_k']), dtype=complex)
        final_x, final_y = k_path[1]
        Z = np.array(k_mesh['Z'])
        for ik_y in range(k_mesh['n_k']):
            path_along_x = [(final_y / (k_mesh['n_k'] - 1) * ik_y + k_mesh['kz'] * Z, final_x + final_y / (k_mesh['n_k'] - 1) * ik_y + k_mesh['kz'] * Z)]
            _, _, e_mat[:,:,:,ik_y] = energy_matrix_on_bz_paths(path_along_x, tb, n_pts=k_mesh['n_k'])
        k_array = k_points = [0,1]
        if add_spin: e_mat = e_mat[2:5,2:5]
        e_mat = np.einsum('ij, jklm -> iklm', np.linalg.inv(change_of_basis), np.einsum('ijkl, jm -> imkl', e_mat, change_of_basis))

    k_mesh = {'k_disc': k_disc.tolist(), 'k_points': k_points.tolist(), 'k_point_labels': k_point_labels, 'k_points_dash': k_mesh['k_path']}
    return k_mesh, e_mat, tb

def get_dmft_bands(n_orb, with_sigma=False, fermi_slice=False, solve=False, orbital_order=['dxz', 'dyz', 'dxy'], **specs):
    
    # dmft output
    if with_sigma:
        sigma_types = ['calc', 'model']
        if isinstance(with_sigma, str):
            if with_sigma not in sigma_types: raise ValueError('Invalid sigma type. Expected one of: {}'.format(sigma_types))
        elif not isinstance(with_sigma, BlockGf):
            raise ValueError('Invalid sigma type. Expected BlockGf.')

        # get sigma
        if with_sigma == 'model': delta_sigma, mu, dft_mu, eta, w_dict = _sigma_from_model(n_orb, orbital_order, **specs)
        # else is from dmft or memory:
        else: delta_sigma, mu, dft_mu, eta, w_dict = sigma_from_dmft(n_orb, orbital_order, with_sigma, **specs)
        
        # calculate alatt
        if not fermi_slice:
            alatt_k_w = _calc_alatt(n_orb, mu, eta, e_mat, delta_sigma, solve, **w_dict)
        else:
            alatt_k_w = _calc_kslice(n_orb, mu, eta, e_mat, delta_sigma, solve, **w_dict)       
    else:
        dft_mu = mu
        w_dict = {}
        w_dict['w_mesh'] = None
        w_dict['window'] = None
        alatt_k_w = None

    
    return {'k_mesh': k_array, 'k_points': k_points, 'k_points_labels': k_points_labels, 'e_mat': e_mat, 'tb': tb}, alatt_k_w, w_dict, dft_mu

