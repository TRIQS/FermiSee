import numpy as np
from itertools import product
from triqs.gf import GfReFreq

import tools.tools as tools

def sigma_analytic_to_data(sigma, w_dict, n_orb):

    w_dict['w_mesh'] = [w.value for w in w_dict['w_mesh']]

    temp_sigma_data = {}
    temp_sigma_data['sigma_re'] = sigma.real.tolist()
    temp_sigma_data['sigma_im'] = sigma.imag.tolist()
    temp_sigma_data['w_dict'] = w_dict
    temp_sigma_data['dmft_mu'] = 0.0
    temp_sigma_data['n_orb'] = n_orb
    temp_sigma_data['use'] = True

    return temp_sigma_data

# for what used to be curry version
# def sigma_analytic_to_gf(c_sigma, n_orb, w_dict, soc, lambdas):

    # Sigma_freq = GfReFreq(target_shape=(n_orb, n_orb), mesh=w_dict['w_mesh'])
    # for w in Sigma_freq.mesh:
        # Sigma_freq[:,:][w] = c_sigma(w.value)(*lambdas) * np.eye(n_orb)

    # sigma_array = np.zeros((n_orb, n_orb, w_dict['n_w']), dtype=complex)
    # for ct1, ct2 in product(range(n_orb), range(n_orb)):
        # if ct1 != ct2 and not soc: continue
        # sigma_array[ct1,ct2] = Sigma_freq.data[:,ct1,ct2].real + 1j * Sigma_freq.data[:,ct1,ct2].imag

    # return sigma_array

def sigma_analytic_to_gf(n_orb, w_dict, Sigma_0, Z, soc):

    # Sigma(w) = (1-1/Z)*w + Sigma_0
    Sigma_freq = GfReFreq(target_shape=(n_orb, n_orb), mesh=w_dict['w_mesh'])
    for w in Sigma_freq.mesh:
        for orb in range(n_orb):
            Sigma_freq[orb,orb][w] = (1-1/Z[orb]) * w.value + Sigma_0[orb]

    sigma_array = np.zeros((n_orb, n_orb, w_dict['n_w']), dtype=complex)
    for ct1, ct2 in product(range(n_orb), range(n_orb)):
        if ct1 != ct2 and not soc: continue
        sigma_array[ct1,ct2] = Sigma_freq.data[:,ct1,ct2].real + 1j * Sigma_freq.data[:,ct1,ct2].imag

    return sigma_array

def reorder_sigma(sigma_data, new_order, old_order):
    """
    This function takes a sigma and rotates into new orbital basis.
    """

    sigma = np.array(sigma_data['sigma_re']) + 1j * np.array(sigma_data['sigma_im'])
    change_of_basis = tools.change_basis(len(new_order), new_order,  old_order)

    sigma = np.einsum('ij, jlk -> ilk', np.linalg.inv(change_of_basis), np.einsum('jki, kl -> jli', sigma, change_of_basis))
    sigma_data['sigma_re'] = sigma.real.tolist()
    sigma_data['sigma_im'] = sigma.imag.tolist()
    sigma_data['orbital_order'] = new_order

    return sigma_data
