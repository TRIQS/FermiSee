import numpy as np
from itertools import product
from triqs.gf import GfReFreq

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

def sigma_analytic_to_gf(c_sigma, n_orb, w_dict, soc, lambdas):

    Sigma_freq = GfReFreq(target_shape=(n_orb, n_orb), mesh=w_dict['w_mesh'])
    for w in Sigma_freq.mesh:
        Sigma_freq[:,:][w] = c_sigma(w.value)(*lambdas) * np.eye(n_orb)

    sigma_array = np.zeros((n_orb, n_orb, w_dict['n_w']), dtype=complex)
    for ct1, ct2 in product(range(n_orb), range(n_orb)):
        if ct1 != ct2 and not soc: continue
        sigma_array[ct1,ct2] = Sigma_freq.data[:,ct1,ct2].real + 1j * Sigma_freq.data[:,ct1,ct2].imag

    return sigma_array


def curry(func):
    def f_w(w):
        curry.__curried_func_name__ = func.__name__
        f_args, f_kwargs = [], {}
        def f_lambda(*args, **kwargs):
            nonlocal f_args, f_kwargs
            if args or kwargs:
                f_args += args
                f_kwargs.update(kwargs)
                return func(w, *f_args, *f_kwargs)
            else:
                result = func(w, *f_args, *f_kwargs)
                f_args, f_kwargs = [], {}
                return result
        return f_lambda
    return f_w

