import numpy as np
import base64
import io
from h5 import HDFArchive
import json

import tools.wannier90 as tb_w90
import tools.calc_akw as calc_akw


def load_config(contents, h5_filename, data):
    data['config_filename'] = h5_filename
    content_type, content_string = contents.split(',')
    h5_bytestream = base64.b64decode(content_string)
    try:
        ar = HDFArchive(h5_bytestream)
    except:
        print('error in loading file')
        data['error'] = True

    if 'tb_data' in ar:
        data['tb_data'] = ar['tb_data']
        data['tb_data']['e_mat_re'] = data['tb_data']['e_mat'].real.tolist()
        data['tb_data']['e_mat_im'] = data['tb_data']['e_mat'].imag.tolist()
        del data['tb_data']['e_mat']
        data['tb_data']['eps_nuk'] = data['tb_data']['eps_nuk'].tolist()
        if 'e_vecs' in data['tb_data'].keys():
            data['tb_data']['evecs_re'] = data['tb_data']['e_vecs'].real
            data['tb_data']['evecs_im'] = data['tb_data']['e_vecs'].imag
            del data['tb_data']['e_vecs']
        data['tb_data']['hopping'] = {str(key): value.tolist() for key, value in data['tb_data']['hopping'].items()}

    if 'sigma_data' in ar:
        data['sigma_data'] = ar['sigma_data']
        data['sigma_data']['sigma_re'] = data['sigma_data']['sigma'].real.tolist()
        data['sigma_data']['sigma_im'] = data['sigma_data']['sigma'].imag.tolist()
        del data['sigma_data']['sigma']
        data['sigma_data']['w_dict']['w_mesh'] = data['sigma_data']['w_dict']['w_mesh'].tolist()
        data['sigma_data']['orbital_order'] = tuple(data['sigma_data']['orbital_order'])

    if not 'sigma_data' in ar and not 'tb_data' in ar:
        print('error in loading file')
        data['error'] = True
    else:
        data['error'] = False

    return data


def load_pythTB_json(contents):
    '''
    read pythTB json file

    Parameters
    ----------
    contents: json string from memory

    Returns
    -------
    norb: int
        number of orbitals
    units: list of tuples
        vectors spanning unit cell
    hopping_dict: dict
        dict of hopping tuples
    '''
    content_type, content_string = contents.split(',')
    data_stream = base64.b64decode(content_string)
    data = json.loads(data_stream)

    lat = data['_lat']
    hoppings = data['_hoppings']
    site_energies = data['_site_energies']
    norb = data['_norb']

    # apparently if _dim_r <= 2 then an array doesnt need to be passed in the hoppings
    if data['_dim_r'] <= 2:
        raise Exception("The pyTB lattice must be greater than 2x2. ie _dim_r > 2")

    # extract the lattice dimensions
    units = []
    for i in lat:
        units.append(tuple(i))

    # extract the hoppings
    #parsing is taken and adapted from triqs/lattice/utils.py TB_from_pythTB
    hopping_dict={}
    m_zero = np.zeros((norb, norb), dtype=complex)
    #on-site energy
    hopping_dict[(0, 0, 0)] = np.eye(norb, dtype=complex) * site_energies
    #hoppings
    for hop, orb_from, orb_to, vector in hoppings:
        if tuple(vector) not in hopping_dict:
            hopping_dict[tuple(vector)] = m_zero.copy()
            # per default pythTB does not explicitly stores -R
            hopping_dict[tuple(-np.array(vector))] = m_zero.copy()

            hopping_dict[tuple(vector)][orb_from, orb_to] += hop
            # fill -R from +R using H_ij(+R)=[H_ji(-R)]*
            # if the user specified -R explicitly we have to sum both hopping
            # matrices
            # according to pythTB documentation
            hopping_dict[tuple(-np.array(vector))][orb_to, orb_from] += np.conj(hop)

    print('--------')
    print(hopping_dict)
    return norb, units, hopping_dict


def load_w90_hr(contents):
    content_type, content_string = contents.split(',')
    w90_hr_stream = base64.b64decode(content_string).decode('utf-8')
    hopping, n_wf = tb_w90.parse_hopping_from_wannier90_hr(w90_hr_stream)
    # print('number of Wannier orbitals {}'.format(num_wann))

    return hopping, n_wf


def load_w90_wout(contents):
    content_type, content_string = contents.split(',')
    w90_wout_stream = base64.b64decode(content_string).decode('utf-8')
    units = tb_w90.parse_lattice_vectors_from_wannier90_wout(w90_wout_stream)

    return units


def load_sigma_h5(contents, filename, orbital_order=None):
    '''
    example to store a suitable sigma:
    with HDFArchive(path,'w') as h5:
        h5.create_group('self_energy')
        h5['self_energy']['Sigma'] = Sigma
        h5['self_energy']['w_mesh'] = getX(Sigma.mesh).tolist()
        h5['self_energy']['n_w'] = len(getX(Sigma.mesh).tolist())
        h5['self_energy']['n_orb'] = Sigma['up_0'].target_shape[0]
        h5['self_energy']['dc'] = dc[0]['up'][0,0]
        h5['self_energy']['dmft_mu'] = dmft_mu
        h5['self_energy']['orbital_order'] = (0,1,2)
    '''
    data = {'config_filename': filename}

    content_type, content_string = contents.split(',')
    h5_bytestream = base64.b64decode(content_string)
    ar = HDFArchive(h5_bytestream)

    # extract from h5
    Sigma = ar['self_energy']['Sigma']
    orbital_order = ar['self_energy']['orbital_order']
    n_orb = ar['self_energy']['n_orb']
    dc = ar['self_energy']['dc']
    dmft_mu = ar['self_energy']['dmft_mu']
    w_mesh = ar['self_energy']['w_mesh']

    # setup w_dict
    w_dict = {'w_mesh': w_mesh,
              'n_w': ar['self_energy']['n_w'],
              'window': [w_mesh[0], w_mesh[-1]]}
    # TODO able to choose these
    spin = 'up'
    block = 0

    # convert orbital order to list:
    sigma_interpolated = calc_akw.sigma_from_dmft(n_orb, orbital_order, Sigma, spin, block, dc, w_dict)

    data['sigma_re'] = sigma_interpolated.real.tolist()
    data['sigma_im'] = sigma_interpolated.imag.tolist()
    data['w_dict'] = w_dict
    data['dmft_mu'] = dmft_mu
    data['orbital_order'] = orbital_order
    data['n_orb'] = n_orb
    print(sigma_interpolated.shape)

    return data
