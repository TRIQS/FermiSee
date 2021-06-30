import numpy as np
import base64
import io
from h5 import HDFArchive

import tools.wannier90 as tb_w90
import tools.calc_akw as calc_akw


def load_config(contents, h5_filename):
    data = {'config_filename': h5_filename}
    if not contents: 
        ar = HDFArchive(h5_filename, 'r')

    else:
        content_type, content_string = contents.split(',')
        h5_bytestream = base64.b64decode(content_string)
        ar = HDFArchive(h5_bytestream)
        
    # Akw data
    # data['Akw_data'] = ar['A_k_w_data'] # contains A_k_w, dft_mu
    data['Akw'] = ar['A_k_w_data']['A_k_w']
    data['dft_mu'] = ar['A_k_w_data']['dft_mu']

    # tb data
    # data['tb_data'] = ar['tb_data'] # e_mat, k_mesh, k_points, k_points_labels
    data['k_points_labels'] = ar['tb_data']['k_points_labels']
    e_mat = ar['tb_data']['e_mat']
    data['k_points'] = ar['tb_data']['k_points']
    data['k_mesh'] = ar['tb_data']['k_mesh']
    # w mesh
    data['freq_mesh'] = ar['w_mesh']['w_mesh']

    if not contents:
        del ar

    # calculate bands and distribute
    data['eps_nuk'], evec_nuk = calc_akw.get_tb_bands(data['k_mesh'], e_mat, data['k_points'])

    # workaround to remove last point in k_mesh
    pts_per_kpt = int(len(data['k_mesh'])/(len(data['k_points'])-1))-1
    # remove last intervall except the first point to incl high symm point
    data['k_mesh'] = data['k_mesh'][:-pts_per_kpt]
    data['Akw'] = data['Akw'][:-pts_per_kpt,:]

    # transform np arrays to lists, to be able to serialize to json
    data['Akw'] = data['Akw'].tolist()
    data['freq_mesh'] = data['freq_mesh'].tolist()
    data['k_mesh'] = data['k_mesh'].tolist()
    data['k_points'] = data['k_points'][:-1].tolist()
    data['eps_nuk'] = (data['eps_nuk'].real - data['dft_mu']).tolist()

    # max value
    data['max_Akw'] = 1.05 * np.max(np.array(data['Akw']))

    return data

def load_w90_hr(contents):
    content_type, content_string = contents.split(',')
    w90_hr_stream = base64.b64decode(content_string).decode('utf-8')
    hopping, n_wf = tb_w90.parse_hopping_from_wannier90_hr(w90_hr_stream)
    #print('number of Wannier orbitals {}'.format(num_wann))

    return hopping, n_wf

def load_w90_wout(contents):
    content_type, content_string = contents.split(',')
    w90_wout_stream = base64.b64decode(content_string).decode('utf-8')
    units = tb_w90.parse_lattice_vectors_from_wannier90_wout(w90_wout_stream)

    return units 

def load_sigma_h5(contents , filename, orbital_order):
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
        h5['self_energy']['orbital_order'] = ['dxz', 'dyz', 'dxy']
    '''
    data = {'config_filename': filename}

    content_type, content_string = contents.split(',')
    h5_bytestream = base64.b64decode(content_string)
    ar = HDFArchive(h5_bytestream)

    # extract from h5
    Sigma = ar['self_energy']['Sigma']
    orbital_order_dmft = ar['self_energy']['orbital_order']
    n_orb = ar['self_energy']['n_orb']
    dc = ar['self_energy']['dc']
    dmft_mu = ar['self_energy']['dmft_mu']
    w_mesh = ar['self_energy']['w_mesh']

    # setup w_dict
    w_dict = {'w_mesh' : w_mesh, 
              'n_w' : ar['self_energy']['n_w'], 
              'window' : [w_mesh[0],w_mesh[-1]]}
    # TODO able to choose these
    spin = 'up'
    block = 0

    # convert orbital order to list:
    orbital_order = list(orbital_order[0].values())
    sigma_interpolated = calc_akw.sigma_from_dmft(n_orb, orbital_order, Sigma, spin, block, orbital_order_dmft, dc, w_dict)

    data['sigma_re'] = sigma_interpolated.real.tolist()
    data['sigma_im'] = sigma_interpolated.imag.tolist()
    data['w_dict'] = w_dict
    data['dmft_mu'] = dmft_mu
    print(sigma_interpolated.shape)

    return data




