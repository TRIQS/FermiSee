import numpy as np
from h5 import HDFArchive

def _get_tb_bands(k_mesh, e_mat, k_points,):
    
    e_val = np.zeros((e_mat.shape[0], k_mesh.shape[0]), dtype=complex)
    e_vec = np.zeros(np.shape(e_mat), dtype=complex)
    for ik in range(np.shape(e_mat)[2]):
        e_val[:,ik], e_vec[:,:,ik] = np.linalg.eigh(e_mat[:,:,ik])

    return e_val, e_vec


def update_data(contents, h5_filename):
    data = {'filename': h5_filename}
    if not contents: 
        h5_file = h5_filename
    else:
        h5_file = h5_filename
        # content_type, content_string = contents.split(',')
        # decoded = base64.b64decode(content_string)
        # h5_file = io.BytesIO(decoded)
        # h5 = HDFArchive(h5_file)
        # print(h5)
        # data['Akw'] = None

        
    with HDFArchive(h5_file, 'r') as ar:
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
    data['eps_nuk'], evec_nuk = _get_tb_bands(data['k_mesh'], e_mat, data['k_points'])

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

