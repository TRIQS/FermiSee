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
        data['sigma_data']['sigma_re'] =  data['sigma_data']['sigma'].real.tolist()
        data['sigma_data']['sigma_im'] =  data['sigma_data']['sigma'].imag.tolist()
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
        content_type, content_string = contents.split(',')
        data_stream = base64.b64decode(content_string)
        data = json.loads(data_stream)

        lat = data['_lat']
        hoppings = data['_hoppings']
        site_energies = data['_site_energies']
        #return norb (number of orbitals), units (lattice dim) and hopping_dict (self explanatory)
        norb = data['_norb']
        units=[]
        for i in lat:
                units.append(tuple(i))
        hopping_dict={}
        #apparently if _dim_r <= 2 then an array doesnt need to be passed
        #not sure why, and need to figure out how to handle these cases
        if data['_dim_r'] <= 2:
                raise Exception("The pyTB lattice must be greater than 2x2. ie _dim_r > 2")
        
        #stores the absolute values of all the keys, used later to add negative hopping vectors
        #Ex: hopping_keys =  [(1,0,0), (0,1,0), (0,-1,0)] -> abs_hopping_keys = [(1,0,0), (0,1,0), (0,1,0)]; (1,0,0) is unique in the abs_hopping_key, therefore the (-1,0,0) vector must be added to the list of vectors. 
        abs_hopping_keys = []
        for i in hoppings:
                abs_hopping_keys.append(tuple(np.absolute(i[3])))
        
        #if a negative vector is manually added but with a different hopping energy then the average energy is stored
        #the dup_key_dict stores the the absolute hopping as a key when there is a duplicate in abs_hopping_keys
        #the value is the (original hop vector, t)
        dup_key_dict={}
        
        for i in hoppings:
                t = i[0]
                hop_vector = i[3]
                abs_hop_vector = tuple(np.absolute(hop_vector))
                #if there is only 1 instance of the absolute vector then the negative vector must be added
                if abs_hopping_keys.count(abs_hop_vector) == 1:
                        #add the vector
                        hopping_dict[tuple(hop_vector)]=t*np.eye(norb) #is np.eye always going to be used?
                        #add the negative vector
                        neg_hop_vector = []
                        for j in hop_vector:
                            neg_hop_vector.append(j*-1)
                        hopping_dict[tuple(neg_hop_vector)]=t*np.eye(norb)
                        #print(t,neg_hop_vector)
                #there are multiple instances of the same absolute vector
                else:
                        #the hop vectors are in dup_key_dict; extract the information
                        if abs_hop_vector in dup_key_dict:
                            other_t = dup_key_dict[abs_hop_vector][1]
                            other_hop_vector = dup_key_dict[abs_hop_vector][0]
                            avg = (t+other_t)/2
                            #add vector
                            hopping_dict[tuple(hop_vector)]=avg*np.eye(norb)
                            #edit the value of the other vector
                            hopping_dict[tuple(other_hop_vector)]=avg*np.eye(norb)
                        #the hop vector is not in dup_key_dict (this is the first duplicate); save it
                        else:
                            dup_key_dict[abs_hop_vector]=(tuple(hop_vector),t)
                            #place holder in the dict
                            hopping_dict[tuple(hop_vector)]=[]
        #site energies are added as hoppings to the origin
        for i in site_energies:
                _matrix = i*np.eye(norb)
                hopping_dict[tuple([0.0,0.0,0.0])]=_matrix
        return norb, units, hopping_dict

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

def load_sigma_h5(contents , filename, orbital_order = None):
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
    w_dict = {'w_mesh' : w_mesh, 
              'n_w' : ar['self_energy']['n_w'], 
              'window' : [w_mesh[0],w_mesh[-1]]}
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




