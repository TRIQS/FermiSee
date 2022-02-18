from h5 import HDFArchive
from triqs.gf import *

def convert_sigma_to_fermisee(solid_dmft_h5, out_h5, it='last_iter'):
    '''
    
    todo:
    * extend to multiple impurities
    * respect orbital order
    * make sure that each GF block is n_orb x n_orb, i.e. convert block structure 
    '''

    with HDFArchive(solid_dmft_h5,'r') as h5:
    
        Sigma = h5['DMFT_results'][it]['Sigma_freq_0']
        mu = h5['DMFT_results'][it]['chemical_potential_post']
        dc = h5['DMFT_results'][it]['DC_pot'][0]['up']
        
    w_mesh = [w.value for w in Sigma.mesh]
    n_orb = Sigma[list(Sigma.indices)[0]].target_shape[0]
    # store to FermiSee readable format
    with HDFArchive(out_h5, 'w') as h5:
        h5.create_group('self_energy')
        h5['self_energy']['Sigma'] = Sigma
        h5['self_energy']['w_mesh'] = w_mesh
        h5['self_energy']['n_w'] = len(w_mesh)
        h5['self_energy']['n_orb'] = n_orb
        h5['self_energy']['dc'] = dc[0,0]
        h5['self_energy']['dmft_mu'] = mu
        h5['self_energy']['orbital_order'] = tuple(range(n_orb))