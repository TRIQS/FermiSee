from tools.wannier90 import *
#from triqs_tprf.tight_binding import *
from triqs.lattice.tight_binding import *
import skimage.measure
import copy
from matplotlib import cm

def extend_wannier90_to_spin(hopping, num_wann):
    hopping_spin = {}
    for key, value in hopping.items():
        #hopping_spin[key] = np.kron(value, np.eye(2))
        hopping_spin[key] = np.kron(np.eye(2), value)
    return hopping_spin, 2 * num_wann

def energies_on_bz_paths(paths, tb_lattice, n_pts=50):

    """ Given a list of k-point paths compute the eigen energies along
    the paths using n_pts discrete points for each sub-path. """

    # -- Get the reciprocal lattice vectors
    bz = BrillouinZone(tb_lattice.bl)
    k_mat = np.array(bz.units())

    n_paths = len(paths)
    n_orb = tb_lattice.NOrbitalsInUnitCell

    k = np.zeros(n_pts * n_paths)
    E = np.zeros((n_orb, n_pts * n_paths))

    k_length = 0. # accumulative k-path length

    for pidx, (ki, kf) in enumerate(paths):

        s, e = pidx * n_pts, (pidx+1) * n_pts
        E[:, s:e] = energies_on_bz_path(tb_lattice.tb, ki, kf, n_pts)

        dk = np.dot(k_mat.T, (ki - kf))
        a = np.linspace(0., 1., num=n_pts, endpoint=False)
        k_vec = a[:, None] * dk[None, :]

        k[s:e] = np.linalg.norm(k_vec, axis=1) + k_length
        k_length += np.linalg.norm(dk)

    #K = np.concatenate((k[::n_pts], [2 * k[-1] - k[-2]])) # add last point for K-grid
    K = np.concatenate((k[::n_pts], [k[-1]])) # add last point for K-grid

    return k, K, E


def energy_matrix_on_bz_paths(paths, TBL, n_pts=50):

    """ Given a list of k-point paths compute the eigen energies along
    the paths using n_pts discrete points for each sub-path. """

    # -- Get the reciprocal lattice vectors
    bz = BrillouinZone(TBL.bl)
    k_mat = np.array(bz.units())

    n_paths = len(paths)
    n_orb = TBL.NOrbitalsInUnitCell

    k = np.zeros(n_pts * n_paths)
    E = np.zeros((n_orb, n_orb, n_pts * n_paths),dtype=complex)

    k_length = 0. # accumulative k-path length

    for pidx, (ki, kf) in enumerate(paths):
        
        # if this is the last section, add the endpoint!
        if pidx == len(paths)-1:
             endpoint = True
        else: 
            endpoint = False

        s, e = pidx * n_pts, (pidx+1) * n_pts
        E[:,:, s:e] = energy_matrix_on_bz_path(TBL.tb, ki, kf, n_pts)

        dk = np.dot(k_mat.T, (ki - kf))
        a = np.linspace(0., 1., num=n_pts, endpoint=endpoint)
        k_vec = a[:, None] * dk[None, :]

        k[s:e] = np.linalg.norm(k_vec, axis=1) + k_length
        k_length += np.linalg.norm(dk)

    K = np.concatenate((k[::n_pts], [k[-1]])) # add last point for K-grid

    return k, K, E

def reg(k) : return tuple( int(x) for x in k)

def fract_ind_to_val(x,ind):
    ind[ind == len(x)-1] = len(x)-1-1e-6
    int_ind = [int(indi) for indi in ind]
    int_ind_p1 = [int(indi)+1 for indi in ind]
    return x[int_ind] + (x[int_ind_p1] - x[int_ind])*(np.array(ind)-np.array(int_ind))

def get_kx_ky_FS(X,Y,Z,tbl,k_trans_back,select=None,N_kxy=10,kz=0.0, fermi=0.0):

    kx = np.linspace(0,0.5,N_kxy)
    ky = np.linspace(0,0.5,N_kxy)

    if select is None: select = np.array(range(tbl.NOrbitalsInUnitCell))

    E_FS = np.zeros((tbl.NOrbitalsInUnitCell,N_kxy,N_kxy))
    for kyi in range(N_kxy):
        path_FS = [(Y/(N_kxy-1)*kyi +kz*Z, X+Y/(N_kxy-1)*kyi+kz*Z)]
        du, du, E_FS[:,:,kyi] = energies_on_bz_paths(path_FS, tbl, n_pts=N_kxy)

    contours = {}
    FS_kx_ky = {}
    char = {}
    for ib in range(np.shape(E_FS)[0]):
        contours[ib] = skimage.measure.find_contours(E_FS[ib,:,:],fermi)

    i = 0   
    for cb in contours:
        for ci in range(np.shape(contours[cb])[0]):
            FS_kx_ky[i] = np.vstack([fract_ind_to_val(kx,contours[cb][ci][:,0]),
				fract_ind_to_val(ky,contours[cb][ci][:,1]),
				kz*Z[2]*np.ones(len(contours[cb][ci][:,0]))]).T.reshape(-1,3)
            char[i] = {}
            for n in range(len(FS_kx_ky[i][:,0])):
                p = np.dot(FS_kx_ky[i][n,:], k_trans_back)
                MAT = energy_matrix_on_bz_path(tbl.tb, p,p, n_pts=1)
                E, v = np.linalg.eigh(MAT[select[:,np.newaxis],select,0])
                idx = np.argmin(np.abs(E))

                char[i][n] = [np.round(np.real(v[ib,idx]*np.conjugate(v[ib,idx])),4) for ib in range(len(select))]
            i += 1
    return FS_kx_ky, char

def load_data_generic(path, name='w2w'):
    hopping, num_wann = parse_hopping_from_wannier90_hr_dat(path + name +'_hr.dat')
    units = parse_lattice_vectors_from_wannier90_wout(path + name +'.wout')
    return hopping, units, num_wann

def get_TBL(path, name='w2w', extend_to_spin=False, add_local=None, add_field=None, renormalize=None):

    hopping, units, num_wann = load_data_generic(path, name=name)
    if extend_to_spin:
    	hopping, num_wann = extend_wannier90_to_spin(hopping, num_wann)
    if add_local is not None:
        hopping[(0,0,0)] += add_local
    if renormalize is not None:
        assert len(np.shape(renormalize)) == 1, 'Give Z as a vector'
        assert len(renormalize) == num_wann, 'Give Z as a vector of size n_orb (times two if SOC)'
        
        Z_mat = np.diag(np.sqrt(renormalize))
        for R in hopping:
            hopping[R] = np.dot(np.dot(Z_mat, hopping[R]), Z_mat)

    if add_field is not None:
        hopping[(0,0,0)] += add_field

    TBL = TBLattice(units = units, hopping = hopping, orbital_positions = [(0,0,0)]*num_wann,
                    orbital_names = [str(i) for i in range(num_wann)])
    return TBL
