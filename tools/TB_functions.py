from triqs.lattice.tight_binding import *
from triqs.lattice.utils import k_space_path
import skimage.measure
import copy
from matplotlib import cm
import numpy as np

def extend_wannier90_to_spin(hopping, num_wann):
    hopping_spin = {}
    for key, value in hopping.items():
        #hopping_spin[key] = np.kron(value, np.eye(2))
        hopping_spin[key] = np.kron(np.eye(2), value)
    return hopping_spin, 2 * num_wann

def reg(k) : return tuple( int(x) for x in k)

def fract_ind_to_val(x,ind):
    ind[ind == len(x)-1] = len(x)-1-1e-6
    int_ind = [int(indi) for indi in ind]
    int_ind_p1 = [int(indi)+1 for indi in ind]
    return x[int_ind] + (x[int_ind_p1] - x[int_ind])*(np.array(ind)-np.array(int_ind))

def get_kx_ky_FS(X,Y,origin,tbl,k_trans_back,select=None,N_kxy=10,kz=0.0, fermi=0.0):
    kx = np.linspace(0,0.5,N_kxy)
    ky = np.linspace(0,0.5,N_kxy)
    Z  = np.zeros(3)

    if select is None: select = np.array(range(tbl.n_orbitals))

    E_FS = np.zeros((tbl.n_orbitals,N_kxy,N_kxy))
    for kyi in range(N_kxy):
        path_FS = [((Y-origin)/(N_kxy-1)*kyi +kz*Z + origin, origin+(X-origin)+(Y-origin)/(N_kxy-1)*kyi+kz*Z)]
        kvecs, k, _ = k_space_path(path_FS, num=N_kxy)
        E_FS[:,:,kyi] = tbl.dispersion(kvecs).transpose()

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
                MAT = tbl.fourier(p)
                E, v = np.linalg.eigh(MAT[select[:,np.newaxis],select])
                idx = np.argmin(np.abs(E))

                char[i][n] = [np.round(np.real(v[ib,idx]*np.conjugate(v[ib,idx])),4) for ib in range(len(select))]
            i += 1
    return FS_kx_ky, char

