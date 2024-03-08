from tools.TB_functions import extend_wannier90_to_spin
from triqs.lattice.tight_binding import TBLattice
import numpy as np


def get_TBL(hopping,
            units,
            n_wf,
            extend_to_spin=False,
            add_local=None,
            add_field=None,
            renormalize=None):
    """
    get triqs tight-binding object from hoppings + units
    different from triqs.lattice.utils.TB_from_wannier90
    because it does not read from file but still needs
    be able to add local etc.
    """
    unit_dim = np.shape(units)[0]
    origin = (0,) * unit_dim
    if extend_to_spin:
        hopping, n_wf = extend_wannier90_to_spin(hopping, n_wf)
    if add_local is not None:
        hopping[origin] += add_local
    if renormalize is not None:
        assert len(np.shape(renormalize)) == 1, 'Give Z as a vector'
        assert len(
            renormalize
        ) == n_wf, 'Give Z as a vector of size n_orb (times two if SOC)'

        Z_mat = np.diag(np.sqrt(renormalize))
        for R in hopping:
            hopping[R] = np.dot(np.dot(Z_mat, hopping[R]), Z_mat)

    if add_field is not None:
        hopping[origin] += add_field

    TBL = TBLattice(units=units,
                    hopping=hopping,
                    orbital_positions=[origin] * n_wf,
                    orbital_names=[str(i) for i in range(n_wf)])
    return TBL


def change_basis(n_orb, orbital_order_to, orbital_order_from):
    """
    Rotation between orbital bases
    """

    change_of_basis = np.eye(n_orb)
    for ct, orb in enumerate(orbital_order_to):
        orb_idx = orbital_order_from.index(orb)
        change_of_basis[orb_idx, :] = np.roll(np.eye(n_orb, 1), ct)[:, 0]

    return change_of_basis


def print_matrix(matrix, n_orb, text):
    """
    Pre-determined print command for matrices
    """

    print('{}:'.format(text))
    fmt = '{:16.4f}' * n_orb
    for row in matrix:
        print((' ' * 4 + fmt).format(*row))


def lambda_matrix_w90_t2g(add_lambda):
    """
    Add local SOC term to H(R) for t2g shell
    """

    lambda_x, lambda_y, lambda_z = add_lambda

    lambda_matrix = np.zeros((6, 6), dtype=complex)
    lambda_matrix[0, 1] = -1j * lambda_z / 2.0
    lambda_matrix[0, 5] = 1j * lambda_x / 2.0
    lambda_matrix[1, 5] = -lambda_y / 2.0
    lambda_matrix[2, 3] = -1j * lambda_x / 2.0
    lambda_matrix[2, 4] = lambda_y / 2.0
    lambda_matrix[3, 4] = 1j * lambda_z / 2.0
    lambda_matrix += np.transpose(np.conjugate(lambda_matrix))

    return lambda_matrix


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

