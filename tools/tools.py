from tools.TB_functions import *
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
    """

    if extend_to_spin:
        hopping, n_wf = extend_wannier90_to_spin(hopping, n_wf)
    if add_local is not None:
        hopping[(0, 0, 0)] += add_local
    if renormalize is not None:
        assert len(np.shape(renormalize)) == 1, 'Give Z as a vector'
        assert len(
            renormalize
        ) == n_wf, 'Give Z as a vector of size n_orb (times two if SOC)'

        Z_mat = np.diag(np.sqrt(renormalize))
        for R in hopping:
            hopping[R] = np.dot(np.dot(Z_mat, hopping[R]), Z_mat)

    if add_field is not None:
        hopping[(0, 0, 0)] += add_field

    TBL = TBLattice(units=units,
                    hopping=hopping,
                    orbital_positions=[(0, 0, 0)] * n_wf,
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


def dichotomy(function,
              x_init,
              y_value,
              precision_on_y,
              delta_x,
              max_loops=1000,
              x_name="",
              y_name="",
              verbosity=1):
    r""" Finds :math:`x` that solves :math:`y = f(x)`.

    Starting at ``x_init``, which is used as the lower upper/bound,
    dichotomy finds first the upper/lower bound by adding/subtracting ``delta_x``.
    Then bisection is used to refine :math:`x` until
    ``abs(f(x) - y_value) < precision_on_y`` or ``max_loops`` is reached.

    Parameters
    ----------

    function : function, real valued
        Function :math:`f(x)`. It must take only one real parameter.
    x_init : double
        Initial guess for x. On success, returns the new value of x.
    y_value : double
        Target value for y.
    precision_on_y : double
        Stops if ``abs(f(x) - y_value) < precision_on_y``.
    delta_x : double
        :math:`\Delta x` added/subtracted from ``x_init`` until the second bound is found.
    max_loops : integer, optional
        Maximum number of loops (default is 1000).
    x_name : string, optional
        Name of variable x used for printing.
    y_name : string, optional
        Name of variable y used for printing.
    verbosity : integer, optional
        Verbosity level.

    Returns
    -------

    (x,y, func_out) : (double, double, Gloc)
        :math:`x` and :math:`y=f(x)`. Returns (None, None) if dichotomy failed.
    """

    print(
        "Dichotomy adjustment of %(x_name)s to obtain %(y_name)s = %(y_value)f +/- %(precision_on_y)f"
        % locals())
    PR = "    "
    if x_name == "" or y_name == "": verbosity = max(verbosity, 1)
    x = x_init
    delta_x = abs(delta_x)

    # First find the bounds
    y1, func_out = function(x)
    eps = np.sign(y1 - y_value)
    x1 = x
    y2 = y1
    x2 = x1
    nbre_loop = 0
    while (nbre_loop <= max_loops) and (y2 - y_value) * eps > 0 and abs(
            y2 - y_value) > precision_on_y:
        nbre_loop += 1
        x2 -= eps * delta_x
        y2, func_out = function(x2)
        if x_name != "" and verbosity > 2:
            print("%(PR)s%(x_name)s = %(x2)f  \n%(PR)s%(y_name)s = %(y2)f" %
                  locals())

    # Make sure that x2 > x1
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    print("%(PR)s%(x1)f < %(x_name)s < %(x2)f" % locals())
    print("%(PR)s%(y1)f < %(y_name)s < %(y2)f" % locals())

    # We found bounds.
    # If one of the two bounds is already close to the solution
    # the bisection will not run. For this case we set x and yfound.
    if abs(y1 - y_value) < abs(y2 - y_value):
        yfound = y1
        x = x1
    else:
        yfound = y2
        x = x2

    # Now let's refine between the bounds
    while (nbre_loop <= max_loops) and (abs(yfound - y_value) >
                                        precision_on_y):
        nbre_loop += 1
        x = x1 + (x2 - x1) * (y_value - y1) / (y2 - y1)
        yfound, func_out = function(x)
        if (y1 - y_value) * (yfound - y_value) > 0:
            x1 = x
            y1 = yfound
        else:
            x2 = x
            y2 = yfound
        if verbosity > 2:
            print("%(PR)s%(x1)f < %(x_name)s < %(x2)f" % locals())
            print("%(PR)s%(y1)f < %(y_name)s < %(y2)f" % locals())
    if abs(yfound - y_value) < precision_on_y:
        if verbosity > 0:
            print("%(PR)s%(x_name)s found in %(nbre_loop)d iterations : " %
                  locals())
            print("%(PR)s%(y_name)s = %(yfound)f;%(x_name)s = %(x)f" %
                  locals())
        return (x, yfound, func_out)
    else:
        if verbosity > 0:
            print(
                "%(PR)sFAILURE to adjust %(x_name)s to the value %(y_value)f after %(nbre_loop)d iterations."
                % locals())
            print("%(PR)sFAILURE returning (None, None) due to failure." %
                  locals())
        return (None, None, None)
