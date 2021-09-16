def _linefit(x, y, interval, spacing=50, addspace=0.0):

    calc_Z = lambda slope: 1/(1-slope)
    
    x = np.array(x)
    lim_l, lim_r = interval
    indices = np.where(np.logical_and(x>=lim_l, x<=lim_r))
    fit = np.polyfit(x[indices], y[indices], 1)
    slope = fit[1]
    Z = calc_Z(slope)
    f_x = np.poly1d(fit)
    x_cont = np.linspace(x[indices][0] - addspace, x[indices][-1] + addspace, spacing)
    
    return x_cont, f_x(x_cont), fit

def _sigma_from_model(n_orb, orbital_order, zeroth_order, first_order, efermi, eta=0.0, **w):
    
    print('Setting model Sigma')

    mu = dft_mu = efermi
    eta = eta * 1j
    
    # set up mesh
    w_dict = w['w_mesh']
    w_mesh = np.linspace(*w_dict['window'], w_dict['n_w'])
    w_dict.update({'w_mesh': w_mesh})

    # interpolate sigma
    sigma_interpolated = np.zeros((n_orb, n_orb, w_dict['n_w']), dtype=complex)
    approximate_sigma = lambda zeroth_order, first_order, orb: zeroth_order[orb] + w_dict['w_mesh'] * first_order[orb]
    for ct, orb in enumerate(orbital_order):
        sigma_interpolated[ct,ct] = approximate_sigma(zeroth_order, first_order, ct)
    
    return sigma_interpolated, mu, dft_mu, eta, w_dict

def get_dmft_bands(n_orb, with_sigma=False, fermi_slice=False, solve=False, orbital_order=(0,1,2), band_basis=False, **specs):
    
    # dmft output
    if with_sigma:
        sigma_types = ['calc', 'model']
        if isinstance(with_sigma, str):
            if with_sigma not in sigma_types: raise ValueError('Invalid sigma type. Expected one of: {}'.format(sigma_types))
        elif not isinstance(with_sigma, BlockGf):
            raise ValueError('Invalid sigma type. Expected BlockGf.')

        # get sigma
        if with_sigma == 'model': delta_sigma, mu, dft_mu, eta, w_dict = _sigma_from_model(n_orb, orbital_order, **specs)
        # else is from dmft or memory:
        else: delta_sigma, mu, dft_mu, eta, w_dict = sigma_from_dmft(n_orb, orbital_order, with_sigma, **specs)
        
        # calculate alatt
        if not fermi_slice:
            alatt_k_w, new_mu = calc_alatt(n_orb, mu, eta, e_mat, delta_sigma, solve, band_basis,**w_dict)
        else:
            alatt_k_w = _calc_kslice(n_orb, mu, eta, e_mat, delta_sigma, solve, band_basis **w_dict)       
    else:
        dft_mu = new_mu
        w_dict = {}
        w_dict['w_mesh'] = None
        w_dict['window'] = None
        alatt_k_w = None
    
    return {'k_mesh': k_array, 'k_points': k_points, 'k_points_labels': k_points_labels, 'e_mat': e_mat, 'tb': tb}, alatt_k_w, w_dict, dft_mu

def plot_bands(fig, ax, alatt_k_w, tb_data, w_dict, n_orb, dft_mu, tb=True, alatt=False, **plot_dict):
    
    if alatt:
        if alatt_k_w is None: raise ValueError('A(k,w) unknown. Specify "with_sigma = True"')
        kw_x, kw_y = np.meshgrid(tb_data['k_mesh'], w_dict['w_mesh'])
        if len(alatt_k_w.shape) > 2:
            for orb in range(n_orb):
                ax.contour(kw_x, kw_y, alatt_k_w[:,:,orb].T, colors=np.array([eval('cm.'+plot_dict['colorscheme_solve'])(0.7)]), levels=1, zorder=2.)
        else:
            graph = ax.pcolormesh(kw_x, kw_y, alatt_k_w.T, cmap=plot_dict['colorscheme_bands'], norm=LogNorm(vmin=plot_dict['vmin'], vmax=np.max(alatt_k_w)))
            colorbar = plt.colorbar(graph)
            colorbar.set_label(r'$A(k, \omega)$')

    if tb:
        #eps_nuk, evec_nuk = _get_tb_bands(**tb_data)
        for band in range(n_orb):
            orbital_projected = [0] if not plot_dict['proj_on_orb'] else np.real(evec_nuk[plot_dict['proj_on_orb'], band] * evec_nuk[plot_dict['proj_on_orb'], band].conjugate())
            ax.scatter(tb_data['k_mesh'], eps_nuk[band].real - dft_mu, c=cm.plasma(orbital_projected), s=1, label=r'tight-binding', zorder=1.)

    _setup_plot_bands(ax, tb_data['k_points'], tb_data['k_points_labels'], w_dict)

def plot_kslice(fig, ax, alatt_k_w, tb_data, w_dict, n_orb, tb_dict, tb=True, alatt=False, quarter=0, dft_mu=0.0, **plot_dict):
    
    sign = [1,-1]
    quarters = np.array([sign,sign])
    
    if alatt:
        if alatt_k_w is None: raise ValueError('A(k,w) unknown. Specify "with_sigma = True"')
        n_kx, n_ky = tb_data['e_mat'].shape[2:4]
        kx, ky = np.meshgrid(range(n_kx), range(n_ky))
        for qrt in list(itertools.product(*quarters))[quarter:quarter+1]:
            if len(alatt_k_w.shape) > 2:
                for orb in range(n_orb):
                    ax.contour(qrt[0] * kx/(n_kx-1), qrt[1] * ky/(n_ky-1), alatt_k_w[:,:,orb].T, colors=np.array([eval('cm.'+plot_dict['colorscheme_solve'])(0.7)]), levels=1, zorder=2)
            else:
                graph = ax.pcolormesh(qrt[0] * kx/(n_kx-1), qrt[1] * ky/(n_ky-1), alatt_k_w.T, cmap=plot_dict['colorscheme_kslice'],
                                      norm=LogNorm(vmin=plot_dict['vmin'], vmax=np.max(alatt_k_w)))
                #colorbar = plt.colorbar(graph)
                #colorbar.set_label(r'$A(k, 0$)')

    if tb:
        quarters *= 2
        eps_nuk, evec_nuk = _get_tb_kslice(tb_data['tb'], dft_mu, **tb_dict)
        for qrt in list(itertools.product(*quarters))[quarter:quarter+1]:
            for band in range(len(eps_nuk)):
                for segment in range(eps_nuk[band].shape[0]):
                    #orbital_projected = evec_nuk[band][segment][plot_dict['proj_on_orb']]
                    ax.plot(qrt[0] * eps_nuk[band][segment:segment+2,0], qrt[1] * eps_nuk[band][segment:segment+2,1], '-',
                            solid_capstyle='round', c=eval('cm.'+plot_dict['colorscheme_kslice'])(1.0), lw=1., zorder=1.)

    _setup_plot_kslice(ax, tb_data['k_points'], w_dict)

    return ax

