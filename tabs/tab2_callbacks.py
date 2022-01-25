import numpy as np
from itertools import product
import dash
from dash import html
from flask import send_file
import plotly.express as px
import plotly.graph_objects as go
import ast
from itertools import permutations, product
import inspect
import base64
from dash_extensions.snippets import send_bytes
from dash.dependencies import Input, Output, State, ALL

from h5 import HDFArchive
from triqs.gf import MeshReFreq

from load_data import load_config, load_w90_hr, load_w90_wout, load_sigma_h5
import tools.calc_tb as tb
import tools.calc_akw as akw
from tabs.id_factory import id_factory


def register_callbacks(app):
    id = id_factory('tab2')
    id_tap = id_factory('tab1')

    # dashboard calculate TB
    @app.callback(
        [Output(id('tb-kslice-data'), 'data'),
         Output(id('tb-bands'), 'on')],
        [Input(id('tb-bands'), 'on'),
         Input(id('calc-tb'), 'n_clicks'),
         Input(id('add-spin'), 'value'),
         Input(id_tap('dft-mu'), 'value'),
         Input(id('n-k'), 'value'),
         Input(id('k-points'), 'data'),
         Input(id('tb-kslice-data'), 'data'),
         Input(id_tap('tb-data'), 'data')],
         prevent_initial_call=True,)
    def calc_tb(tb_switch, click_tb, add_spin, dft_mu, n_k, k_points, tb_kslice_data, tb_data):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***calc_tb***:'), trigger_id)

        if trigger_id == id('calc-tb'):
        ## if not used before, copy data from tb_data
        #if tb_kslice_data['use'] != tb_data['use']:

            for key in tb_data.keys():
                if key not in ['k_mesh', 'k_disc', 'e_mat', 'eps_nuk', 'evecs_re', 'evecs_im', 'bnd_low', 'bnd_high']:
                    tb_kslice_data[key] = tb_data[key]

            kz = 0.
            k_mesh = {'n_k': int(n_k), 'k_path': k_points, 'kz': kz}
            k_mesh['Z'] = np.array([+0.25, +0.25, -0.25])
            add_local = [0.] * tb_kslice_data['n_wf']

            tb_kslice_data['k_mesh'], e_mat, e_vecs, tbl = tb.calc_tb_bands(tb_kslice_data, add_spin, add_local, k_mesh, fermi_slice=True)
            # calculate Hamiltonian
            tb_kslice_data['e_mat_re'] = e_mat.real.tolist()
            tb_kslice_data['e_mat_im'] = e_mat.imag.tolist()
            tb_kslice_data['eps_nuk'], evec_nuk = tb.get_tb_kslice(tbl, k_mesh, dft_mu)
            tb_kslice_data['use'] = True

            tb_switch = {'on': True}

        return tb_kslice_data, tb_switch

    # upload akw data
    @app.callback(
        [Output(id('ak0-data'), 'data'),
         Output(id('akw-bands'), 'on'),
         Output(id('tb-alert'), 'is_open')],
        [Input(id('ak0-data'), 'data'),
         Input(id('tb-kslice-data'), 'data'),
         Input(id_tap('sigma-data'), 'data'),
         Input(id_tap('akw-data'), 'data'),
         Input(id('akw-bands'), 'on'),
         Input(id_tap('dft-mu'), 'value'),
         Input(id('k-points'), 'data'),
         Input(id('n-k'), 'value'),
         Input(id('calc-akw'), 'n_clicks'),
         Input(id('calc-tb'), 'n_clicks'),
         Input(id('akw-mode'), 'value'),
         Input(id('band-basis'), 'on')],
         State(id('tb-alert'), 'is_open'),
         prevent_initial_call=True
        )
    def update_ak0(ak0_data, tb_kslice_data, sigma_data, akw_data, akw_switch, dft_mu, k_points, n_k, click_tb, click_akw, akw_mode, band_basis, tb_alert):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***update_ak0***:'), trigger_id)

        if trigger_id == id('dft-mu') and not sigma_data['use']:
            return ak0_data, akw_switch, tb_alert

        elif trigger_id in (id('calc-akw'), id('n-k'), id('akw-mode')) or ( trigger_id == id('k-points') and click_akw > 0 ):
            if not sigma_data['use'] or not tb_kslice_data['use']:
                return ak0_data, akw_switch, not tb_alert

            solve = True if akw_mode == 'QP dispersion' else False
            ak0_data['dmft_mu'] = akw_data['dmft_mu']
            ak0_data['eta'] = 0.01
            ak0, ak0_data['dmft_mu'] = akw.calc_kslice(tb_kslice_data, sigma_data, ak0_data, solve, band_basis)
            ak0_data['Akw'] = ak0.tolist()
            ak0_data['use'] = True
            ak0_data['solve'] = solve

            akw_switch = {'on': True}

        return ak0_data, akw_switch, tb_alert

    # dashboard colors
    @app.callback(
        Output(id('colorscale'), 'options'),
        Input(id('colorscale-mode'), 'value'),
        prevent_initial_call=False,
        )
    def update_colorscales(mode):
        colorscales = [name for name, body in inspect.getmembers(getattr(px.colors, mode))
                if isinstance(body, list) and len(name.rsplit('_')) == 1]
        return [{'label': key, 'value': key} for key in colorscales]

    # upload akw data
    @app.callback(
        Output(id('Ak0'), 'figure'),
        [Input(id('tb-bands'), 'on'),
         Input(id('akw-bands'), 'on'),
         Input(id('colorscale'), 'value'),
         Input(id('tb-kslice-data'), 'data'),
         Input(id('ak0-data'), 'data'),
         Input(id_tap('sigma-data'), 'data')],
         prevent_initial_call=True)
    def plot_ak0(tb_switch, akw_switch, colorscale, tb_kslice_data, ak0_data, sigma_data):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***update_ak0***:'), trigger_id)

        # initialize general figure environment
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', rangeslider_visible=False,
                         showticklabels=True, spikedash='solid')
        fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showticklabels=True, spikedash='solid')
        fig.update_traces(xaxis='x', hoverinfo='none')
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40}, clickmode='event+select', hovermode='closest',
                          xaxis_range=[0, 1], yaxis_range=[0, 1], xaxis_title='kxa/π', yaxis_title='kya/π', font=dict(size=16))

        if not tb_kslice_data['use']:
            return fig

        sign = [1,-1]
        quarter = 0
        quarters = np.array([sign,sign])
        k_mesh = tb_kslice_data['k_mesh']
        if tb_switch:
            quarters *= 2
            eps_nuk = {int(key): np.array(value) for key, value in tb_kslice_data['eps_nuk'].items()}
            for qrt in list(product(*quarters))[quarter:quarter+1]:
                for band in range(len(eps_nuk)):
                    for segment in range(eps_nuk[band].shape[0]):
                        #orbital_projected = evec_nuk[band][segment][plot_dict['proj_on_orb']]
                        fig.add_trace(go.Scattergl(x=qrt[0] * eps_nuk[band][segment:segment+2,0], y=qrt[1] * eps_nuk[band][segment:segment+2,1],
                                                   mode='lines', line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=False,
                                                   text=f'tb band {band}', hoverinfo='x+y+text'))

        if not ak0_data['use']:
            return fig

        if akw_switch:
            n_kx, n_ky = np.array(tb_kslice_data['e_mat_re']).shape[2:4]
            ak0 = np.array(ak0_data['Akw'])
            kx = np.linspace(0, 1, ak0.shape[0])
            ky = np.linspace(0, 1, ak0.shape[1])
            for qrt in list(product(*quarters))[quarter:quarter+1]:
                if ak0_data['solve']:
                    for ik1 in range(len(ky)):
                        for orb in range(tb_kslice_data['n_wf']):
                            fig.add_trace(go.Scattergl(x=kx, y=ak0[:,ik1,orb].T, showlegend=False, mode='markers',
                                                       marker_color=px.colors.sequential.Viridis[0]))
                else:
                    fig.add_trace(go.Heatmap(x=kx, y=ky, z=ak0.T, colorscale=colorscale, reversescale=False, showscale=False, zmin=np.min(ak0), zmax=np.max(ak0)))

        return fig
