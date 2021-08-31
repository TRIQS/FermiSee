import numpy as np
from itertools import product
import dash
import dash_html_components as html
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
from tabs.id_factory import id_factory


def register_callbacks(app):
    id = id_factory('tab2')
    id_tap = id_factory('tab1')

    # dashboard calculate TB
    @app.callback(
        [Output(id('tb-kslice-data'), 'data')],
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

            tb_kslice_data['k_mesh'], e_mat, e_vecs, tbl = tb.calc_tb_bands(tb_kslice_data, add_spin, float(dft_mu), add_local, k_mesh, fermi_slice=True)
            # calculate Hamiltonian
            tb_kslice_data['e_mat'] = e_mat.real.tolist()
            tb_kslice_data['eps_nuk'], evec_nuk = tb.get_tb_kslice(tbl, k_mesh, dft_mu)
            tb_kslice_data['use'] = True

        return [tb_kslice_data]

    # upload akw data
    @app.callback(
        Output(id('Ak0'), 'figure'),
        [Input(id('tb-bands'), 'on'),
         Input(id('akw-bands'), 'on'),
         Input(id('colorscale'), 'value'),
         Input(id('tb-kslice-data'), 'data'),
         Input(id_tap('akw-data'), 'data'),
         Input(id_tap('sigma-data'), 'data')],
         prevent_initial_call=True)
    def plot_ak0(tb_switch, akw, colorscale, tb_kslice_data, akw_data, sigma_data):
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

        if not tb_kslice_data['use']:
            return fig
    
        k_mesh = tb_kslice_data['k_mesh']
        if tb_switch:
            sign = [1,-1]
            quarter = 0
            quarters = 2* np.array([sign,sign])
            eps_nuk = {int(key): np.array(value) for key, value in tb_kslice_data['eps_nuk'].items()}
            for qrt in list(product(*quarters))[quarter:quarter+1]:
                for band in range(len(eps_nuk)):
                    for segment in range(eps_nuk[band].shape[0]):
                        #orbital_projected = evec_nuk[band][segment][plot_dict['proj_on_orb']]
                        fig.add_trace(go.Scattergl(x=qrt[0] * eps_nuk[band][segment:segment+2,0], y=qrt[1] * eps_nuk[band][segment:segment+2,1],
                                                   mode='lines', line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=False,
                                                   text=f'tb band {band}', hoverinfo='x+y+text'))

        return fig
