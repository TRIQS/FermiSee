import numpy as np
from itertools import product
import dash
import dash_html_components as html
from flask import send_file
import plotly.express as px
import plotly.graph_objects as go
import ast
from itertools import permutations
import inspect
import base64
from dash_extensions.snippets import send_bytes
from dash.dependencies import Input, Output, State, ALL

from h5 import HDFArchive
from triqs.gf import MeshReFreq

from load_data import load_config, load_w90_hr, load_w90_wout, load_sigma_h5
from tabs.id_factory import id_factory


def register_callbacks(app):
    id = id_factory('tab2')
    id_tap = id_factory('tab1')

    # upload akw data
    @app.callback(
        Output(id('Ak0'), 'figure'),
        [Input(id('tb-bands'), 'on'),
         Input(id('akw-bands'), 'on'),
         Input(id('colorscale'), 'value'),
         Input(id_tap('tb-data'), 'data'),
         Input(id('tb-kslice-data'), 'data'),
         Input(id_tap('akw-data'), 'data'),
         Input(id_tap('sigma-data'), 'data')],
         prevent_initial_call=True)
    def plot_ak0(tb_bands, akw, colorscale, tb_data, tb_kslice_data, akw_data, sigma_data):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***update_akw***:'), trigger_id)

        for key in tb_data.keys():
            if key not in ['k_mesh', 'e_mat', 'eps_nuk', 'evecs_re', 'evecs_im', 'bnd_low', 'bnd_high']:
                tb_kslice_data[key] = tb_data[key]
        print(tb_kslice_data.keys())
        
        # initialize general figure environment
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', rangeslider_visible=False, 
                         showticklabels=True, spikedash='solid')
        fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showticklabels=True, spikedash='solid')
        fig.update_traces(xaxis='x', hoverinfo='none')

        # decide which data to show for TB
        if tb_data['use']: tb_temp = tb_data
        print('tb_temp' in locals())
        if not 'tb_temp' in locals():
            return fig

        k_mesh = tb_temp['k_mesh']
        print(k_mesh)
        if tb_bands:
            for band in range(len(tb_temp['eps_nuk'])):
                fig.add_trace(go.Scattergl(x=k_mesh['k_disc'], y=tb_temp['eps_nuk'][band], mode='lines',
                            line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=False, text=f'tb band {band}',
                            hoverinfo='x+y+text'
                            ))

        return fig
