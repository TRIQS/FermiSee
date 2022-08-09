import numpy as np
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_daq as daq

from tabs.dashboard import make_dashboard
from tabs.id_factory import id_factory

id = id_factory('tab1')

# layout
def layout(tb_data, tb_kslice_data, akw_data, ak0_data, sigma_data, loaded_data):
    return dcc.Tab(
        label='spectral function A(k,ω)',
        children=[
            # column 1
            make_dashboard(tb_data, tb_kslice_data, akw_data, ak0_data, sigma_data, loaded_data, 1),
            # column 2
            html.Div([
                html.A(children=html.H3('A(k,ω)', id=id('Akw-title'), style={'textAlign': 'center'}),
                       id=id('download-csv'),
                       href="",
                       download="Akw_rawdata.csv",
                       target="_blank",
                       style={"color":"black", "text-decoration":"none"}),
                dbc.Tooltip("Click to download the data as CSV", target=id('Akw-title')),
                dcc.Graph(
                    id=id('Akw'),
                    style={'height': '84vh'},
                    config={'displaylogo': False, 'toImageButtonOptions': {'scale' : 1, 'width': 1618, 'height': 1000}},
                    clickData={'points': []}
                )
            ], style={
                'display': 'inline-block',
                'width': '41%',
                'padding-right': '1%',
                'vertical-align': 'top'
                }
            ),

            # column 3
            html.Div([
                html.Div([
                    html.H3('EDC', style={'textAlign': 'center'}),
                    dcc.Graph(
                        id='EDC',
                        style={'height': '100%'},
                        config={'displaylogo': False, 'toImageButtonOptions': {'scale' : 1, 'width': 3000, 'height': 1000}}
                       ),
                    dcc.Slider(
                        id='kpt_edc',
                        min=0,
                        #max=len(data['k_mesh'])-1,
                        value=0,
                        #marks={str(year): str(year) for year in df['Year'].unique()},
                        step=1,
                        marks=None,
                        #handleLabel={'showCurrentValue': True, 'label': 'value'},
                        updatemode='drag',
                        ),
                ], style={
                    'padding-right': '1%',
                    'width': '99%',
                    'padding-bottom': '5%',
                    'height': '40vh',
                    'vertical-align': 'top',
                    'borderBottom': 'thin lightgrey solid'
                    }
                ),
                html.Div([
                    html.H3('MDC', style={'textAlign': 'center', 'padding': '0%'}),
                    dcc.Graph(
                        id='MDC',
                        style={'height': '100%', 'padding': '0%'},
                        config={'displaylogo': False, 'toImageButtonOptions': {'scale' : 1, 'width': 3000, 'height': 1000}}
                       ),
                    dcc.Slider(
                        id='w_mdc',
                        min=0,
                        #max=len(data['freq_mesh'])-1,
                        #value=np.argmin(np.abs(np.array(data['freq_mesh']))),
                        #marks={str(year): str(year) for year in df['Year'].unique()},
                        step=1,
                        marks=None,
                        updatemode='drag',
                        ),
                ], style={
                    'padding-right': '1%',
                    'padding-top': '5%',
                    'width': '99%',
                    'height': '40vh'
                    }
                ),
            ], style={
                'display': 'inline-block',
                'width': '41%',
                'padding-right': '1%',
                'vertical-align': 'top'
                }
            )
            ]
        )
