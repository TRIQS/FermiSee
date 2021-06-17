import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

from tabs.id_factory import id_factory

# layout
def make_dashboard(data, tab_number):
    id = id_factory(f'tab{tab_number}')
    col_part = '#F8F9F9'
    return dcc.Tab(
        label='spectral function A(k,ω)',
        children=[
            # column 1
            html.Div([
                    #html.Hr(),
                    html.H3('Data input'),
                    html.Div(children=[
                        html.H5('TB Hamiltonian'),
                        dcc.Upload(
                            id=id('upload-file'),
                            children=html.Div(['drop or ', html.A('select files')]),
                            style={
                                'width': '90%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        html.Div([
                            html.P('show TB bands:',style={'width' : '130px','display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                ),
                            daq.BooleanSwitch(
                                id=id('tb-bands'),
                                on=True,
                                color='#005eb0',
                                style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                            ),
                        ], style={'padding': '5px 5px'}
                        ),
                    ], style={'backgroundColor': col_part,
                               'borderRadius': '15px',
                               'padding': '10px'}),
                    html.Hr(),
                    html.Div(children=[
                        html.H5('self-energy Σ(k,ω)'),
                        html.Div('Choose Σ(k,ω):'),
                        dcc.RadioItems(
                            id=id('choose-sigma'),
                            options=[{'label': i, 'value': i} for i in ['upload', 'enter manually']],
                            value='upload',
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.Div(id=id('sigma-upload'), children=[
                            dcc.Upload(
                                id=id('sigma-upload-box'),
                                children=html.Div(['drop or ', html.A('select files')]),
                                style={
                                    'width': '90%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                        ]),
                        html.Div(id=id('sigma-function'), children=[
                            dcc.Textarea(
                                id=id('sigma-function-input'),
                                #value='Enter a python function for Σ(k,ω):\n\ndef sigma(ω, a, b):\n\treturn a + b * ω',
                                placeholder='Enter a python function',
                                value='def sigma(w): return -1 * w',
                                style={'width': '100%', 'height': '80px'}
                                ),
                            html.Button('Submit', id=id('sigma-function-button'), n_clicks=0),
                            html.Div(id=id('sigma-function-output'), style={'whiteSpace': 'pre-line'}),
                        ]),
                    ], style={'backgroundColor': col_part,
                               'borderRadius': '15px',
                               'padding': '10px'}),
                    html.Hr(),
                    html.Div(children=[
                        html.H5('spectral function A(k,ω)'),
                        html.Div([
                            html.P('show A(k,ω):',style={'width' : '130px','display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                ),
                            daq.BooleanSwitch(
                                id=id('akw'),
                                on=True,
                                color='#005eb0',
                                style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                            ),
                        ], style={'padding': '5px 5px'}
                        ),
                        html.Div('Colorscale:'),
                        dcc.RadioItems(
                            id=id('colorscale-mode'),
                            options=[{'label': i, 'value': i} for i in ['sequential', 'diverging']],
                            value='diverging',
                            labelStyle={'display': 'inline-block'}
                        ),
                        dcc.Dropdown(
                            id=id('colorscale'),
                            value='Tealrose',
                            placeholder='Select colorscale'
                        ),
                    ], style={'backgroundColor': col_part,
                               'borderRadius': '15px',
                               'padding': '10px'}),
                    #html.Hr(),
                    dcc.Store(id=id('data-storage'), data = data),
            ], style={
                'padding-left': '1%',
                'padding-right': '1%',
                'display': 'inline-block',
                'width': '14%',
                'vertical-align': 'top'
                }
            ),
            ]
        )
