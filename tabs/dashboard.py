import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_table

from tabs.id_factory import id_factory

# layout
def make_dashboard(data, tb_data, akw_data, sigma_data, tab_number):
    id = id_factory(f'tab{tab_number}')
    col_part = '#F8F9F9'
    return dcc.Tab(
        label='spectral function A(k,ω)',
        children=[
            html.Div([
                    html.H3('Data input'),
                    # section 1
                    html.Div(children=[
                        html.H5('Upload config'),
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
                    ], style={'backgroundColor': col_part,
                               'borderRadius': '15px',
                               'padding': '10px'}),
                    # section 2
                    html.Hr(),
                    html.Div(children=[
                        html.H5('TB Hamiltonian'),
                        html.Div([
                            html.Div([dcc.Upload(
                                id=id('upload-w90-hr'),
                                children=html.A('w90_hr'),
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
                                multiple=False)], style={'width': '49%', 'display': 'inline-block'}),
                            html.Div([dcc.Upload(
                                id=id('upload-w90-wout'),
                                children=html.A('w90_wout'),
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
                                multiple=False)], style={'width': '49%', 'display': 'inline-block'})
                        ]),
                        html.Div([
                            html.P('add spin:',style={'width' : '130px','display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                ),
                            daq.BooleanSwitch(
                                id=id('add-spin'),
                                on=False,
                                color='#005eb0',
                                style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                            ),
                        ], style={'padding': '5px 5px'}
                        ),
                        dcc.Input(id=id('dft-mu'),
                            type='number',
                            value='0.',
                            step='0.01',
                            placeholder='chemical potential μ'),
                        # html.Div('orbital order'),
                        html.Div([
                        dash_table.DataTable(
                            id=id('dft-orbital-order'), editable=True,
                            columns=[{'name': f'orbital order', 'id': id(f'oo-{i}'), 'clearable': False, 'presentation': 'dropdown'} for i in range(3)],
                            data=[{id(f'oo-{i}'): key for i, key in enumerate(['dxz', 'dyz', 'dxy'])}],
                            dropdown={id('oo-{}'.format(i)): {
                                'options': [{'label': key, 'value': key} for key in ['dxz', 'dyz', 'dxy']]
                                } for i in range(3)},
                            merge_duplicate_headers=True,
                            style_header= {'textAlign': 'left'}
                            )], style={'padding': '5px 5px'}),
                        html.Div('k-points'),
                        dash_table.DataTable(
                            id=id('k-points'),
                            columns=[{
                                'name': k,
                                'id' : id('column-{}'.format(i)),
                                'deletable': False,
                                } for i, k in enumerate(['label', 'kx', 'ky', 'kz'])],
                            data=[
                                {id('column-{}'.format(i)): k for i, k in enumerate(['G', 0,0,0])},
                                {id('column-{}'.format(i)): k for i, k in enumerate(['M', 0.5,0.5,0])}],
                            editable=True,
                            row_deletable=True
                            ),
                        html.Button('Add k-point', id=id('add-kpoint'), n_clicks=0),
                        html.Button('Calculate TB bands', id=id('calc-tb'), n_clicks=0),
                    ], style={'backgroundColor': col_part,
                               'borderRadius': '15px',
                               'padding': '10px'}),
                    # section 3
                    html.Hr(),
                    html.Div(children=[
                        html.H5('Self-energy'),
                        dcc.RadioItems(
                            id=id('choose-sigma'),
                            options=[{'label': i, 'value': i} for i in ['upload', 'enter manually']],
                            value='upload',
                            inputStyle={"margin-right": "5px"},
                            labelStyle={'display': 'inline-block', 'margin-left':'5px'}
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
                        dcc.Input(id=id('eta'),
                            type='number',
                            value='0.01',
                            step='0.005',
                            placeholder='broadening η',
                            style={'width': '80%','margin-bottom': '10px'}),
                        dbc.Alert('Complete TB section first.', id=id('tb-alert'), dismissable=True, 
                                  color='warning', fade=False, is_open=False),
                        html.Button('Calculate A(k,ω)', id=id('calc-akw'), n_clicks=0),
                    ], style={'backgroundColor': col_part,
                               'borderRadius': '15px',
                               'padding': '10px'}
                    ),
                    # section 4
                    html.Hr(),
                    html.Div(children=[
                        html.H5('Layout'),
                        html.Div([
                            html.P('show TB bands:',style={'width' : '130px','display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                ),
                            daq.BooleanSwitch(
                                id=id('tb-bands'),
                                on=False,
                                color='#005eb0',
                                style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                            ),
                        ], style={'padding': '5px 5px'}
                        ),
                        html.Div([
                            html.P('show A(k,ω):',style={'width' : '130px','display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                ),
                            daq.BooleanSwitch(
                                id=id('akw-bands'),
                                on=False,
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
                            inputStyle={"margin-right": "5px"},
                            labelStyle={'display': 'inline-block', 'margin-left':'5px'}
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
                    dcc.Store(id=id('tb-data'), data = tb_data),
                    dcc.Store(id=id('akw-data'), data = akw_data),
                    dcc.Store(id=id('full-data'), data = data),
                    dcc.Store(id=id('sigma-data'), data = sigma_data),
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
