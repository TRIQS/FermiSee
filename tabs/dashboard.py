import numpy as np
from itertools import permutations
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import dash_daq as daq
from dash import dash_table

from tabs.id_factory import id_factory

# layout


def make_dashboard(tb_data, tb_kslice_data, akw_data, ak0_data, sigma_data, loaded_data, tab_number):
    id = id_factory(f'tab{tab_number}')
    col_part = '#F8F9F9'
    button_style = {'margin': '5px', 'padding': '0px 5px 0px 3px', 'text-transform': 'none'}
    section_button_style = {'font-size': '15px', 'width': '100%', 'display': 'inline-block', 'margin-bottom': '10px',
                            'height': '37px', 'verticalAlign': 'center', 'textAlign': 'center', 'text-transform': 'none'}
    section_box_style = {'backgroundColor': col_part,
                         'borderRadius': '10px',
                         'padding': '10px',
                         'margin-top': "10px",
                         'margin-bottom': "10px"}
    tb_input_style = {'width': '90%',
                             'height': '37px',
                             'lineHeight': '37px',
                             'borderWidth': '1px',
                             'borderStyle': 'dashed',
                             'borderRadius': '5px',
                             'textAlign': 'center',
                             'margin': '10px'}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Section 1: Includes the upload and download buttons ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    upload_dwnload = html.Div(children=[
                    dbc.Alert('file corrupt or no file', id=id('config-alert'), dismissable=True,
                              color='warning', fade=False, is_open=False, duration=3000),
                    dcc.Upload(
                        id=id('upload-file'),
                        children=html.Div([html.A('Upload config')], style={'font-size': '20px'}),
                        style={'font-size': '15px', 'width': '100%', 'display': 'inline-block', 'margin-bottom': '10px',
                               'height': '37px', 'verticalAlign': 'center', 'textAlign': 'center', 'text-transform': 'none', 'borderStyle': 'dashed', 'borderWidth': '1px', 'borderRadius': '5px', 'lineHeight': '37px'},
                        multiple=False
                    ),
                    html.Button("Download config", id=id('dwn_button'),
                                style=section_button_style),
                    dcc.Download(id=id('download_h5')),
                ], style=section_box_style)

    loading_component = dcc.Loading(id=id("loading"),
                                    children=[html.Div(id=id("loading-tb")),
                                              html.Div(id=id("loading-plot")),
                                              html.Div(id=id("loading-akw"))
                                              ], color='#7E588A', type="cube", fullscreen=True, style={'backgroundColor': 'transparent'})

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Section 2: Tight binding hamilatonian ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Selection between uploading w90 or a pythTB JSON file
    w90_vs_pythTB = dcc.RadioItems(
                                id=id('choose-TB-method'),
                                options=[{'label': i, 'value': i} for i in ['wannier-90', 'pythTB']],
                                value='wannier-90',
                                inputStyle={"margin-right": "5px"},
                                labelStyle={'display': 'inline-block', 'margin-left': '5px'}
                            )
        #w90hr and w90wout file upload buttons
    w90_input = html.Div( id=id('w90-buttons'), children=[
                                html.Div([dcc.Upload(
                                    id=id('upload-w90-hr'),
                                    children=html.A('w90_hr'),
                                    style=tb_input_style,
                                    multiple=False)], style={'width': '49%', 'display': 'inline-block'}),
                                html.Div([dcc.Upload(
                                    id=id('upload-w90-wout'),
                                    children=html.A('w90_wout'),
                                    style=tb_input_style,
                                    multiple=False)], style={'width': '49%', 'display': 'inline-block'})
                            ])
        #pythTB JSON upload button
    pythTB_input = html.Div(id=id('pythTB-button'), children=[
                                dcc.Upload(
                                id=id('upload-pythTB-json'),
                                children=html.A('pythTB .json'),
                                style=tb_input_style,
                                multiple=False)], style={'display': 'none'})
        #add spin
        #Shonel: what does this do?
    add_spin = html.Div([html.P('add spin:', style={'width': '130px', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}),
                         daq.BooleanSwitch(
                            id=id('add-spin'),
                            on=False,
                            color='#005eb0',
                            style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                            ),
                        ],
                        # style={'padding': '5px 5px'}
                        style={'display': 'None'}
                       )
        #this is the section that calculates the chemical potential based on the number of electrons in the system
    electron_section = html.Div([
                                html.P('#electrons: ', style={'width': '50%', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'center'}
                                       ),
                                dcc.Input(id=id('gf-filling'), type='number', value='0.', step='0.001',
                                          debounce=True, placeholder='number of electrons', style={'width': '50%'}),
                                html.P('μ (eV):', style={'width': '50%', 'display': 'inline-block',
                                                    'text-align': 'left',
                                                    'vertical-align':
                                                    'center'}),
                                html.Div(id=id('dft-mu'), children=html.P('0'),
                                style={'width': '50%',
                                       'display':'inline-block',
                                       'margin': '5px 0px',
                                       'padding': '5px 0px 0px 10px',
                                       'background': 'white',
                                       'border-radius':'5px',
                                      }),
                                ],
                                style={'padding': '5px 5px'})

    electron_no_edits = html.Div([
                                html.P('#electrons: ', style={'width': '50%', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'center'}
                                       ),
                                html.Div(id=id('gf-filling'), children=html.P('0'),
                                         style={'width': '50%',
                                       'display':'inline-block',
                                       'margin': '5px 0px',
                                       'padding': '5px 5px 0px 10px',
                                       'background': 'white',
                                       'border-radius':'5px'
                                               }),
                                html.P('μ (eV):', style={'width': '50%', 'display': 'inline-block',
                                                    'text-align': 'left',
                                                    'vertical-align':
                                                    'center'}),
                                html.Div(id=id('dft-mu'), children=html.P('0'),
                                style={'width': '50%',
                                       'display':'inline-block',
                                       'margin': '5px 0px',
                                       'padding': '5px 0px 0px 10px',
                                       'background': 'white',
                                       'border-radius':'5px',
                                      }),
                                ],
                                style={'padding': '5px 5px'} )

    # k table: table for the k points
    ktable = html.Div([
                                dash_table.DataTable(
                                    id=id('k-points'),
                                    columns=[{
                                        'name': k,
                                        'id': id('column-{}'.format(i)),
                                        'deletable': False,
                                    } for i, k in enumerate(['label', 'kx', 'ky', 'kz'])],
                                    data=[
                                        {id('column-{}'.format(i)): k for i, k in enumerate(['G', 0, 0, 0])},
                                        {id('column-{}'.format(i)): k for i, k in enumerate(['X', 0.5, 0.0, 0])},
                                        {id('column-{}'.format(i)): k for i, k in enumerate(['M', 0.5, 0.5, 0])}],
                                    editable=True,
                                    row_deletable=True
                                )
                            ],
                            id=id('k-table')
                            )
        #k table for tab2 fermi slice (fslice)
    fslice_ktable = html.Div([
                                dash_table.DataTable(
                                    id=id('fslice-k-points'),
                                    columns=[{
                                        'name': k,
                                        'id': id('column-{}'.format(i)),
                                        'deletable': False,
                                    } for i, k in enumerate(['label', 'kx', 'ky', 'kz'])],
                                    data=[
                                        {id('column-{}'.format(i)): k for i, k in enumerate([ 'origin', 0, 0, 0])},
                                        {id('column-{}'.format(i)): k for i, k in enumerate([ 'lower right', 0.5, 0.0, 0])},
                                        {id('column-{}'.format(i)): k for i, k in enumerate([ 'upper left', 0, 0.5, 0])},
                                        # {id('column-{}'.format(i)): k for i, k in enumerate([ 'Z', 0, 0, 0.5])},
                                    ],
                                    editable=True,
                                    row_deletable=False
                                ),
                            ],
                            id=id('fslice-k-table'),
                            )
        # button to add k points to the k table
    add_kpoint = html.Button('Add k-point', id=id('add-kpoint'), n_clicks=0, style=button_style)
        #input the number of k-points
    num_kpoint = html.Div([
                                html.P('#k-points:', style={'width': '50%', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'center'}
                                       ),
                                dcc.Input(id=id('n-k'), value='20', step=1, placeholder='number of k-points',
                                          type='number', debounce=True, style={'width': '50%'}),
                            ], style={'padding': '5px 5px'}
                            )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Section 3: Self-Energy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # radio select option between uploading an h5 archive for the self energy or manually inputting the function values
    upload_vs_input = dcc.RadioItems(
                                id=id('choose-sigma'),
                                options=[{'label': i, 'value': i} for i in ['upload', 'enter manually']],
                                value='upload',
                                inputStyle={"margin-right": "5px"},
                                labelStyle={'display': 'inline-block', 'margin-left': '5px'}
                            )
    # the sigma upload button
    sigma_upload = html.Div(id=id('sigma-upload'), children=[
                                dcc.Upload(
                                    id=id('sigma-upload-box'),
                                    children=html.Div(['drop or ', html.A('select files')]),
                                    style={
                                        'width': '90%',
                                        'height': '40px',
                                        'lineHeight': '40px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                    },
                                    multiple=False
                                ),
                            ])
    #manually input sigma function
    sigma_function = html.Div(id=id('sigma-function'), children=[
                                # dcc.Textarea(
                                # id=id('sigma-function-input'),
                                # placeholder='Enter a python function',
                                # value='def sigma(w, Z, A): return (1-1/Z)*w - 1j*A*w**2',
                                # style={'width': '100%', 'height': '80px'}
                                # ),
                                html.P('Σ(ω, Z, Σ₀, η) = (1-1/Z)*ω - iη + Σ₀ ',
                                       style={'width': '240px', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                       ),
                                dash_table.DataTable(
                                    id=id('sigma-params'),
                                    columns=[{
                                        'name': k,
                                        'id': id('column-{}'.format(i)),
                                        'deletable': False,
                                    } for i, k in enumerate(['param', 'orb1', 'orb2', 'orb3'])],
                                    data=[
                                        {id('column-{}'.format(i)): k for i, k in enumerate(['Z', 1.0, 1.0, 1.0])},
                                        {id('column-{}'.format(i)): k for i, k in enumerate(['Sigma0', 0.0, 0.0, 0.0])}],
                                    editable=True,
                                ),
                                html.Button('Submit', id=id('sigma-function-button'), n_clicks=0, style=button_style),
                                html.Div(id=id('sigma-function-output'), style={'whiteSpace': 'pre-line'}),
                            ])
    # input value for ata η
    input_ata = html.Div([
                                html.P('η (eV):', style={'width': '50%', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'center'}
                                       ),
                                dcc.Input(id=id('eta'), type='number', value='0.010', step='0.001',
                                          placeholder='broadening',
                                          style={'width': '50%'}),
                            ], style={'padding': '5px 5px'}
                            )
    # select orbital order
    orbital_order = html.Div(
                                dcc.Dropdown(id=id('orbital-order'), value='(0,1,2)', placeholder='Select orbital order',
                                             options=[{'label': str(k), 'value': str(k)} for i, k in enumerate(list(permutations([0, 1, 2])))],
                                             style={'width': '100%'}), id=id('orbital-order-tooltip')
                            )
    akw_alert_tab2 = html.Div(dbc.Alert('Calculate A(k,w) in tab 1 first', id=id('akw-alert'), dismissable=True,
                                      color='warning', fade=True, is_open=False, duration=3000)
                               )
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Construct Dashboard  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Tab 2 Fermi Surface:
    if tab_number == 2:
        #does not include section 1
        upload_dwnload = html.Div(style = {'display' : 'None'})
        #in section two does not include the inputs, different k table
        sec2_components = [add_spin,
                           electron_no_edits,
                           html.Div('k-points'),
                           fslice_ktable,
                           num_kpoint
                          ]
        sec3_components = [ dbc.Alert('Complete TB section first and calc mu', id=id('tb-alert'), dismissable=True,
                                      color='warning', fade=True, is_open=False, duration=3000),
                            dbc.Alert('# of orbitals does not match (Σ vs. H(r))', id=id('orb-alert'), dismissable=True,
                                      color='warning', fade=True, is_open=False, duration=3000),
                           akw_alert_tab2,
                            html.Button('Calculate A(k,w)', id=id('calc-akw'), n_clicks=0, style=button_style)]
    else:
        #the default dashboard will be the tab 1 dashboard
        sec2_components = [w90_vs_pythTB,
                           w90_input,
                           pythTB_input,
                           add_spin,
                           electron_section,
                           html.Div('k-points'),
                           ktable,
                           add_kpoint,
                           num_kpoint
                          ]

        sec3_components = [ upload_vs_input,
                            sigma_upload,
                            sigma_function,
                            input_ata,
                            orbital_order,
                            dbc.Tooltip('Select orbital order of Σ with respect to W90 input Hamiltonian',
                                        target=id('orbital-order-tooltip'),
                                        style={'maxWidth': 300, 'width': 300, 'font-size': 14}),
                            dbc.Alert('Complete TB section first and calc mu', id=id('tb-alert'), dismissable=True,
                                      color='warning', fade=True, is_open=False, duration=3000),
                            dbc.Alert('# of orbitals does not match (Σ vs. H(r))', id=id('orb-alert'), dismissable=True,
                                      color='warning', fade=True, is_open=False, duration=3000),
                            html.Button('Calculate A(k,w)', id=id('calc-akw'), n_clicks=0, style=button_style)
                          ]

    #this is here to make the dashboard invisible for the tabs that are under construction
    section_style = section_box_style
    if tab_number == 3:
        section_style = {'display' :' None'}
        upload_dwnload = html.Div(style = {'display' : 'None'})


    return dcc.Tab(
        label='spectral function A(k,ω)',
        children=[
            html.Div([
                # section 1
                upload_dwnload,
                loading_component,
                # section 2
                html.Div(children=[
                    html.Button(
                        "TB Hamiltonian",
                        id=id('sec2-collapse-button'),
                        style=section_button_style,
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        # html div for collapse first:
                        html.Div(
                            # now body of collapse:
                            children = sec2_components
                        ),
                        id=id('sec2-collapse'),
                        is_open=True,
                    ),
                ], style=section_style),

                # section 3
                html.Div(children=[
                    html.Button(
                        "Self-energy",
                        id=id('sec3-collapse-button'),
                        style=section_button_style,
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        # html div for collapse first:
                        html.Div(children=sec3_components),
                        id=id('sec3-collapse'),
                        is_open=False,
                    ),
                ], style=section_style),

                # section 4
                html.Div(children=[
                    html.Button(
                        "Options",
                        id=id('sec4-collapse-button'),
                        style=section_button_style,
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        # html div for collapse first:
                        html.Div(children=[
                            # now body of collapse:
                            html.Div([
                                html.P('select orbitals: ', style={'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'center', 'width': '50%'}
                                       ),
                                dcc.Input(id=id('select-orbitals'),
                                          type='text',
                                          value='',
                                          debounce=True,
                                          placeholder='Ex: 0,1,2',
                                          style={'width': '50%'}),
                            ],id=id('input-select-orbital'), style={'display':'none'}),
                            html.Div([
                                html.P('show TB bands:', style={'width': '130px', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                       ),
                                dcc.Slider(0,2,step=None,
                                           marks={
                                               0 : 'none',
                                               1 : 'bands',
                                               2 : 'projection'
                                           },
                                           value=1,
                                           id=id('band-slider')
                                          )
                            ],
                            id=id('input-band-slider'), style={'padding': '5px 1px'}
                            ),
                            html.Div([
                                html.P('show A(k,ω):', style={'width': '130px', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                       ),
                                dcc.Slider(0,2,step=None,
                                           marks={
                                               0 : 'none',
                                               1 : 'A(k,ω)',
                                               2 : 'projection'
                                           },
                                           value=0,
                                           id=id('akw-slider')
                                          )
                            ],
                            id=id('input-akw-slider'), style={'padding': '5px 1px'}
                            ),
                            html.Div([
                                html.P('show A(ω):', style={'width': '130px', 'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                                       ),
                                daq.BooleanSwitch(
                                    id=id('sum-edc'),
                                    on=False,
                                    color='#005eb0',
                                    style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                                ),
                            ], style={'padding': '5px 5px'}
                            ),
                            dcc.RadioItems(
                                id=id('akw-mode'),
                                options=[{'label': i, 'value': i} for i in ['A(k,ω)', 'QP dispersion']],
                                value='A(k,ω)',
                                inputStyle={"margin-right": "5px"},
                                labelStyle={'display': 'inline-block', 'margin-left': '5px'},
                                style={'padding': '0px 0px 10px 0px'}
                            ),
                            html.Div('Colorscale:'),
                            dcc.RadioItems(
                                id=id('colorscale-mode'),
                                options=[{'label': i, 'value': i} for i in ['sequential', 'diverging']],
                                value='diverging',
                                inputStyle={"margin-right": "5px"},
                                labelStyle={'display': 'inline-block', 'margin-left': '5px'},
                                style={'padding': '0px 0px 10px 0px'}
                            ),
                            dcc.Dropdown(
                                id=id('colorscale'),
                                value='Tealrose',
                                placeholder='Select colorscale'
                            ),
                        ]),
                        id=id('sec4-collapse'),
                        is_open=False,
                    ),
                ], style=section_style),

                # support sec
                html.Div(children=[
                    html.P('for help and tutorials visit:'),
                    html.A('github.com/TRIQS/FermiSee', href='https://github.com/TRIQS/FermiSee', target='_blank'),
                ], style=section_box_style),

                dcc.Store(id=id('tb-data'), data=tb_data),
                dcc.Store(id=id('tb-kslice-data'), data=tb_kslice_data),
                dcc.Store(id=id('akw-data'), data=akw_data),
                dcc.Store(id=id('ak0-data'), data=ak0_data),
                dcc.Store(id=id('sigma-data'), data=sigma_data),
                dcc.Store(id=id('loaded-data'), data=loaded_data),
                dcc.Store(id=id('data-name')),
            ], style={
                'padding-left': '1%',
                'padding-right': '1%',
                'display': 'inline-block',
                'width': '14%',
                'min-width': '260px',
                'vertical-align': 'top'
            }
            ),
        ]
    )
