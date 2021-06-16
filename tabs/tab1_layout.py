import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

# layout
def layout(data):
    return dcc.Tab(
        label='spectral function A(k,ω)',
        children=[
            # column 1
            html.Div([
                    html.Hr(),
                    html.H4('TB Hamiltonian'),
                    dcc.Upload(
                        id='upload-file',
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
                            id='tb-bands',
                            on=True,
                            color='#005eb0',
                            style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                        ),
                    ], style={'padding': '5px 5px'}
                    ),
                    html.Hr(),
                    html.H4('self-energy Σ(k,ω)'),
                    html.Div('Choose Σ(k,ω):'),
                    dcc.RadioItems(
                        id='choose-sigma',
                        options=[{'label': i, 'value': i} for i in ['upload', 'enter manually']],
                        value='upload',
                        labelStyle={'display': 'inline-block'}
                    ),
                    html.Div(id='sigma-upload', children=[
                        dcc.Upload(
                            id='sigma-upload-box',
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
                    html.Div(id='sigma-function', children=[
                        dcc.Textarea(
                            id='sigma-function-input',
                            #value='Enter a python function for Σ(k,ω):\n\ndef sigma(ω, a, b):\n\treturn a + b * ω',
                            placeholder='Enter a python function',
                            value='def sigma(w, a, b): return a + b * w',
                            style={'width': '100%', 'height': 300}
                            ),
                        html.Button('Submit', id='sigma-function-button', n_clicks=0),
                        html.Div(id='sigma-function-output', style={'whiteSpace': 'pre-line'}),
                    ]),
                    html.Hr(),
                    html.H4('spectral function A(k,ω)'),
                    html.Div([
                        html.P('show A(k,ω):',style={'width' : '130px','display': 'inline-block', 'text-align': 'left', 'vertical-align': 'top'}
                            ),
                        daq.BooleanSwitch(
                            id='akw',
                            on=True,
                            color='#005eb0',
                            style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle'}
                        ),
                    ], style={'padding': '5px 5px'}
                    ),
                    dcc.Dropdown(
                        id='choose-plot',
                        options=[{'label': 'A(k,ω)', 'value': 3}],
                        value='Fertility rate, total (births per woman)'
                    ),
                    html.Hr(),
                    dcc.Store(id="data-storage", data = data),
            ], style={
                'borderBottom': 'thin lightgrey solid',
                'padding-left': '1%',
                'padding-right': '1%',
                'display': 'inline-block',
                'width': '14%',
                'vertical-align': 'top'
                }
            ),

            # column 2
            html.Div([
                dcc.Graph(
                    id='Akw',
                    style={'height': '84vh'},
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
                    dcc.Graph(
                        id='EDC',
                        style={'height': '95%'}
                       ),
                    dcc.Slider(
                        id='kpt_edc',
                        min=0,
                        max=len(data['k_mesh'])-1,
                        value=0,
                        #marks={str(year): str(year) for year in df['Year'].unique()},
                        step=1,
                        #handleLabel={'showCurrentValue': True, 'label': 'value'},
                        updatemode='drag',
                        ),
                ], style={
                    'padding-right': '1%',
                    'width': '99%',
                    'padding-top': '3%',
                    'height': '40vh',
                    'vertical-align': 'top',
                    'borderBottom': 'thin lightgrey solid'
                    }
                ),
                html.Div([
                    dcc.Graph(
                        id='MDC',
                        style={'height': '95%'}
                       ),
                    dcc.Slider(
                        id='w_mdc',
                        min=0,
                        max=len(data['freq_mesh'])-1,
                        value=np.argmin(np.abs(np.array(data['freq_mesh']))),
                        #marks={str(year): str(year) for year in df['Year'].unique()},
                        step=1,
                        updatemode='drag',
                        verticalHeight=100),
                ], style={
                    'padding-right': '1%',
                    'padding-top': '3%',
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
