import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from h5 import HDFArchive

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def _get_tb_bands(k_mesh, e_mat, k_points,):
    
    e_val = np.zeros((e_mat.shape[0], k_mesh.shape[0]), dtype=complex)
    e_vec = np.zeros(np.shape(e_mat), dtype=complex)
    for ik in range(np.shape(e_mat)[2]):
        e_val[:,ik], e_vec[:,:,ik] = np.linalg.eigh(e_mat[:,:,ik])

    return e_val, e_vec


def update_data(h5_file):
    data = {'file': h5_file}
    with HDFArchive(h5_file, 'r') as ar:
        # Akw data
        # data['Akw_data'] = ar['A_k_w_data'] # contains A_k_w, dft_mu
        data['Akw'] = ar['A_k_w_data']['A_k_w']
        data['dft_mu'] = ar['A_k_w_data']['dft_mu']

        # tb data
        # data['tb_data'] = ar['tb_data'] # e_mat, k_mesh, k_points, k_points_labels
        data['k_points_labels'] = ar['tb_data']['k_points_labels']
        e_mat = ar['tb_data']['e_mat']
        data['k_points'] = ar['tb_data']['k_points']
        data['k_mesh'] = ar['tb_data']['k_mesh']
        # w mesh
        data['freq_mesh'] = ar['w_mesh']['w_mesh']
    data['eps_nuk'], evec_nuk = _get_tb_bands(data['k_mesh'], e_mat, data['k_points'])

    # workaround to remove last point in k_mesh
    pts_per_kpt = int(len(data['k_mesh'])/(len(data['k_points'])-1))-1
    # remove last intervall except the first point to incl high symm point
    data['k_mesh'] = data['k_mesh'][:-pts_per_kpt]
    data['Akw'] = data['Akw'][:-pts_per_kpt,:]

    # transform np arrays to lists, to be able to serialize to json
    data['Akw'] = data['Akw'].tolist()
    data['freq_mesh'] = data['freq_mesh'].tolist()
    data['k_mesh'] = data['k_mesh'].tolist()
    data['k_points'] = data['k_points'][:-1].tolist()
    data['eps_nuk'] = (data['eps_nuk'].real - data['dft_mu']).tolist()

    # max value
    data['max_Akw'] = 1.05 * np.max(np.array(data['Akw']))

    return data

# init data
data = update_data('example.h5')

# layout
app.layout = html.Div([
    # column 1
    html.Div([
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
            dcc.RadioItems(
                id='choose-sigma',
                options=[{'label': i, 'value': i} for i in ['Σ(ω)', 'linearize']],
                value='Σ(ω)',
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Store(id="data-storage", data = data)
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
            style={'height': '84vh'}
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
])

# make connections
@app.callback(
    [dash.dependencies.Output('Akw', 'figure'),
    dash.dependencies.Output('data-storage', 'data')],
    [dash.dependencies.Input('tb-bands', 'on'),
     dash.dependencies.Input('akw', 'on'),
     dash.dependencies.Input('upload-file', 'filename'),
     dash.dependencies.Input('data-storage', 'data')])
#
def update_Akw(tb_bands, akw, filename, data):
    layout = go.Layout(title={'text':'A(k,ω)', 'xanchor': 'center', 'x':0.5})
    fig = go.Figure(layout=layout)

    if filename != None and not filename == data['file']:
        data = update_data(filename)

    fig.add_shape(type = 'line', x0=0, y0=0, x1=max(data['k_mesh']), y1=0, line=dict(color='gray', width=0.8))

    if akw:
        # kw_x, kw_y = np.meshgrid(data.tb_data['k_mesh'], w_mesh['w_mesh'])
        z_data = np.log(np.array(data['Akw']).T)
        fig.add_trace(go.Heatmap(x=data['k_mesh'], y=data['freq_mesh'], z=z_data,
                      colorscale='Tealrose',reversescale=False, showscale=False,
                      zmin=np.min(z_data), zmax=np.max(z_data)))


    if tb_bands:
        for band in range(3):
            fig.add_trace(go.Scattergl(x=data['k_mesh'], y=data['eps_nuk'][band], mode='lines',
                          line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=False, text=f'tb band {band}',
                          hoverinfo='x+y+text'
                          ))
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                      hovermode='closest',
                      yaxis_range=[data['freq_mesh'][0], data['freq_mesh'][-1]],
                      yaxis_title='ω (eV)',
                      xaxis=dict(ticktext=['γ' if k == 'g' else k for k in data['k_points_labels']],tickvals=data['k_points']),
                      font=dict(size=16))
    return fig, data


@app.callback(
    [dash.dependencies.Output('EDC', 'figure'),
    dash.dependencies.Output('kpt_edc', 'max')],
    [dash.dependencies.Input('kpt_edc', 'value'),
    dash.dependencies.Input('data-storage', 'data')]
    )
#
def update_EDC(kpt_edc, data):
    layout = go.Layout(title={'text':'EDC', 'xanchor': 'center', 'x':0.5})
    fig = go.Figure(layout=layout)

    
    fig.add_trace(go.Scattergl(x=data['freq_mesh'], y=np.array(data['Akw'])[kpt_edc,:], mode='lines',
        line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=True, name='k = {:.3f}'.format(data['k_mesh'][kpt_edc]),
                          hoverinfo='x+y+text'
                          ))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                      hovermode='closest',
                      xaxis_range=[data['freq_mesh'][0], data['freq_mesh'][-1]],
                      yaxis_range=[0, data['max_Akw']],
                      xaxis_title='ω (eV)',
                      yaxis_title='A(ω)',
                      font=dict(size=16),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                      )
    return fig, len(data['k_mesh'])-1

@app.callback(
    [dash.dependencies.Output('MDC', 'figure'),
    dash.dependencies.Output('w_mdc', 'max')],
    [dash.dependencies.Input('w_mdc', 'value'),
     dash.dependencies.Input('data-storage', 'data')]
    )
#
def update_MDC(w_mdc, data):
    layout = go.Layout(title={'text':'MDC', 'xanchor': 'center', 'x':0.5})
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scattergl(x=data['k_mesh'], y=np.array(data['Akw'])[:, w_mdc], mode='lines',
        line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=True, name='ω = {:.3f} eV'.format(data['freq_mesh'][w_mdc]),
                          hoverinfo='x+y+text'
                          ))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                      hovermode='closest',
                      xaxis_range=[data['k_mesh'][0], data['k_mesh'][-1]],
                    #   yaxis_range=[0, 1.05 * np.max(np.array(data['Akw'])[:, w_mdc])],
                      yaxis_range=[0, data['max_Akw']],
                      xaxis_title='k',
                      yaxis_title='A(k)',
                      font=dict(size=16),
                      xaxis=dict(ticktext=['γ' if k == 'g' else k for k in data['k_points_labels']],tickvals=data['k_points']),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                      )
    return fig, len(data['freq_mesh'])-1

if __name__ == '__main__':
    app.run_server(debug=True, port=9375, host='0.0.0.0')
