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

def _get_tb_bands(k_mesh, e_mat, k_points, **specs):

    e_val = np.zeros((e_mat.shape[0], k_mesh.shape[0]), dtype=complex)
    e_vec = np.zeros(np.shape(e_mat), dtype=complex)
    for ik in range(np.shape(e_mat)[2]):
        e_val[:,ik], e_vec[:,:,ik] = np.linalg.eigh(e_mat[:,:,ik])

    return e_val, e_vec

# load example
with HDFArchive('example.h5', 'r') as ar:
    Akw_data = ar['A_k_w_data'] # contains A_k_w, dft_mu
    tb_data = ar['tb_data'] # e_mat, k_mesh, k_points, k_points_labels
    w_mesh = ar['w_mesh']
eps_nuk, evec_nuk = _get_tb_bands(**tb_data)

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
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'padding': '10px 0px',
        'display': 'inline-block',
        'width': '14%',
        'vertical-align': 'top'
        }
    ),

    # column 2
    html.Div([
        dcc.Graph(
            id='Akw',
            style={'height': '75vh'}
        )
    ], style={
        'display': 'inline-block',
        'width': '42%',
        'padding': '0 20',
        'vertical-align': 'top'
        }
    ),

    # column 3
    html.Div([
        dcc.Graph(
            id='EDC',
            style={'height': '40vh'}
           ),
        dcc.Slider(
            id='kpt_edc',
            min=0,
            max=len(tb_data['k_mesh'])-1,
            value=0,
            #marks={str(year): str(year) for year in df['Year'].unique()},
            step=1,
            verticalHeight=200,
            #handleLabel={'showCurrentValue': True, 'label': 'value'},
            updatemode='drag',
            ),
        dcc.Graph(
            id='MDC',
            style={'height': '40vh'}
           ),
        dcc.Slider(
            id='w_mdc',
            min=0,
            max=len(w_mesh['w_mesh'])-1,
            value=0,
            #marks={str(year): str(year) for year in df['Year'].unique()},
            step=1,
            updatemode='drag',
            verticalHeight=200),
    ], style={
        'padding': '10px 0px',
        'display': 'inline-block',
        'width': '42%',
        'vertical-align': 'top'
        }
    ),
])

# make connections
@app.callback(
    dash.dependencies.Output('Akw', 'figure'),
    [dash.dependencies.Input('tb-bands', 'on'),
     dash.dependencies.Input('akw', 'on')])
#
def update_Akw(tb_bands, akw):
    layout = go.Layout(title={'text':'A(k,ω)', 'xanchor': 'center', 'x':0.5})
    fig = go.Figure(layout=layout)
    freq_mesh = w_mesh['w_mesh']
    fig.add_shape(type = 'line', x0=0, y0=0, x1=tb_data['k_mesh'].max(), y1=0, line=dict(color='gray', width=0.8))
    #fig.add_hline(y=0.0, line_color='gray', line_width=0.8)

    if akw:
        # kw_x, kw_y = np.meshgrid(tb_data['k_mesh'], w_mesh['w_mesh'])
        kw_x = tb_data['k_mesh']
        z_data = np.log(Akw_data['A_k_w'].T)
        fig.add_trace(go.Heatmap(x=kw_x, y=freq_mesh, z=z_data,
                      colorscale='Tealrose',reversescale=False, showscale=False,
                      zmin=np.min(z_data), zmax=np.max(z_data)))


    if tb_bands:
        for band in range(3):
            fig.add_trace(go.Scatter(x=tb_data['k_mesh'], y=eps_nuk[band].real - Akw_data['dft_mu'], mode='lines',
                          line=go.scatter.Line(color=px.colors.sequential.Viridis[0]), showlegend=False, text=f'tb band {band}',
                          hoverinfo='x+y+text'
                          ))
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                      hovermode='closest',
                      yaxis_range=[freq_mesh[0], freq_mesh[-1]],
                      yaxis_title='ω (eV)',
                      xaxis=dict(ticktext=['γ' if k == 'g' else k for k in tb_data['k_points_labels']],tickvals=tb_data['k_points']),
                      font=dict(size =14))
    return fig


@app.callback(
    dash.dependencies.Output('EDC', 'figure'),
    [dash.dependencies.Input('kpt_edc', 'value')]
    )
#
def update_EDC(kpt_edc):
    layout = go.Layout(title={'text':'EDC', 'xanchor': 'center', 'x':0.5})
    fig = go.Figure(layout=layout)

    freq_mesh = w_mesh['w_mesh']
    k_mesh = tb_data['k_mesh']
    Akw = Akw_data['A_k_w']
    fig.add_trace(go.Scatter(x=freq_mesh, y=Akw[kpt_edc,:], mode='lines',
        line=go.scatter.Line(color=px.colors.sequential.Viridis[0]), showlegend=True, name=f'k = {k_mesh[kpt_edc]:.3f}',
                          hoverinfo='x+y+text'
                          ))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                      hovermode='closest',
                      xaxis_range=[freq_mesh[0], freq_mesh[-1]],
                      yaxis_range=[0, 1.05 * np.max(Akw[kpt_edc, :])],
                      xaxis_title='ω (eV)',
                      yaxis_title='A(ω)',
                      font=dict(size =14),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                      )
    return fig

@app.callback(
    dash.dependencies.Output('MDC', 'figure'),
    [dash.dependencies.Input('w_mdc', 'value')]
    )
#
def update_MDC(w_mdc):
    layout = go.Layout(title={'text':'MDC', 'xanchor': 'center', 'x':0.5})
    fig = go.Figure(layout=layout)

    freq_mesh = w_mesh['w_mesh']
    k_mesh = tb_data['k_mesh']
    Akw = Akw_data['A_k_w']
    fig.add_trace(go.Scatter(x=k_mesh, y=Akw[:, w_mdc], mode='lines',
        line=go.scatter.Line(color=px.colors.sequential.Viridis[0]), showlegend=True, name=f'ω = {freq_mesh[w_mdc]:.3f} eV',
                          hoverinfo='x+y+text'
                          ))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                      hovermode='closest',
                      xaxis_range=[k_mesh[0], k_mesh[-1]],
                      yaxis_range=[0, 1.05 * np.max(Akw[:, w_mdc])],
                      xaxis_title='k',
                      yaxis_title='A(k)',
                      font=dict(size =14),
                      xaxis=dict(ticktext=['γ' if k == 'g' else k for k in tb_data['k_points_labels']],tickvals=tb_data['k_points']),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                      )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=9375, host='0.0.0.0')
