import numpy as np
import dash
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import ast
import inspect
from dash.dependencies import Input, Output, State

from load_data import load_config, load_w90_hr, load_w90_wout
from tools.calc_akw import calc_tb_bands, get_tb_bands 
from tabs.id_factory import id_factory

def register_callbacks(app):
    id = id_factory('tab1')

    # upload data
    @app.callback(
        Output(id('full-data'), 'data'),
        [Input(id('full-data'), 'data'),
         Input(id('upload-file'), 'contents'),
         Input(id('upload-file'), 'filename')])
    def update_data(data, config_contents, config_filename):

        if config_filename != None and not config_filename == data['config_filename']:
            print('loading config file from h5...')
            data = load_config(config_contents, config_filename)
            data['use'] = True

        return data

    # dashboard calculate TB
    @app.callback(
        [Output(id('tb-data'), 'data'),
         Output(id('upload-w90-hr'), 'children'),
         Output(id('upload-w90-wout'), 'children'),
         Output(id('tb-bands'), 'on')],
        [Input(id('upload-w90-hr'), 'contents'),
         Input(id('upload-w90-hr'), 'filename'),
         Input(id('upload-w90-hr'), 'children'),
         Input(id('upload-w90-wout'), 'contents'),
         Input(id('upload-w90-wout'), 'filename'),
         Input(id('upload-w90-wout'), 'children'),
         Input(id('tb-bands'), 'on'),
         Input(id('calc-tb'), 'n_clicks'),
         Input(id('tb-data'), 'data'),
         Input(id('add-spin'), 'value'),
         Input(id('dft-mu'), 'value'),
         Input(id('k-points'), 'data'),
         Input(id('dft-orbital-order'), 'data')])
    def calc_tb(w90_hr, w90_hr_name, w90_hr_button, w90_wout, w90_wout_name,
                w90_wout_button, tb_switch, n_clicks, tb_data, add_spin, dft_mu, k_points, dft_orbital_order):

        if w90_hr != None and not 'loaded_hr' in tb_data:
            print('loading w90 hr file...')
            hopping, num_wann = load_w90_hr(w90_hr)
            hopping = {str(key): value.real.tolist() for key, value in hopping.items()}
            tb_data['num_wann'] = num_wann
            tb_data['hopping'] = hopping
            tb_data['loaded_hr'] = True

            return tb_data, html.Div([w90_hr_name]), w90_wout_button, tb_switch 

        if w90_wout != None and not 'loaded_wout' in tb_data:
            print('loading w90 wout file...')
            tb_data['units'] = load_w90_wout(w90_wout)
            tb_data['loaded_wout'] = True

            return tb_data, w90_hr_button, html.Div([w90_wout_name]), tb_switch

        if n_clicks > 0:
            n_orb = 3
            add_local = [0.] * 3
            k_mesh = {'n_k': 20, 'k_path': k_points, 'kz': 0.0}
            fermi_slice = False
            tb_data['k_mesh'], e_mat, tb = calc_tb_bands(tb_data, n_orb, add_spin, float(dft_mu), add_local, dft_orbital_order, k_mesh, fermi_slice)
            tb_data['eps_nuk'], evec_nuk = get_tb_bands(e_mat)
            tb_data['eps_nuk'] = tb_data['eps_nuk'].tolist()
            tb_data['use'] = True

        return tb_data, w90_hr_button, w90_wout_button, {'on': True}

    # dashboard k-points
    @app.callback(
        Output(id('k-points'), 'data'),
        Input(id('add-kpoint'), 'n_clicks'),
        State(id('k-points'), 'data'),
        State(id('k-points'), 'columns'))
    def add_row(n_clicks, rows, columns):
        if n_clicks > 0:
            rows.append({c['id']: '' for c in columns})
        return rows
    
    # dashboard sigma upload
    @app.callback(
         [Output(id('sigma-function'), 'style'),
          Output(id('sigma-upload'), 'style')],
         Input(id('choose-sigma'), 'value')
        )
    def toggle_update_sigma(sigma_radio_item):
        if sigma_radio_item == 'upload':
            return [{'display': 'none'}] * 1 + [{'display': 'block'}]
        else:
            return [{'display': 'block'}] * 1 + [{'display': 'none'}]

    # dashboard enter sigma
    @app.callback(
        #[Output('sigma-function-output', 'children'),
        # Output('sigma-function', 'sigma')],
        Output(id('sigma-function-output'), 'children'),
        Input(id('sigma-function-button'), 'n_clicks'),
        State(id('sigma-function-input'), 'value')
        )
    def update_sigma(n_clicks, value):
        if n_clicks > 0:
            # parse function
            tree = ast.parse(value, mode='exec')
            code = compile(tree, filename='test', mode='exec')
            namespace = {} 
            exec(code, namespace)

            return 'You have entered: \n{}'.format(value)

    # dashboard colors
    @app.callback(
        Output(id('colorscale'), 'options'),
        Input(id('colorscale-mode'), 'value')
        )
    def update_colorscales(mode):
        colorscales = [name for name, body in inspect.getmembers(getattr(px.colors, mode))
                if isinstance(body, list) and len(name.rsplit('_')) == 1]
        return [{'label': key, 'value': key} for key in colorscales]

    # plot A(k,w)
    @app.callback(
        Output('Akw', 'figure'),
        [Input(id('tb-bands'), 'on'),
         Input(id('akw'), 'on'),
         Input(id('colorscale'), 'value'),
         Input(id('tb-data'), 'data'),
         Input(id('full-data'), 'data')])
    def update_Akw(tb_bands, akw, colorscale, tb_data, full_data):
        
        # initialize general figure environment
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', rangeslider_visible=False, 
                         showticklabels=True, spikedash='solid')
        fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showticklabels=True, spikedash='solid')
        fig.update_traces(xaxis='x', hoverinfo='none')

        # decide which data to show for TB
        if full_data['use']: tb_temp = full_data
        if tb_data['use']: tb_temp = tb_data
        if not 'tb_temp' in locals():
            return fig
    
        k_mesh = tb_temp['k_mesh']
        fig.add_shape(type = 'line', x0=0, y0=0, x1=max(k_mesh['k_disc']), y1=0, line=dict(color='gray', width=0.8))
    
        if tb_bands:
            for band in range(len(tb_temp['eps_nuk'])):
                fig.add_trace(go.Scattergl(x=k_mesh['k_disc'], y=tb_temp['eps_nuk'][band], mode='lines',
                              line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=False, text=f'tb band {band}',
                              hoverinfo='x+y+text'
                              ))

        # decide which data to show for A(k,w)
        if full_data['use']: akw_temp = full_data
        #if akw_data['use']: akw_temp = akw_data
        if not 'akw_temp' in locals():
            return fig

        if akw:
            # kw_x, kw_y = np.meshgrid(data.tb_data['k_mesh'], w_mesh['w_mesh'])
            z_data = np.log(np.array(awk_temp['Akw']).T)
            fig.add_trace(go.Heatmap(x=awk_temp['k_mesh'], y=akw_temp['freq_mesh'], z=z_data,
                          colorscale=colorscale,reversescale=False, showscale=False,
                          zmin=np.min(z_data), zmax=np.max(z_data)))
    
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                          clickmode='event+select',
                          hovermode='closest',
                          yaxis_range=[data['freq_mesh'][0], data['freq_mesh'][-1]],
                          yaxis_title='ω (eV)',
                          xaxis=dict(ticktext=['γ' if k == 'g' else k for k in data['k_points_labels']],tickvals=data['k_points']),
                          font=dict(size=16))
    
        return fig
    
    ## plot EDC 
    #@app.callback(
    #    [Output('EDC', 'figure'),
    #     Output('kpt_edc', 'value'),
    #     Output('kpt_edc', 'max')],
    #    [Input('kpt_edc', 'value'),
    #     Input(id('full-data'), 'data'),
    #     Input('Akw', 'clickData')]
    #    )
    #def update_EDC(kpt_edc, data, click_coordinates):
    #    layout = go.Layout()
    #    fig = go.Figure(layout=layout)
    #    ctx = dash.callback_context
    #    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #    if trigger_id == 'Akw':
    #        new_kpt = click_coordinates['points'][0]['x']
    #        kpt_edc = np.argmin(np.abs(np.array(data['k_mesh']) - new_kpt))
    #    
    #    fig.add_trace(go.Scattergl(x=data['freq_mesh'], y=np.array(data['Akw'])[kpt_edc,:], mode='lines',
    #        line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=True, name='k = {:.3f}'.format(data['k_mesh'][kpt_edc]),
    #                          hoverinfo='x+y+text'
    #                          ))
    #
    #    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
    #                      hovermode='closest',
    #                      xaxis_range=[data['freq_mesh'][0], data['freq_mesh'][-1]],
    #                      yaxis_range=[0, data['max_Akw']],
    #                      xaxis_title='ω (eV)',
    #                      yaxis_title='A(ω)',
    #                      font=dict(size=16),
    #                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    #                      )
    #    return fig, kpt_edc, len(data['k_mesh'])-1
    #
    ## plot MDC
    #@app.callback(
    #    [Output('MDC', 'figure'),
    #     Output('w_mdc', 'value'),
    #     Output('w_mdc', 'max')],
    #    [Input('w_mdc', 'value'),
    #     Input(id('full-data'), 'data'),
    #     Input('Akw', 'clickData')]
    #    )
    #def update_MDC(w_mdc, data, click_coordinates):
    #    layout = go.Layout()
    #    fig = go.Figure(layout=layout)
    #    ctx = dash.callback_context
    #    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #    if trigger_id == 'Akw':
    #        new_w = click_coordinates['points'][0]['y']
    #        w_mdc = np.argmin(np.abs(np.array(data['freq_mesh']) - new_w))
    #
    #    fig.add_trace(go.Scattergl(x=data['k_mesh'], y=np.array(data['Akw'])[:, w_mdc], mode='lines',
    #        line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=True, name='ω = {:.3f} eV'.format(data['freq_mesh'][w_mdc]),
    #                          hoverinfo='x+y+text'
    #                          ))
    #
    #    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
    #                      hovermode='closest',
    #                      xaxis_range=[data['k_mesh'][0], data['k_mesh'][-1]],
    #                    #   yaxis_range=[0, 1.05 * np.max(np.array(data['Akw'])[:, w_mdc])],
    #                      yaxis_range=[0, data['max_Akw']],
    #                      xaxis_title='k',
    #                      yaxis_title='A(k)',
    #                      font=dict(size=16),
    #                      xaxis=dict(ticktext=['γ' if k == 'g' else k for k in data['k_points_labels']],tickvals=data['k_points']),
    #                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    #                      )
    #    return fig, w_mdc, len(data['freq_mesh'])-1


