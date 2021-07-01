import numpy as np
from itertools import product
import dash
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import ast
import inspect
import io
from dash.dependencies import Input, Output, State
from h5 import HDFArchive

from load_data import load_config, load_w90_hr, load_w90_wout, load_sigma_h5
from tools.calc_akw import calc_tb_bands, get_tb_bands, calc_alatt 
from tabs.id_factory import id_factory


def register_callbacks(app):
    id = id_factory('tab1')

    # upload akw data
    @app.callback(
        [Output(id('akw-data'), 'data'),
         Output(id('akw-bands'), 'on'),
         Output(id('tb-alert'), 'is_open')],
        [Input(id('akw-data'), 'data'),
         Input(id('upload-file'), 'contents'),
         Input(id('upload-file'), 'filename'),
         Input(id('tb-data'), 'data'),
         Input(id('sigma-data'), 'data'),
         Input(id('akw-bands'), 'on'),
         Input(id('dft-mu'), 'value'),
         Input(id('k-points'), 'data'),
         Input(id('calc-akw'), 'n_clicks')],
         State(id('tb-alert'), 'is_open'),
         prevent_initial_call=True
        )
    def update_data(akw_data, config_contents, config_filename, tb_data, 
                    sigma_data, akw_switch, dft_mu, k_points, click_akw, tb_alert):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print(trigger_id)

        if config_filename != None:
            print('loading config file from h5...')
            akw_data = load_config(config_contents, config_filename)
            akw_data['use'] = True

        if trigger_id == id('dft-mu') and not sigma_data['use']:
            return akw_data, akw_switch, tb_alert

        if trigger_id in (id('calc-akw'), id('dft-mu')) or ( trigger_id == id('k-points') and click_akw > 0 ):
            if not sigma_data['use'] or not tb_data['use']:
                return akw_data, akw_switch, not tb_alert

            solve = False
            akw_data['dmft_mu'] = sigma_data['dmft_mu']
            akw_data['eta'] = 0.01
            akw = calc_alatt(tb_data, sigma_data, akw_data, solve)
            akw_data['Akw'] = akw.tolist()
            akw_data['use'] = True

            akw_switch = {'on': True}

        return akw_data, akw_switch, tb_alert 

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
         Input(id('dft-orbital-order'), 'data')],
         prevent_initial_call=True,)
    def calc_tb(w90_hr, w90_hr_name, w90_hr_button, w90_wout, w90_wout_name,
                w90_wout_button, tb_switch, click_tb, tb_data, add_spin, dft_mu, k_points, dft_orbital_order):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print(trigger_id)

        #if w90_hr != None and not 'loaded_hr' in tb_data:
        if trigger_id == id('upload-w90-hr'):
            print('loading w90 hr file...')
            hopping, n_wf = load_w90_hr(w90_hr)
            hopping = {str(key): value.real.tolist() for key, value in hopping.items()}
            tb_data['n_wf'] = n_wf
            tb_data['hopping'] = hopping
            tb_data['loaded_hr'] = True

            return tb_data, html.Div([w90_hr_name]), w90_wout_button, tb_switch 

        #if w90_wout != None and not 'loaded_wout' in tb_data:
        if trigger_id == id('upload-w90-wout'):
            print('loading w90 wout file...')
            tb_data['units'] = load_w90_wout(w90_wout)
            tb_data['loaded_wout'] = True

            return tb_data, w90_hr_button, html.Div([w90_wout_name]), tb_switch

        else:
            if not click_tb > 0:
                return tb_data, w90_hr_button, w90_wout_button, tb_switch
            if np.any([k_val in ['', None] for k in k_points for k_key, k_val in k.items()]):
                return tb_data, w90_hr_button, w90_wout_button, tb_switch

            if not isinstance(dft_mu, (float, int)):
                dft_mu = 0.0

            add_local = [0.] * tb_data['n_wf']

            k_mesh = {'n_k': 30, 'k_path': k_points, 'kz': 0.0}
            tb_data['k_mesh'], e_mat, tb = calc_tb_bands(tb_data, add_spin, float(dft_mu), add_local, dft_orbital_order, k_mesh, fermi_slice=False)
            # calculate Hamiltonian
            tb_data['e_mat'] = e_mat.real.tolist()
            # compute eigenvalues too
            tb_data['eps_nuk'], evec_nuk = get_tb_bands(e_mat)
            tb_data['eps_nuk'] = tb_data['eps_nuk'].tolist()
            tb_data['bnd_low'] = np.min(np.array(tb_data['eps_nuk'][0]))
            tb_data['bnd_high'] = np.max(np.array(tb_data['eps_nuk'][-1]))
            tb_data['use'] = True

            return tb_data, w90_hr_button, w90_wout_button, {'on': True}

    # dashboard k-points
    @app.callback(
        Output(id('k-points'), 'data'),
        [Input(id('add-kpoint'), 'n_clicks'),
         Input(id('k-points'), 'data')],
        State(id('k-points'), 'data'),
        State(id('k-points'), 'columns'),
        prevent_initial_call=True)
    def add_row(n_clicks, data, rows, columns):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        for row, col in product(rows, range(1,4)):
            try:
                row[id(f'column-{col}')] = float(row[id(f'column-{col}')])
            except:
                row[id(f'column-{col}')] = None

        if trigger_id == id('add-kpoint'):
            rows.append({c['id']: '' for c in columns})
        
        return rows
    
    # dashboard sigma upload
    @app.callback(
         [Output(id('sigma-data'), 'data'),
          Output(id('sigma-function'), 'style'),
          Output(id('sigma-upload'), 'style'),
          Output(id('sigma-upload-box'), 'children')
        ],
         [Input(id('sigma-data'), 'data'),
         Input(id('choose-sigma'), 'value'),
         Input(id('sigma-upload-box'), 'contents'),
         Input(id('sigma-upload-box'), 'filename'),
         Input(id('sigma-upload-box'), 'children'),
         Input(id('dft-orbital-order'), 'data')],
         prevent_initial_call=False
        )
    def toggle_update_sigma(sigma_data, sigma_radio_item, sigma_content, sigma_filename, sigma_button, dft_orbital_order):
        if sigma_radio_item == 'upload':

            if sigma_content != None:
                print('loading Sigma from file...')
                sigma_data = load_sigma_h5(sigma_content, sigma_filename, dft_orbital_order)
                print('successfully loaded sigma from file')
                sigma_data['use'] = True
                sigma_button = html.Div([sigma_filename])
            else:
                sigma_button = sigma_button
            
            return sigma_data, {'display': 'none'}, {'display': 'block'}, sigma_button
        else:
            return sigma_data, {'display': 'block'}, {'display': 'none'}, sigma_button

    # dashboard enter sigma
    @app.callback(
        #[Output('sigma-function-output', 'children'),
        # Output('sigma-function', 'sigma')],
        Output(id('sigma-function-output'), 'children'),
        Input(id('sigma-function-button'), 'n_clicks'),
        State(id('sigma-function-input'), 'value'),
    prevent_initial_call=True,
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
        Input(id('colorscale-mode'), 'value'),
    prevent_initial_call=True,
        )
    def update_colorscales(mode):
        colorscales = [name for name, body in inspect.getmembers(getattr(px.colors, mode))
                if isinstance(body, list) and len(name.rsplit('_')) == 1]
        return [{'label': key, 'value': key} for key in colorscales]

    # plot A(k,w)
    @app.callback(
        Output('Akw', 'figure'),
        [Input(id('tb-bands'), 'on'),
         Input(id('akw-bands'), 'on'),
         Input(id('colorscale'), 'value'),
         Input(id('tb-data'), 'data'),
         Input(id('akw-data'), 'data'),
         Input(id('sigma-data'), 'data')],
         prevent_initial_call=True)
    def update_Akw(tb_bands, akw, colorscale, tb_data, akw_data, sigma_data):
        print(akw_data.keys())
        
        # initialize general figure environment
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', rangeslider_visible=False, 
                         showticklabels=True, spikedash='solid')
        fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showticklabels=True, spikedash='solid')
        fig.update_traces(xaxis='x', hoverinfo='none')

        # decide which data to show for TB
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
            if not akw:
                fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                          clickmode='event+select',
                          hovermode='closest',
                          xaxis_range=[k_mesh['k_disc'][0], k_mesh['k_disc'][-1]],
                          yaxis_range=[tb_data['bnd_low']- 0.02*abs(tb_data['bnd_low']) , 
                                       tb_data['bnd_high']+ 0.02*abs(tb_data['bnd_high'])],
                          yaxis_title='ω (eV)',
                          xaxis=dict(ticktext=['γ' if k == 'g' else k for k in k_mesh['k_point_labels']],tickvals=k_mesh['k_points']),
                          font=dict(size=16))

        # decide which data to show for A(k,w)
        if akw_data['use']: akw_temp = akw_data
        if not 'akw_temp' in locals():
            return fig

        if akw:
            # kw_x, kw_y = np.meshgrid(data.tb_data['k_mesh'], w_mesh['w_mesh'])
            z_data = np.log(np.array(akw_temp['Akw']).T)
            w_mesh = sigma_data['w_dict']['w_mesh']
            fig.add_trace(go.Heatmap(x=k_mesh['k_disc'], y=w_mesh, z=z_data,
                                     colorscale=colorscale, reversescale=False, showscale=False,
                                     zmin=np.min(z_data), zmax=np.max(z_data)))

            fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                              clickmode='event+select',
                              hovermode='closest',
                              yaxis_range=[w_mesh[0], w_mesh[-1]],
                              yaxis_title='ω (eV)',
                              xaxis_range=[k_mesh['k_disc'][0], k_mesh['k_disc'][-1]],
                              xaxis=dict(ticktext=['γ' if k == 'g' else k for k in k_mesh['k_point_labels']], tickvals=k_mesh['k_points']),
                              font=dict(size=16))
    
        return fig
    
    # plot EDC 
    @app.callback(
       [Output('EDC', 'figure'),
        Output('kpt_edc', 'value'),
        Output('kpt_edc', 'max')],
       [Input(id('tb-bands'), 'on'),
        Input(id('akw-bands'), 'on'),
        Input('kpt_edc', 'value'),
        Input(id('akw-data'), 'data'),
        Input(id('tb-data'), 'data'),
        Input('Akw', 'clickData'),
        Input(id('sigma-data'), 'data')],
        prevent_initial_call=True)
    def update_EDC(tb_bands, akw, kpt_edc, akw_data, tb_data, click_coordinates, sigma_data):
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if tb_bands:
            # decide which data to show for TB
            if tb_data['use']: tb_temp = tb_data
            k_mesh = tb_temp['k_mesh']
            if not 'tb_temp' in locals():
                return fig
            for band in range(len(tb_temp['eps_nuk'])):
                if band == 0:
                    fig.add_trace(go.Scattergl(x=[tb_temp['eps_nuk'][band][kpt_edc],tb_temp['eps_nuk'][band][kpt_edc]], 
                                         y=[0,300], mode="lines", line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), 
                                         showlegend=True, name='k = {:.3f}'.format(tb_temp['k_mesh']['k_disc'][kpt_edc]),
                                    hoverinfo='x+y+text'
                                    ))
                else:
                    fig.add_trace(go.Scattergl(x=[tb_temp['eps_nuk'][band][kpt_edc],tb_temp['eps_nuk'][band][kpt_edc]], 
                                            y=[0,300], mode="lines", line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), 
                                            showlegend=False, hoverinfo='x+y+text'
                                        ))
            if not akw:
                fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                                hovermode='closest',
                                xaxis_range=[tb_data['bnd_low']- 0.02*abs(tb_data['bnd_low']) , 
                                             tb_data['bnd_high']+ 0.02*abs(tb_data['bnd_high'])],
                                yaxis_range=[0, 1],
                                xaxis_title='ω (eV)',
                                yaxis_title='A(ω)',
                                font=dict(size=16),
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                                )       
        if akw:
            w_mesh = sigma_data['w_dict']['w_mesh']
            if trigger_id == 'Akw':
                new_kpt = click_coordinates['points'][0]['x']
                kpt_edc = np.argmin(np.abs(np.array(k_mesh['k_disc']) - new_kpt))
            
            fig.add_trace(go.Scattergl(x=w_mesh, y=np.array(akw_data['Akw'])[kpt_edc,:], mode='lines',
                line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=True, name='k = {:.3f}'.format(k_mesh['k_disc'][kpt_edc]),
                                    hoverinfo='x+y+text'
                                    ))
            
            fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                                hovermode='closest',
                                xaxis_range=[w_mesh[0], w_mesh[-1]],
                                yaxis_range=[0, 1.05 * np.max(np.array(akw_data['Akw']))],
                                xaxis_title='ω (eV)',
                                yaxis_title='A(ω)',
                                font=dict(size=16),
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                                )
        if akw:
            return fig, kpt_edc, len(k_mesh['k_disc'])-1
        elif tb_bands:
            return fig, kpt_edc, len(k_mesh['k_disc'])-1
        else:
            return fig, 0, 1
    #
    # plot MDC
    @app.callback(
       [Output('MDC', 'figure'),
        Output('w_mdc', 'value'),
        Output('w_mdc', 'max')],
       [Input(id('tb-bands'), 'on'),
        Input(id('akw-bands'), 'on'),
        Input('w_mdc', 'value'),
        Input(id('akw-data'), 'data'),
        Input(id('tb-data'), 'data'),
        Input('Akw', 'clickData'),
        Input(id('sigma-data'), 'data')],
        prevent_initial_call=True)
    def update_MDC(tb_bands, akw, w_mdc, akw_data, tb_data, click_coordinates, sigma_data):
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if tb_bands:
            k_mesh = tb_data['k_mesh']
        #     # decide which data to show for TB
        #     if tb_data['use']: tb_temp = tb_data
        #     if not 'tb_temp' in locals():
        #         return fig

        #     k_mesh = tb_temp['k_mesh']
        #     for band in range(len(tb_temp['eps_nuk'])):
        #         if band == 0:
        #             y = np.min(np.abs(np.array(tb_temp['eps_nuk'][band]) - w_mdc))
        #             fig.add_trace(go.Scattergl(x=[k_mesh['k_disc'][0],k_mesh['k_disc'][-1]], 
        #                                  y=[y,y], mode="lines", line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), 
        #                                  showlegend=True, name='ω = {:.3f} eV'.format(akw_data['freq_mesh'][w_mdc]),
        #                             hoverinfo='x+y+text'
        #                             ))
        #         # else:
        #         #     fig.add_trace(go.Scattergl(x=[tb_temp['eps_nuk'][band][kpt_edc],tb_temp['eps_nuk'][band][kpt_edc]], 
        #         #                             y=[0,300], mode="lines", line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), 
        #         #                             showlegend=False, hoverinfo='x+y+text'
        #                                 # ))
        #     if not akw:
        #         fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
        #                         hovermode='closest',
        #                         xaxis_range=[k_mesh['k_points'][0], k_mesh['k_points'][-1]],
        #                         yaxis_range=[0, 1],
        #                         xaxis_title='ω (eV)',
        #                         yaxis_title='A(ω)',
        #                         font=dict(size=16),
        #                         legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        #                         )       

        if akw:
            w_mesh = sigma_data['w_dict']['w_mesh']
            if trigger_id == 'Akw':
                new_w = click_coordinates['points'][0]['y']
                w_mdc = np.argmin(np.abs(np.array(w_mesh) - new_w))
        
            fig.add_trace(go.Scattergl(x=k_mesh['k_disc'], y=np.array(akw_data['Akw'])[:, w_mdc], mode='lines', line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]),
                                       showlegend=True, name='ω = {:.3f} eV'.format(w_mesh[w_mdc]), hoverinfo='x+y+text'))
        
            fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                              hovermode='closest',
                              xaxis_range=[k_mesh['k_disc'][0], k_mesh['k_disc'][-1]],
                        #     yaxis_range=[0, 1.05 * np.max(np.array(akw_data['Akw'])[:, w_mdc])],
                              yaxis_range=[0, 1.05 * np.max(np.array(akw_data['Akw']))],
                              xaxis_title='k',
                              yaxis_title='A(k)',
                              font=dict(size=16),
                              xaxis=dict(ticktext=['γ' if k == 'g' else k for k in k_mesh['k_point_labels']], tickvals=k_mesh['k_points']),
                              legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                              )
        if akw:
            return fig, w_mdc, len(w_mesh)-1
        # elif tb_bands:
        #     return fig, w_mdc, np.max(np.array(tb_temp['eps_nuk'][-1]))+(0.03*np.max(np.array(tb_temp['eps_nuk'][-1])))
        else:
            return fig, 0, 1
    
    @app.callback(
    Output(id('download_h5'), "data"),
    [Input(id('dwn_button'), "n_clicks"),
     Input(id('akw-data'), 'data'),
     Input(id('tb-data'), 'data'),
     Input(id('sigma-data'), 'data')],
     prevent_initial_call=True,
    )
    def download_data(n_clicks,akw_data, tb_data, sigma_data):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        # check if the download button was pressed
        if trigger_id == id('dwn_button'):
            # return_data = io.BytesIO()
            return_data = HDFArchive()
            return_data['test'] = 0.0
            return dcc.send_file(return_data, attachment_filename='spectrometer.h5')
        else:
            return None
        


