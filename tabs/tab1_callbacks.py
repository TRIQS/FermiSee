import dash
from dash import html
from flask import send_file
import plotly.express as px
import plotly.graph_objects as go
from ast import literal_eval
from itertools import permutations
import inspect
import base64
from dash.dependencies import Input, Output, State, ALL
import numpy as np
from itertools import product

from h5 import HDFArchive
from triqs.gf import MeshReFreq, GfReFreq

from load_data import load_config, load_w90_hr, load_w90_wout, load_sigma_h5, load_pythTB_json
import tools.calc_tb as tb
import tools.calc_akw as akw
import tools.gf_helpers as gf
import tools.tools as tools
from tabs.id_factory import id_factory


def register_callbacks(app):
    id = id_factory('tab1')

    @app.callback(
        [Output(id('loaded-data'),'data'),
         Output(id('config-alert'), 'is_open')],
        [Input(id('upload-file'), 'contents'),
         Input(id('upload-file'), 'filename'),
         Input(id('loaded-data'), 'data')],
         State(id('config-alert'), 'is_open'),
         prevent_initial_call=True
    )
    def upload_config(config_contents, config_filename, loaded_data, config_alert):
        if not config_filename:
            return loaded_data, True

        print('loading config file from h5...')
        loaded_data = load_config(config_contents, config_filename, loaded_data)
        if loaded_data['error']:
            return loaded_data, True

        return loaded_data, False


    # upload akw data
    @app.callback(
        [Output(id('akw-data'), 'data'),
         Output(id('akw-bands'), 'on'),
         Output(id('tb-alert'), 'is_open')],
        [Input(id('akw-data'), 'data'),
         Input(id('tb-data'), 'data'),
         Input(id('sigma-data'), 'data'),
         Input(id('akw-bands'), 'on'),
         Input(id('dft-mu'), 'children'),
         Input(id('k-points'), 'data'),
         Input(id('n-k'), 'value'),
         Input(id('calc-tb'), 'n_clicks'),
         Input(id('calc-akw'), 'n_clicks'),
         Input(id('akw-mode'), 'value'),
         Input(id('eta'), 'value'),
         Input(id('band-basis'), 'on')],
         State(id('tb-alert'), 'is_open'),
         prevent_initial_call=True
        )
    def update_akw(akw_data, tb_data, sigma_data, akw_switch, dft_mu, k_points, n_k, click_tb, click_akw, akw_mode, eta, band_basis, tb_alert):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***update_akw***:'), trigger_id)

        if trigger_id == id('dft-mu') and not sigma_data['use']:
            return akw_data, akw_switch, tb_alert

        elif trigger_id in (id('calc-akw'), id('n-k'), id('akw-mode')) or ( trigger_id == id('k-points') and click_akw > 0 ):
            if not sigma_data['use'] or not tb_data['use'] or not tb_data['dft_mu']:
                return akw_data, akw_switch, not tb_alert

            solve = True if akw_mode == 'QP dispersion' else False
            if not 'dmft_mu' in akw_data.keys():
                akw_data['dmft_mu'] = sigma_data['dmft_mu']

            akw_data['eta'] = float(eta)
            alatt, Aw, akw_data['dmft_mu'] = akw.calc_alatt(tb_data, sigma_data, akw_data, solve, band_basis)
            akw_data['Akw'] = alatt.tolist()
            akw_data['Aw'] = Aw.tolist()
            akw_data['use'] = True
            akw_data['solve'] = solve

            akw_switch = {'on': True}

        return akw_data, akw_switch, tb_alert
    
    #toggle the TB options
    @app.callback(
    [Output(id('w90-buttons'), 'style'),
     Output(id('pythTB-button'), 'style')],
    [Input(id('choose-TB-method'), 'value')]
    )
    def display_TB_upload_method(radio_selection):
            if radio_selection == 'pythTB':
                    return {'display': 'none'}, {'display': 'block'}
            return {'display': 'block'}, {'display': 'none'}
    
    #change button color after something is uploaded
    @app.callback(
        Output(id('upload-pythTB-json'), 'style'),
        [Input(id('upload-pythTB-json'),'style'),
         Input(id('upload-pythTB-json'), 'children')],
        prevent_initial_call = True,
    )
    def change_button(pythTB_style, pythTB_filename):

        #Here I'll make a new div with a different style that will be returned and change the style of the w90_hr and w90_out when a file is loaded
        loaded_style={
                            'width': '90%',
                            'height': '37px',
                            'background-color': 'green',
                            'lineHeight': '37px',
                            'borderWidth': '1px',
                            'borderStyle': 'solid',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
        }

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == id('upload-pythTB-json'):
            return pythTB_style
        return pythTB_style
    
    # dashboard calculate TB
    @app.callback(
        [Output(id('tb-data'), 'data'),
         Output(id('upload-w90-hr'), 'children'),
         Output(id('upload-w90-wout'), 'children'),
         Output(id('upload-pythTB-json'), 'children'),
         Output(id('tb-bands'), 'on'),
         Output(id('dft-mu'), 'children'),
         Output(id('gf-filling'), 'value'),
         Output(id('orbital-order'), 'options'),
         Output(id('band-basis'), 'on')],
        [Input(id('upload-w90-hr'), 'contents'),
         Input(id('upload-w90-hr'), 'filename'),
         Input(id('upload-w90-hr'), 'children'),
         Input(id('upload-w90-wout'), 'contents'),
         Input(id('upload-w90-wout'), 'filename'),
         Input(id('upload-w90-wout'), 'children'),
         Input(id('upload-pythTB-json'), 'contents'),
         Input(id('upload-pythTB-json'), 'filename'),
         Input(id('upload-pythTB-json'), 'children'),
         Input(id('tb-bands'), 'on'),
         Input(id('calc-tb'), 'n_clicks'),
         Input(id('gf-filling'), 'value'),
         Input(id('calc-tb-mu'), 'n_clicks'),
         Input(id('tb-data'), 'data'),
         Input(id('add-spin'), 'value'),
         Input(id('dft-mu'), 'children'),
         Input(id('n-k'), 'value'),
         Input(id('k-points'), 'data'),
         Input(id('loaded-data'), 'data'),
         Input(id('orbital-order'),'options'),
         Input(id('eta'), 'value'),
         Input(id('band-basis'), 'on')],
         prevent_initial_call=True,)
    def calc_tb(w90_hr, w90_hr_name, w90_hr_button, w90_wout, w90_wout_name,
                w90_wout_button, pythTB, pythTB_name, pythTB_button, tb_switch, click_tb, n_elect, click_tb_mu, tb_data, add_spin, dft_mu, n_k,
                k_points, loaded_data, orb_options, eta, band_basis):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***calc_tb***:'), trigger_id)
        
        if trigger_id == id('tb-bands'):
            return tb_data, w90_hr_button, w90_wout_button, pythTB_button, tb_switch, dft_mu, n_elect, orb_options, band_basis

        # if w90_hr != None and not 'loaded_hr' in tb_data:
        if trigger_id == id('upload-w90-hr'):
            print('loading w90 hr file...')
            hopping, n_wf = load_w90_hr(w90_hr)
            hopping = {str(key): value.real.tolist() for key, value in hopping.items()}
            tb_data['n_wf'] = n_wf
            tb_data['hopping'] = hopping
            tb_data['loaded_hr'] = True
            tb_data['dft_mu'] = 0.0
            orb_options = [{'label': str(k), 'value': str(k)} for i, k in enumerate(list(permutations([i for i in range(tb_data['n_wf'])])))]
            return tb_data, html.Div([w90_hr_name]), w90_wout_button, pythTB_button, tb_switch, dft_mu, n_elect, orb_options, band_basis

        # if w90_wout != None and not 'loaded_wout' in tb_data:
        if trigger_id == id('upload-w90-wout'):
            print('loading w90 wout file...')
            tb_data['units'] = load_w90_wout(w90_wout)
            tb_data['loaded_wout'] = True

            return tb_data, w90_hr_button, html.Div([w90_wout_name]), pythTB_button, tb_switch, dft_mu, n_elect, orb_options, band_basis

        # if pythTB is being loaded
        if trigger_id == id('upload-pythTB-json'):
            print('loading pythTB .json file...')
            print(pythTB_name)
            n_orb, units, hoppings = load_pythTB_json(pythTB)
            hoppings = {str(key): value.real.tolist() for key, value in hoppings.items()}

            tb_data['n_wf'] = n_orb
            tb_data['units'] = units
            tb_data['hopping'] = hoppings
            tb_data['dft_mu'] = 0.0

            # a little hack to turn the flags true to continue using the existing functionality
            tb_data['loaded_hr'] = True
            tb_data['loaded_wout'] = True

            return tb_data, w90_hr_button, w90_wout_button, html.Div([pythTB_name]), tb_switch, dft_mu, n_elect, orb_options, band_basis

        # if a full config has been uploaded
        if trigger_id == id('loaded-data'):
            print('set uploaded data as tb_data')
            tb_data = loaded_data['tb_data']
            tb_data['use'] = True
            orb_options = [{'label': str(k), 'value': str(k)} for i, k in enumerate(list(permutations([i for i in range(tb_data['n_wf'])])))]

            return tb_data, w90_hr_button, w90_wout_button, pythTB_button, {'on': True}, html.P('{:.4f}'.format(tb_data['dft_mu'])), tb_data['n_elect'], orb_options, tb_data['band_basis']

        if trigger_id == id('calc-tb-mu') and ((tb_data['loaded_hr'] and tb_data['loaded_wout']) or tb_data['use']):
            if float(n_elect) == 0.0:
                print('please specify filling')
                return tb_data, w90_hr_button, w90_wout_button, pythTB_button, tb_switch, dft_mu, n_elect, orb_options, band_basis

            add_local = [0.] * tb_data['n_wf']
            tb_data['dft_mu'], tb_data['eps_min_max'] = akw.calc_mu(tb_data, float(n_elect), add_spin, add_local, eta=float(eta))

            return tb_data, w90_hr_button, w90_wout_button, pythTB_button, tb_switch, html.P('{:.4f}'.format(tb_data['dft_mu'])), n_elect, orb_options, band_basis

        else:
            if not click_tb > 0 and not tb_data['use']:
                return tb_data, w90_hr_button, w90_wout_button, pythTB_button, tb_switch, dft_mu, n_elect, orb_options, band_basis
            if np.any([k_val in ['', None] for k in k_points for k_key, k_val in k.items()]):
                return tb_data, w90_hr_button, w90_wout_button, pythTB_button, tb_switch, dft_mu, n_elect, orb_options, band_basis



            if not isinstance(n_k, int):
                n_k = 20

            add_local = [0.] * tb_data['n_wf']

            k_mesh = {'n_k': int(n_k), 'k_path': k_points, 'kz': 0.0}
            tb_data['k_mesh'], e_mat, e_vecs, tbl = tb.calc_tb_bands(tb_data, add_spin, add_local, k_mesh, fermi_slice=False, band_basis=band_basis)

            # calculate Hamiltonian
            tb_data['e_mat_re'] = e_mat.real.tolist()
            tb_data['e_mat_im'] = e_mat.imag.tolist()
            if band_basis:
                tb_data['evecs_re'] = e_vecs.real.tolist()
                tb_data['evecs_im'] = e_vecs.imag.tolist()
                tb_data['eps_nuk'] = np.einsum('iij -> ij', e_mat-tb_data['dft_mu']).real.tolist()
            else:
                tb_data['eps_nuk'], evec_nuk = tb.get_tb_bands(e_mat, tb_data['dft_mu'])
                tb_data['eps_nuk'] = tb_data['eps_nuk'].tolist()
            tb_data['bnd_low'] = np.min(np.array(tb_data['eps_nuk'][0])).real
            tb_data['bnd_high'] = np.max(np.array(tb_data['eps_nuk'][-1])).real
            tb_data['n_elect'] = float(n_elect)
            tb_data['band_basis'] = band_basis
            if not add_spin:
                tb_data['add_spin'] = False
            else:
                tb_data['add_spin'] = True
            tb_data['use'] = True

            return tb_data, w90_hr_button,w90_wout_button, pythTB_button, {'on': True}, dft_mu, n_elect, orb_options, band_basis

    # dashboard k-points
    @app.callback(
        [Output(id('k-points'), 'data'),
         Output(id('n-k'), 'value')],
        [Input(id('add-kpoint'), 'n_clicks'),
         Input(id('n-k'), 'value'),
         Input(id('loaded-data'), 'data')],
        State(id('k-points'), 'data'),
        State(id('k-points'), 'columns'),
        prevent_initial_call=True)
    def add_row(n_clicks, n_k, loaded_data, rows, columns):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # if a full config has been uploaded update kpoints
        if trigger_id == id('loaded-data'):
            rows = loaded_data['tb_data']['k_mesh']['k_points_dash']
            n_k = int(len(loaded_data['tb_data']['k_mesh']['k_disc'])/(len(loaded_data['tb_data']['k_mesh']['k_points'])-1))
            return rows, n_k

        for row, col in product(rows, range(1,4)):
            try:
                row[id(f'column-{col}')] = float(row[id(f'column-{col}')])
            except:
                row[id(f'column-{col}')] = None

        if trigger_id == id('add-kpoint'):
            rows.append({c['id']: '' for c in columns})

        return rows, n_k

    # dashboard sigma upload
    @app.callback(
         [Output(id('sigma-data'), 'data'),
          Output(id('sigma-function'), 'style'),
          Output(id('sigma-upload'), 'style'),
          Output(id('sigma-upload-box'), 'children'),
          Output(id('orbital-order'), 'value'),
          Output(id('orb-alert'), 'is_open'),
          Output(id('sigma-params'), 'data'),
          Output(id('sigma-params'), 'columns'),
          Output(id('sigma-function-output'), 'children')],
         [Input(id('sigma-data'), 'data'),
          Input(id('tb-data'), 'data'),
          Input(id('choose-sigma'), 'value'),
          Input(id('sigma-upload-box'), 'contents'),
          Input(id('sigma-upload-box'), 'filename'),
          Input(id('sigma-upload-box'), 'children'),
          Input(id('loaded-data'), 'data'),
          Input(id('sigma-function-button'), 'n_clicks'),
          Input(id('orbital-order'), 'value')],
          # Input({'type': 'sigma-lambdas', 'index': ALL}, 'value')],
         [State(id('orb-alert'), 'is_open'),
          State(id('sigma-params'), 'data'),
          State(id('sigma-params'), 'columns')],
         prevent_initial_call=False
        )
    def toggle_update_sigma(sigma_data, tb_data, sigma_radio_item, sigma_content, sigma_filename,
                            sigma_button, loaded_data, n_clicks_sigma, orbital_order, orb_alert, sigma_params, sigma_columns):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***update_sigma***:'), trigger_id)

        return_f_sigma = None
        # orbital_order = tuple(int(i) for i in orbital_order.strip('()').split(','))
        orbital_order = literal_eval(orbital_order)

        # update Sigma params if n_wf has changed
        if trigger_id == id('tb-data') and tb_data['n_wf'] != len(sigma_columns)-1:
            print(sigma_columns)
            n_orb_table = len(sigma_columns)-1
            orbital_order = tuple(range(tb_data['n_wf']))
            if n_orb_table > tb_data['n_wf']:
                # remove columns
                for i in range(n_orb_table - tb_data['n_wf']):
                    sigma_columns.pop()
                    sigma_params[0].popitem()
                    sigma_params[1].popitem()
            else:
                for i in range(tb_data['n_wf'] - n_orb_table):
                    i_orb = int(sigma_columns[-1]['id'].split('-')[-1]) + 1
                    new_col = {
                        'name': 'orb{}'.format(i_orb),
                        'id': 'tab1-column-{}'.format(i_orb),
                        'deletable': False
                    }
                    sigma_columns.append(new_col)
                    sigma_params[0][new_col['id']] = '1.0'
                    sigma_params[1][new_col['id']] = '0.0'

            return sigma_data, {'display': 'none'}, {'display': 'block'}, sigma_button, str(tuple(orbital_order)), orb_alert, sigma_params, sigma_columns, return_f_sigma

        if trigger_id == id('loaded-data') and 'sigma_data' in loaded_data.keys():
            print('set uploaded data as sigma_data')
            sigma_data = loaded_data['sigma_data']
            sigma_data['use'] = True
            sigma_button = html.Div([loaded_data['config_filename']])

            # somehow the orbital order is transformed back to lists all the time, so make sure here that it is a tuple!
            return sigma_data, {'display': 'none'}, {'display': 'block'}, sigma_button, str(tuple(sigma_data['orbital_order'])), orb_alert, sigma_params, sigma_columns, return_f_sigma

        if trigger_id == id('orbital-order'):
            print('the orbital order has changed', orbital_order)
            if sigma_data['use'] == True:
                sigma_data = gf.reorder_sigma(sigma_data, new_order=orbital_order, old_order=sigma_data['orbital_order'])
            return sigma_data, {'display': 'none'}, {'display': 'block'}, sigma_button, str(tuple(orbital_order)), orb_alert, sigma_params, sigma_columns, return_f_sigma

        if sigma_radio_item == 'upload':

            if sigma_content != None and trigger_id == id('sigma-upload-box'):
                print('loading Sigma from file...')
                sigma_data = load_sigma_h5(sigma_content, sigma_filename)
                # check if number of orbitals match and reject data if no match
                if sigma_data['n_orb'] != tb_data['n_wf']:
                    return sigma_data, {'display': 'none'}, {'display': 'block'}, sigma_button, str(tuple(orbital_order)), not orb_alert, sigma_params, sigma_columns, return_f_sigma
                print('successfully loaded sigma from file')
                sigma_data['use'] = True
                orbital_order = sigma_data['orbital_order']
                sigma_button = html.Div([sigma_filename])

            return sigma_data, {'display': 'none'}, {'display': 'block'}, sigma_button, str(tuple(orbital_order)), orb_alert, sigma_params, sigma_columns, return_f_sigma

        if trigger_id == id('sigma-function-button'):

            if not tb_data['use']:
                # TODO: create warning before return
                return sigma_data, {'display': 'block'}, {'display': 'none'}, sigma_button, str(tuple(orbital_order)), orb_alert, sigma_params, sigma_columns, return_f_sigma

            if np.any([S_val in ['', None] for params in sigma_params for S_key, S_val in params.items()]):
                return sigma_data, {'display': 'block'}, {'display': 'none'}, sigma_button, str(tuple(orbital_order)), orb_alert, sigma_params, sigma_columns, return_f_sigma

                # append numpy and math
                # import_np = 'import numpy as np'
                # import_math = 'import math'
                # imports = [import_np, import_math]
                # namespace = {}
                # # parse function
                # for to_parse in [*imports, f_sigma]:
                # tree = ast.parse(to_parse, mode='exec')
                # code = compile(tree, filename='test', mode='exec')
                # exec(code, namespace)
                # # curry
                # c_sigma = tools.curry(namespace['sigma'])
                # # get lambdas from dashboard if trigger, else default values
                # lambda_values = sigma_lambdas if '"type":"sigma-lambdas"' in trigger_id else [1, 1]
                # lambda_tuples = [key for key in zip(inspect.getfullargspec(namespace['sigma'])[0][1:], lambda_values)]
                # lambda_list = _build_lambda_children(True, lambdas=lambda_tuples)
                # lambda_view = {'display': 'inline-block'}

            if n_clicks_sigma > 0:
                # TODO: create warning, see above
                n_orb = tb_data['n_wf']
                w_min = tb_data['bnd_low'] - 0.2*abs(tb_data['bnd_low'])
                w_max = tb_data['bnd_high'] + 0.2*abs(tb_data['bnd_high'])
                n_w = 1001
                soc = False

                w_mesh = MeshReFreq(omega_min=w_min, omega_max=w_max, n_max=n_w)
                w_dict = {'w_mesh' : w_mesh, 'n_w' : n_w, 'window' : [w_min, w_max]}

                # Sigma(w) = (1-1/Z) w + Sigma_0
                # Sigma_0 : correlated crystal field
                Z = []
                Sigma_0 = []
                Z = list(map(float, [val for val in sigma_params[0].values()][1:]))
                Sigma_0 = list(map(float, [val for val in sigma_params[1].values()][1:]))
                assert (len(Z) == n_orb and len(Sigma_0) == n_orb), 'Z and Sigma_0 do not have the right length'

                sigma_mu_analytic = gf.sigma_analytic_to_gf(n_orb, w_dict, Sigma_0, [1 for i in range(n_orb)], soc)
                sigma_data['sigma_mu_re'] = sigma_mu_analytic.real.tolist()
                sigma_data['sigma_mu_im'] = sigma_mu_analytic.imag.tolist()

                sigma_analytic = gf.sigma_analytic_to_gf(n_orb, w_dict, Sigma_0, Z, soc)
                sigma_data.update(gf.sigma_analytic_to_data(sigma_analytic, w_dict, n_orb))
                # initial dmft_mu to dft_mu
                sigma_data['dmft_mu'] = tb_data['dft_mu']
                sigma_data['orbital_order'] = orbital_order

                return_f_sigma = 'You have entered: \nZ={}\nΣ₀={}'.format(Z,Sigma_0)

            return sigma_data, {'display': 'block'}, {'display': 'none'}, sigma_button, str(tuple(orbital_order)), orb_alert, sigma_params, sigma_columns, return_f_sigma

        return sigma_data, {'display': 'block'}, {'display': 'none'}, sigma_button, str(tuple(orbital_order)), orb_alert, sigma_params, sigma_columns, return_f_sigma

    # dashboard colors
    @app.callback(
        Output(id('colorscale'), 'options'),
        Input(id('colorscale-mode'), 'value'),
        prevent_initial_call=False,
        )
    def update_colorscales(mode):
        colorscales = [name for name, body in inspect.getmembers(getattr(px.colors, mode))
                if isinstance(body, list) and len(name.rsplit('_')) == 1]
        return [{'label': key, 'value': key} for key in colorscales]

    # plot A(k,w)
    @app.callback(
        Output(id('Akw'), 'figure'),
        [Input(id('tb-bands'), 'on'),
         Input(id('akw-bands'), 'on'),
         Input(id('colorscale'), 'value'),
         Input(id('tb-data'), 'data'),
         Input(id('akw-data'), 'data'),
         Input(id('sigma-data'), 'data')],
         prevent_initial_call=True)
    def plot_Akw(tb_switch, akw_switch, colorscale, tb_data, akw_data, sigma_data):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***plot_Akw***:'), trigger_id)

        # initialize general figure environment
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', rangeslider_visible=False,
                         showticklabels=True, spikedash='solid')
        fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showticklabels=True, spikedash='solid')
        fig.update_traces(xaxis='x', hoverinfo='none')

        if not tb_data['use']:
            return fig

        k_mesh = tb_data['k_mesh']
        fig.add_shape(type = 'line', x0=0, y0=0, x1=max(k_mesh['k_disc']), y1=0, line=dict(color='gray', width=0.8))
        print(k_mesh['k_points_dash'])
        if tb_switch:
            for band in range(len(tb_data['eps_nuk'])):
                fig.add_trace(go.Scattergl(x=k_mesh['k_disc'], y=tb_data['eps_nuk'][band], mode='lines',
                            line=go.scattergl.Line(color=px.colors.sequential.Viridis[0]), showlegend=False, text=f'tb band {band}',
                            hoverinfo='x+y+text'
                            ))
            if not akw_switch:
                fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                          clickmode='event+select',
                          hovermode='closest',
                          xaxis_range=[k_mesh['k_disc'][0], k_mesh['k_disc'][-1]],
                          yaxis_range=[tb_data['bnd_low']- 0.02*abs(tb_data['bnd_low']) ,
                                       tb_data['bnd_high']+ 0.02*abs(tb_data['bnd_high'])],
                                  yaxis_title='ω(eV)',
                          xaxis=dict(ticktext=['γ' if k == 'g' else k for k in k_mesh['k_point_labels']],tickvals=k_mesh['k_points']),
                          font=dict(size=20))

        if not akw_data['use']:
            return fig

        if akw_switch:
            w_mesh = sigma_data['w_dict']['w_mesh']
            if akw_data['solve']:
                z_data = np.array(akw_data['Akw'])
                for orb in range(z_data.shape[1]):
                    #fig.add_trace(go.Contour(x=k_mesh['k_disc'], y=w_mesh, z=z_data[:,:,orb].T,
                    #    colorscale=colorscale, contours=dict(start=0.1, end=1.5, coloring='lines'), ncontours=1, contours_coloring='lines'))
                    fig.add_trace(go.Scattergl(x=k_mesh['k_disc'], y=z_data[:,orb].T, showlegend=False, mode='markers',
                                               marker_color=px.colors.sequential.Viridis[0]))
            else:
                # z_data = np.log(np.array(akw_data['Akw']).T)
                z_data = np.array(akw_data['Akw']).T
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
                              font=dict(size=20))

        return fig

    # plot EDC
    @app.callback(
       [Output('EDC', 'figure'),
        Output('kpt_edc', 'value'),
        Output('kpt_edc', 'max')],
       [Input(id('tb-bands'), 'on'),
        Input(id('akw-bands'), 'on'),
        Input(id('sum-edc'), 'on'),
        Input('kpt_edc', 'value'),
        Input(id('akw-data'), 'data'),
        Input(id('tb-data'), 'data'),
        Input(id('Akw'), 'clickData'),
        Input(id('sigma-data'), 'data')],
        prevent_initial_call=True)
    def update_EDC(tb_bands, akw_bands, sum_edc, kpt_edc, akw_data, tb_data, click_coordinates, sigma_data):
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***update_EDC***:'), trigger_id)

        if tb_data['use']: tb_temp = tb_data
        if not 'tb_temp' in locals():
            return fig, 0, 1

        k_mesh = tb_temp['k_mesh']

        if tb_bands:
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
            if not akw_bands:
                fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                                hovermode='closest',
                                xaxis_range=[tb_data['bnd_low']- 0.02*abs(tb_data['bnd_low']) ,
                                             tb_data['bnd_high']+ 0.02*abs(tb_data['bnd_high'])],
                                yaxis_range=[0, 1],
                                xaxis_title='ω (eV)',
                                yaxis_title='A(ω)',
                                font=dict(size=20),
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                                )
        if akw_bands:
            w_mesh = sigma_data['w_dict']['w_mesh']
            if sum_edc:
                fig.add_trace(go.Scattergl(x=w_mesh, y=np.array(akw_data['Aw']), mode='lines',
                    line=go.scattergl.Line(color='#AB63FA'), showlegend=False,
                                           hoverinfo='x+y+text'
                                        ))
                fig.update_layout(yaxis_title='A(ω) summed')
            else:
                if trigger_id == id('Akw'):
                    new_kpt = click_coordinates['points'][0]['x']
                    kpt_edc = np.argmin(np.abs(np.array(k_mesh['k_disc']) - new_kpt))

                fig.add_trace(go.Scattergl(x=w_mesh, y=np.array(akw_data['Akw'])[kpt_edc,:], mode='lines',
                    line=go.scattergl.Line(color='#AB63FA'), showlegend=False,
                                           hoverinfo='x+y+text'
                                        ))
                fig.update_layout(yaxis_title='A(ω)')

            fig.update_layout(
                margin={
                    'l': 40,
                    'b': 40,
                    't': 10,
                    'r': 40
                },
                hovermode='closest',
                xaxis_range=[w_mesh[0], w_mesh[-1]],
                yaxis_range=[0, 1.01 * np.max(np.array(akw_data['Akw']))],
                xaxis_title='ω (eV)',
                font=dict(size=20),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

        if akw_bands:
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
        Input(id('Akw'), 'clickData'),
        Input(id('sigma-data'), 'data')],
        prevent_initial_call=True)
    def update_MDC(tb_bands, akw_bands, w_mdc, akw_data, tb_data, click_coordinates, sigma_data):
        layout = go.Layout()
        fig = go.Figure(layout=layout)
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***update_MDC***:'), trigger_id)

        if tb_data['use']: tb_temp = tb_data
        if not 'tb_temp' in locals():
            return fig, 0, 1

        k_mesh = tb_temp['k_mesh']

        #if tb_bands:
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

        if akw_bands:
            w_mesh = sigma_data['w_dict']['w_mesh']
            if trigger_id == id('Akw'):
                new_w = click_coordinates['points'][0]['y']
                w_mdc = np.argmin(np.abs(np.array(w_mesh) - new_w))

            fig.add_trace(go.Scattergl(x=k_mesh['k_disc'], y=np.array(akw_data['Akw'])[:, w_mdc], mode='lines', line=go.scattergl.Line(color='#AB63FA'),
                                       showlegend=True, name='ω = {:.3f} eV'.format(w_mesh[w_mdc]), hoverinfo='x+y+text'))

            fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
                              hovermode='closest',
                              xaxis_range=[k_mesh['k_disc'][0], k_mesh['k_disc'][-1]],
                        #     yaxis_range=[0, 1.05 * np.max(np.array(akw_data['Akw'])[:, w_mdc])],
                              yaxis_range=[0, 1.01 * np.max(np.array(akw_data['Akw']))],
                              xaxis_title='k',
                              yaxis_title='A(k)',
                              font=dict(size=20),
                              xaxis=dict(ticktext=['γ' if k == 'g' else k for k in k_mesh['k_point_labels']], tickvals=k_mesh['k_points']),
                              legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                              )
        if akw_bands:
            return fig, w_mdc, len(w_mesh)-1
        # elif tb_bands:
        #     return fig, w_mdc, np.max(np.array(tb_temp['eps_nuk'][-1]))+(0.03*np.max(np.array(tb_temp['eps_nuk'][-1])))
        else:
            return fig, 0, 1

    @app.callback(
    Output(id('download_h5'), "data"),
    [Input(id('dwn_button'), "n_clicks"),
     Input(id('tb-data'), 'data'),
     Input(id('akw-data'), 'data'),
     Input(id('sigma-data'), 'data'),
     Input(id('band-basis'), 'on')],
     prevent_initial_call=True,
    )
    def download_data(n_clicks, tb_data, akw_data, sigma_data, band_basis):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print('{:20s}'.format('***download_data***:'), trigger_id)
        # check if the download button was pressed
        if trigger_id == id('dwn_button'):
            return_data = HDFArchive(descriptor = None, open_flag='a')

            # store everything as np arrays not as list to enable compression in h5 write!
            tb_data_store = tb_data.copy()
            tb_data_store['e_mat'] = np.array(tb_data['e_mat_re']) + 1j * np.array(tb_data['e_mat_im'])
            if band_basis:
                tb_data_store['e_vecs'] = np.array(tb_data['evecs_re']) + 1j * np.array(tb_data['evecs_im'])
                del tb_data_store['evecs_re']
                del tb_data_store['evecs_im']
            tb_data_store['eps_nuk'] = np.array(tb_data['eps_nuk'])
            tb_data_store['hopping'] = {str(key): np.array(value) for key, value in tb_data_store['hopping'].items()}
            return_data['tb_data'] = tb_data_store

            if akw_data['use']:
                sigma_data_store = sigma_data.copy()
                sigma_data_store['sigma'] = np.array(sigma_data['sigma_re']) + 1j * np.array(sigma_data['sigma_im'])
                del sigma_data_store['sigma_re']
                del sigma_data_store['sigma_im']
                sigma_data_store['w_dict']['w_mesh'] = np.array(sigma_data['w_dict']['w_mesh'])
                sigma_data_store['dmft_mu']  = akw_data['dmft_mu']
                return_data['sigma_data'] = sigma_data_store

            content = base64.b64encode(return_data.as_bytes()).decode()

            return dict(content=content, filename='fermisee.h5', base64=True)
        else:
            return None

    @app.callback(
    Output(id('sec2-collapse'), "is_open"),
    [Input(id('sec2-collapse-button'), "n_clicks")],
    [State(id('sec2-collapse'), "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
    Output(id('sec3-collapse'), "is_open"),
    [Input(id('sec3-collapse-button'), "n_clicks")],
    [State(id('sec3-collapse'), "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
    Output(id('sec4-collapse'), "is_open"),
    [Input(id('sec4-collapse-button'), "n_clicks")],
    [State(id('sec4-collapse'), "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    return
