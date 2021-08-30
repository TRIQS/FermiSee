import dash_core_components as dcc
import dash_html_components as html

from tabs.dashboard import make_dashboard
from tabs.id_factory import id_factory

id = id_factory('tab2')

def layout(tb_data, akw_data, sigma_data, loaded_data):
    return dcc.Tab(
        label='Fermi surface A(k,0)',
        children=[
            # column 1
            make_dashboard(tb_data, akw_data, sigma_data, loaded_data, 2),
            # column 2
            html.Div([
                html.H3('A(k,0)', style={'textAlign': 'center'}),
                dcc.Graph(
                    id=id('Ak0'),
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
            ]
        )
