import dash_core_components as dcc
import dash_html_components as html

from tabs.dashboard import make_dashboard

def layout(tb_data, akw_data, sigma_data):
    return dcc.Tab(
        label='Fermi surface A(k,0)',
        children=[
            # column 1
            make_dashboard(tb_data, akw_data, sigma_data, 2),
            # column 2
            html.Div([
                dcc.Graph(
                    id='A0k',
                    style={'height': '60vh'},
                    )
                ],
                style={
                    'display': 'inline-block',
                    'width': '70%',
                    'textAlign': 'center',
                    }
                )
            ]
        )
