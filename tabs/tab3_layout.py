import dash_core_components as dcc
import dash_html_components as html

from tabs.dashboard import make_dashboard

def layout(tb_data, tb_kslice_data, akw_data, sigma_data, loaded_data):
    return dcc.Tab(
        label='Optical spectroscopy',
        children=[
            # column 1
            make_dashboard(tb_data, tb_kslice_data, akw_data, sigma_data, loaded_data, 3),
            # column 2
            html.Div([
                html.H1('Page under construction 🚧'),
                ],
                style={
                    'display': 'inline-block',
                    'width': '84%',
                    'textAlign': 'center',
                    'margin-top': '40%'
                    }
                )
            ]
        )
