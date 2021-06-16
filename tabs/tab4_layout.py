import dash_core_components as dcc
import dash_html_components as html

def layout(data):
    return dcc.Tab(
            label='Raman spectroscopy',
            children=[
                html.Div([
                    html.H1('Page under construction 🚧'),
                    ],
                    style={
                        'display': 'inline-block',
                        'width': '100%',
                        'textAlign': 'center',
                        'margin-top': '40%'
                        }
                    )
                ]
            )
