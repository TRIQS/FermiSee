import dash
import dash_bootstrap_components as dbc
from load_data import load_config
from layout import layout
from tabs.tab1_callbacks import register_callbacks as tab1_callbacks
from tabs.tab2_callbacks import register_callbacks as tab2_callbacks
from flask import Flask

server = Flask(__name__)

external_stylesheets = [dbc.themes.FLATLY, 'https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,server=server, prevent_initial_callbacks=True)
app.title = 'triqs_spectrometer'


tb_data = {'use': False, 'loaded_hr': False, 'loaded_wout' : False}
tb_kslice_data = dict(tb_data)
sigma_data = {'use': False}
akw_data = {'use': False}
ak0_data = dict(akw_data)
loaded_data = {}
app.layout = layout(tb_data, tb_kslice_data, akw_data, ak0_data, sigma_data, loaded_data)
tab1_callbacks(app)
tab2_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True, port=9375, host='0.0.0.0')
