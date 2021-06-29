import dash
import dash_bootstrap_components as dbc
from load_data import load_config
from layout import layout
from tabs.tab1_callbacks import register_callbacks as tab1_callbacks
from flask import Flask

server = Flask(__name__)

external_stylesheets = [dbc.themes.FLATLY, 'https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,server=server)



data = {'use': False}
tb_data = {'use': False}
sigma_data = {'use': False}
akw_data = {'use': False}
app.layout = layout(data, tb_data, akw_data, sigma_data)
tab1_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True, port=9375, host='0.0.0.0')
