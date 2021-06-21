import dash
from load_data import load_config
from layout import layout
from tabs.tab1_callbacks import register_callbacks as tab1_callbacks
from flask import Flask

server = Flask(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,server=server)
data = load_config(None, 'example.h5')
app.layout = layout(data)
tab1_callbacks(app, data)

if __name__ == '__main__':
    app.run_server(debug=True, port=9375, host='0.0.0.0')
