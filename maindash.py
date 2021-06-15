import dash
from load_data import update_data
from layout import layout
from tabs.tab1_callbacks import register_callbacks as tab1_callbacks

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
data = update_data(None, 'example.h5')
app.layout = layout(data)
tab1_callbacks(app, data)
