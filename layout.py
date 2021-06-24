import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

from tabs.tab1_layout import layout as tab1_layout
from tabs.tab2_layout import layout as tab2_layout
from tabs.tab3_layout import layout as tab3_layout

def layout(data, tb_data, akw_data, sigma_data):
    return html.Div([
    dcc.Tabs([
        tab1_layout(data, tb_data, akw_data, sigma_data),
        tab2_layout(data, tb_data, akw_data, sigma_data),
        tab3_layout(data, tb_data, akw_data, sigma_data),
        ])
    ])
