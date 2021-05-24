import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

available_indicators = df['Indicator Name'].unique()

app.layout = html.Div([
    # column 1
    html.Div([
            dcc.Upload(
                id='upload-file',
        	children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '90%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Fertility rate, total (births per woman)'
            ),
            dcc.RadioItems(
                id='linearize',
                options=[{'label': i, 'value': i} for i in ['Σ(ω)', 'linearize']],
                value='Σ(ω)',
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Life expectancy at birth, total (years)'
            ),
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'padding': '10px 5px',
        'display': 'inline-block',
        'width': '19%',
        'vertical-align': 'top'
        }
    ),

    # column 2
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={
        'display': 'inline-block',
        'padding': '0 20'
        }
    ),

    # column 3
    html.Div([
        dcc.Graph(id='x-time-series',
            #config={'fillFrame': True},
            ),
        dcc.Slider(
            id='crossfilter2-year--slider',
            min=df['Year'].min(),
            max=df['Year'].max(),
            value=df['Year'].max(),
            marks={str(year): str(year) for year in df['Year'].unique()},
            step=None,
            verticalHeight=200),
        dcc.Graph(id='y-time-series'),
        dcc.Slider(
            id='crossfilter-year--slider',
            min=df['Year'].min(),
            max=df['Year'].max(),
            value=df['Year'].max(),
            marks={str(year): str(year) for year in df['Year'].unique()},
            step=None),
    ], style={
        'padding': '10px 0px',
        'display': 'inline-block',
        'width': '39%',
        'vertical-align': 'top'
        }
    ),
])


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('linearize', 'value'),
     dash.dependencies.Input('crossfilter-year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Year'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
            )

    fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

#    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
#
#    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')
#
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title):

    fig = px.scatter(dff, x='Year', y='Value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


#@app.callback(
#    dash.dependencies.Output('x-time-series', 'figure'),
#    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
#def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#    country_name = hoverData['points'][0]['customdata']
#    dff = df[df['Country Name'] == country_name]
#    dff = dff[dff['Indicator Name'] == xaxis_column_name]
#    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#    return create_time_series(dff, axis_type, title)


#@app.callback(
#    dash.dependencies.Output('y-time-series', 'figure'),
#    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
#def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
#    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
#    dff = dff[dff['Indicator Name'] == yaxis_column_name]
#    return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(debug=True, port=9375, host='0.0.0.0' ,threaded=True)
