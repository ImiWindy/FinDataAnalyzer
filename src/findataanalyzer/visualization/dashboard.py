"""Dashboard interface for financial data visualization."""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import base64
import io

from findataanalyzer.core.analyzer import DataAnalyzer
from findataanalyzer.core.predictor import Predictor
from findataanalyzer.core.data_loader import DataLoader
from findataanalyzer.visualization.charts import ChartGenerator


class Dashboard:
    """Dashboard for financial data visualization using Dash."""
    
    def __init__(self, title: str = "FinDataAnalyzer Dashboard"):
        """
        Initialize the dashboard.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = title
        self.data_loader = DataLoader()
        self.analyzer = DataAnalyzer()
        self.predictor = Predictor()
        self.chart_generator = ChartGenerator()
        
        # Set up the layout
        self._setup_layout()
        # Set up callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(self.title),
                html.P("آنالیز و نمایش داده‌های مالی")
            ], className="header"),
            
            # Data upload section
            html.Div([
                html.H2("بارگذاری داده"),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        "فایل را به اینجا بکشید یا ",
                        html.A("انتخاب کنید")
                    ]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px"
                    },
                    multiple=False
                ),
                html.Div(id="upload-status")
            ], className="section"),
            
            # Data source URL
            html.Div([
                html.H2("یا از URL استفاده کنید"),
                dcc.Input(
                    id="data-url",
                    type="text",
                    placeholder="آدرس URL داده (CSV, Excel, JSON)",
                    style={"width": "100%", "padding": "8px"}
                ),
                html.Button(
                    "بارگذاری",
                    id="load-url-button",
                    style={"margin-top": "10px"}
                )
            ], className="section"),
            
            # Data preview
            html.Div([
                html.H2("پیش‌نمایش داده"),
                html.Div(id="data-preview-container")
            ], className="section"),
            
            # Visualization options
            html.Div([
                html.H2("نمودارها"),
                
                # Tabs for different visualizations
                dcc.Tabs([
                    dcc.Tab(label="سری زمانی", children=[
                        html.Div([
                            html.H3("تنظیمات نمودار سری زمانی"),
                            html.Label("ستون زمان:"),
                            dcc.Dropdown(id="time-series-time-col"),
                            html.Label("ستون‌های مقدار:"),
                            dcc.Dropdown(id="time-series-value-cols", multi=True),
                            html.Button("نمایش نمودار", id="plot-time-series-button"),
                            html.Div(id="time-series-container")
                        ], className="tab-content")
                    ]),
                    dcc.Tab(label="شمعی", children=[
                        html.Div([
                            html.H3("تنظیمات نمودار شمعی"),
                            html.Label("ستون تاریخ:"),
                            dcc.Dropdown(id="candlestick-date-col"),
                            html.Label("ستون قیمت باز:"),
                            dcc.Dropdown(id="candlestick-open-col"),
                            html.Label("ستون قیمت بالا:"),
                            dcc.Dropdown(id="candlestick-high-col"),
                            html.Label("ستون قیمت پایین:"),
                            dcc.Dropdown(id="candlestick-low-col"),
                            html.Label("ستون قیمت بسته:"),
                            dcc.Dropdown(id="candlestick-close-col"),
                            html.Button("نمایش نمودار", id="plot-candlestick-button"),
                            html.Div(id="candlestick-container")
                        ], className="tab-content")
                    ]),
                    dcc.Tab(label="همبستگی", children=[
                        html.Div([
                            html.H3("نمودار همبستگی"),
                            html.Label("انتخاب ستون‌ها:"),
                            dcc.Dropdown(id="correlation-cols", multi=True),
                            html.Button("نمایش نمودار", id="plot-correlation-button"),
                            html.Div(id="correlation-container")
                        ], className="tab-content")
                    ]),
                    dcc.Tab(label="توزیع", children=[
                        html.Div([
                            html.H3("نمودار توزیع"),
                            html.Label("انتخاب ستون:"),
                            dcc.Dropdown(id="distribution-col"),
                            html.Button("نمایش نمودار", id="plot-distribution-button"),
                            html.Div(id="distribution-container")
                        ], className="tab-content")
                    ])
                ])
            ], className="section"),
            
            # Analysis section
            html.Div([
                html.H2("تحلیل داده‌ها"),
                html.Button("انجام تحلیل", id="run-analysis-button"),
                html.Div(id="analysis-results-container")
            ], className="section"),
            
            # Prediction section
            html.Div([
                html.H2("پیش‌بینی"),
                html.Label("ستون هدف:"),
                dcc.Dropdown(id="prediction-target-col"),
                html.Label("افق پیش‌بینی (تعداد گام‌های آینده):"),
                dcc.Input(
                    id="prediction-horizon",
                    type="number",
                    value=5,
                    min=1,
                    max=30
                ),
                html.Label("روش پیش‌بینی:"),
                dcc.RadioItems(
                    id="prediction-method",
                    options=[
                        {"label": "رگرسیون خطی", "value": "linear"},
                        {"label": "ARIMA", "value": "arima"}
                    ],
                    value="linear"
                ),
                html.Button("انجام پیش‌بینی", id="run-prediction-button"),
                html.Div(id="prediction-results-container")
            ], className="section"),
            
            # Store components for data
            dcc.Store(id="stored-data"),
            dcc.Store(id="stored-analysis-results"),
            dcc.Store(id="stored-prediction-results")
        ], className="dashboard-container")
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        @self.app.callback(
            [Output("stored-data", "data"),
             Output("upload-status", "children"),
             Output("time-series-time-col", "options"),
             Output("time-series-value-cols", "options"),
             Output("candlestick-date-col", "options"),
             Output("candlestick-open-col", "options"),
             Output("candlestick-high-col", "options"),
             Output("candlestick-low-col", "options"),
             Output("candlestick-close-col", "options"),
             Output("correlation-cols", "options"),
             Output("distribution-col", "options"),
             Output("prediction-target-col", "options")],
            [Input("upload-data", "contents"),
             Input("load-url-button", "n_clicks")],
            [State("upload-data", "filename"),
             State("data-url", "value")]
        )
        def update_data(contents, n_clicks, filename, data_url):
            """Update data when uploaded or loaded from URL."""
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return [None, "", [], [], [], [], [], [], [], [], [], []]
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            try:
                if trigger_id == "upload-data" and contents:
                    # Parse the contents
                    content_type, content_string = contents.split(",")
                    decoded = base64.b64decode(content_string)
                    df = self.data_loader.load_from_content(decoded)
                    status = html.Div([
                        html.P(f"فایل بارگذاری شد: {filename}")
                    ], style={"color": "green"})
                
                elif trigger_id == "load-url-button" and data_url:
                    # Load data from URL
                    df = self.data_loader.load_data(data_url)
                    status = html.Div([
                        html.P(f"داده از URL بارگذاری شد")
                    ], style={"color": "green"})
                
                else:
                    return [None, "", [], [], [], [], [], [], [], [], [], []]
                
                # Generate options for dropdowns
                all_cols = [{"label": col, "value": col} for col in df.columns]
                num_cols = [{"label": col, "value": col} for col in df.select_dtypes(include=['number']).columns]
                
                # Convert DataFrame to dictionary for storage
                data_dict = df.to_dict("records")
                
                return [
                    data_dict, 
                    status, 
                    all_cols,  # time-series-time-col
                    num_cols,  # time-series-value-cols
                    all_cols,  # candlestick-date-col
                    num_cols,  # candlestick-open-col
                    num_cols,  # candlestick-high-col
                    num_cols,  # candlestick-low-col
                    num_cols,  # candlestick-close-col
                    num_cols,  # correlation-cols
                    num_cols,  # distribution-col
                    num_cols   # prediction-target-col
                ]
                
            except Exception as e:
                return [
                    None, 
                    html.Div([html.P(f"خطا: {str(e)}")], style={"color": "red"}),
                    [], [], [], [], [], [], [], [], [], []
                ]
        
        @self.app.callback(
            Output("data-preview-container", "children"),
            Input("stored-data", "data")
        )
        def update_data_preview(data):
            """Update data preview when data is loaded."""
            if not data:
                return html.P("هیچ داده‌ای بارگذاری نشده است.")
            
            # Convert stored data back to DataFrame
            df = pd.DataFrame(data)
            
            # Create a preview table
            return html.Div([
                html.P(f"ابعاد داده: {df.shape[0]} سطر × {df.shape[1]} ستون"),
                html.Div([
                    html.Table(
                        # Header
                        [html.Tr([html.Th(col) for col in df.columns])] +
                        # Rows
                        [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
                         for i in range(min(10, len(df)))]
                    )
                ], style={"overflow-x": "auto"})
            ])
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """
        Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        self.app.run_server(debug=debug, port=port) 