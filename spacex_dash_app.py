
import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLEAN_FILE = os.path.join(DATA_DIR, "spacex_clean.csv")

def load_clean(path: str = CLEAN_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean data not found at {path}. Run data_wrangling.py first.")
    return pd.read_csv(path)

app = Dash(__name__)
app.title = "SpaceX Landing Dashboard"

df = load_clean()

site_cols = [c for c in df.columns if c.startswith("launch_site__")]
sites = ["All Sites"] + [c.replace("launch_site__", "") for c in site_cols]

app.layout = html.Div([
    html.H1("SpaceX Falcon 9 First Stage Landing — Dashboard"),
    html.Div([
        html.Label("Launch Site"),
        dcc.Dropdown(id="site-dropdown", options=[{"label": s, "value": s} for s in sites], value="All Sites"),
    ], style={"width": "30%"}),
    html.Div([
        html.Label("Payload Count Range"),
        dcc.RangeSlider(id="payload-slider",
                        min=int(df["payload_count"].min()),
                        max=int(df["payload_count"].max()),
                        step=1,
                        value=[int(df["payload_count"].min()), int(df["payload_count"].max())])
    ], style={"marginTop": 20}),
    dcc.Graph(id="success-pie-chart"),
    dcc.Graph(id="success-scatter-chart"),
], style={"maxWidth": "1000px", "margin": "0 auto"})

@app.callback(
    Output("success-pie-chart", "figure"),
    Input("site-dropdown", "value"),
)
def update_pie(site):
    tmp = df.copy()
    if site != "All Sites":
        col = f"launch_site__{site}"
        if col in tmp.columns:
            tmp = tmp[tmp[col] == 1]
    fig = px.pie(tmp, names="Class", title="Success (1) vs Failure (0)")
    return fig

@app.callback(
    Output("success-scatter-chart", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_scatter(site, payload_range):
    lo, hi = payload_range
    tmp = df[(df["payload_count"] >= lo) & (df["payload_count"] <= hi)].copy()
    if site != "All Sites":
        col = f"launch_site__{site}"
        if col in tmp.columns:
            tmp = tmp[tmp[col] == 1]
    # Simple scatter: payload_count vs year colored by Class
    fig = px.scatter(tmp, x="payload_count", y="year", color="Class", title="Payload Count vs Year")
    return fig

if __name__ == "__main__":
    app.run_server(debug=False)
