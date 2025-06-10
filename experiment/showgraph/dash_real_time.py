import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np  # 用于 log10 计算

# CSV 文件路径
csv_file = "change.csv"

# 创建 Dash 应用
app = dash.Dash(__name__)

# Web 界面布局
app.layout = html.Div([
    html.H1("实时 NFA vs PACEP 中间匹配结果数量变化"),
    html.H2("窗口大小 18，数据量 1000"),
    dcc.Graph(id='live-graph'),
    html.H2("关键要点:"),
    html.Ul([
    html.Li("NFA 处理需要 13 s， 匹配结果量为 609;"),
    html.Li("PACEP 处理需要 2 s， 匹配结果量为 609;"),
    html.Li("NFA 中间匹配结果数量增加到10^3;"),
    html.Li("PACEP 中间匹配数量在30以内。")  # 注意这里原用户的分号用了中文；
    ],
        style={'fontSize': '24px'} ),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)  # 每秒更新一次
])

# 定义回调函数，实时更新数据
@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # 读取最新的 CSV 数据
    try:
        df = pd.read_csv(csv_file, names=["event_index", "nums", "models"], header=None)

        # 如果数据为空，返回空图
        if df.empty:
            return go.Figure()

        # 先将 `nums` 取 log10，避免 `log10(0)` 错误
        df["nums"] = df["nums"].replace(0, 1e-9)  # 避免 log(0) 报错
        df["nums"] = np.log10(df["nums"])  # 计算 log10

        # 创建绘图数据
        fig = go.Figure()

        # 按 models 分组绘制不同颜色的线条
        for model, subset in df.groupby("models"):
            fig.add_trace(go.Scatter(
                x=subset["event_index"],
                y=subset["nums"],
                mode='lines',
                name=str(model)  # 添加图例
            ))

        fig.update_layout(
            title="NFA vs PACEP 实时数据变化（log10 变换）",
            xaxis_title="Event Index",
            yaxis_title="log10(Nums)",
            legend_title="Models",
            template="plotly_dark"
        )

        return fig

    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return go.Figure()

# 启动 Dash 服务器
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)