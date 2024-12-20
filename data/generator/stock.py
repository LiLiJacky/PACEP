import pandas as pd
import time

def process_csv(input_file, output_file):
    # 读取 CSV 文件，假设文件有标题行
    df = pd.read_csv(input_file)

    # 确保列名符合预期
    df.columns = [
        "date",          # 日期
        "open_price",    # 开盘价
        "close_price",   # 收盘价
        "volume",        # 成交量
        "peak_price",    # 最高价
        "lowest_price",  # 最低价
        "adjusted_close",  # 调整后收盘价
        "ticker"         # 股票代码
    ]

    # 将日期转换为时间戳（毫秒级）
    df["timestamp"] = pd.to_datetime(df["date"], format="%Y-%m-%d").astype(int) // 10**6

    # 添加实际到达时间字段（纳秒级时间戳）
    df["actualArrivalTime"] = int(time.time_ns())

    # 添加事件类型字段
    df["eventType"] = "realStock"

    # 调整字段顺序
    df = df[[
        "ticker",         # 股票代码作为 key
        "timestamp",      # 时间戳
        "open_price",     # 开盘价
        "peak_price",     # 最高价
        "lowest_price",   # 最低价
        "close_price",    # 收盘价
        "volume",         # 成交量
    ]]

    # 保存到输出文件
    df.to_csv(output_file, index=False)

# 输入和输出文件路径
input_csv = "../compare/goog_msft.csv"
output_csv = "../compare/goog_msft_other.csv"

# 调用处理函数
process_csv(input_csv, output_csv)