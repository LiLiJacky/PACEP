import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import matplotlib

# 强制使用 TkAgg 后端，确保 PyCharm 可用
matplotlib.use("TkAgg")

# 读取 CSV 文件路径
csv_file = "change.csv"

# 启用交互模式
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

# 先显示窗口，防止窗口不出现
plt.show(block=False)

while True:
    # 检查文件是否存在且非空
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        print("文件为空，等待数据写入...")
        time.sleep(1)  # 等待 1 秒后继续检查
        continue

    # 读取 CSV 数据，并手动设置列名
    df = pd.read_csv(csv_file, names=["event_index", "nums", "models"], header=None)

    # 如果数据为空，则跳过绘图
    if df.empty or df.isnull().values.all():
        print("CSV 文件无有效数据，等待更新...")
        time.sleep(1)
        continue

    # 清除当前图像
    ax.clear()

    # 按 models 分组绘制不同线条
    for model in df["models"].dropna().unique():
        subset = df[df["models"] == model]
        if not subset.empty:
            ax.plot(subset["event_index"], subset["nums"], marker='o', linestyle='-', label=str(model))

    # 设置标题和标签
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Nums")
    ax.set_title("NFA vs PACEP")

    # 确保 legend 存在数据
    if len(df["models"].dropna().unique()) > 0:
        ax.legend()

    ax.grid(True)

    # 强制刷新窗口
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(1)  # 每秒刷新一次