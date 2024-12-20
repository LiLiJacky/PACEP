import pandas as pd
import matplotlib.pyplot as plt

# 读取 PACEP 的 CSV 数据
data_pacep = pd.read_csv("../output/experiment/compare/a+b.csv")

# 将CORE的数据整理成字典列表
data_core = [
    {"max_window_time": 10, "total_time_sec": 0.674438666, "event_count": 1000, "total_matches": 1058,
     "used_memory_mb": 3, "max_latency_ns": 650878875, "min_latency_ns": 333370875,
     "avg_latency_ns": 402194123, "throughput": 1482},
    {"max_window_time": 20, "total_time_sec": 1.352475041, "event_count": 1000, "total_matches": 3940,
     "used_memory_mb": 3, "max_latency_ns": 1294289416, "min_latency_ns": 270783375,
     "avg_latency_ns": 884239089, "throughput": 739},
    {"max_window_time": 30, "total_time_sec": 5.561037834, "event_count": 1000, "total_matches": 8564,
     "used_memory_mb": 5, "max_latency_ns": 5476640125, "min_latency_ns": 235644417,
     "avg_latency_ns": 3202505675, "throughput": 179},
    {"max_window_time": 40, "total_time_sec": 2.78342225, "event_count": 1000, "total_matches": 14900,
     "used_memory_mb": 6, "max_latency_ns": 2698844209, "min_latency_ns": 257974791,
     "avg_latency_ns": 1702680090, "throughput": 359},
    {"max_window_time": 50, "total_time_sec": 7.276508708, "event_count": 1000, "total_matches": 22868,
     "used_memory_mb": 13, "max_latency_ns": 7165425042, "min_latency_ns": 212752167,
     "avg_latency_ns": 4032118900, "throughput": 137},
    {"max_window_time": 100, "total_time_sec": 25.338638125, "event_count": 1000, "total_matches": 85487,
     "used_memory_mb": 42, "max_latency_ns": 25050801209, "min_latency_ns": 162166709,
     "avg_latency_ns": 10056895601, "throughput": 39},
    {"max_window_time": 150, "total_time_sec": 103.942929167, "event_count": 1000, "total_matches": 184107,
     "used_memory_mb": 100, "max_latency_ns": 103335195666, "min_latency_ns": 134923500,
     "avg_latency_ns": 33856727760, "throughput": 9},
    {"max_window_time": 200, "total_time_sec": 359.261633584, "event_count": 1000, "total_matches": 318264,
     "used_memory_mb": 209, "max_latency_ns": 358098296708, "min_latency_ns": 218291584,
     "avg_latency_ns": 111805157784, "throughput": 2},
]
df_core = pd.DataFrame(data_core)

# 设置输出文件夹
file_output_dir = "../output/experiment/compare/png/"

# 绘制第一张图：Total Execution Time for 1000 Stock Events
plt.figure()
plt.plot(data_pacep["max_window_time"], data_pacep["total_time_sec"], marker='o', label="PACEP")
plt.plot(df_core["max_window_time"], df_core["total_time_sec"], marker='o', label="CORE")
plt.title("Total Execution Time for 10000 Stock Events")
plt.xlabel("Window Length")
plt.ylabel("Time (sec)")
plt.legend()
plt.grid(True)
plt.savefig(file_output_dir + "Total_Execution_Time.png")
plt.show()

# 绘制第二张图：Memory Usage for 1000 Stock Events
plt.figure()
plt.plot(data_pacep["max_window_time"], data_pacep["used_memory_mb"], marker='o', label="PACEP")
plt.plot(df_core["max_window_time"], df_core["used_memory_mb"], marker='o', label="CORE")
plt.title("Memory Usage for 10000 Stock Events")
plt.xlabel("Window Length")
plt.ylabel("Memory in MB")
plt.legend()
plt.grid(True)
plt.savefig(file_output_dir + "Memory_Usage.png")
plt.show()

# 绘制第三张图：Maximum and Average Latency for 1000 Stock Events
plt.figure()
plt.plot(data_pacep["max_window_time"], data_pacep["max_latency_ns"] / 1e9, marker='*', label="PACEP Max Latency")
plt.plot(data_pacep["max_window_time"], data_pacep["avg_latency_ns"] / 1e9, marker='x', label="PACEP Avg Latency")
plt.plot(df_core["max_window_time"], df_core["max_latency_ns"] / 1e9, marker='*', label="CORE Max Latency")
plt.plot(df_core["max_window_time"], df_core["avg_latency_ns"] / 1e9, marker='x', label="CORE Avg Latency")
plt.title("Maximum and Average Latency for 10000 Stock Events")
plt.xlabel("Window Length")
plt.ylabel("Latency (sec)")
plt.legend()
plt.grid(True)
plt.savefig(file_output_dir + "Max_Avg_Latency.png")
plt.show()