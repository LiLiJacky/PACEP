import random
import pandas as pd
from datetime import datetime, timedelta

# 定义类型
log_types = ['reducer_start', 'load_std', 'reducer_end', 'map_start', 'map_end', 'shuffle_start', 'shuffle_end']


# 定义生成模拟数据的函数
def generate_log_data(num_logs=10000):
    logs = []
    others = []

    # 当前时间（起始时间）
    current_time = datetime.now()
    begin_time = current_time

    for _ in range(num_logs):
        log_type = random.choice(log_types)

        # 根据type生成不同的value
        if log_type == 'reducer_start':
            value = random.randint(1, 1000)  # 模拟一个随机的 reducer 开始处理的任务ID
        elif log_type == 'reducer_end':
            value = random.randint(1, 1000)  # 模拟一个随机的 reducer 完成任务ID
        elif log_type == 'load_std':
            value = random.randint(1, 100)  # 模拟加载标准差的整数值（从1到100的整数）
        elif log_type == 'map_start':
            value = random.randint(1, 1000)  # 模拟一个随机的 map 开始处理的任务ID
        elif log_type == 'map_end':
            value = random.randint(1, 1000)  # 模拟一个随机的 map 完成任务ID
        elif log_type == 'shuffle_start':
            value = random.randint(1, 100)  # 模拟 shuffle 操作的整数延时
        elif log_type == 'shuffle_end':
            value = random.randint(1, 100)  # 模拟 shuffle 操作的结束延时（整数）

        # 生成时间戳，基于当前时间递增生成
        current_time += timedelta(seconds=random.randint(1, 5))  # 时间间隔在1到5秒之间
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # 计算 current_time 和 begin_time 之间的差值，单位为秒
        time_difference = int((current_time - begin_time).total_seconds())

        # 生成日志记录
        logs.append([log_type, value, timestamp])

        others.append([time_difference, time_difference, log_types.index(log_type), 1, value])

    return logs, others


# 生成10000条模拟数据
log_data, others = generate_log_data(10000)

# 将数据转换为DataFrame
df = pd.DataFrame(log_data, columns=['type', 'value', 'timestamp'])
others_df = pd.DataFrame(others, columns=['id', 'timestamp', 'type', 'group', 'value'])

# 保存为CSV文件
df.to_csv('../compare/hadoop_log_data.csv', index=False)
others_df.to_csv('../compare/hadoop_log_data_other.csv', index=False)

print("CSV文件已保存为 'hadoop_log_data_ordered.csv'")