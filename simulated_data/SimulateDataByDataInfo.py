import os
import time
import random
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import statsmodels.api as sm

from models.Data import Data
from models.DataInfo import DataInfo


def data_producer(data_info, output_queue, *causal):
    if causal:
        while True:
            # create 60 second data
            # 模拟一个自回归序列 AR(1) 模型
            ar_params = np.array([0.999])  # AR 系数
            ar = np.r_[1, -ar_params]  # 构造 AR 模型的系数
            ma = np.array([1])  # MA(0)，即没有移动平均部分
            n = 100  # 序列长度

            # 生成 AR(1) 模型的时间序列数据
            x1 = 10 + sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n)

            # 生成干预数据
            index = 0
            while index < len(x1):
                if random.random() < data_info.probability:
                    # 生成 4 ～ 10 个干预数据
                    intervention_length = random.randint(4, 10)
                    offset = 0
                    while offset < intervention_length and offset + index < len(x1):
                        x1[index + offset] += 50
                        offset += 1
                    index += intervention_length
                else:
                    index += 1

            # 生成 y 值，y = 1.2 * x1 + 随机噪声
            y = 2 * x1 + np.random.normal(size=n)

            # 讲数据写入队列
            index = 0
            time_list = []
            while index < len(x1):
                timestamp = datetime.now()
                time_list.append(timestamp)
                data_item_a = Data(data_info.names, x1[index], timestamp)

                data_item_b = Data("B", y[index], timestamp)
                output_queue.put(data_item_b)
                output_queue.put(data_item_a)
                time.sleep(data_info.frequency)
                index += 1

            df = pd.DataFrame({'timestamp': time_list,'x1': x1, 'y': y})
            df.to_csv('../data/simulate_data/causal_simulate_data.csv', mode='a',
                      header=not os.path.exists('../data/simulate_data/causal_simulate_data.csv'), index=False)
    else:
        """
        Function to produce data for a given variable and put it in the output queue.
        """
        while True:
            # Calculate the number of data items to generate in 60 seconds
            num_items = int(60 / data_info.frequency)

            for _ in range(num_items):
                # Decide whether to generate normal or matched data based on probability
                if random.random() < data_info.probability:
                    for matched_value in data_info.matched:
                        # Create a data item with the matched data
                        timestamp = datetime.now()
                        data_item = Data(data_info.names, matched_value, timestamp)
                        # Put the generated data item in the queue
                        output_queue.put(data_item)
                        # Sleep for the specified frequency
                        time.sleep(data_info.frequency)
                else:
                    data_value = random.choice(data_info.normal)
                    # Create a data item with the normal data
                    timestamp = datetime.now()
                    data_item = Data(data_info.names, data_value, timestamp)
                    # Put the generated data item in the queue
                    output_queue.put(data_item)

                    # Sleep for the specified frequency
                    time.sleep(data_info.frequency)


def data_consumer(input_queue):
    """
    Function to consume data from the input queue and process it.
    """
    while True:
        try:
            data_item = input_queue.get()  # Blocking call, waits for data
            if data_item:
                # Process the data item (e.g., print it, store it, send it)
                print(data_item)
        except KeyboardInterrupt:
            break


class DataSource:
    def __init__(self, data_info_list, *args):
        self.data_info_list = data_info_list
        self.causal_ground_truth = None
        if args:
            self.causal_ground_truth = args[0]

    def start_data_generation(self, data_queue):
        """
        Starts data generation processes for each variable based on the provided data info list.
        """
        processes = []

        if self.causal_ground_truth is not None:
            # start causal data generate process
            process = Process(target=data_producer, args=(self.data_info_list[0], data_queue, self.causal_ground_truth))
            processes.append(process)
            process.start()
        else:
            # Start producer processes for each data info
            for data_info in self.data_info_list:
                process = Process(target=data_producer, args=(data_info, data_queue))
                processes.append(process)
                process.start()

        return processes
