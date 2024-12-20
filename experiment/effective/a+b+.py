import copy
import os
import time
from datetime import datetime, timedelta
from itertools import combinations
from queue import Empty

import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from aftermatch.NoSkipStrategy import NoSkipStrategy
from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from generator.DataGenerator import generate_data_with_pandas_sorted
from lazy_calculate.TableTool import TableTool
from models.ConstraintCollection import ConstraintCollection
from models.DataInfo import DataInfo
from models.TimeConstarint import TimeConstraint
from models.TypeConstraint import TypeConstraint
from models.ValueConstraint import ValueConstraint
from nfa.ComputationState import ComputationState
from nfa.NFA import NFA, TimerService, EventWrapper
from nfa.NFAState import NFAState
from nfa.RegexToState import RegexToState
from nfa.State import State
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from simulated_data.SimulateData import Data
from sharedbuffer.EventId import EventId
from sharedbuffer.NodeId import NodeId
from multiprocessing import Queue
from simulated_data.SimulateDataByFile import DataSource

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # 返回内存占用，单位为 MB

def nfa(valid_states, window_times, max_window_time, data_source, rate, selection_strategy):
    nfa = NFA(valid_states, window_times, max_window_time, handle_timeout=True, selection_strategy = selection_strategy)

    # 初始化 SharedBuffer 和 NFA
    config = SharedBufferCacheConfig.from_config('../../config.ini')
    shared_buffer = SharedBuffer(config)
    accessor = SharedBufferAccessor(shared_buffer)

    # 创建初始 NFA 状态
    nfa_state = nfa.create_initial_nfa_state()

    # Initialize the data queue
    data_queue = Queue()

    # Initialize and start data generation
    data_source = DataSource(csv_file=data_source, rate=rate)
    processes = data_source.start_data_generation_test(data_queue)

    modify_begin_time = time.time()
    total_matched = set()
    last_handle_time = modify_begin_time
    table_tool = TableTool()

    record = []
    memory_usage_total = 0
    events_nums = 0

    # 模拟数据流处理
    try:
        while True:
            data_item = data_queue.get(timeout=2)
            if data_item:
                # 封装事件并处理
                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                # 修剪超过时间窗口限制的部分匹配结果
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                # 处理事件并更新 NFA 状态
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                      TimerService())

                if matches:
                    for match in matches:
                        lista = []
                        for a in match[0]['A']:
                            atime = pd.to_datetime(a.event.get_timestamp(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                            lista.append(atime)

                        listb = []
                        for b in match[0]['B']:
                            btime = pd.to_datetime(b.event.get_timestamp(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                            listb.append(btime)

                        stringId = table_tool.ensure_hashable(match)
                        if stringId not in total_matched:
                            record.append({"a": lista, "b": listb})
                            total_matched.add(stringId)

                last_handle_time = time.time()

            memory_usage_total += get_memory_usage()
            events_nums += 1
            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:
        # print("Total handle time:" + str(last_handle_time - modify_begin_time))
        # print("Total matches:" + str(len(total_matched)))
        #
        # print(record)

        return last_handle_time - modify_begin_time, len(total_matched), memory_usage_total / events_nums

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def postponing(valid_states, window_times, max_window_time, data_source, rate, lazy_calculate_model, selection_strategy):

    nfa = NFA(valid_states, window_times, max_window_time, handle_timeout=True, lazy_model=True, lazy_calculate_model=lazy_calculate_model, selection_strategy = selection_strategy)

    # 初始化 SharedBuffer 和 NFA
    config = SharedBufferCacheConfig.from_config('../../config.ini')
    shared_buffer = SharedBuffer(config)
    accessor = SharedBufferAccessor(shared_buffer)

    # 创建初始 NFA 状态
    nfa_state = nfa.create_initial_nfa_state()

    # Initialize the data queue
    data_queue = Queue()

    # Initialize and start data generation
    data_source = DataSource(csv_file=data_source, rate=rate)
    processes = data_source.start_data_generation_test(data_queue)
    table_tool = TableTool()

    modify_begin_time = time.time()
    total_matched = set()
    last_handle_time = modify_begin_time
    result_list = []
    record = []

    memory_usage_total = 0
    events_nums = 0

    # 模拟数据流处理
    try:
        while True:
            data_item = data_queue.get(timeout=2)
            if data_item:
                # 封装事件并处理
                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                # 修剪超过时间窗口限制的部分匹配结果
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                # 处理事件并更新 NFA 状态
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                      TimerService())

                if matches:
                    for match in matches:
                        lista = []
                        for a in match[0]['A']:
                            atime = pd.to_datetime(a.event.get_timestamp(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                            lista.append(atime)

                        listb = []
                        for b in match[0]['B']:
                            btime = pd.to_datetime(b.event.get_timestamp(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                            listb.append(btime)

                        stringId = table_tool.ensure_hashable(match)
                        if stringId not in total_matched:
                            record.append({"a": lista, "b": listb})
                            total_matched.add(stringId)

                last_handle_time = time.time()

            memory_usage_total += get_memory_usage()
            events_nums += 1
            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:
        # print("Total handle time:" + str(last_handle_time - modify_begin_time))
        # print("Total matches:" + str(len(total_matched)))
        #
        # print(record)

        return last_handle_time - modify_begin_time, len(total_matched), memory_usage_total / events_nums

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()

def test(regex, min_window_time, max_window_time, data_source, rate, lazy_model, lazy_calculate_model, contiguity_strategy):
    constraints_dict = {
        'value_constrain': [
            {'variables': ['A', 'A[i]'], 'expression': '0 <= A[i] - last(A) <= 1000'},
            {'variables': ['B', 'B[i]'], 'expression': '0 <= B[i] - last(B) <= 1000'}
        ],
        'time_constrain': [
            {'variables': ['A', 'B'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['a']},
            {'variables': ['B'], 'variables_name': ['b']},
        ]
    }

    contiguity_strategy_copy = copy.deepcopy(contiguity_strategy)

    constraint_collection = ConstraintCollection()

    # 转换 value_constraints
    for vc in constraints_dict['value_constrain']:
        value_constraint = ValueConstraint(variables=vc['variables'], expression=vc['expression'])
        constraint_collection.add_constraint(value_constraint)

    # 转换 time_constraints
    for tc in constraints_dict['time_constrain']:
        time_constraint = TimeConstraint(variables=tc['variables'], min_time=tc['min_time'], max_time=tc['max_time'])
        constraint_collection.add_constraint(time_constraint)

    # 添加 type_constraints
    for tsc in constraints_dict['type_constrain']:
        type_constraint = TypeConstraint(variables=tsc['variables'], variables_name=tsc['variables_name'])
        constraint_collection.add_constraint(type_constraint)

    constraint_collection._window_time = max_window_time

    # 现在 constraint_collection 包含了从字典转换过来的所有约束
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy, lazy_model, lazy_calculate_model)
    regex_to_state.generate_states_and_transitions()
    if lazy_model:
        regex_to_state.add_lazy_handle()
    valid_states = regex_to_state.states


    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}

    if lazy_model:
        return postponing(valid_states, window_times, max_window_time, data_source, rate, lazy_calculate_model, contiguity_strategy_copy)
    else:
        # print("Test nfa handle:")
        return nfa(valid_states, window_times, max_window_time, data_source, rate, contiguity_strategy_copy)




if __name__ == "__main__":
    regex = "A{2,} B{2,}"

    rate = 400000

    # 设置起始时间
    start_time = datetime(2024, 11, 11, 11, 11, 11)

    results = []

    # 运行100轮实验的结果存储
    experiment_results = []

    # 生成 a 从 0.1 到 0.9，b 从 0.9 到 0.1 的数据
    for i in range(10):  # 10轮实验
        begin_time = time.time()
        run_results = []
        for a in np.arange(0.2, 1.0, 0.2):
            print(time.time())
            b = 1 - a  # b 是 a 的补数
            event_dict = {"a": a, "b": b}
            output_name = f"minimal_test_{a:.1f}_{b:.1f}"

            # 调用生成数据的方法
            generate_data_with_pandas_sorted(event_dict, 1000, output_name, start_time, time_interval_seconds=1)

            max_window_times = [4, 6, 8, 10, 12, 14, 16, 18]
            min_window_time = 0

            data_source = '../../data/minimal_test/' + output_name + '.csv'

            for max_window_time in max_window_times:
                # 连续匹配策略
                contiguity_strategys = {
                    "STAM": {
                        'STRICT': set(),
                        'RELAXED': set(),
                        'NONDETERMINISTICRELAXED': {'A', 'B'}
                    }
                }

                out_limit = False
                for event_selection_strategy, contiguity_strategy in contiguity_strategys.items():
                    nfa_begin_time = time.time()
                    # NFA 测试
                    nfa_handle_time, nfa_matches, nfa_memory = test(regex, min_window_time, max_window_time, data_source, rate,
                                                        False, "", contiguity_strategy)

                    if time.time() - nfa_begin_time > 600:
                        out_limit = True

                    # Lazy 测试
                    lazy_handle_time, lazy_matches, lazy_memory = test(regex, min_window_time, max_window_time, data_source, rate,
                                                          True, "TABLECALCULATE", contiguity_strategy)

                    # NFA 结果
                    nfa_result = {
                        "a": a,
                        "b": b,
                        "window_time": max_window_time,
                        "selection_strategy": event_selection_strategy,
                        "engine": "nfa",
                        "handle_time": nfa_handle_time,
                        "matches": nfa_matches,
                        "memory_usage": nfa_memory
                    }

                    # Lazy 结果
                    lazy_result = {
                        "a": a,
                        "b": b,
                        "window_time": max_window_time,
                        "selection_strategy": event_selection_strategy,
                        "engine": "pacep",
                        "handle_time": lazy_handle_time,
                        "matches": lazy_matches,
                        "memory_usage": lazy_memory
                    }

                    run_results.append(nfa_result)
                    run_results.append(lazy_result)
                    if out_limit:
                        break

                if out_limit:
                    break

        experiment_results.append(run_results)

        running_time = time.time() - begin_time
        print("round " + str(i) + " cost time: " + str(running_time))

    # 将 experiment_results 转换为适合存储的格式
    flat_results = [
        {
            "a": result.get("a", None),
            "b": result.get("b", None),
            "window_time": result["window_time"],
            "selection_strategy": result["selection_strategy"],
            "engine": result["engine"],
            "handle_time": result["handle_time"],
            "matches": result["matches"],
            "memory_usage": result["memory_usage"]
        }
        for run in experiment_results
        for result in run
    ]

    # 创建 DataFrame
    results_df = pd.DataFrame(flat_results)

    # 保存为 CSV 文件
    output_path = "../../output/minimal_test_result.csv"
    results_df.to_csv(output_path, index=False)

    # 按窗口时间和频率分组计算平均值
    grouped_results = results_df.groupby(['window_time', 'a', 'engine']).agg(
        avg_handle_time=('handle_time', 'mean'),
        avg_memory_usage=('memory_usage', 'mean')
    ).reset_index()

    # 定义颜色映射（使用推荐的方式）
    frequencies = sorted(grouped_results['a'].unique())
    cmap = plt.colormaps['tab10']  # 使用新的推荐方式获取颜色映射
    color_map = {freq: cmap(i / len(frequencies)) for i, freq in enumerate(frequencies)}

    # 绘图
    plt.figure(figsize=(12, 8))

    # 绘制每个频率的曲线
    window_times = sorted(grouped_results['window_time'].unique())

    for freq in frequencies:
        nfa_data = grouped_results[(grouped_results['a'] == freq) & (grouped_results['engine'] == 'nfa')]
        pacep_data = grouped_results[(grouped_results['a'] == freq) & (grouped_results['engine'] == 'pacep')]

        color = color_map[freq]

        if not nfa_data.empty:
            plt.plot(
                nfa_data['window_time'],
                np.log10(nfa_data['avg_handle_time']),
                linestyle='--', marker='o', color=color, label=f'NFA (freqA={freq:.1f})'
            )
        if not pacep_data.empty:
            plt.plot(
                pacep_data['window_time'],
                np.log10(pacep_data['avg_handle_time']),
                linestyle='-', marker='o', color=color, label=f'PACEP (freqA={freq:.1f})'
            )

    # 设置图表信息
    plt.xlabel('Window Time')
    plt.ylabel('Log10 of Average Handle Time (seconds)')
    plt.title('Average Handle Time vs Window Time (per Frequency)')
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()

    # 保存图片
    output_image_path = "../../output/a+b+_average_handle_time.png"
    plt.savefig(output_image_path)

    # 绘图
    plt.figure(figsize=(12, 8))

    # 绘制每个频率的曲线
    window_times = sorted(grouped_results['window_time'].unique())

    for freq in frequencies:
        nfa_data = grouped_results[(grouped_results['a'] == freq) & (grouped_results['engine'] == 'nfa')]
        pacep_data = grouped_results[(grouped_results['a'] == freq) & (grouped_results['engine'] == 'pacep')]

        color = color_map[freq]

        if not nfa_data.empty:
            plt.plot(
                nfa_data['window_time'],
                np.log10(nfa_data['avg_memory_usage']),
                linestyle='--', marker='o', color=color, label=f'NFA (freqA={freq:.1f})'
            )
        if not pacep_data.empty:
            plt.plot(
                pacep_data['window_time'],
                np.log10(pacep_data['avg_memory_usage']),
                linestyle='-', marker='o', color=color, label=f'PACEP (freqA={freq:.1f})'
            )

    # 设置图表信息
    plt.xlabel('Window Time')
    plt.ylabel('Log10 of Average Memory Usage (MB)')
    plt.title('Average Memory Usage vs Window Time (per Frequency)')
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()

    # 保存图片
    output_image_path = "../../output/a+b+_memory_usage.png"
    plt.savefig(output_image_path)