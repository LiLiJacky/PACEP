import time
from datetime import datetime, timedelta
from itertools import combinations
from queue import Empty

import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt

from aftermatch.NoSkipStrategy import NoSkipStrategy
from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from models.ConstraintCollection import ConstraintCollection
from models.DataInfo import DataInfo
from models.TimeConstarint import TimeConstraint
from models.TypeConstraint import TypeConstraint
from models.ValueConstraint import ValueConstraint
from nfa.ComputationState import ComputationState
from nfa.NFA import NFA, TimerService, EventWrapper
from nfa.NFAState import NFAState
from nfa.RegexToStateV100 import RegexToState
from nfa.State import State
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from simulated_data.SimulateData import Data
from sharedbuffer.EventId import EventId
from sharedbuffer.NodeId import NodeId
from multiprocessing import Queue
from simulated_data.SimulateDataByFile import DataSource


def nfa(regex, min_window_time, max_window_time, data_source, rate):
    constraints_dict = {
        'value_constrain':[],
        'time_constrain': [
            {'variables': ['A', 'B[i]', 'C'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['0']},
            {'variables': ['B'], 'variables_name': ['1']},
            {'variables': ['C'], 'variables_name': ['0']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {},
        'NONDETERMINISTICRELAXED': {"A", "B", "C"}
    }

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
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
    regex_to_state.generate_states_and_transitions()
    valid_states = regex_to_state.states

    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}

    nfa = NFA(valid_states, window_times, max_window_time, handle_timeout=True)

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
    processes = data_source.start_data_generation_stock(data_queue)

    start_time = time.time()
    total_matched = 0
    event_count = 0
    latency_list = []
    # 模拟数据流处理
    try:
        while True:
            data_item = data_queue.get(timeout=2)
            if data_item:
                event_count += 1
                event_start_time = time.time_ns()  # 使用纳秒时间戳
                # 封装事件并处理
                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                # 修剪超过时间窗口限制的部分匹配结果
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                # 处理事件并更新 NFA 状态
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                      TimerService())

                if matches:
                    total_matched += len(matches)

                event_latency_ns = time.time_ns() - event_start_time
                latency_list.append(event_latency_ns)
            time.sleep(1 / rate)
    except Empty:
        total_time = time.time() - start_time
        print("nfa:")
        print(total_time)
        avg_latency_ns = sum(latency_list) // len(latency_list) if latency_list else 0
        max_latency_ns = max(latency_list) if latency_list else 0
        min_latency_ns = min(latency_list) if latency_list else 0
        throughput = event_count / total_time if total_time > 0 else 0
        used_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # 内存使用以 MB 为单位

        # 返回统计结果
        return {
            "max_window_time": max_window_time,
            "total_time_sec": total_time,
            "event_count": event_count,
            "total_matches": total_matched,
            "used_memory_mb": used_memory,
            "max_latency_ns": max_latency_ns,
            "min_latency_ns": min_latency_ns,
            "avg_latency_ns": avg_latency_ns,
            "throughput": throughput,
            "type": "nfa"
        }
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def postponing(regex, min_window_time, max_window_time, data_source, rate):
    constraints_dict = {
        'value_constrain':[],
        'time_constrain': [
            {'variables': ['A', 'B[i]', 'C'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['0']},
            {'variables': ['B'], 'variables_name': ['1']},
            {'variables': ['C'], 'variables_name': ['0']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "B", "C"},
        'NONDETERMINISTICRELAXED': {}
    }

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
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
    regex_to_state.generate_states_and_transitions()
    regex_to_state.add_lazy_handle()
    valid_states = regex_to_state.states

    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}

    nfa = NFA(valid_states, window_times, max_window_time, handle_timeout=True, lazy_model=True, selection_strategy = contiguity_strategy)

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
    processes = data_source.start_data_generation_stock(data_queue)

    start_time = time.time()
    total_matched = 0
    event_count = 0
    latency_list = []

    try:
        while True:
            data_item = data_queue.get(timeout=2)
            if data_item:
                event_count += 1
                event_start_time = time.time_ns()  # 使用纳秒时间戳

                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                      TimerService())

                if matches:
                    total_matched += len(matches)

                event_latency_ns = time.time_ns() - event_start_time
                latency_list.append(event_latency_ns)
            time.sleep(1 / rate)
    except Empty:
        total_time = time.time() - start_time
        print("lazy:")
        print(total_time)
        avg_latency_ns = sum(latency_list) // len(latency_list) if latency_list else 0
        max_latency_ns = max(latency_list) if latency_list else 0
        min_latency_ns = min(latency_list) if latency_list else 0
        throughput = event_count / total_time if total_time > 0 else 0
        used_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # 内存使用以 MB 为单位
        print(throughput)

        # 返回统计结果
        return {
            "max_window_time": max_window_time,
            "total_time_sec": total_time,
            "event_count": event_count,
            "total_matches": total_matched,
            "used_memory_mb": used_memory,
            "max_latency_ns": max_latency_ns,
            "min_latency_ns": min_latency_ns,
            "avg_latency_ns": avg_latency_ns,
            "throughput": throughput,
            "type": "lazy"
        }

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def postponing_with_value_constrain(regex, min_window_time, max_window_time, data_source, rate):
    constraints_dict = {
        'value_constrain':[{'variables': ['A'], 'expression': '1'},
                           {'variables': ['B[i]'], 'expression': '2'},
                           {'variables': ['C'], 'expression': '1'}
                           ],
        'time_constrain': [
            {'variables': ['A', 'B[i]', 'C'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['0']},
            {'variables': ['B'], 'variables_name': ['1']},
            {'variables': ['C'], 'variables_name': ['0']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "B", "C"},
        'NONDETERMINISTICRELAXED': {}
    }

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
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
    regex_to_state.generate_states_and_transitions()
    regex_to_state.add_lazy_handle()
    valid_states = regex_to_state.states

    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}

    nfa = NFA(valid_states, window_times, max_window_time, handle_timeout=True, lazy_model=True)

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
    processes = data_source.start_data_generation_stock(data_queue)

    start_time = time.time()
    total_matched = 0
    event_count = 0
    latency_list = []

    try:
        while True:
            data_item = data_queue.get(timeout=2)
            if data_item:
                event_count += 1
                event_start_time = time.time_ns()  # 使用纳秒时间戳

                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                      TimerService())

                if matches:
                    total_matched += len(matches)

                event_latency_ns = time.time_ns() - event_start_time
                latency_list.append(event_latency_ns)
            time.sleep(1 / rate)
    except Empty:
        total_time = time.time() - start_time
        print("lazy:")
        avg_latency_ns = sum(latency_list) // len(latency_list) if latency_list else 0
        max_latency_ns = max(latency_list) if latency_list else 0
        min_latency_ns = min(latency_list) if latency_list else 0
        throughput = event_count / total_time if total_time > 0 else 0
        used_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # 内存使用以 MB 为单位
        print(throughput)

        # 返回统计结果
        return {
            "max_window_time": max_window_time,
            "total_time_sec": total_time,
            "event_count": event_count,
            "total_matches": total_matched,
            "used_memory_mb": used_memory,
            "max_latency_ns": max_latency_ns,
            "min_latency_ns": min_latency_ns,
            "avg_latency_ns": avg_latency_ns,
            "throughput": throughput,
            "type": "lazy"
        }

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()



if __name__ == "__main__":
    regex = "A B+ C"

    max_window_times = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400]
    min_window_time = 0

    data_source = '../../data/simulate_data/stock.stream'

    rate = 400000

    # print("Test nfa handle:")
    # nfa(regex, min_window_time, max_window_time, data_source, rate)

    # 用于存储所有统计结果
    results = []

    for max_window_time in max_window_times:
        print(max_window_time)
        # stats_nfa = nfa(regex, min_window_time, max_window_time, data_source, rate)
        # if stats_nfa:
        #     results.append(stats_nfa)
        stats = postponing(regex, min_window_time, max_window_time, data_source, rate)
        if stats:
            results.append(stats)


    # 使用 Pandas 将结果保存为 CSV 文件
    df = pd.DataFrame(results)
    df.to_csv("../../output/experiment/compare/ab+c.csv", index=False)

    print("Results saved to ab+c.csv")
