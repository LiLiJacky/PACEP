import time
from datetime import datetime, timedelta
from itertools import combinations
from queue import Empty

import numpy as np
import pandas as pd
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


def peak(source, rate):
    regex = "A B C"
    constraints_dict = {
        'value_constrain': [{'variables': ['A', 'B'], 'expression': '0 < B - A <= 100'},
                            {'variables': ['B', 'C'], 'expression': '0 < B - C <= 100'}],
        'time_constrain': [
            {'variables': ['A', 'B', 'C'], 'min_time': 3 * 86400, 'max_time':  3 * 86400}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['CLOSE']},
            {'variables': ['C'], 'variables_name': ['CLOSE']},
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    result_list = []
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
                    total_matched += len(matches)
                    for match in matches:
                        normal_time = datetime.utcfromtimestamp(match['A'][0].timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        print(normal_time)

                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:

        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def trough(source, rate):
    regex = "A B C"
    constraints_dict = {
        'value_constrain': [{'variables': ['A', 'B'], 'expression': '0 < A - B <= 100'},
                            {'variables': ['B', 'C'], 'expression': '0 < C - B <= 100'}],
        'time_constrain': [
            {'variables': ['A', 'B', 'C'], 'min_time': 3 * 86400, 'max_time':  3 * 86400}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['CLOSE']},
            {'variables': ['C'], 'variables_name': ['CLOSE']},
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    result_list = []
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
                    total_matched += len(matches)

                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:

        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()

def rising_fall(regex, min_window_time, max_window_time, source, rate):
    constraints_dict = {
        'value_constrain': [
                            {'variables': ['D'], 'expression': '0.1 <= least_squares(D) <= 100'},
                            {'variables': ['D', 'A'], 'expression': '0 < A - max(D) <= 100'},
                            {'variables': ['A', 'B'], 'expression': '0 < B - A <= 100'},
                            {'variables': ['B', 'C'], 'expression': '0 < B - C <= 100'},
                            {'variables': ['E'], 'expression': '-100 <= least_squares(E) <= -0.1'},
                            {'variables': ['C', 'E'], 'expression': '0 < C - max(E) <= 100'},
                            ],
        'time_constrain': [
            {'variables': ['D[i]', 'A', 'B', 'C', 'E[i]'], 'min_time':min_window_time, 'max_time':  max_window_time},
            {'variables': ['A', 'B', 'C'], 'min_time': 3 * 86400, 'max_time':  3 * 86400},
        ],
        'type_constrain': [
            {'variables': ['D'], 'variables_name': ['CLOSE']},
            {'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['CLOSE']},
            {'variables': ['C'], 'variables_name': ['CLOSE']},
            {'variables': ['E'], 'variables_name': ['CLOSE']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "B", "C"},
        'NONDETERMINISTICRELAXED': {"D", "E"}
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    result_list = []
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
                    total_matched += len(matches)
                    # for match in matches:
                    #     normal_time = datetime.utcfromtimestamp(match['A'][0].timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    #     print(normal_time)

                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:

        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def optimizer_by_lazy_handle(regex, min_window_time, max_window_time, source, rate):
    constraints_dict = {
        'value_constrain': [
                            {'variables': ['A', 'B'], 'expression': '0 < B - A <= 100'},
                            {'variables': ['B', 'C'], 'expression': '0 < B - C <= 100'},
                            ],
        'time_constrain': [
            {'variables': ['D[i]', 'A', 'B', 'C', 'E[i]'], 'min_time':min_window_time, 'max_time':  max_window_time},
            {'variables': ['A', 'B', 'C'], 'min_time': 3 * 86400, 'max_time':  3 * 86400},
        ],
        'type_constrain': [
            {'variables': ['D'], 'variables_name': ['CLOSE']},
            {'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['CLOSE']},
            {'variables': ['C'], 'variables_name': ['CLOSE']},
            {'variables': ['E'], 'variables_name': ['CLOSE']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "B", "C"},
        'NONDETERMINISTICRELAXED': {"D", "E"}
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    result_list = []
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
                    total_matched += len(matches)
                    for match in matches:
                        d_values = []
                        for d_accessor in match['D[i]']:
                            d_values.append([d_accessor.event.get_value(), d_accessor.event.get_timestamp()])
                        a_value = match['A'][0].event.get_value()
                        c_value = match['C'][0].event.get_value()

                        # 生成至少5个元素的组合
                        for n in range(2, len(d_values) + 1):
                            for combo in combinations(d_values, n):
                                data = np.array(combo)
                                # Convert Unix timestamps to ordinal dates
                                time_stamp = np.array([pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                values = data[:, 0]
                                n = len(time_stamp)
                                sum_timestamp = np.sum(time_stamp)
                                sum_value = np.sum(values)
                                sum_tv = np.sum(values * time_stamp)
                                sum_timestamp_squared = np.sum(time_stamp ** 2)

                                numerator = n * sum_tv - sum_value * sum_timestamp
                                denominator = n * sum_timestamp_squared - sum_timestamp ** 2

                                result = numerator / denominator if denominator != 0 else 0
                                if result >= 0.1:
                                    max_value_in_combo = max([item[0] for item in combo])  # 获取当前组合的最大值
                                    if max_value_in_combo <= a_value:
                                        # 生成至少5个元素的组合
                                        e_values = []
                                        for e_accessor in match['E[i]']:
                                            e_values.append(
                                                [e_accessor.event.get_value(), e_accessor.event.get_timestamp()])

                                        for n in range(2, len(e_values) + 1):
                                            for combo in combinations(e_values, n):
                                                data = np.array(combo)
                                                # Convert Unix timestamps to ordinal dates
                                                time_stamp = np.array(
                                                    [pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                                values = data[:, 0]
                                                n = len(time_stamp)
                                                sum_timestamp = np.sum(time_stamp)
                                                sum_value = np.sum(values)
                                                sum_tv = np.sum(values * time_stamp)
                                                sum_timestamp_squared = np.sum(time_stamp ** 2)

                                                numerator = n * sum_tv - sum_value * sum_timestamp
                                                denominator = n * sum_timestamp_squared - sum_timestamp ** 2

                                                result = numerator / denominator if denominator != 0 else 0
                                                if result <= -0.1:
                                                    max_value_in_combo = max([item[0] for item in combo])  # 获取当前组合的最大值
                                                    if max_value_in_combo <= c_value:
                                                        total_matched += 1


                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:

        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def optimizer_by_lazy_handle_and_calculate_plan(regex, min_window_time, max_window_time, source, rate):
    
    constraints_dict = {
        'value_constrain': [
                            {'variables': ['A', 'B'], 'expression': '0 < B - A <= 100'},
                            {'variables': ['B', 'C'], 'expression': '0 < B - C <= 100'},
                            ],
        'time_constrain': [
            {'variables': ['D[i]', 'A', 'B', 'C', 'E[i]'], 'min_time':min_window_time, 'max_time':  max_window_time},
            {'variables': ['A', 'B', 'C'], 'min_time': 3 * 86400, 'max_time':  3 * 86400},
        ],
        'type_constrain': [
            {'variables': ['D'], 'variables_name': ['CLOSE']},
            {'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['CLOSE']},
            {'variables': ['C'], 'variables_name': ['CLOSE']},
            {'variables': ['E'], 'variables_name': ['CLOSE']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "B", "C"},
        'NONDETERMINISTICRELAXED': {"D", "E"}
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    result_list = []
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
                    total_matched += len(matches)
                    for match in matches:
                        d_values = []
                        for d_accessor in match['D[i]']:
                            d_values.append([d_accessor.event.get_value(), d_accessor.event.get_timestamp()])
                        a_value = match['A'][0].event.get_value()
                        c_value = match['C'][0].event.get_value()

                        if a_value <= min([d[0] for d in d_values]):
                            continue

                        e_values = []
                        for e_accessor in match['E[i]']:
                            e_values.append(
                                [e_accessor.event.get_value(), e_accessor.event.get_timestamp()])

                        if c_value <= min([e[0] for e in e_values]):
                            continue

                        # 生成至少5个元素的组合
                        for n in range(2, len(d_values) + 1):
                            for combo in combinations(d_values, n):
                                if a_value <= min([c[0] for c in combo]):
                                    continue
                                data = np.array(combo)
                                # Convert Unix timestamps to ordinal dates
                                time_stamp = np.array([pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                values = data[:, 0]
                                n = len(time_stamp)
                                sum_timestamp = np.sum(time_stamp)
                                sum_value = np.sum(values)
                                sum_tv = np.sum(values * time_stamp)
                                sum_timestamp_squared = np.sum(time_stamp ** 2)

                                numerator = n * sum_tv - sum_value * sum_timestamp
                                denominator = n * sum_timestamp_squared - sum_timestamp ** 2

                                result = numerator / denominator if denominator != 0 else 0
                                if result >= 0.1:
                                    max_value_in_combo = max([item[0] for item in combo])  # 获取当前组合的最大值
                                    if max_value_in_combo <= a_value:
                                        if c_value <= min([c[0] for c in combo]):
                                            continue
                                        # 生成至少5个元素的组合
                                        for n in range(2, len(e_values) + 1):
                                            for combo in combinations(e_values, n):
                                                data = np.array(combo)
                                                # Convert Unix timestamps to ordinal dates
                                                time_stamp = np.array(
                                                    [pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                                values = data[:, 0]
                                                n = len(time_stamp)
                                                sum_timestamp = np.sum(time_stamp)
                                                sum_value = np.sum(values)
                                                sum_tv = np.sum(values * time_stamp)
                                                sum_timestamp_squared = np.sum(time_stamp ** 2)

                                                numerator = n * sum_tv - sum_value * sum_timestamp
                                                denominator = n * sum_timestamp_squared - sum_timestamp ** 2

                                                result = numerator / denominator if denominator != 0 else 0
                                                if result <= -0.1:
                                                    max_value_in_combo = max([item[0] for item in combo])  # 获取当前组合的最大值
                                                    if max_value_in_combo <= c_value:
                                                        total_matched += 1


                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:

        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()

def optimizer_by_latency_handle_and_calculate_plan_and_prefix_optimizer(regex, min_window_time, max_window_time, source, rate):
    regex = "A B C E{2,5}"
    constraints_dict = {
        'value_constrain': [
            {'variables': ['A', 'B'], 'expression': '0 < B - A <= 100'},
            {'variables': ['B', 'C'], 'expression': '0 < B - C <= 100'},
        ],
        'time_constrain': [
            {'variables': ['A', 'B', 'C'], 'min_time':min_window_time, 'max_time': max_window_time},
            {'variables': ['A', 'B', 'C'], 'min_time': 3 * 86400, 'max_time': 3 * 86400},
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['CLOSE']},
            {'variables': ['C'], 'variables_name': ['CLOSE']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "B", "C"},
        'NONDETERMINISTICRELAXED': {"D", "E"}
    }

    constraint_collection = ConstraintCollection()

    # 转换 value_constraints
    for vc in constraints_dict['value_constrain']:
        value_constraint = ValueConstraint(variables=vc['variables'], expression=vc['expression'])
        constraint_collection.add_constraint(value_constraint)

    # 转换 time_constraints
    for tc in constraints_dict['time_constrain']:
        time_constraint = TimeConstraint(variables=tc['variables'], min_time=tc['min_time'],
                                         max_time=tc['max_time'])
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    result_list = []
    d_buffer = []
    e_buffer = []
    # 模拟数据流处理
    try:
        while True:
            data_item = data_queue.get(timeout=2)
            if data_item:
                if data_item.get_id() == 'CLOSE':
                    d_buffer.append([data_item.value, data_item.timestamp])
                    e_buffer.append([data_item.value, data_item.timestamp])
                # 封装事件并处理
                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)


                # 修剪超过时间窗口限制的部分匹配结果
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)
                for d in d_buffer:
                    if data_item.timestamp - d[1] > max_window_time:
                        d_buffer.remove(d)
                for e in e_buffer:
                    if data_item.timestamp - e[1] > max_window_time:
                        e_buffer.remove(e)

                # 处理事件并更新 NFA 状态
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp,
                                      NoSkipStrategy.INSTANCE,
                                      TimerService())

                if matches:
                    total_matched += len(matches)
                    for match in matches:
                        start_timestamp = match['A'][0].event.timestamp
                        end_timestamp = match['C'][0].event.timestamp
                        d_values = []
                        for d in d_buffer:
                            if d[1] < end_timestamp - max_window_time:
                                continue
                            if d[1] <= start_timestamp:
                                d_values.append(d)
                            else:
                                break
                        if len(d_values) < 2:
                            break

                        a_value = match['A'][0].event.get_value()
                        c_value = match['C'][0].event.get_value()

                        if a_value <= min([d[0] for d in d_values]):
                            continue

                        e_values = []
                        for e_accessor in match['E[i]']:
                            e_values.append(
                                [e_accessor.event.get_value(), e_accessor.event.get_timestamp()])

                        if c_value <= min([e[0] for e in e_values]):
                            continue

                        # 生成至少5个元素的组合
                        for n in range(2, len(d_values) + 1):
                            for combo in combinations(d_values, n):
                                if a_value <= min([c[0] for c in combo]):
                                    continue
                                data = np.array(combo)
                                # Convert Unix timestamps to ordinal dates
                                time_stamp = np.array(
                                    [pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                values = data[:, 0]
                                n = len(time_stamp)
                                sum_timestamp = np.sum(time_stamp)
                                sum_value = np.sum(values)
                                sum_tv = np.sum(values * time_stamp)
                                sum_timestamp_squared = np.sum(time_stamp ** 2)

                                numerator = n * sum_tv - sum_value * sum_timestamp
                                denominator = n * sum_timestamp_squared - sum_timestamp ** 2

                                result = numerator / denominator if denominator != 0 else 0
                                if result >= 0.1:
                                    max_value_in_combo = max([item[0] for item in combo])  # 获取当前组合的最大值
                                    if max_value_in_combo <= a_value:
                                        if c_value <= min([c[0] for c in combo]):
                                            continue
                                        # 生成至少5个元素的组合
                                        for n in range(2, len(e_values) + 1):
                                            for combo in combinations(e_values, n):
                                                data = np.array(combo)
                                                # Convert Unix timestamps to ordinal dates
                                                time_stamp = np.array(
                                                    [pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                                values = data[:, 0]
                                                n = len(time_stamp)
                                                sum_timestamp = np.sum(time_stamp)
                                                sum_value = np.sum(values)
                                                sum_tv = np.sum(values * time_stamp)
                                                sum_timestamp_squared = np.sum(time_stamp ** 2)

                                                numerator = n * sum_tv - sum_value * sum_timestamp
                                                denominator = n * sum_timestamp_squared - sum_timestamp ** 2

                                                result = numerator / denominator if denominator != 0 else 0
                                                if result <= -0.1:
                                                    max_value_in_combo = max(
                                                        [item[0] for item in combo])  # 获取当前组合的最大值
                                                    if max_value_in_combo <= c_value:
                                                        total_matched += 1

                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:

        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()

def m_mulity_kleene(regex, min_window_time, max_window_time, source, rate):
    constraints_dict = {
        'value_constrain': [{'variables': ['A'], 'expression': '0.1 <= least_squares(A) <= 100'},
                            {'variables': ['B'], 'expression': '-100 <= B - min(A[i]) <= 0'},
                            {'variables': ['B', 'C'], 'expression': '0.5 <= C - B <= 100'},
                            {'variables': ['C'], 'expression': '-100 <= C - max(A[i]) <= 0'},],
        'time_constrain': [
            {'variables': ['A[i]', 'B', 'C'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['LOW']},
            {'variables': ['C'], 'variables_name': ['HIGH']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"B", "C"},
        'NONDETERMINISTICRELAXED': {"A"}
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    result_list = []
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
                    total_matched += len(matches)

                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:

        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    rise_fall_regex = "D{2,5} A B C E{2,5}"
    regex = "A{5,} B{3,3} C{5,} D{3,3} E{5,} F{3,3}"

    max_window_time = 864000
    min_window_time = 432000

    data_source = '../../data/nasdaq/ALCO.csv'
    data_source_demo = '../../data/nasdaq/flash_crash_data_succeed_demo.csv'
    output = '../../output/experiment/nasdaq'

    raise_fall_demo = '../../data/nasdaq/raise_fall_data_succeed_demo.csv'

    rate = 40000

    # print("Peak:")
    # peak(raise_fall_demo, rate)

    # print("Trough:")
    # trough(raise_fall_demo, rate)

    print("Rising then fall with NFA_b:")
    rising_fall(rise_fall_regex, min_window_time, max_window_time, data_source, rate)

    print("Rising then fall with lazy handle:")
    optimizer_by_lazy_handle(rise_fall_regex, min_window_time, max_window_time, data_source, rate)

    print("With prefix optimizer:")
    optimizer_by_lazy_handle_and_calculate_plan(rise_fall_regex, min_window_time, max_window_time, data_source, rate)

    # print("With NFA_b:")
    #m_mulity_kleene(regex, min_window_time, max_window_time, data_source, rate)
