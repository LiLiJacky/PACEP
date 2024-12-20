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

def flash_crash_single_kleene(regex, min_window_time, max_window_time, source, rate):
    constraints_dict = {
        'value_constrain': [{'variables': ['A'], 'expression': '0.1 <= sum_square_difference(A) <= 100'},
                            {'variables': ['A', 'B'], 'expression': '-100 <= B - min(A) <= 0'},
                            {'variables': ['B', 'C'], 'expression': '0.5 <= C - B <= 100'},
                            {'variables': ['A', 'C'], 'expression': '0 <= C - max(A) <= 100'},],
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
    regex_to_state.draw()

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
                        crash_timestamp = match['B'][0].event.timestamp
                        flash_timestamp_list = []
                        for a_accessor in match['A[i]']:
                            flash_timestamp_list.append(a_accessor.event.get_timestamp())

                        begin_date = pd.to_datetime(min(flash_timestamp_list), unit='s').strftime('%Y-%m-%d')
                        end_date = pd.to_datetime(crash_timestamp
                                                  , unit='s').strftime('%Y-%m-%d')

                        # 提取 flash_stage 和 crash_stage 的时间戳列表
                        flash_stage = [pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') for ts
                                       in flash_timestamp_list]
                        crash_stage = end_date

                        # 将结果添加到结果列表
                        result_list.append({
                            "begin_date": begin_date,
                            "end_date": end_date,
                            "flash_stage": ";".join(flash_stage),
                            "crash_stage": crash_stage
                        })


                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:
        df = pd.DataFrame(result_list)
        df.to_csv(output + '/nfa.csv', index=False)
        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def optimizer_by_lazy_handle(regex, min_window_time, max_window_time, source, rate, output):
    # 探究模式计算代价对计算延迟的影响,当处理延迟超过两倍生成速率时，认为处理已经无法满足实时处理需求
    # 注意约束衍生，当不允许任意子集创建后，有可能匹配不成功
    constraints_dict = {
        'value_constrain': [
                            #{'variables': ['A'], 'expression': '0.1 <= least_squares(A) <= 100'},
                            # 约束衍生
                            {'variables': ['A', 'B'], 'expression': '-100 <= B - max(A) <= 0'},
                            {'variables': ['B', 'C'], 'expression': '0.5 <= C - B <= 100'},
                            # 约束衍生
                            {'variables': ['A', 'C'], 'expression': '0 <= C - min(A) <= 100'},
        ],
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
                    for match in matches:
                        a_values = []
                        for a_accessor in match['A[i]']:
                            a_values.append([a_accessor.event.get_value(), a_accessor.event.get_timestamp()])
                        
                        b_value = match['B'][0].event.get_value()
                        c_value = match['C'][0].event.get_value()

                        # 生成至少5个元素的组合
                        for n in range(5, len(a_values) + 1):
                            for combo in combinations(a_values, n):
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
                                    min_value_in_combo = min([item[0] for item in combo])  # 获取当前组合的最大值
                                    if max_value_in_combo <= c_value and min_value_in_combo >= b_value:
                                        total_matched += 1
                                        # 提取组合的起始和结束日期
                                        begin_date = pd.to_datetime(min(data[:, 1]), unit='s').strftime('%Y-%m-%d')
                                        end_date = pd.to_datetime(match['C'][0].event.get_timestamp()
                                                                     , unit='s').strftime('%Y-%m-%d')

                                        # 提取 flash_stage 和 crash_stage 的时间戳列表
                                        flash_stage = [pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') for ts
                                                       in data[:, 1]]
                                        crash_stage = end_date

                                        # 将结果添加到结果列表
                                        result_list.append({
                                            "begin_date": begin_date,
                                            "end_date": end_date,
                                            "flash_stage": ";".join(flash_stage),
                                            "crash_stage": crash_stage
                                        })


                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:
        df = pd.DataFrame(result_list)
        df.to_csv(output + '/lazy_handle.csv', index=False)
        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def optimizer_by_lazy_handle_and_calculate_plan(regex, min_window_time, max_window_time, source, rate):
    # 探究模式计算代价对计算延迟的影响,当处理延迟超过两倍生成速率时，认为处理已经无法满足实时处理需求
    constraints_dict = {
        'value_constrain': [
                            #{'variables': ['A'], 'expression': '0.1 <= least_squares(A) <= 100'},
                            {'variables': ['A', 'B'], 'expression': '-100 <= B - max(A) <= 0'},
                            {'variables': ['B', 'C'], 'expression': '0.5 <= C - B <= 100'},
                            {'variables': ['A', 'C'], 'expression': '0 <= C - min(A) <= 100'},
        ],
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
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
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
                        a_values = []
                        for a_accessor in match['A[i]']:
                            a_values.append([a_accessor.event.get_value(),a_accessor.event.get_timestamp()])

                        b_value = match['B'][0].event.get_value()
                        c_value = match['C'][0].event.get_value()

                        # 生成至少5个元素的组合
                        for n in range(5, len(a_values) + 1):
                            for combo in combinations(a_values, n):
                                data = np.array(combo)
                                # Convert Unix timestamps to ordinal dates
                                time_stamp = np.array([pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                # calculate plan
                                max_value_in_combo = max([item[0] for item in combo])  # 获取当前组合的最大值
                                min_value_in_combo = min([item[0] for item in combo])  # 获取当前组合的最大值
                                if max_value_in_combo <= c_value and min_value_in_combo >= b_value:
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


def optimizer_by_lazy_handle_and_calculate_plan_and_prefix_optimizer(regex, min_window_time, max_window_time, source, rate, output):
    regex_main = "B C"
    regex_prefix = "A{5,}"

    constraints_dict_main = {
        'value_constrain': [
                            #{'variables': ['A'], 'expression': '0.1 <= least_squares(A) <= 100'},
                            #{'variables': ['B'], 'expression': '-100 <= B - min(A[i]) <= 0'},
                            {'variables': ['B', 'C'], 'expression': '0.5 <= C - B <= 100'},
                            #{'variables': ['C'], 'expression': '-100 <= C - max(A[i]) <= 0'},
        ],
        'time_constrain': [
            {'variables': ['B', 'C'], 'min_time': 0, 'max_time': 1}
        ],
        'type_constrain': [
            #{'variables': ['A'], 'variables_name': ['CLOSE']},
            {'variables': ['B'], 'variables_name': ['LOW']},
            {'variables': ['C'], 'variables_name': ['HIGH']},
        ]
    }

    constraints_dict_prefix = {
        'value_constrain': [
        ],
        'time_constrain': [
            {'variables': ['A'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['CLOSE']},
        ]
    }


    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "B", "C"},
        'NONDETERMINISTICRELAXED': {}
    }

    constraint_collection_main = ConstraintCollection()

    # 转换 value_constraints
    for vc in constraints_dict_main['value_constrain']:
        value_constraint = ValueConstraint(variables=vc['variables'], expression=vc['expression'])
        constraint_collection_main.add_constraint(value_constraint)

    # 转换 time_constraints
    for tc in constraints_dict_main['time_constrain']:
        time_constraint = TimeConstraint(variables=tc['variables'], min_time=tc['min_time'], max_time=tc['max_time'])
        constraint_collection_main.add_constraint(time_constraint)

    # 添加 type_constraints
    for tsc in constraints_dict_main['type_constrain']:
        type_constraint = TypeConstraint(variables=tsc['variables'], variables_name=tsc['variables_name'])
        constraint_collection_main.add_constraint(type_constraint)

    constraint_collection_main._window_time = max_window_time

    # 现在 constraint_collection 包含了从字典转换过来的所有约束
    regex_to_state_main = RegexToState(regex_main, constraint_collection_main, contiguity_strategy)
    regex_to_state_main.generate_states_and_transitions()
    valid_states_main = regex_to_state_main.states

    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}

    nfa_main = NFA(valid_states_main, window_times, max_window_time, handle_timeout=True)

    # 初始化 SharedBuffer 和 NFA
    config = SharedBufferCacheConfig.from_config('../../config.ini')
    shared_buffer = SharedBuffer(config)
    accessor = SharedBufferAccessor(shared_buffer)

    # 创建初始 NFA 状态
    nfa_state_main = nfa_main.create_initial_nfa_state()

    # Initialize the data queue
    data_queue = Queue()

    # Initialize and start data generation
    data_source = DataSource(csv_file=source, rate=rate)
    processes = data_source.start_data_generation(data_queue)

    modify_begin_time = time.time()
    total_matched = 0
    last_handle_time = modify_begin_time
    a_buffer = []
    result_list = []
    # 模拟数据流处理
    try:
        while True:
            data_item = data_queue.get(timeout=2)
            if data_item:
                if data_item.get_id() == 'CLOSE':
                    a_buffer.append([data_item.value, data_item.timestamp])
                # 封装事件并处理
                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                # 修剪超过时间窗口限制的部分匹配结果
                nfa_main.advance_time(accessor, nfa_state_main, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)
                for a in a_buffer:
                    if data_item.timestamp - a[1] >= max_window_time:
                        a_buffer.remove(a)
                    else:
                        break

                # 处理事件并更新 NFA 状态
                matches_main = nfa_main.process(accessor, nfa_state_main, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                      TimerService())


                if matches_main:
                    for match in matches_main:
                        start_timestamp = match['B'][0].event.timestamp
                        end_timestamp = match['C'][0].event.timestamp
                        a_values = []
                        max_a = float('-inf')
                        min_a = float('inf')
                        # 判断缓冲数据中有哪些可能参与匹配
                        for a in a_buffer:
                            if a[1] <= end_timestamp - max_window_time:
                                continue
                            if a[1] < start_timestamp:
                                a_values.append(a)
                                max_a = max(max_a, a[0])
                                min_a = min(min_a, a[0])
                            else:
                                break
                        if len(a_values) < 5:
                            break
                        b_value = match['B'][0].event.get_value()
                        c_value = match['C'][0].event.get_value()

                        # 衍生约束
                        if b_value >= max_a or c_value <= min_a:
                            continue

                        # 生成至少5个元素的组合
                        for n in range(5, len(a_values) + 1):
                            for combo in combinations(a_values, n):
                                data = np.array(combo)
                                # Convert Unix timestamps to ordinal dates
                                time_stamp = np.array([pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 1]])
                                # calculate plan
                                max_value_in_combo = max([item[0] for item in combo])  # 获取当前组合的最大值
                                min_value_in_combo = min([item[0] for item in combo])  # 获取当前组合的最大值
                                if max_value_in_combo <= c_value and min_value_in_combo >= b_value:
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
                                        total_matched += 1
                                        # 提取组合的起始和结束日期
                                        begin_date = pd.to_datetime(min(data[:, 1]), unit='s').strftime('%Y-%m-%d')
                                        end_date = pd.to_datetime(match['C'][0].event.get_timestamp()
                                                                  , unit='s').strftime('%Y-%m-%d')

                                        # 提取 flash_stage 和 crash_stage 的时间戳列表
                                        flash_stage = [pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') for ts
                                                       in data[:, 1]]
                                        crash_stage = end_date

                                        # 将结果添加到结果列表
                                        result_list.append({
                                            "begin_date": begin_date,
                                            "end_date": end_date,
                                            "flash_stage": ";".join(flash_stage),
                                            "crash_stage": crash_stage
                                        })

                last_handle_time = time.time()

            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:
        df = pd.DataFrame(result_list)
        df.to_csv(output + '/prefix_optimization.csv', index=False)
        print("total handle time:" + str(last_handle_time - modify_begin_time))
        print("total matches:" + str(total_matched))
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    regex = "A{5,} B C"

    max_window_time = 1864000
    min_window_time = 432000

    data_source = '../../data/nasdaq/ALCO.csv'
    data_source_demo = '../../data/nasdaq/flash_crash_data_succeed_demo.csv'
    output = '../../output/experiment/nasdaq'

    rate = 400000

    # print("With NFA_b:")
    # flash_crash_single_kleene(regex, min_window_time, max_window_time, data_source, rate)
    print("With lazy handle:")
    optimizer_by_lazy_handle(regex, min_window_time, max_window_time, data_source, rate, output)
    # print("With calculate plan:")
    # optimizer_by_lazy_handle_and_calculate_plan(regex, min_window_time, max_window_time, data_source, rate)

    # print("With prefix optimizer:")
    # optimizer_by_lazy_handle_and_calculate_plan_and_prefix_optimizer(regex, min_window_time, max_window_time,
    #                                                                      data_source, rate, output)
