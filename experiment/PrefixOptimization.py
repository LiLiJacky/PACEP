import itertools
import time
from datetime import datetime, timedelta

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
from simulated_data.SimulateDataByDataInfo import DataSource
from util.CSVCompare import plot_results_handle_time_partial_nums


def test_prefix_normal(max_items, max_time):
    # 探究相互独立分布的两个子模式，频率对组合数的影响
    regex = "A{2,} B C{2,} D"

    window_time = 10
    probability = 0.01

    frequency_A = {0.5, 1, 2}
    frequency_B = {2}

    constraints_dict = {
        'value_constrain': [{'variables': ['B'], 'expression': '10 <= B <= 20'},
                            {'variables': ['D'], 'expression': '10 <= D <= 20'},
                            {'variables': ['A'], 'expression': '10 <= average(A) <= 20'},
                            {'variables': ['C'], 'expression': '10 <= average(C) <= 20'}],
        'time_constrain': [
            {'variables': ['A[i]', 'B', 'C[i]', 'D'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['A']},
            {'variables': ['B'], 'variables_name': ['B']},
            {'variables': ['C'], 'variables_name': ['C']},
            {'variables': ['D'], 'variables_name': ['D']}
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {},
        'NONDETERMINISTICRELAXED': {"B", "A", "C", "D"}
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

    constraint_collection._window_time = window_time

    # 现在 constraint_collection 包含了从字典转换过来的所有约束
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
    regex_to_state.generate_states_and_transitions()
    valid_states = regex_to_state.states

    partial_match_nums = {fb: {fa: [] for fa in frequency_A} for fb in frequency_B}
    result = []


    for fa in frequency_A:
        data_info_A = DataInfo(
            names='A',
            normal=list(range(0, 11)),
            matched=[12, 13],
            frequency=fa,
            probability=probability / 2)
        data_info_C = DataInfo(
            names='C',
            normal=list(range(0, 11)),
            matched=[12, 13],
            frequency=fa,
            probability=probability / 2)
        for fb in frequency_B:
            data_info_B = DataInfo(
                names='B',
                normal=list(range(0, 11)),
                matched=[12, 13],
                frequency=fb,
                probability=probability / 2)
            data_info_D = DataInfo(
                names='B',
                normal=list(range(0, 11)),
                matched=[12, 13],
                frequency=fb,
                probability=probability / 2)

            data_info_list = [data_info_A, data_info_B, data_info_C, data_info_D]

            # 假设 window_times 和 window_time 是已知的或从配置中获得的
            window_times = {}

            nfa = NFA(valid_states, window_times, window_time, handle_timeout=True)

            # 初始化 SharedBuffer 和 NFA
            config = SharedBufferCacheConfig.from_config('../config.ini')
            shared_buffer = SharedBuffer(config)
            accessor = SharedBufferAccessor(shared_buffer)

            # 创建初始 NFA 状态
            nfa_state = nfa.create_initial_nfa_state()

            # Initialize the data queue
            data_queue = Queue()

            # Initialize and start data generation
            data_source = DataSource(data_info_list)
            processes = data_source.start_data_generation(data_queue)

            max_len = 0
            # 模拟数据流处理
            try:
                item_count = 0
                while True:
                    data_item = data_queue.get()
                    if data_item:
                        begin_time = time.time()
                        item_count += 1
                        # 封装事件并处理
                        event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                        # 修剪超过时间窗口限制的部分匹配结果
                        nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                        # 处理事件并更新 NFA 状态
                        matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp,
                                              NoSkipStrategy.INSTANCE,
                                              TimerService())

                        partial_match_count = len(nfa_state.partial_matches)
                        partial_match_nums[fb][fa].append(partial_match_count)

                        handle_time = time.time() - begin_time

                        result.append({
                            'type': 'normal',
                            'fa': fa,
                            'fb': fb,
                            'item_num': item_count,
                            'partial_match_nums': partial_match_count,
                            'handle_time': handle_time
                        })

                        if item_count >= max_items or handle_time > max_time * (fa + fb):  # 假设处理 100 个数据项后停止
                            break

                    time.sleep(0.01)  # To prevent the main loop from consuming too much CPU
            except KeyboardInterrupt:
                for process in processes:
                    process.terminate()
                    process.join()

    # 将结果保存为 CSV 文件
    df_name = '../output/experiment/csv/pattern_with_prefix_kleene.csv'
    df = pd.DataFrame(result)
    df.to_csv(df_name, index=False)

    return df_name


def test_prefix_optimization_postponing_handle(max_items, max_time):
    # 探究相互独立分布的两个子模式，频率对组合数的影响
    regex = "A{2,} B C{2,} D"

    frequency_A = {0.5, 1, 2}
    frequency_B = {2}

    window_time = 10
    probability = 0.01

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A", "C"},
        'NONDETERMINISTICRELAXED': {"B", "D"}
    }

    constraints_dict_a = {
        'value_constrain': [{'variables': ['A'], 'expression': '10 <= average(A) <= 20'}],
        'time_constrain': [
            {'variables': ['A[i]'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['A']}
        ]
    }

    constraints_dict_c = {
        'value_constrain': [{'variables': ['C'], 'expression': '10 <= average(C) <= 20'}],
        'time_constrain': [
            {'variables': ['C[i]'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['C'], 'variables_name': ['C']}
        ]
    }

    constraints_dict_b = {
        'value_constrain': [{'variables': ['B'], 'expression': '10 <= B <= 20'}],
        'time_constrain': [
            {'variables': ['B'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['B'], 'variables_name': ['B']}
        ]
    }

    constraints_dict_d = {
        'value_constrain': [{'variables': ['D'], 'expression': '10 <= D <= 20'}],
        'time_constrain': [
            {'variables': ['D'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['D'], 'variables_name': ['D']}
        ]
    }
    basic_constraints_dict_list = {
        'value_constrain': [{'variables': ['B'], 'expression': '10 <= B <= 20'},
                            {'variables': ['D'], 'expression': '10 <= D <= 20'},
                            {'variables': ['A'], 'expression': '10 <= max(A) <= 20'},# 衍生约束
                            {'variables': ['C'], 'expression': '10 <= max(C) <= 20'}],
        'time_constrain': [
            {'variables': ['A[i]', 'B', 'C[i]', 'D'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['A']},
            {'variables': ['B'], 'variables_name': ['B']},
            {'variables': ['C'], 'variables_name': ['C']},
            {'variables': ['D'], 'variables_name': ['D']}
        ]
    }

    postponing_constraint_collection_list = {
        "name": ["A", "C"],
        "constraints": [constraints_dict_a, constraints_dict_c],
        "regex": ["A{2,}", "C{2,}"]
    }
    valid_states_list = {}

    constraint_collection = ConstraintCollection()

    # 转换 value_constraints
    for vc in basic_constraints_dict_list['value_constrain']:
        value_constraint = ValueConstraint(variables=vc['variables'], expression=vc['expression'])
        constraint_collection.add_constraint(value_constraint)

    # 转换 time_constraints
    for tc in basic_constraints_dict_list['time_constrain']:
        time_constraint = TimeConstraint(variables=tc['variables'], min_time=tc['min_time'],
                                         max_time=tc['max_time'])
        constraint_collection.add_constraint(time_constraint)

    # 添加 type_constraints
    for tsc in basic_constraints_dict_list['type_constrain']:
        type_constraint = TypeConstraint(variables=tsc['variables'], variables_name=tsc['variables_name'])
        constraint_collection.add_constraint(type_constraint)

    constraint_collection._window_time = window_time

    # 现在 constraint_collection 包含了从字典转换过来的所有约束
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
    regex_to_state.generate_states_and_transitions()
    valid_states = regex_to_state.states


    partial_match_nums = {fb: {fa: [] for fa in frequency_A} for fb in frequency_B}
    result = []

    for fa in frequency_A:
        data_info_A = DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[12, 13],
                frequency=fa,
                probability=probability / 2)
        data_info_C = DataInfo(
            names='C',
            normal=list(range(0, 11)),
            matched=[12, 13],
            frequency=fa,
            probability=probability / 2)
        for fb in frequency_B:
            data_info_B = DataInfo(
                names='B',
                normal=list(range(0, 11)),
                matched=list(range(11, 20)),
                frequency=fb,
                probability=probability)

            data_info_D = DataInfo(
                names='D',
                normal=list(range(0, 11)),
                matched=list(range(11, 20)),
                frequency=fb,
                probability=probability)

            data_info_list = [data_info_A, data_info_B, data_info_C, data_info_D]

            # 假设 window_times 和 window_time 是已知的或从配置中获得的
            window_times = {}

            nfa = NFA(valid_states, window_times, window_time, handle_timeout=True)

            # 初始化 SharedBuffer 和 NFA
            config = SharedBufferCacheConfig.from_config('../config.ini')
            shared_buffer = SharedBuffer(config)
            accessor = SharedBufferAccessor(shared_buffer)

            # 创建初始 NFA 状态
            nfa_state = nfa.create_initial_nfa_state()

            # Initialize the data queue
            data_queue = Queue()

            # Initialize and start data generation
            data_source = DataSource(data_info_list)
            processes = data_source.start_data_generation(data_queue)

            max_len = 0
            partial_match_nums = 0
            # 模拟数据流处理
            try:
                item_count = 0
                while True:
                    data_item = data_queue.get()
                    if data_item:
                        begin_time = time.time()
                        item_count += 1
                        # 封装事件并处理
                        event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                        # 修剪超过时间窗口限制的部分匹配结果
                        nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                        # 处理事件并更新 NFA 状态
                        matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                              TimerService())
                        if len(matches) > 0:
                            for match in matches:
                                a_values = []
                                for a_accessor in match['A[i]']:
                                    a_values.append(a_accessor.event.get_value())
                                c_values = []
                                for c_accessor in match['C[i]']:
                                    c_values.append(c_accessor.event.get_value())

                                answer = {"A":[], "C":[]}
                                # 遍历所有可能的组合长度
                                for r in range(1, len(a_values) + 1):
                                    # 生成所有长度为 r 的组合
                                    for combination in itertools.combinations(a_values, r):
                                        avg = sum(combination) / len(combination)
                                        if 10 <= avg <= 20:
                                            answer["A"] = combination
                                            break
                                for r in range(1, len(c_values) + 1):
                                    # 生成所有长度为 r 的组合
                                    for combination in itertools.combinations(c_values, r):
                                        avg = sum(combination) / len(combination)
                                        if 10 <= avg <= 20:
                                            answer["C"] = combination
                                            break

                        partial_match_count = len(nfa_state.partial_matches)

                        handle_time = time.time() - begin_time

                        result.append({
                            'type': 'independence',
                            'fa': fa,
                            'fb': fb,
                            'item_num': item_count,
                            'partial_match_nums': partial_match_count,
                            'handle_time': handle_time
                        })

                        if item_count >= max_items or handle_time > max_time * (fa + fb):  # 假设处理 100 个数据项后停止
                            break

                        max_len = max(max_len, len(nfa_state.partial_matches))

                    time.sleep(0.01)  # To prevent the main loop from consuming too much CPU
            except KeyboardInterrupt:
                for process in processes:
                    process.terminate()
                    process.join()


    # 将结果保存为 CSV 文件
    df_name = '../output/experiment/csv/pattern_with_prefix_kleene_postponing.csv'
    df = pd.DataFrame(result)
    df.to_csv(df_name, index=False)

    return df_name



def test_prefix_optimization_two_table_join(max_items, max_time):
    # 探究相互独立分布的两个子模式，频率对组合数的影响
    regex_a = "A{2,}"
    regex_b = "B"

    frequency_A = {0.5, 1, 2}
    frequency_B = {2}

    window_time = 10
    probability = 0.01

    constraints_dict_a = {
        'value_constrain': [{'variables': ['A'], 'expression': '10 <= average(A) <= 20'}],
        'time_constrain': [
            {'variables': ['A[i]'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['A']}
        ]
    }

    constraints_dict_b = {
        'value_constrain': [{'variables': ['B'], 'expression': '10 <= B <= 20'}],
        'time_constrain': [
            {'variables': ['B'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['B'], 'variables_name': ['B']}
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {"A"},
        'NONDETERMINISTICRELAXED': {"B"}
    }

    constraints_dict_list = {
        "name": ["A", "B"],
        "constraints": [constraints_dict_a, constraints_dict_b],
        "regex": [regex_a, regex_b]}
    constraint_collection_list = {}
    valid_states_list = {}

    for i in range(0, len(constraints_dict_list["name"])):
        constraints_dict = constraints_dict_list["constraints"][i]
        current_regex = constraints_dict_list["regex"][i]
        name = constraints_dict_list["name"][i]
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

        constraint_collection._window_time = window_time

        # 现在 constraint_collection 包含了从字典转换过来的所有约束
        regex_to_state = RegexToState(current_regex, constraint_collection, contiguity_strategy)
        regex_to_state.generate_states_and_transitions()
        valid_states = regex_to_state.states

        constraint_collection_list[name] = constraint_collection
        valid_states_list[name] = valid_states

    partial_match_nums = {fb: {fa: [] for fa in frequency_A} for fb in frequency_B}
    result = []

    for fa in frequency_A:
        data_info_A = DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[12, 13],
                frequency=fa,
                probability=probability / 2)
        for fb in frequency_B:
            data_info_B = DataInfo(
                names='B',
                normal=list(range(0, 11)),
                matched=list(range(11, 20)),
                frequency=fb,
                probability=probability)

            data_info_list = [data_info_A, data_info_B]

            # 假设 window_times 和 window_time 是已知的或从配置中获得的
            window_times = {}

            nfa_a = NFA(valid_states_list["A"], window_times, window_time, handle_timeout=True)
            nfa_b = NFA(valid_states_list["B"], window_times, window_time, handle_timeout=True)

            # 初始化 SharedBuffer 和 NFA
            config = SharedBufferCacheConfig.from_config('../config.ini')
            shared_buffer_a = SharedBuffer(config)
            accessor_a = SharedBufferAccessor(shared_buffer_a)
            shared_buffer_b = SharedBuffer(config)
            accessor_b = SharedBufferAccessor(shared_buffer_b)
            accessor_list = {"A": accessor_a, "B": accessor_b}

            # 创建初始 NFA 状态
            nfa_state_a = nfa_a.create_initial_nfa_state()
            nfa_state_b = nfa_b.create_initial_nfa_state()
            nfa_state_list = {"A": nfa_state_a, "B": nfa_state_b}

            # Initialize the data queue
            data_queue = Queue()

            # Initialize and start data generation
            data_source = DataSource(data_info_list)
            processes = data_source.start_data_generation(data_queue)

            max_len = 0
            current_matches_a = []
            partial_match_nums_a = 0
            partial_match_nums_b = 0
            # 模拟数据流处理
            try:
                item_count = 0
                while True:
                    data_item = data_queue.get()
                    if data_item:

                        begin_time = time.time()
                        item_count += 1
                        di_name = data_item.variable_name
                        accessor = accessor_list[di_name]
                        nfa_state = nfa_state_list[data_item.get_id()]
                        nfa = nfa_a if di_name == "A" else nfa_b
                        # 封装事件并处理
                        event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                        # 修剪超过时间窗口限制的部分匹配结果
                        nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                        # 处理事件并更新 NFA 状态
                        matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                              TimerService())
                        if len(matches) > 0:
                            current_matches_a = [match for match in current_matches_a if
                                                 event_wrapper.timestamp - match["A[i]"][0].timestamp < window_time]
                            if di_name == "A":
                                current_matches_a.extend(matches)
                            elif di_name == "B":
                                for match_b in matches:
                                    valid_matches = [{"A": match, "B": match_b} for match in current_matches_a if
                                                 match["A[i]"][0].timestamp >=  event_wrapper.timestamp - window_time and
                                                     match["A[i]"][-1].timestamp <= match_b["B"][0].timestamp]

                        partial_match_count = len(nfa_state.partial_matches)
                        if di_name == "A":
                            partial_match_nums_a = partial_match_count
                        elif di_name == "B":
                            partial_match_nums_b = partial_match_count
                        partial_match_count = partial_match_nums_a + partial_match_nums_b

                        handle_time = time.time() - begin_time

                        result.append({
                            'type': 'independence',
                            'fa': fa,
                            'fb': fb,
                            'item_num': item_count,
                            'partial_match_nums': partial_match_count,
                            'handle_time': handle_time
                        })

                        if item_count >= max_items or handle_time > max_time * (fa + fb):  # 假设处理 100 个数据项后停止
                            break

                        max_len = max(max_len, len(nfa_state.partial_matches))

                    time.sleep(0.01)  # To prevent the main loop from consuming too much CPU
            except KeyboardInterrupt:
                for process in processes:
                    process.terminate()
                    process.join()


    # 将结果保存为 CSV 文件
    df_name = '../output/experiment/csv/pattern_with_prefix_kleene_two_table_join.csv'
    df = pd.DataFrame(result)
    df.to_csv(df_name, index=False)

    return df_name

if __name__ == "__main__":
    max_items = 128
    max_time = 10
    without_handle = test_prefix_normal(max_items, max_time)
    with_handle_postponing = test_prefix_optimization_postponing_handle(max_items, max_time)
    with_handle_two_table_join = test_prefix_optimization_two_table_join(max_items, max_time)

    # 需要测试最大吞吐量

    # 处理延迟

    plot_results_handle_time_partial_nums(without_handle, with_handle_postponing)


