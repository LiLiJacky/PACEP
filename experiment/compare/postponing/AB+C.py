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
from nfa.RegexToState import RegexToState
from nfa.State import State
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from simulated_data.SimulateData import Data
from sharedbuffer.EventId import EventId
from sharedbuffer.NodeId import NodeId
from multiprocessing import Queue
from simulated_data.SimulateDataByFile import DataSource


def nfa(valid_states, window_times, max_window_time, data_source, rate):
    nfa = NFA(valid_states, window_times, max_window_time, handle_timeout=True)

    # 初始化 SharedBuffer 和 NFA
    config = SharedBufferCacheConfig.from_config('../../../config.ini')
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

        #print("Result counts:", result_counts)
        print("Total handle time:" + str(last_handle_time - modify_begin_time))
        print("Total matches:" + str(total_matched))

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


def postponing(valid_states, window_times, max_window_time, data_source, rate, lazy_calculate_model):

    nfa = NFA(valid_states, window_times, max_window_time, handle_timeout=True, lazy_model=True, lazy_calculate_model=lazy_calculate_model)

    # 初始化 SharedBuffer 和 NFA
    config = SharedBufferCacheConfig.from_config('../../../config.ini')
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
        #print("Result counts:", result_counts)
        print("Total handle time:" + str(last_handle_time - modify_begin_time))
        print("Total matches:" + str(total_matched))

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()

def test(regex, min_window_time, max_window_time, data_source, rate, lazy_model, lazy_calculate_model):
    constraints_dict = {
        'value_constrain': [
            {'variables': ['B', 'B[i]'], 'expression': '0 <= B[i] - last(B) <= 1000'}
        ],
        'time_constrain': [
            {'variables': ['A', 'B', 'C'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['a']},
            {'variables': ['B'], 'variables_name': ['b']},
            {'variables': ['C'], 'variables_name': ['c']},
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': set(),
        'RELAXED': set(),
        'NONDETERMINISTICRELAXED': {'A', 'B', 'C'}
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
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy, lazy_model)
    regex_to_state.generate_states_and_transitions()
    if lazy_model:
        regex_to_state.add_lazy_handle()
    valid_states = regex_to_state.states


    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}

    if lazy_model:
        postponing(valid_states, window_times, max_window_time, data_source, rate, lazy_calculate_model)
    else:
        # print("Test nfa handle:")
        nfa(valid_states, window_times, max_window_time, data_source, rate)




if __name__ == "__main__":
    regex = "A B+ C"

    # window_time = 50 ; matches = 949 ; time = 2.03
    # data_source = '../../../data/A+B+C_test2.csv'

    data_source = '../../../data/compare/core.csv'

    rate = 400000

    max_window_times = [4, 8, 16, 24, 32]
    min_window_time = 0

    data_source = '../../../data/compare/core.csv'
    rate = 400000

    for max_window_time in max_window_times:
        print(f"Testing with max_window_time = {max_window_time}")

        print("Test lazy handle with INCREMENTCALCULATE:")
        test(regex, min_window_time, max_window_time, data_source, rate, True, "INCREMENTCALCULATE")

        print("Test lazy handle with TABLECALCULATE:")
        test(regex, min_window_time, max_window_time, data_source, rate, True, "TABLECALCULATE")
        #
        # print("Test nfa handle:")
        # test(regex, min_window_time, max_window_time, data_source, rate, False, "")

