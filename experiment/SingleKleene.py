import time
from datetime import datetime, timedelta

import numpy as np
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
from simulated_data.SimulateDataByDataInfo import DataSource

def test_algorithm_influence_latency():
    # 探究模式计算代价对计算延迟的影响,当处理延迟超过两倍生成速率时，认为处理已经无法满足实时处理需求
    regex = "A{4,}"

    frequency = 0.1
    window_time = 10
    # 设计不同的algorithm
    value_constraint_set = {
        "o(1)": {
            "value_constrain": [{'variables': ['A'], 'expression': '10 < last(A) <= 20'}],
            "data_info": DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[20],
                frequency=frequency,
                probability=0.1)
        },
        "o(n)": {
            "value_constrain": [{'variables': ['A'], 'expression': '10 <= average(A) <= 20'}],
            "data_info": DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[12, 13, 15, 16],
                frequency=frequency,
                probability=0.1 / 4)
        },
        "o(n_log_n)": {
            "value_constrain": [{'variables': ['A'], 'expression': '13 <= merge_sort(A)[2] <= 15'}],
            "data_info": DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[12, 13, 15, 17],
                frequency=frequency,
                probability=0.1 / 4)
        },
        "o(n^2)": {
            "value_constrain": [{'variables': ['A'], 'expression': '13 <= bubble_sort(A)[2] <= 15'}],
            "data_info": DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[12, 13, 15, 17],
                frequency=frequency,
                probability=0.1 / 4)
        },
        "o(n^3)": {
            "value_constrain": [{'variables': ['A'], 'expression': '40 <= triplet_sum(A) <= 70'}],
            "data_info": DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[15, 16, 17, 18],
                frequency=frequency,
                probability=0.1 / 4)
        },
        "o(n_factorial)": {
            "value_constrain": [{'variables': ['A'], 'expression': '15 <= permutations(A) <= 20'}],
            "data_info": DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[20],
                frequency=frequency,
                probability=0.1)
        },
        "o(2^n)": {
            "value_constrain": [{'variables': ['A'], 'expression': '1100 <= subset_sum(A) <= 1296'}],
            "data_info": DataInfo(
                names='A',
                normal=list(range(0, 11)),
                matched=[15, 16, 17, 18],
                frequency=frequency,
                probability=0.1 / 4)
        }
    }

    latency_results = {al: [] for al in value_constraint_set.keys()}

    for al, vc in value_constraint_set.items():
        constraints_dict = {
            'value_constrain': vc["value_constrain"],
            'time_constrain': [
                {'variables': ['A[i]'], 'min_time': 0, 'max_time': window_time}
            ],
            'type_constrain': [
                {'variables': ['A'], 'variables_name': ['A']}
            ]
        }

        data_info_A = vc['data_info']

        data_info_list = [data_info_A]

        # 连续匹配策略
        contiguity_strategy = {
            'STRICT': {},
            'RELAXED': {},
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

        constraint_collection._window_time = window_time

        # 现在 constraint_collection 包含了从字典转换过来的所有约束
        regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
        regex_to_state.generate_states_and_transitions()
        valid_states = regex_to_state.states
        regex_to_state.draw()

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

        modify_begin_time = time.time()
        # 模拟数据流处理
        try:
            while True:
                data_item = data_queue.get()
                if data_item:
                    # 封装事件并处理
                    event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                    start_time = time.time()

                    # 修剪超过时间窗口限制的部分匹配结果
                    nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                    # 处理事件并更新 NFA 状态
                    matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                          TimerService())

                    end_time = time.time()
                    latency = end_time - start_time
                    latency_results[al].append((len(nfa_state.partial_matches), latency))

                    if latency > 20 * data_info_A.frequency or end_time - modify_begin_time > 300:
                        print(latency_results[al])
                        break

                time.sleep(0.01)  # To prevent the main loop from consuming too much CPU
        except KeyboardInterrupt:
            for process in processes:
                process.terminate()
                process.join()


    for al, latencies in latency_results.items():
        latencies.sort()
        x = [item[0] for item in latencies]
        y = [item[1] for item in latencies]
        y_avg = np.mean(y)
        plt.plot(x, y, label=f"{al} (avg: {y_avg:.6f}s)")

    plt.xlabel("Number of Partial Matches")
    plt.ylabel("Computation Latency (s)")
    plt.title("Algorithm Influence on Computation Latency")
    plt.legend()
    plt.show()
    plt.savefig('../output/experiment/Algorithm_Influence_on_Computation_Latency.png')


if __name__ == "__main__":
    test_algorithm_influence_latency()