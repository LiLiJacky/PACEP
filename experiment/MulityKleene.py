import csv
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

def test_frequency_influence_latency():
    # 探究相互独立分布的两个子模式，频率对组合数的影响
    regex = "C A{2,} B{2,}"

    frequency_A = {0.5, 1}
    frequency_B = {0.5, 1}
    frequency_C = {1}

    window_time = 10
    probability_a = 0.001
    probability_b = 0.0001
    probability_c = 0.001

    # constraints_dict = {
    #     'value_constrain': [{'variables': ['C'], 'expression': '10 <= C <= 20'},
    #                         {'variables': ['A'], 'expression': '10 <= average(A) <= 20'},
    #                         {'variables': ['B'], 'expression': '10 <= average(B) <= 20'}],
    #     'time_constrain': [
    #         {'variables': ['C', 'A[i]', 'B[i]'], 'min_time': 0, 'max_time': window_time}
    #     ],
    #     'type_constrain': [
    #         {'variables': ['C'], 'variables_name': ['C']},
    #         {'variables': ['A'], 'variables_name': ['A']},
    #         {'variables': ['B'], 'variables_name': ['B']}
    #     ]
    # }

    constraints_dict = {
        'value_constrain': [{'variables': ['C'], 'expression': '10 <= C <= 20'},
                            {'variables': ['A'], 'expression': '10 <= sum_square_difference(A) <= 20'},
                            {'variables': ['B'], 'expression': '10 <= average(B) <= 20'},
                            {'variables': ['A', 'B'], 'expression': 'max(B) + average(A) >= 25'}],
        'time_constrain': [
            {'variables': ['C', 'A[i]', 'B[i]'], 'min_time': 0, 'max_time': window_time}
        ],
        'type_constrain': [
            {'variables': ['C'], 'variables_name': ['C']},
            {'variables': ['A'], 'variables_name': ['A']},
            {'variables': ['B'], 'variables_name': ['B']}
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
    regex_to_state.draw()

    partial_match_nums = {fb: {fa: [] for fa in frequency_A} for fb in frequency_B}
    result = []

    for fc in frequency_C:
        data_info_C = DataInfo(
            names='C',
            normal=list(range(0, 11)),
            matched=[12],
            frequency=fc,
            probability=probability_c)
        for fa in frequency_A:
            data_info_A = DataInfo(
                    names='A',
                    normal=list(range(0, 11)),
                    matched=[12, 13],
                    frequency=fa,
                    probability=probability_a / 2)
            for fb in frequency_B:
                data_info_B = DataInfo(
                    names='B',
                    normal=list(range(0, 11)),
                    matched=[12, 13],
                    frequency=fb,
                    probability=probability_b / 2)

                data_info_list = [data_info_C, data_info_A, data_info_B]

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
                            matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                                  TimerService())

                            partial_match_count = len(nfa_state.partial_matches)
                            partial_match_nums[fb][fa].append(partial_match_count)

                            handle_time = time.time() - begin_time

                            result.append({
                                'type': 'normal',
                                'fa': fa,
                                'fb': fb,
                                'item_num': item_count,
                                #'partial_match_nums': partial_match_count,
                                'handle_time': handle_time
                            })

                            if item_count >= 40 or handle_time > 100 * (fa + fb):  # 假设处理 100 个数据项后停止
                                break

                        time.sleep(0.01)  # To prevent the main loop from consuming too much CPU
                except KeyboardInterrupt:
                    for process in processes:
                        process.terminate()
                        process.join()

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequency_B)))

    for color, (fb, fa_dict) in zip(colors, partial_match_nums.items()):
        for fa, matches in fa_dict.items():
            plt.plot(range(1, len(matches) + 1), matches, label=f"fa={fa}, fb={fb}", color=color)

    plt.xlabel("Processed Data Items")
    plt.ylabel("Handle Time")
    plt.title("Handle Time for Different Frequencies")
    plt.legend()
    plt.show()

    plt.savefig('../output/experiment/Frequency_Influence_on_Partial_Matches.png')

    # 将结果保存为 CSV 文件
    df = pd.DataFrame(result)
    df.to_csv('../output/experiment/csv/frequency_influence_partial_matches_time.csv', index=False)



if __name__ == "__main__":
    test_frequency_influence_latency()