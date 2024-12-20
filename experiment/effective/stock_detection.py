import copy
import os
import time
import psutil
from datetime import datetime, timedelta
from itertools import combinations
from queue import Empty

import numpy as np
import pandas as pd
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
from shared_calculate.SimpleBloomFilterManager import SimpleBloomFilterManager
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from simulated_data.SimulateData import Data
from sharedbuffer.EventId import EventId
from sharedbuffer.NodeId import NodeId
from multiprocessing import Queue
from simulated_data.SimulateDataByFile import DataSource


# 获取当前进程的内存占用
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # 返回内存占用，单位为 MB


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
    matcount = 0

    bf = SimpleBloomFilterManager(10000000)

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
                        # lista = []
                        # for a in match[0]['A']:
                        #     atime = pd.to_datetime(a.event.get_timestamp(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                        #     lista.append(atime)
                        #
                        # listb = []
                        # for b in match[0]['B']:
                        #     btime = pd.to_datetime(b.event.get_timestamp(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                        #     listb.append(btime)

                        stringId = table_tool.ensure_hashable(match)

                        # if stringId not in total_matched:
                        #     total_matched.add(stringId)

                        if not bf.check(stringId):
                            bf.add(stringId)
                            matcount += 1
                            # result_list.append({"a":lista,"b":listb})

                last_handle_time = time.time()

            memory_usage_total += get_memory_usage()
            events_nums += 1
            if events_nums % 10 == 0:
                memory_usage_total /= 10
            time.sleep(1 / rate)  # To prevent the main loop from consuming too much CPU
    except Empty:
        # print("Total handle time:" + str(last_handle_time - modify_begin_time))
        # print("Total matches:" + str(len(total_matched)))
        #
        # print(record)

        #return last_handle_time - modify_begin_time, len(total_matched), memory_usage_total
        print(matcount)
        # print(result_list)
        return last_handle_time - modify_begin_time, matcount, memory_usage_total

    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()

def test(regex, min_window_time, max_window_time, data_source, rate, lazy_model, lazy_calculate_model, contiguity_strategy):
    constraints_dict = {
        'value_constrain': [
            {'variables': ['A', 'A[i]'], 'expression': '0 <= A[i] - last(A) <= 1000'},
            {'variables': ['A'], 'expression': '0 <= last(A) - 1.1118 * first(A) <= 1000'},
            {'variables': ['B', 'B[i]'], 'expression': '0 <= B[i] - last(B) <= 1000'},
            {'variables': ['A', 'B'], 'expression': '0 <= last(B) - first(B) - last(A) + first(A) <= 1000'}
        ],
        'time_constrain': [
            {'variables': ['A', 'B'], 'min_time': min_window_time, 'max_time': max_window_time}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['GOOG']},
            {'variables': ['B'], 'variables_name': ['MSFT']},
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




if __name__ == "__main__":
    regex = "A{5,} B{5,}"

    rate = 400000

    results = []

    # 运行100轮实验的结果存储
    experiment_results = []

    begin_time = time.time()
    run_results = []

    output_name = f"goog_msft_small"

    max_window_times = [10, 14, 18, 22, 26, 30]
    #max_window_times = [10]
    min_window_time = 0

    data_source = '../../data/compare/' + output_name + '.csv'

    for max_window_time in max_window_times:
        # 连续匹配策略
        contiguity_strategys = {
            "STAM": {
                'STRICT': set(),
                'RELAXED': set(),
                'NONDETERMINISTICRELAXED': {'A', 'B'}
            }
        }
        print("Window size:" + str(max_window_time))
        max_window_time = max_window_time * 60 * 60 * 24
        out_limit = False
        for event_selection_strategy, contiguity_strategy in contiguity_strategys.items():
            # Lazy 测试
            lazy_handle_time, lazy_matches, lazy_memory = test(regex, min_window_time, max_window_time + 1, data_source, rate,
                                                  True, "TABLECALCULATE", contiguity_strategy)
            print("Lazy handle cost time:" + str(lazy_handle_time))

            lazy_increment_begin_time = time.time()
            # Lazy 优化测试
            lazy_increment_handle_time, lazy_increment_matches, lazy_increment_memory = test(regex, min_window_time, max_window_time + 1,
                                                                      data_source, rate,
                                                                      True, "INCREMENTCALCULATE",
                                                                      contiguity_strategy)
            print("Lazy increment handle cost time:" + str(lazy_increment_handle_time))

            # Lazy 结果
            lazy_result = {
                "window_time": max_window_time,
                "selection_strategy": event_selection_strategy,
                "engine": "pacep",
                "handle_time": lazy_handle_time,
                "matches": lazy_matches,
                "memory_usage": lazy_memory
            }
            # increment 结果
            lazy_increment_result = {
                "window_time": max_window_time,
                "selection_strategy": event_selection_strategy,
                "engine": "pacep_increment",
                "handle_time": lazy_increment_handle_time,
                "matches": lazy_increment_matches,
                "memory_usage": lazy_increment_memory
            }

            run_results.append(lazy_result)
            run_results.append(lazy_increment_result)

    experiment_results.append(run_results)


    # 将 experiment_results 转换为适合存储的格式
    flat_results = [
        {
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
    output_path = "../../output/stock_detection_result.csv"
    results_df.to_csv(output_path, index=False)
