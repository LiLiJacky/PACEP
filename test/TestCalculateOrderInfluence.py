import time
from datetime import datetime, timedelta

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

def pattern_recognize_test():

    # 通过正则表达式生成状态和状态转移
    regex = "A{2,3} B{1,2}"
    constraints_dict = {
        'value_constrain': [
            {'variables': ['A'], 'expression': '10 <= last(A) - first(A) <= 20'},
            {'variables': ['B'], 'expression': '5 <= B[i] <= 10'}
        ],
        'time_constrain': [
            {'variables': ['A[i]', 'B[i]'], 'min_time': 0, 'max_time': 10}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['A']},
            {'variables': ['B'], 'variables_name': ['B']}
        ]
    }

    data_info_A = DataInfo(
        names='A',
        normal=list(range(0, 11)),  # 0 to 10
        matched=[20],  # matched data is 20
        frequency= 1,  # example frequency
        probability=0.1  # example probability
    )

    data_info_B = DataInfo(
        names='B',
        normal=list(range(0, 5)),  # 0 to 4
        matched=[7],  # matched data is 7
        frequency= 1,  # example frequency
        probability=0.1  # example probability
    )

    data_info_list = [data_info_A, data_info_B]

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {},
        'NONDETERMINISTICRELAXED': {"A", "B"}
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

    constraint_collection._window_time = 10

    # 现在 constraint_collection 包含了从字典转换过来的所有约束
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
    regex_to_state.generate_states_and_transitions()
    valid_states = regex_to_state.states
    regex_to_state.draw()

    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}
    window_time = 10

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

    # 模拟数据流处理
    try:
        while True:
            data_item = data_queue.get()
            print(data_item)
            if data_item:
                # 封装事件并处理
                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)

                # 修剪超过时间窗口限制的部分匹配结果
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

                # 处理事件并更新 NFA 状态
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,
                                      TimerService())
                print(f"Now PartialMatches: {len(nfa_state.partial_matches)}")

                if matches:
                    print(f"Match found: {matches}")
                    #break  # 如果匹配成功，可以选择终止处理，或者继续处理其他事件
            time.sleep(0.001)  # To prevent the main loop from consuming too much CPU
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    pattern_recognize_test()