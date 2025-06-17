from datetime import datetime, timedelta

from aftermatch.NoSkipStrategy import NoSkipStrategy
from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from models.ConstraintCollection import ConstraintCollection
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


def generate_matching_event_data():
    now = datetime.now()

    event_data = [
        Data(variable_name='A', timestamp=now, value=50),  # 满足 A
        Data(variable_name='C', timestamp=now + timedelta(seconds=1), value=100),  # 满足 C
        Data(variable_name='E', timestamp=now + timedelta(seconds=2), value=200),  # 满足 E
        Data(variable_name='D', timestamp=now + timedelta(seconds=3), value=100),  # 满足 D
        Data(variable_name='G', timestamp=now + timedelta(seconds=4), value=10),  # 满足 G
        Data(variable_name='G', timestamp=now + timedelta(seconds=5), value=12),  # 满足 G
        Data(variable_name='G', timestamp=now + timedelta(seconds=6), value=11),  # 满足 G
        Data(variable_name='F', timestamp=now + timedelta(seconds=8), value=45),  # 满足 F
        Data(variable_name='F', timestamp=now + timedelta(seconds=9), value=45),  # 满足 F
        Data(variable_name='B', timestamp=now + timedelta(seconds=10), value=50),  # 满足 B
    ]

    return event_data


def pattern_recognize_test():
    # 生成匹配的事件数据
    event_data = generate_matching_event_data()

    # 初始化 SharedBuffer 和 NFA
    config = SharedBufferCacheConfig.from_config('../config.ini')
    shared_buffer = SharedBuffer(config)
    accessor = SharedBufferAccessor(shared_buffer)

    # 通过正则表达式生成状态和状态转移
    regex = "A C E D G{3,} F{1,} B"
    constraints_dict = {
        'value_constrain': [
            {'variables': ['D', 'B'], 'expression': '0 < D + B < 159'},
            {'variables': ['A', 'C', 'E', 'D', 'F', 'B'], 'expression': '0 < A + C + E + D + sum(F) + B < 1000'},
            {'variables': ['G'], 'expression': '9 < G[i] <= 12'}
        ],
        'time_constrain': [
            {'variables': ['F', 'C', 'B', 'G', 'E'], 'min_time': 0, 'max_time': 5}
        ],
        'type_constrain': [
            {'variables': ['A'], 'variables_name': ['A']},
            {'variables': ['C'], 'variables_name': ['C']},
            {'variables': ['E'], 'variables_name': ['E']},
            {'variables': ['D'], 'variables_name': ['D']},
            {'variables': ['G'], 'variables_name': ['G']},
            {'variables': ['F'], 'variables_name': ['F']},
            {'variables': ['B'], 'variables_name': ['B']}
        ]
    }

    # 连续匹配策略
    contiguity_strategy = {
        'STRICT': {},
        'RELAXED': {},
        'NONDETERMINISTICRELAXED': {"A", "C", "E", "D", "G", "F", "B"}
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

    constraint_collection._window_time = 5

    # 现在 constraint_collection 包含了从字典转换过来的所有约束
    regex_to_state = RegexToState(regex, constraint_collection, contiguity_strategy)
    regex_to_state.generate_states_and_transitions()
    valid_states = regex_to_state.states
    regex_to_state.draw()

    # 假设 window_times 和 window_time 是已知的或从配置中获得的
    window_times = {}
    window_time = 5


    nfa = NFA(valid_states, window_times, window_time, handle_timeout=True)

    # 创建初始 NFA 状态
    nfa_state = nfa.create_initial_nfa_state()

    # 模拟数据流处理
    for value in event_data:
        # 封装事件并处理
        event_wrapper = EventWrapper(value, value.timestamp, accessor)

        # 处理事件并更新 NFA 状态
        matches = nfa.process(accessor, nfa_state, event_wrapper, value.timestamp, NoSkipStrategy.INSTANCE, TimerService())

        # 修剪超过时间窗口限制的部分匹配结果
        nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)

        print(f"Now PartialMatches: {len(nfa_state.partial_matches)}")

        if matches:
            print(f"Match found: {matches}")
            break  # 如果匹配成功，可以选择终止处理，或者继续处理其他事件

        print(f"Processed event: {value}, no match found yet.")

    print(nfa_state)

if __name__ == "__main__":
    pattern_recognize_test()