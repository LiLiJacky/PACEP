from models.CountConstraint import CountConstraint
from models.TimeConstarint import TimeConstraint
from models.ValueConstraint import ValueConstraint
from nfa.State import State
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction


def test_state_transition():
    # 创建状态对象
    start_state = State("Start", "Start")
    end_state = State("End", "Final")

    # 创建状态转换动作
    action_take = StateTransitionAction.TAKE
    action_ignore = StateTransitionAction.IGNORE

    # 创建条件（约束）
    count_constraint = CountConstraint(["var1", "var2"], 1, 3)
    time_constraint = TimeConstraint(["time1", "time2"], 5, 10)
    value_constraint = ValueConstraint(["var3"], "var3 == 'expected_value'")

    # 创建状态转换对象
    state_transition1 = StateTransition(start_state, action_take, end_state, count_constraint)
    state_transition2 = StateTransition(start_state, action_ignore, end_state, time_constraint)

    # 测试 get_action 方法
    assert state_transition1.get_action() == action_take, "get_action 方法失败"
    assert state_transition2.get_action() == action_ignore, "get_action 方法失败"

    # 测试 get_target_state 方法
    assert state_transition1.get_target_state() == end_state, "get_target_state 方法失败"
    assert state_transition2.get_target_state() == end_state, "get_target_state 方法失败"

    # 测试 get_source_state 方法
    assert state_transition1.get_source_state() == start_state, "get_source_state 方法失败"
    assert state_transition2.get_source_state() == start_state, "get_source_state 方法失败"

    # 测试 get_condition 方法
    assert state_transition1.get_condition() == count_constraint, "get_condition 方法失败"
    assert state_transition2.get_condition() == time_constraint, "get_condition 方法失败"

    # 测试 set_condition 方法
    state_transition1.set_condition(value_constraint)
    assert state_transition1.get_condition() == value_constraint, "set_condition 方法失败"

    # 测试 __eq__ 方法
    assert state_transition1 != state_transition2, "__eq__ 方法失败"

    # 测试 __hash__ 方法
    assert hash(state_transition1) == hash(StateTransition(start_state, action_take, end_state, value_constraint)), "__hash__ 方法失败"

    # 测试 __str__ 方法
    assert str(state_transition1) == f"StateTransition({action_take}, from {start_state.get_name()} to {end_state.get_name()}, with condition)", "__str__ 方法失败"

    print("所有测试通过!")

# 运行测试方法
test_state_transition()