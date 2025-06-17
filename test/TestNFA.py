import unittest
from unittest.mock import MagicMock

from nfa.ComputationState import ComputationState
from nfa.NFA import ConditionContext, NFA, TimerService, EventWrapper
from nfa.NFAState import NFAState
from nfa.State import State
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction
from sharedbuffer.EventId import EventId
from sharedbuffer.NodeId import NodeId
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from util.DeweyNumber import DeweyNumber


class TestNFA(unittest.TestCase):

    def setUp(self):
        # 创建EventId、NodeId、DeweyNumber对象
        self.event_id_1 = EventId(1, 1000)
        self.event_id_2 = EventId(2, 2000)
        self.node_id_1 = NodeId(self.event_id_1, "Page1")
        self.node_id_2 = NodeId(self.event_id_2, "Page2")
        self.version_1 = DeweyNumber(1)
        self.version_2 = DeweyNumber(2)

        # 创建ComputationState对象
        self.state_1 = ComputationState.create_state(
            "state_1", self.node_id_1, self.version_1, 1000, 1500, self.event_id_1
        )
        self.state_2 = ComputationState.create_state(
            "state_2", self.node_id_2, self.version_2, 2000, 2500, self.event_id_2
        )

        # 创建NFAState对象
        self.nfa_state = NFAState([self.state_1, self.state_2])

        # 创建状态转换的模拟
        self.transition_1 = MagicMock(spec=StateTransition)
        self.transition_1.get_action.return_value = StateTransitionAction.TAKE
        self.transition_1.get_target_state.return_value = MagicMock(spec=State)
        self.transition_1.get_target_state().get_name.return_value = "state_2"
        self.transition_1.get_condition.return_value = None

        self.transition_2 = MagicMock(spec=StateTransition)
        self.transition_2.get_action.return_value = StateTransitionAction.IGNORE
        self.transition_2.get_target_state.return_value = MagicMock(spec=State)
        self.transition_2.get_target_state().get_name.return_value = "state_1"
        self.transition_2.get_condition.return_value = None

        self.state_1_obj = MagicMock(spec=State)
        self.state_1_obj.get_state_transitions.return_value = [self.transition_1, self.transition_2]
        self.state_1_obj.get_name.return_value = "state_1"
        self.state_1_obj.is_start.return_value = True
        self.state_1_obj.is_final.return_value = False

        self.state_2_obj = MagicMock(spec=State)
        self.state_2_obj.get_state_transitions.return_value = []
        self.state_2_obj.get_name.return_value = "state_2"
        self.state_2_obj.is_start.return_value = False
        self.state_2_obj.is_final.return_value = True

        # 创建NFA对象
        self.nfa = NFA([self.state_1_obj, self.state_2_obj], {"state_1": 1000, "state_2": 2000}, 3000, True)

    def test_initial_nfa_state(self):
        # 测试创建初始NFA状态
        initial_state = self.nfa.create_initial_nfa_state()
        self.assertEqual(len(initial_state.get_partial_matches()), 1)
        self.assertTrue(initial_state.is_state_changed() == False)

    def test_process(self):
        # 模拟SharedBufferAccessor对象
        shared_buffer_accessor = MagicMock(spec=SharedBufferAccessor)
        shared_buffer_accessor.extract_patterns.return_value = [{"state_1": [self.event_id_1]}]
        shared_buffer_accessor.put.return_value = self.node_id_2

        # 模拟AfterMatchSkipStrategy对象
        after_match_skip_strategy = MagicMock()

        # 处理事件并测试状态
        event = "event"
        result = self.nfa.process(shared_buffer_accessor, self.nfa_state, event, 1000, after_match_skip_strategy, TimerService())
        self.assertTrue(len(result) > 0)

    def test_advance_time(self):
        # 模拟SharedBufferAccessor对象
        shared_buffer_accessor = MagicMock(spec=SharedBufferAccessor)
        shared_buffer_accessor.extract_patterns.return_value = [{"state_1": [self.event_id_1]}]
        shared_buffer_accessor.put.return_value = self.node_id_2

        # 模拟AfterMatchSkipStrategy对象
        after_match_skip_strategy = MagicMock()

        # 测试时间推进并检查状态变化
        result, timeout_result = self.nfa.advance_time(shared_buffer_accessor, self.nfa_state, 3500, after_match_skip_strategy)
        self.assertTrue(len(timeout_result) > 0)

    def test_compute_next_states(self):
        # 模拟SharedBufferAccessor对象
        shared_buffer_accessor = MagicMock(spec=SharedBufferAccessor)
        event_wrapper = MagicMock(spec=EventWrapper)
        timer_service = TimerService()

        # 测试计算下一个状态
        next_states = self.nfa.compute_next_states(shared_buffer_accessor, self.state_1, event_wrapper, timer_service)
        self.assertTrue(len(next_states) > 0)

    def test_find_final_state_after_proceed(self):
        # 测试寻找最终状态
        event = "event"
        final_state = self.nfa.find_final_state_after_proceed(ConditionContext(MagicMock(), self.state_1, TimerService(), 1000), self.state_1_obj, event)
        self.assertIsNotNone(final_state)

if __name__ == "__main__":
    unittest.main()