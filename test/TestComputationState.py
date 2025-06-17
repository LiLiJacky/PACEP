import unittest

from nfa.ComputationState import ComputationState
from sharedbuffer.EventId import EventId
from sharedbuffer.NodeId import NodeId
from util.DeweyNumber import DeweyNumber


class TestComputationState(unittest.TestCase):

    def setUp(self):
        # 创建测试数据
        self.version = DeweyNumber(1)  # 假设 DeweyNumber 已经定义
        self.event_id_1 = EventId(1, 1000)
        self.event_id_2 = EventId(2, 1000)
        self.node_id_1 = NodeId(self.event_id_1, "Page1")
        self.node_id_2 = NodeId(self.event_id_2, "Page2")
        self.start_timestamp = 1625097600000
        self.previous_timestamp = 1625097605000
        self.state_name = "state_1"

    def test_create_start_state(self):
        # 测试 create_start_state 方法
        state = ComputationState.create_start_state(self.state_name)
        self.assertEqual(state.current_state_name, self.state_name)
        self.assertEqual(state.version, DeweyNumber(1))
        self.assertEqual(state.start_timestamp, -1)
        self.assertEqual(state.previous_timestamp, -1)
        self.assertIsNone(state.previous_buffer_entry)
        self.assertIsNone(state.start_event_id)

    def test_create_state(self):
        # 测试 create_state 方法
        state = ComputationState.create_state(
            self.state_name,
            self.node_id_1,
            self.version,
            self.start_timestamp,
            self.previous_timestamp,
            self.event_id_1
        )
        self.assertEqual(state.current_state_name, self.state_name)
        self.assertEqual(state.version, self.version)
        self.assertEqual(state.start_timestamp, self.start_timestamp)
        self.assertEqual(state.previous_timestamp, self.previous_timestamp)
        self.assertEqual(state.previous_buffer_entry, self.node_id_1)
        self.assertEqual(state.start_event_id, self.event_id_1)

    def test_equals(self):
        # 测试 equals 方法
        state1 = ComputationState.create_state(
            self.state_name,
            self.node_id_1,
            self.version,
            self.start_timestamp,
            self.previous_timestamp,
            self.event_id_1
        )
        state2 = ComputationState.create_state(
            self.state_name,
            self.node_id_1,
            self.version,
            self.start_timestamp,
            self.previous_timestamp,
            self.event_id_1
        )
        self.assertEqual(state1, state2)

    def test_hash(self):
        # 测试 hash 方法
        state = ComputationState.create_state(
            self.state_name,
            self.node_id_1,
            self.version,
            self.start_timestamp,
            self.previous_timestamp,
            self.event_id_1
        )
        self.assertIsInstance(hash(state), int)

    def test_to_string(self):
        # 测试 __str__ 方法
        state = ComputationState.create_state(
            self.state_name,
            self.node_id_1,
            self.version,
            self.start_timestamp,
            self.previous_timestamp,
            self.event_id_1
        )
        expected_str = (
            f"ComputationState{{"
            f"currentStateName='{self.state_name}', "
            f"version={self.version}, "
            f"startTimestamp={self.start_timestamp}, "
            f"previousTimestamp={self.previous_timestamp}, "
            f"previousBufferEntry={self.node_id_1}, "
            f"startEventID={self.event_id_1}"
            f"}}"
        )
        self.assertEqual(str(state), expected_str)

if __name__ == '__main__':
    unittest.main()