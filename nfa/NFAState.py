from dataclasses import dataclass, field
from typing import List, Optional
import heapq

from nfa.ComputationState import ComputationState
from sharedbuffer.EventId import EventId
from sharedbuffer.NodeId import NodeId
from util.DeweyNumber import DeweyNumber


@dataclass
class NFAState:
    partial_matches: List[ComputationState] = field(default_factory=list, compare=False)
    completed_matches: List[ComputationState] = field(default_factory=list, compare=False)
    state_changed: bool = field(default=False, compare=False)
    is_new_start_partial_match: bool = field(default=False, compare=False)

    def __post_init__(self):
        # 将列表转换为优先队列
        heapq.heapify(self.partial_matches)
        heapq.heapify(self.completed_matches)

    def is_state_changed(self) -> bool:
        return self.state_changed

    def reset_state_changed(self):
        self.state_changed = False

    def set_state_changed(self):
        self.state_changed = True

    def get_partial_matches(self) -> List[ComputationState]:
        return self.partial_matches

    def get_completed_matches(self) -> List[ComputationState]:
        return self.completed_matches

    def set_new_partial_matches(self, new_partial_matches: List[ComputationState]):
        heapq.heapify(new_partial_matches)
        self.partial_matches = new_partial_matches

    def is_new_start_partial_match(self) -> bool:
        return self.is_new_start_partial_match

    def reset_new_start_partial_match(self):
        self.is_new_start_partial_match = False

    def set_new_start_partial_match(self):
        self.is_new_start_partial_match = True

    def __eq__(self, other):
        if not isinstance(other, NFAState):
            return NotImplemented
        return (sorted(self.partial_matches) == sorted(other.partial_matches) and
                sorted(self.completed_matches) == sorted(other.completed_matches))

    def __hash__(self):
        return hash((tuple(self.partial_matches), tuple(self.completed_matches)))

    def __str__(self):
        return (f"NFAState{{"
                f"partialMatches={self.partial_matches}, "
                f"completedMatches={self.completed_matches}, "
                f"stateChanged={self.state_changed}"
                f"}}")

# 示例测试代码
if __name__ == "__main__":
    # 假设 EventId 和 NodeId 已经定义
    event_id = EventId(1, 1000)
    node_id = NodeId(event_id, "Page1")

    # 创建 ComputationState 示例对象
    state1 = ComputationState.create_state(
        current_state="state_1",
        previous_entry=node_id,
        version=DeweyNumber(1),
        start_timestamp=1625097600000,
        previous_timestamp=1625097605000,
        start_event_id=event_id
    )
    state2 = ComputationState.create_state(
        current_state="state_2",
        previous_entry=None,
        version=DeweyNumber(2),
        start_timestamp=1625097601000,
        previous_timestamp=1625097606000,
        start_event_id=None
    )

    # 输出 ComputationState 的字符串表示
    print(state1)
    print(state2)

    # 检查状态相等性
    print("State1 equals State2:", state1 == state2)

    # 测试哈希值
    print("Hash of State1:", hash(state1))
    print("Hash of State2:", hash(state2))

    # 创建初始状态
    start_state = ComputationState.create_start_state("initial_state")
    print("Start State:", start_state)

    # 测试 get 方法
    print("State1's start event ID:", state1.get_start_event_id())
    print("State1's previous buffer entry:", state1.get_previous_buffer_entry())
    print("State1's start timestamp:", state1.get_start_timestamp())
    print("State1's previous timestamp:", state1.get_previous_timestamp())
    print("State1's current state name:", state1.get_current_state_name())
    print("State1's version:", state1.get_version())