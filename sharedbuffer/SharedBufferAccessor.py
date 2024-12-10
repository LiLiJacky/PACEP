from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from models.ValueConstraint import ValueConstraint
from sharedbuffer.Lockable import Lockable
from sharedbuffer.NodeId import NodeId
from sharedbuffer.ShareBufferNode import SharedBufferNode
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferEdge import SharedBufferEdge

from typing import Any, List, Dict, Optional, Tuple, Deque
from collections import defaultdict, deque

class SharedBufferAccessor:
    def __init__(self, shared_buffer: 'SharedBuffer'):
        self.shared_buffer = shared_buffer

    def advance_time(self, timestamp: int) -> None:
        self.shared_buffer.advance_time(timestamp)

    def register_event(self, value: Any, timestamp: int) -> 'EventId':
        return self.shared_buffer.register_event(value, timestamp)

    def put(
        self,
        state_name: str,
        event_id: 'EventId',
        previous_node_id: Optional['NodeId'],
        version: 'DeweyNumber',
        lazy_constraints: List[ValueConstraint]
    ) -> 'NodeId':
        if previous_node_id is not None:
            self.lock_node(previous_node_id, version)

        current_node_id = NodeId(event_id, state_name)
        current_node = self.shared_buffer.get_entry(current_node_id)
        if current_node is None:
            current_node = Lockable(SharedBufferNode(), 0)
            self.lock_event(event_id)

        current_node.get_element().add_edge(SharedBufferEdge(previous_node_id, version, lazy_constraints))
        self.shared_buffer.upsert_entry(current_node_id, current_node)

        return current_node_id

    def extract_patterns(
        self, node_id: 'NodeId', version: 'DeweyNumber'
    ) -> Tuple[List[Dict[str, List['EventId']]], List['ValueConstraint']]:
        result = []

        extraction_states = deque()
        entry_lock = self.shared_buffer.get_entry(node_id)
        latency_constrain = []

        if entry_lock is not None:
            entry = entry_lock.get_element()
            extraction_states.append(self.ExtractionState((node_id, entry), version, deque()))

            while extraction_states:
                extraction_state = extraction_states.pop()
                current_path = extraction_state.getPath()
                current_entry = extraction_state.getEntry()

                if current_entry is None:
                    complete_path = defaultdict(list)

                    while current_path:
                        current_path_entry = current_path.pop()[0]
                        page = current_path_entry.page_name
                        complete_path[page].append(current_path_entry.event_id)
                    result.append(complete_path)
                else:
                    current_path.append(current_entry)
                    first_match = True
                    for lockable_edge in current_entry[1].get_edges():
                        edge = lockable_edge.get_element()
                        current_version = extraction_state.getVersion()
                        if current_version.is_compatible_with(edge.get_dewey_number()):
                            target = edge.get_target()
                            new_path = current_path if first_match else deque(current_path)
                            first_match = False

                            if edge.lazy_calculate_value_constrain:
                                latency_constrain.extend(edge.lazy_calculate_value_constrain)
                            extraction_states.append(
                                self.ExtractionState(
                                    (target, self.shared_buffer.get_entry(target).get_element()) if target else None,
                                    edge.get_dewey_number(),
                                    new_path
                                )
                            )
        return result, latency_constrain

    def materialize_match(self, match: Dict[str, List['EventId']]) -> Dict[str, List[Any]]:
        materialized_match = defaultdict(list)

        for pattern, event_ids in match.items():
            events = []
            for event_id in event_ids:
                event = self.shared_buffer.get_event(event_id).get_element()
                events.append(event)
            materialized_match[pattern] = events

        return materialized_match

    def lock_node(self, node: 'NodeId', version: 'DeweyNumber') -> None:
        shared_buffer_node = self.shared_buffer.get_entry(node)
        if shared_buffer_node is not None:
            shared_buffer_node.lock()
            for edge in shared_buffer_node.get_element().get_edges():
                if version.is_compatible_with(edge.get_element().get_dewey_number()):
                    edge.lock()
            self.shared_buffer.upsert_entry(node, shared_buffer_node)

    def release_node(self, node: 'NodeId', version: 'DeweyNumber') -> None:
        nodes_to_examine = deque([node])
        versions_to_examine = deque([version])

        while nodes_to_examine:
            cur_node = nodes_to_examine.pop()
            cur_buffer_node = self.shared_buffer.get_entry(cur_node)

            if cur_buffer_node is None:
                break

            current_version = versions_to_examine.pop()
            edges = cur_buffer_node.get_element().get_edges()
            edges_to_remove = []
            edges_iterator = iter(edges)
            while True:
                try:
                    shared_buffer_edge = next(edges_iterator)
                    edge = shared_buffer_edge.get_element()
                    if current_version.is_compatible_with(edge.get_dewey_number()):
                        if shared_buffer_edge.release():
                            edges_to_remove.append(shared_buffer_edge)
                            target_id = edge.get_target()
                            if target_id is not None:
                                nodes_to_examine.append(target_id)
                                versions_to_examine.append(edge.get_dewey_number())
                except StopIteration:
                    break

            # Remove edges after iteration to avoid modifying list during iteration
            for edge in edges_to_remove:
                edges.remove(edge)

            if cur_buffer_node.release():
                self.shared_buffer.remove_entry(cur_node)
                self.release_event(cur_node.event_id)
            else:
                self.shared_buffer.upsert_entry(cur_node, cur_buffer_node)

    def lock_event(self, event_id: 'EventId') -> None:
        event_wrapper = self.shared_buffer.get_event(event_id)
        if event_wrapper is None:
            raise RuntimeError(f"Referring to non-existent event with id {event_id}")
        event_wrapper.lock()
        self.shared_buffer.upsert_event(event_id, event_wrapper)

    def release_event(self, event_id: 'EventId') -> None:
        event_wrapper = self.shared_buffer.get_event(event_id)
        if event_wrapper is not None:
            if event_wrapper.release():
                self.shared_buffer.remove_event(event_id)
            else:
                self.shared_buffer.upsert_event(event_id, event_wrapper)

    def close(self) -> None:
        self.shared_buffer.flush_cache()

    class ExtractionState:
        def __init__(
            self,
            entry: Tuple['NodeId', 'SharedBufferNode'],
            version: 'DeweyNumber',
            path: Deque
        ):
            self.entry = entry
            self.version = version
            self.path = path

        def getEntry(self) -> Tuple['NodeId', 'SharedBufferNode']:
            return self.entry

        def getPath(self) -> Deque:
            return self.path

        def getVersion(self) -> 'DeweyNumber':
            return self.version

        def __str__(self):
            return f"ExtractionState({self.entry}, {self.version}, {list(self.path)})"

# 示例使用
# config.ini 示例内容：
# [SharedBufferCacheConfig]
# events_buffer_cache_slots = 100
# entry_cache_slots = 50
# cache_statistics_interval = 300  # 单位是秒
if __name__ == "__main__":
    config = SharedBufferCacheConfig.from_config('../config.ini')
    shared_buffer = SharedBuffer(config)
    accessor = SharedBufferAccessor(shared_buffer)