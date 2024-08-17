from abc import ABC, abstractmethod
from typing import Collection, Dict, List

from nfa.ComputationState import ComputationState
from sharedbuffer.EventId import EventId
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor


class AfterMatchSkipStrategy(ABC):
    @abstractmethod
    def is_skip_strategy(self) -> bool:
        pass

    def prune(
        self,
        matches_to_prune: Collection[ComputationState],
        matched_result: Collection[Dict[str, List[EventId]]],
        shared_buffer_accessor: SharedBufferAccessor
    ) -> None:
        if not self.is_skip_strategy():
            return

        pruning_id = self.get_pruning_id(matched_result)
        if pruning_id:
            discard_states = []
            for computation_state in matches_to_prune:
                start_event_id = computation_state.get_start_event_id()
                if start_event_id and self.should_prune(start_event_id, pruning_id):
                    shared_buffer_accessor.release_node(
                        computation_state.previous_buffer_entry,
                        computation_state.version
                    )
                    discard_states.append(computation_state)

            for state in discard_states:
                matches_to_prune.remove(state)

    @abstractmethod
    def should_prune(self, start_event_id: EventId, pruning_id: EventId) -> bool:
        pass

    @abstractmethod
    def get_pruning_id(self, match: Collection[Dict[str, List[EventId]]]) -> EventId:
        pass

    def get_pattern_name(self):
        return None  # Returns None by default, can be overridden if needed

    @staticmethod
    def max(o1: EventId, o2: EventId) -> EventId:
        if o2 is None:
            return o1
        if o1 is None:
            return o2
        return o1 if o1.compare_to(o2) >= 0 else o2

    @staticmethod
    def min(o1: EventId, o2: EventId) -> EventId:
        if o2 is None:
            return o1
        if o1 is None:
            return o2
        return o1 if o1.compare_to(o2) <= 0 else o2