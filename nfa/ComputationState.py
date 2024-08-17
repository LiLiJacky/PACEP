from dataclasses import dataclass
from typing import Optional, Tuple

from sharedbuffer import NodeId, EventId
from util.DeweyNumber import DeweyNumber


@dataclass(frozen=True)
class ComputationState:
    current_state_name: str
    version: 'DeweyNumber'
    start_timestamp: int
    previous_timestamp: int
    previous_buffer_entry: Optional['NodeId'] = None
    start_event_id: Optional['EventId'] = None

    def get_start_event_id(self) -> Optional['EventId']:
        return self.start_event_id

    def get_previous_buffer_entry(self) -> Optional['NodeId']:
        return self.previous_buffer_entry

    def get_start_timestamp(self) -> int:
        return self.start_timestamp

    def get_previous_timestamp(self) -> int:
        return self.previous_timestamp

    def get_current_state_name(self) -> str:
        return self.current_state_name

    def get_version(self) -> 'DeweyNumber':
        return self.version

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ComputationState):
            return (
                    self.current_state_name == other.current_state_name
                    and self.version == other.version
                    and self.start_timestamp == other.start_timestamp
                    and self.previous_timestamp == other.previous_timestamp
                    and self.start_event_id == other.start_event_id
                    and self.previous_buffer_entry == other.previous_buffer_entry
            )
        return False

    def __hash__(self) -> int:
        return hash((
            self.current_state_name,
            self.version,
            self.start_timestamp,
            self.previous_timestamp,
            self.start_event_id,
            self.previous_buffer_entry
        ))

    def __str__(self) -> str:
        return (
            f"ComputationState{{"
            f"currentStateName='{self.current_state_name}', "
            f"version={self.version}, "
            f"startTimestamp={self.start_timestamp}, "
            f"previousTimestamp={self.previous_timestamp}, "
            f"previousBufferEntry={self.previous_buffer_entry}, "
            f"startEventID={self.start_event_id}"
            f"}}"
        )

    @staticmethod
    def create_start_state(state: str, version: Optional['DeweyNumber'] = None) -> 'ComputationState':
        if version is None:
            version = DeweyNumber(1)
        return ComputationState.create_state(state, None, version, -1, -1, None)

    @staticmethod
    def create_state(
            current_state: str,
            previous_entry: Optional['NodeId'],
            version: 'DeweyNumber',
            start_timestamp: int,
            previous_timestamp: int,
            start_event_id: Optional['EventId']
    ) -> 'ComputationState':
        return ComputationState(
            current_state_name=current_state,
            version=version,
            start_timestamp=start_timestamp,
            previous_timestamp=previous_timestamp,
            previous_buffer_entry=previous_entry,
            start_event_id=start_event_id
        )

    def _compare(self, other: 'ComputationState') -> Tuple[bool, Optional[bool]]:
        if self.start_event_id is None and other.start_event_id is None:
            return False, None
        if self.start_event_id is None:
            return False, False
        if other.start_event_id is None:
            return False, True
        if self.start_event_id.timestamp != other.start_event_id.timestamp:
            return True, self.start_event_id.timestamp < other.start_event_id.timestamp
        return True, self.start_event_id.id < other.start_event_id.id

    def __lt__(self, other: 'ComputationState') -> bool:
        _, result = self._compare(other)
        return result if result is not None else False

    def __le__(self, other: 'ComputationState') -> bool:
        _, result = self._compare(other)
        return result if result is not None else False or self == other
