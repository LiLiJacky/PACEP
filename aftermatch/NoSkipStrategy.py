from typing import Collection, Dict, List

from aftermatch.AfterMatchSkipStrategy import AfterMatchSkipStrategy
from sharedbuffer.EventId import EventId


class NoSkipStrategy(AfterMatchSkipStrategy):
    INSTANCE = None

    def __init__(self):
        if not NoSkipStrategy.INSTANCE:
            NoSkipStrategy.INSTANCE = self

    def is_skip_strategy(self) -> bool:
        return False

    def should_prune(self, start_event_id: EventId, pruning_id: EventId) -> bool:
        raise RuntimeError("This should never happen. Please file a bug.")

    def get_pruning_id(self, match: Collection[Dict[str, List[EventId]]]) -> EventId:
        raise RuntimeError("This should never happen. Please file a bug.")

    def __str__(self):
        return "NoSkipStrategy{}"


NoSkipStrategy.INSTANCE = NoSkipStrategy()