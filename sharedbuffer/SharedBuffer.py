import logging
from datetime import timedelta
from cachetools import LRUCache
from collections import defaultdict
from typing import Any, Dict, Optional, Iterator

from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from sharedbuffer.EventId import EventId
from sharedbuffer.Lockable import Lockable
from sharedbuffer.NodeId import NodeId


class SharedBuffer:
    def __init__(self, config: SharedBufferCacheConfig):
        self.events_buffer: Dict[EventId, Lockable] = {}
        self.entries: Dict[NodeId, Lockable] = {}
        self.events_count: Dict[int, int] = defaultdict(int)

        self.events_buffer_cache = LRUCache(maxsize=config.events_buffer_cache_slots)
        self.entry_cache = LRUCache(maxsize=config.entry_cache_slots)

        self.cache_statistics_interval = config.cache_statistics_interval
        self._setup_cache_statistics()

    def _setup_cache_statistics(self):
        from threading import Timer

        def log_cache_statistics():
            logging.info("Statistics details of eventsBufferCache: %s", self.events_buffer_cache)
            logging.info("Statistics details of entryCache: %s", self.entry_cache)
            self._setup_cache_statistics()

        Timer(self.cache_statistics_interval.total_seconds(), log_cache_statistics).start()

    def get_accessor(self):
        # 延迟导入
        from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
        return SharedBufferAccessor(self)

    def advance_time(self, timestamp: int):
        keys_to_remove = [key for key in self.events_count if key < timestamp]
        for key in keys_to_remove:
            del self.events_count[key]

        if not self.events_count:
            self.events_count.clear()

    def register_event(self, value: Any, timestamp: int) -> EventId:
        id = self.events_count[timestamp]
        event_id = EventId(id, timestamp)
        lockable_value = Lockable(value, 1)
        self.events_count[timestamp] += 1
        self.events_buffer_cache[event_id] = lockable_value
        return event_id

    def is_empty(self) -> bool:
        return not self.events_buffer_cache and not self.events_buffer

    def release_cache_statistics_timer(self):
        # This would stop the logging timer; implementation is dependent on the timer mechanism
        pass

    def upsert_event(self, event_id: EventId, event: Lockable):
        self.events_buffer_cache[event_id] = event

    def upsert_entry(self, node_id: NodeId, entry: Lockable):
        self.entry_cache[node_id] = entry

    def remove_event(self, event_id: EventId):
        if event_id in self.events_buffer_cache:
            del self.events_buffer_cache[event_id]
        if event_id in self.events_buffer:
            del self.events_buffer[event_id]

    def remove_entry(self, node_id: NodeId):
        if node_id in self.entry_cache:
            del self.entry_cache[node_id]
        if node_id in self.entries:
            del self.entries[node_id]

    def get_entry(self, node_id: NodeId) -> Optional[Lockable]:
        lockable_from_cache = self.entry_cache.get(node_id)
        if lockable_from_cache is not None:
            return lockable_from_cache
        lockable_from_state = self.entries.get(node_id)
        if lockable_from_state is not None:
            self.entry_cache[node_id] = lockable_from_state
        return lockable_from_state

    def get_event(self, event_id: EventId) -> Optional[Lockable]:
        lockable_from_cache = self.events_buffer_cache.get(event_id)
        if lockable_from_cache is not None:
            return lockable_from_cache
        lockable_from_state = self.events_buffer.get(event_id)
        if lockable_from_state is not None:
            self.events_buffer_cache[event_id] = lockable_from_state
        return lockable_from_state

    def flush_cache(self):
        self.entries.update(self.entry_cache)
        self.entry_cache.clear()
        self.events_buffer.update(self.events_buffer_cache)
        self.events_buffer_cache.clear()


if __name__ == "__main__":
    # 示例使用
    config = SharedBufferCacheConfig.from_config('../config.ini')
    buffer = SharedBuffer(config)