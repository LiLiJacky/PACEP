import configparser
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class SharedBufferCacheConfig:
    events_buffer_cache_slots: int
    entry_cache_slots: int
    cache_statistics_interval: timedelta

    @classmethod
    def from_config(cls, config_path: str) -> 'SharedBufferCacheConfig':
        config = configparser.ConfigParser()
        config.read(config_path)

        events_buffer_cache_slots = config.getint('SharedBufferCacheConfig', 'events_buffer_cache_slots')
        entry_cache_slots = config.getint('SharedBufferCacheConfig', 'entry_cache_slots')
        cache_statistics_interval = timedelta(seconds=config.getint('SharedBufferCacheConfig', 'cache_statistics_interval'))

        return cls(events_buffer_cache_slots, entry_cache_slots, cache_statistics_interval)

    def get_events_buffer_cache_slots(self) -> int:
        return self.events_buffer_cache_slots

    def get_entry_cache_slots(self) -> int:
        return self.entry_cache_slots

    def get_cache_statistics_interval(self) -> timedelta:
        return self.cache_statistics_interval


if __name__ == "__main__":
    # 示例使用
    config = SharedBufferCacheConfig.from_config('../config.ini')
    print(f"Events Buffer Cache Slots: {config.get_events_buffer_cache_slots()}")
    print(f"Entry Cache Slots: {config.get_entry_cache_slots()}")
    print(f"Cache Statistics Interval: {config.get_cache_statistics_interval()}")