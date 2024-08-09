from dataclasses import dataclass
from functools import total_ordering

@dataclass(frozen=True)
@total_ordering
class EventId:
    id: int
    timestamp: int

    def __eq__(self, other):
        if isinstance(other, EventId):
            return self.id == other.id and self.timestamp == other.timestamp
        return False

    def __lt__(self, other):
        if isinstance(other, EventId):
            if self.timestamp == other.timestamp:
                return self.id < other.id
            return self.timestamp < other.timestamp
        return NotImplemented

    def __hash__(self):
        return hash((self.id, self.timestamp))

    def __str__(self):
        return f"EventId{{id={self.id}, timestamp={self.timestamp}}}"


if __name__ == "__main__":
    # 示例使用
    event1 = EventId(1, 1000)
    event2 = EventId(2, 1000)
    event3 = EventId(1, 2000)

    print(event1)  # 输出: EventId{id=1, timestamp=1000}
    print(event2)  # 输出: EventId{id=2, timestamp=1000}
    print(event3)  # 输出: EventId{id=1, timestamp=2000}

    print(event1 == event2)  # 输出: False
    print(event1 == EventId(1, 1000))  # 输出: True
    print(event1 < event2)  # 输出: True
    print(event3 > event1)  # 输出: True

    # 排序示例
    events = [event1, event2, event3]
    events.sort()
    print(events)  # 输出: [EventId{id=1, timestamp=1000}, EventId{id=2, timestamp=1000}, EventId{id=1, timestamp=2000}]