from dataclasses import dataclass

from sharedbuffer.EventId import EventId


@dataclass(frozen=True)
class NodeId:
    event_id: EventId
    page_name: str

    def __eq__(self, other):
        if isinstance(other, NodeId):
            return self.event_id == other.event_id and self.page_name == other.page_name
        return False

    def __hash__(self):
        return hash((self.event_id, self.page_name))

    def __str__(self):
        return f"NodeId{{event_id={self.event_id}, page_name='{self.page_name}'}}"

if __name__ == "__main__":
    # 示例使用
    event1 = EventId(1, 1000)
    node1 = NodeId(event1, "Page1")
    node2 = NodeId(event1, "Page1")
    node3 = NodeId(EventId(2, 1000), "Page2")

    print(node1)  # 输出: NodeId{event_id=EventId{id=1, timestamp=1000}, page_name='Page1'}
    print(node2)  # 输出: NodeId{event_id=EventId{id=1, timestamp=1000}, page_name='Page1'}
    print(node3)  # 输出: NodeId{event_id=EventId{id=2, timestamp=1000}, page_name='Page2'}

    print(node1 == node2)  # 输出: True
    print(node1 == node3)  # 输出: False
    print(hash(node1))  # 输出: 哈希值
    print(hash(node2))  # 输出: 相同的哈希值
    print(hash(node3))  # 输出: 不同的哈希值