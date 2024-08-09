from dataclasses import dataclass, field
from typing import List

from sharedbuffer.Lockable import Lockable
from sharedbuffer.NodeId import NodeId
from sharedbuffer.SharedBufferEdge import SharedBufferEdge
from util.DeweyNumber import DeweyNumber


@dataclass
class SharedBufferNode:
    edges: List['Lockable[SharedBufferEdge]'] = field(default_factory=list)

    def get_edges(self) -> List['Lockable[SharedBufferEdge]']:
        return self.edges

    def add_edge(self, edge: 'SharedBufferEdge'):
        self.edges.append(Lockable(edge, 0))

    def __str__(self) -> str:
        return f"SharedBufferNode{{edges={self.edges}}}"

    def __eq__(self, other) -> bool:
        if isinstance(other, SharedBufferNode):
            return self.edges == other.edges
        return False

    def __hash__(self) -> int:
        return hash(tuple(self.edges))

if __name__ == "__main__":
    # 示例使用
    # 假设 Lockable 和 SharedBufferEdge 已经定义
    node = SharedBufferNode()
    edge = SharedBufferEdge(target=NodeId("node1", "example1"), dewey_number=DeweyNumber([1, 2, 3]))
    node.add_edge(edge)

    print(node.get_edges())  # 输出: [Lockable(element=SharedBufferEdge(target=NodeId(id='node1'), dewey_number=DeweyNumber(dewey_number=[1, 2, 3])), ref_counter=0)]
    print(node)  # 输出: SharedBufferNode{edges=[Lockable(element=SharedBufferEdge(target=NodeId(id='node1'), dewey_number=DeweyNumber(dewey_number=[1, 2, 3])), ref_counter=0)]}

    # 检查相等性
    node2 = SharedBufferNode([Lockable(edge, 0)])
    print(node == node2)  # 输出: True

    # 哈希值
    print(hash(node))  # 输出: 哈希值
    print(hash(node2))  # 输出: 相同的哈希值