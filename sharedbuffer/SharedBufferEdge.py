from dataclasses import dataclass

from sharedbuffer.NodeId import NodeId
from util.DeweyNumber import DeweyNumber


@dataclass(frozen=True)
class SharedBufferEdge:
    target: 'NodeId'
    dewey_number: 'DeweyNumber'

    def get_target(self):
        return self.target

    def get_dewey_number(self):
        return self.dewey_number

    def __str__(self):
        return f"SharedBufferEdge{{target={self.target}, dewey_number={self.dewey_number}}}"

    def __eq__(self, other):
        if isinstance(other, SharedBufferEdge):
            return self.target == other.target and self.dewey_number == other.dewey_number
        return False

    def __hash__(self):
        return hash((self.target, self.dewey_number))


if __name__ == "__main__":
    # 示例使用
    node = NodeId("node1", "example1")
    dewey_number = DeweyNumber([1, 2, 3])
    edge = SharedBufferEdge(target=node, dewey_number=dewey_number)

    print(edge.get_target())  # 输出: NodeId(id='node1')
    print(edge.get_dewey_number())  # 输出: DeweyNumber(dewey_number=[1, 2, 3])
    print(edge)  # 输出: SharedBufferEdge{target=NodeId(id='node1'), dewey_number=DeweyNumber(dewey_number=[1, 2, 3])}

    # 检查相等性
    edge2 = SharedBufferEdge(target=node, dewey_number=DeweyNumber([1, 2, 3]))
    print(edge == edge2)  # 输出: True

    # 哈希值
    print(hash(edge))  # 输出: 哈希值
    print(hash(edge2))  # 输出: 相同的哈希值