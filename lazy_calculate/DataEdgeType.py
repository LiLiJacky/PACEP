from enum import Enum


class EdgeType(Enum):
    """
    表示 DataBlock 之间连接的边类型。
    """
    INNER = "Inner"       # 同一个 state_name 中的连接
    EXTERNAL = "External" # 不同 state_name 之间的连接