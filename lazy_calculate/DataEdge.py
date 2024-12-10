from lazy_calculate.DataEdgeType import EdgeType


class Edge:
    _id_counter = 0  # 全局计数器，用于生成唯一的 Edge ID

    def __init__(self, source_block, target_block, label=None):
        self.source_block = source_block
        self.target_block = target_block
        self.label = label
        self.unique_id = self._generate_unique_id()

        # 自动判断边类型
        self.edge_type = self._determine_edge_type()

        # 自动将边添加到对应的 DataBlock 的出边和入边列表
        source_block.add_out_edge(self)
        target_block.add_in_edge(self)

    @classmethod
    def _generate_unique_id(cls):
        cls._id_counter += 1
        return f"Edge_{cls._id_counter}"

    def _determine_edge_type(self):
        """
        根据源块和目标块的 state_name 判断边类型。
        :return: EdgeType
        """
        if self.source_block.state_name == self.target_block.state_name:
            return EdgeType.INNER
        else:
            return EdgeType.EXTERNAL

    def __repr__(self):
        return (f"Edge(unique_id={self.unique_id}, "
                f"source={self.source_block.unique_id}, "
                f"target={self.target_block.unique_id}, "
                f"label={self.label}, "
                f"edge_type={self.edge_type})")