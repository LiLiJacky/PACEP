from lazy_calculate.DataEdge import Edge


class DataBlock:
    _id_counter = {}  # 用于记录每个 state_name 的子增 ID

    def __init__(self, state_name, content, event_list=None, in_edge_list=None, out_edge_list=None, lazy_calculate_algorithm=None):
        """
        初始化 DataBlock 类实例
        :param state_name: 状态名称
        :param content: 内容（如列表片段）
        :param event_list: 事件列表（默认为空列表）
        :param in_edge_list: 入边列表（默认为空列表）
        :param out_edge_list: 出边列表（默认为空列表）
        """
        self.state_name = state_name
        self.content = content  # Block 的主要内容
        self.event_list = event_list if event_list is not None else []
        self.in_edge_list = in_edge_list if in_edge_list is not None else []
        self.out_edge_list = out_edge_list if out_edge_list is not None else []
        self.unique_id = self._generate_unique_id(state_name)
        self.lazy_calculate_algorithm = lazy_calculate_algorithm if lazy_calculate_algorithm is not None else []

    @classmethod
    def _generate_unique_id(cls, state_name):
        if state_name not in cls._id_counter:
            cls._id_counter[state_name] = 0
        cls._id_counter[state_name] += 1
        return f"{state_name}_{cls._id_counter[state_name]}"

    def add_event(self, event):
        self.event_list.append(event)

    def add_in_edge(self, edge):
        """添加入边"""
        self.in_edge_list.append(edge)

    def add_out_edge(self, edge):
        """添加出边"""
        self.out_edge_list.append(edge)

    def split(self, split_point):
        """
        按指定分割点将当前块分为两部分：
        左侧部分保留入边，右侧部分保留出边
        :param split_point: 分割点（列表索引）
        :return: (左块, 右块)
        """
        left_content = self.content[:split_point]
        right_content = self.content[split_point:]

        # 创建左块和右块
        left_block = DataBlock(self.state_name, left_content, in_edge_list=self.in_edge_list)
        right_block = DataBlock(self.state_name, right_content, out_edge_list=self.out_edge_list)

        # 添加相互连接的边
        Edge(left_block, right_block, label="SplitEdge")

        return left_block, right_block

    def __repr__(self):
        return (f"DataBlock(unique_id={self.unique_id}, "
                f"state_name={self.state_name}, "
                f"content={self.content}, "
                f"in_edge_list={[edge.unique_id for edge in self.in_edge_list]}, "
                f"out_edge_list={[edge.unique_id for edge in self.out_edge_list]})")

