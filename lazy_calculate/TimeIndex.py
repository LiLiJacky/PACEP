from rtree import index

class TimeIndex:
    def __init__(self):
        # 每个 variable 对应一个 R‑tree 索引，用于高效的二维范围查询
        self.indexes = {}
        # 每个 variable 对应一个字典，记录每个时间区间对应的 (entry_id, keys_set)
        self.entries = {}
        # 每个 variable 对应一个 id 到时间区间的映射，方便查询时根据 id 找到具体区间
        self.id_to_time_range = {}
        # 全局自增 id，用于 R‑tree 内部的唯一标识
        self.counter = 0

    def add_to_time_index(self, variable, start_time, end_time, key):
        """
        将一个 <start_time, end_time, key> 插入到指定 variable 对应的时间索引中。
        如果同一时间区间已经存在，则将 key 添加到该区间对应的集合中。
        """
        if variable not in self.indexes:
            # 初始化 R‑tree 以及对应的辅助数据结构
            p = index.Property()
            self.indexes[variable] = index.Index(properties=p)
            self.entries[variable] = {}         # { (start_time, end_time): (entry_id, {key1, key2, ...}) }
            self.id_to_time_range[variable] = {}  # { entry_id: (start_time, end_time) }

        time_range = (start_time, end_time)
        if time_range in self.entries[variable]:
            # 如果时间区间已存在，直接更新 key 集合
            entry_id, keys_set = self.entries[variable][time_range]
            keys_set.add(key)
        else:
            # 分配一个全局唯一的 id，并插入 R‑tree
            entry_id = self.counter
            self.counter += 1
            keys_set = {key}
            self.entries[variable][time_range] = (entry_id, keys_set)
            self.id_to_time_range[variable][entry_id] = time_range
            # 在 R‑tree 中将时间区间以一个点插入
            self.indexes[variable].insert(entry_id, (start_time, end_time, start_time, end_time))

    def query_time_range(self, variable, query_start, query_end):
        """
        查询指定 variable 中所有时间范围完全位于 [query_start, query_end] 内的 key。
        返回所有满足：start_time >= query_start 且 end_time <= query_end 的记录中的 key。
        """
        if variable not in self.indexes:
            return []

        result_keys = []
        # 构造查询矩形：二维范围为 x, y 均在 [query_start, query_end]
        # 由于我们将每条记录作为一个点插入，所以只会返回完全满足条件的记录
        candidate_ids = list(self.indexes[variable].intersection((query_start, query_start, query_end, query_end)))
        for entry_id in candidate_ids:
            # 根据 id 找到对应的时间区间
            time_range = self.id_to_time_range[variable][entry_id]
            # 这里理论上 candidate_ids 返回的点均已满足条件，但再做一次校验也无妨
            if time_range[0] >= query_start and time_range[1] <= query_end:
                keys = self.entries[variable][time_range][1]
                result_keys.extend(keys)
        return result_keys


# -------------------------------
# 测试示例
if __name__ == "__main__":
    ti = TimeIndex()

    # 模拟插入一些数据，注意相同时间区间可对应多个 key
    ti.add_to_time_index('var1', 10, 20, 'key_A')
    ti.add_to_time_index('var1', 10, 20, 'key_A2')  # 与 key_A 同一时间区间
    ti.add_to_time_index('var1', 15, 25, 'key_B')
    ti.add_to_time_index('var1', 12, 18, 'key_C')
    ti.add_to_time_index('var1', 30, 40, 'key_D')

    # 查询所有完全在 [11, 22] 内的记录
    # 分析：
    #   - 时间区间 [10,20] 不满足要求，因为 10 < 11 （即 key_A 和 key_A2 均不满足）
    #   - 时间区间 [15,25] 不满足，因为 25 > 22 （即 key_B 不满足）
    #   - 时间区间 [12,18] 符合要求（即 key_C 满足）
    #   - [30,40] 明显不满足
    result = ti.query_time_range('var1', 11, 22)
    print("查询结果:", result)  # 输出：查询结果: ['key_C']