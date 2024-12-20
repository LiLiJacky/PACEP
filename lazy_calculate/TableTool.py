import itertools
from collections import defaultdict


class TableTool:
    def __init__(self):
        self.shared_var_index = defaultdict(lambda: defaultdict(list))  # 倒排索引

    def ensure_hashable(self, value):
        """
        递归将 value 的所有 list, dict, set 等不可哈希类型转换为可哈希类型。
        """
        if isinstance(value, list):
            return tuple(self.ensure_hashable(v) for v in value)
        elif isinstance(value, dict):
            return tuple(sorted((self.ensure_hashable(k), self.ensure_hashable(v)) for k, v in value.items()))
        elif isinstance(value, set):
            return tuple(sorted(self.ensure_hashable(v) for v in value))
        elif isinstance(value, tuple):
            return tuple(self.ensure_hashable(v) for v in value)
        elif isinstance(value, (int, float, str)):
            return value
        else:
            try:
                hash(value)
                return value
            except TypeError:
                raise TypeError(f"Value of type {type(value)} is not hashable: {value}")

    def create_new_table(self, variables, combinations):
        """
        创建新的组合表，解决嵌套 list 不可哈希问题
        """
        new_table = defaultdict(list)
        for combo in combinations:
            row = {var: value for var, value in zip(variables, combo)}
            hashable_row = {key: self.ensure_hashable(value) for key, value in row.items()}
            new_table[tuple(hashable_row.items())] = row
        return new_table

    def build_shared_var_index(self, combination_table, shared_vars):
        """
        为组合表构建共享变量的倒排索引
        """
        self.shared_var_index.clear()  # 重置索引
        for row_id, row in enumerate(combination_table.values()):
            index_key = tuple(row[var] for var in shared_vars)
            self.shared_var_index[index_key].append(row_id)

    def join_with_table(self, combination_table, variables, valid_combinations, shared_vars):
        """
        将新的组合与现有表格连接，当 shared_vars 为空时直接进行笛卡尔积
        """
        if not shared_vars:
            # 无共享变量，直接进行笛卡尔积
            new_table = defaultdict(list)
            for existing_row in combination_table.values():
                for combo in valid_combinations:
                    combo_dict = {var: value for var, value in zip(variables, combo)}
                    merged_row = {**existing_row, **combo_dict}
                    hashable_row = {key: self.ensure_hashable(value) for key, value in merged_row.items()}
                    new_table[tuple(hashable_row.items())] = merged_row
            return new_table

        # 有共享变量，利用索引加速连接
        self.build_shared_var_index(combination_table, shared_vars)
        new_table = defaultdict(list)

        for combo in valid_combinations:
            combo_dict = {var: value for var, value in zip(variables, combo)}
            index_key = tuple(combo_dict[var] for var in shared_vars)

            # 利用共享变量索引快速定位需要匹配的行
            if index_key in self.shared_var_index:
                for row_id in self.shared_var_index[index_key]:
                    existing_row = list(combination_table.values())[row_id]
                    merged_row = {**existing_row, **combo_dict}
                    hashable_row = {key: self.ensure_hashable(value) for key, value in merged_row.items()}
                    new_table[tuple(hashable_row.items())] = merged_row

        return new_table

    def generate_combinations(self, variables, simplified_result):
        """
        生成指定变量的所有可能组合
        :param variables: 要生成组合的变量列表
        :param simplified_result: 轻量化的结果数据
        :return: 各变量的所有可能组合
        """
        return [
            [[data[idx] for idx in indices]
             for indices in itertools.chain.from_iterable(
                itertools.combinations(range(len(data)), r) for r in range(1, len(data) + 1)
            )]
            for var, data in simplified_result.items() if var in variables
        ]

    def ensure_hashable(self, value):
        """
        递归将复杂数据结构转换为可哈希的形式
        :param value: 待处理的值
        :return: 可哈希形式的值
        """
        # 延迟导入 EventWrapper
        from nfa.NFA import EventWrapper

        if isinstance(value, list):
            # 如果是EventWrapper对象列表，利用每个EventWrapper的timestamp生成一个哈希
            if all(isinstance(v, EventWrapper) for v in value):
                # 使用位移合并所有时间戳
                hash_key = 0
                shift_factor = 31  # 可调整的位移因子
                for wrapper in value:
                    timestamp = wrapper.event.timestamp
                    hash_key = (hash_key << shift_factor) + timestamp  # 左移并加上当前时间戳
                return hash_key
            else:
                # 对普通列表递归确保每个元素可哈希
                return tuple(self.ensure_hashable(v) for v in value)
        elif isinstance(value, dict):
            return tuple(sorted((self.ensure_hashable(k), self.ensure_hashable(v)) for k, v in value.items()))
        elif isinstance(value, set):
            return tuple(sorted(self.ensure_hashable(v) for v in value))
        elif isinstance(value, tuple):
            return tuple(self.ensure_hashable(v) for v in value)
        elif isinstance(value, EventWrapper):
            # 对单个 EventWrapper 对象，根据 timestamp 哈希
            return self.ensure_hashable(value.event.timestamp)
        elif isinstance(value, (int, float, str)):
            return value
        else:
            try:
                hash(value)
                return value
            except TypeError:
                raise TypeError(f"Value of type {type(value)} is not hashable: {value}")

    def simplify_basic_result(self, basic_result):
        """
        将 basic_result 转换为轻量化数据结构
        """
        return {
            key: [[event.event.timestamp, event.event.value] for event in events]
            for key, events in basic_result.items()
        }

    def restore_to_event_wrapper(self, combination_table, original_result):
        """
        将组合表恢复为嵌套的列表结构，包含 defaultdict 格式
        """
        restored_results = []

        reverse_mapping = {
            tuple(self.ensure_hashable([event.event.timestamp, event.event.value])): event
            for key, events in original_result.items()
            for event in events
        }

        for row in combination_table.values():
            restored_combination = defaultdict(list)
            for var, value in row.items():
                for single_value in value:
                    hashable_value = tuple(self.ensure_hashable(single_value))
                    event_wrapper = reverse_mapping.get(hashable_value)
                    if event_wrapper is not None:
                        restored_combination[var].append(event_wrapper)
                    else:
                        print(f"Warning: Could not find EventWrapper for value {hashable_value}")
            restored_results.append([restored_combination, []])

        return restored_results