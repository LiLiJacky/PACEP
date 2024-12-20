import itertools
from collections import defaultdict



class TableTool:
    def __init__(self):
        self.shared_var_index = {}

    # def ensure_hashable(self, value):
    #     """
    #     递归将 value 的所有 list, dict, set 等不可哈希类型转换为可哈希类型。
    #     """
    #     if isinstance(value, list):
    #         return tuple(self.ensure_hashable(v) for v in value)
    #     elif isinstance(value, dict):
    #         return tuple(sorted((self.ensure_hashable(k), self.ensure_hashable(v)) for k, v in value.items()))
    #     elif isinstance(value, set):
    #         return tuple(sorted(self.ensure_hashable(v) for v in value))
    #     elif isinstance(value, tuple):
    #         return tuple(self.ensure_hashable(v) for v in value)
    #     elif isinstance(value, (int, float, str)):
    #         return value
    #     else:
    #         try:
    #             hash(value)
    #             return value
    #         except TypeError:
    #             raise TypeError(f"Value of type {type(value)} is not hashable: {value}")

    def create_new_table(self, variables, combinations):
        """
        创建新的组合表，解决嵌套 list 不可哈希问题
        """
        new_table = {}
        for combo in combinations:
            row = {var: value for var, value in zip(variables, combo)}
            hashable_row = {key: self.ensure_hashable(value) for key, value in row.items()}
            new_table[tuple(hashable_row.items())] = row
        return new_table

    def build_shared_var_index(self, combination_table, shared_vars):
        self.shared_var_index.clear()  # 重置索引
        for row_id, row in enumerate(combination_table.values()):
            index = tuple(row[var] for var in shared_vars)
            index_key = self.ensure_hashable(index)
            if index_key in self.shared_var_index.keys():
                self.shared_var_index[index_key].append(row_id)
            else:
                self.shared_var_index[index_key] = [row_id]

    def join_with_table(self, combination_table, variables, valid_combinations, has_combination_state):
        """
        将新的组合与现有表格连接，当 shared_vars 为空时直接进行笛卡尔积
        """
        shared_vars = [ var for var in variables if var in has_combination_state]
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
        new_table = {}

        unsolved_states = [var for var in variables if var not in shared_vars]
        for combo in valid_combinations:
            combo_dict = {var: value for var, value in zip(unsolved_states, combo)}
            index = tuple(combo_dict[var] for var in shared_vars)
            index_key = self.ensure_hashable(index)

            # 利用共享变量索引快速定位需要匹配的行
            if index_key in self.shared_var_index:
                for row_id in self.shared_var_index[index_key]:
                    existing_row = list(combination_table.values())[row_id]
                    merged_row = {**existing_row, **combo_dict}
                    hashable_row = {key: self.ensure_hashable(value) for key, value in merged_row.items()}
                    new_table[tuple(hashable_row.items())] = merged_row

        return new_table

    def generate_combinations(self, variables, simplified_result, has_calculate_variables, selection_strategy, min_times = None):
        """
        生成指定变量的所有可能组合
        :param variables: 要生成组合的变量列表
        :param simplified_result: 轻量化的结果数据
        :return: 各变量的所有可能组合
        """
        return [
            [
                # 对于 NONDETERMINISTICRELAXED，生成所有的索引组合
                [data[idx] for idx in indices]
                for indices in itertools.chain.from_iterable(
                itertools.combinations(range(len(data)), r)
                for r in range(min_times[var] if var in min_times else 1, len(data) + 1)
            )
            ]
            if var in selection_strategy['NONDETERMINISTICRELAXED']
            else
            [
                # 对于 RELAXED，生成所有连续子序列的索引组合
                [data[idx] for idx in range(i, i + r)]
                for r in range(min_times[var] if var in min_times else 1, len(data) + 1)
                for i in range(len(data) - r + 1)
            ]
            for var, data in simplified_result.items()
            if var in variables and var not in has_calculate_variables
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
            # 如果是 EventWrapper 对象列表，利用每个 EventWrapper 的 timestamp 生成一个哈希
            if all(isinstance(v, EventWrapper) for v in value):
                # 使用位移合并所有时间戳
                hash_key = 0
                shift_factor = 31  # 可调整的位移因子
                for wrapper in value:
                    timestamp = wrapper.event.timestamp
                    hash_key = (hash_key << shift_factor) + timestamp  # 左移并加上当前时间戳
                return str(hash_key)  # 确保返回的是字符串类型
            else:
                # 对普通列表递归确保每个元素可哈希
                return str(tuple(self.ensure_hashable(v) for v in value))  # 将列表转换为元组并确保每个元素哈希后转为字符串
        elif isinstance(value, dict):
            # 对字典递归确保每个键和值可哈希
            return str(tuple(sorted((self.ensure_hashable(k), self.ensure_hashable(v)) for k, v in value.items())))
        elif isinstance(value, set):
            # 对集合递归确保每个元素可哈希
            return str(tuple(sorted(self.ensure_hashable(v) for v in value)))
        elif isinstance(value, tuple):
            # 对元组递归确保每个元素可哈希
            return str(tuple(self.ensure_hashable(v) for v in value))
        elif isinstance(value, EventWrapper):
            # 对单个 EventWrapper 对象，根据 timestamp 哈希
            return str(self.ensure_hashable(value.event.timestamp))
        elif isinstance(value, (int, float, str)):
            return str(value)  # 确保返回字符串
        else:
            try:
                hash(value)
                return str(value)  # 如果能够哈希，确保返回字符串
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
            self.ensure_hashable([event.event.timestamp, float(event.event.value)]): event
            for key, events in original_result.items()
            for event in events
        }

        for row in combination_table.values():
            restored_combination = defaultdict(list)
            for var, value in row.items():
                for single_value in value:
                    hashable_value = self.ensure_hashable([single_value[0], float(single_value[1])])
                    event_wrapper = reverse_mapping.get(hashable_value)
                    if event_wrapper is not None:
                        restored_combination[var].append(event_wrapper)
                    else:
                        print(f"Warning: Could not find EventWrapper for value {hashable_value}")
            restored_results.append([restored_combination, []])

        return restored_results

    def simple_result_to_event_wrapper(self, raw_data, simple_result):
        result = []
        # 构建关于元数据的时间戳和事件的映射
        for time_key in simple_result:

            result.append(raw_data)