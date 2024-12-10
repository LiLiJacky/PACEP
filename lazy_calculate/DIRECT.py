import itertools
from collections import defaultdict
from math import comb
from typing import List
import re

from factories.AlgorithmFactory import AlgorithmFactory
from models.ValueConstraint import ValueConstraint


class DIRECT:
    def __init__(self):
        self.AlgorithmFactory = AlgorithmFactory()

    def calculate_combination_cost(self, algorithm_name, n):
        """
        计算从 C(n, 1) 到 C(n, n) 的总复杂度
        """
        total_cost = 0
        for k in range(1, n + 1):
            combination_count = comb(n, k)
            complexity = self.AlgorithmFactory.get_algorithm_calculate_time_complexity(algorithm_name, k)
            total_cost += combination_count * complexity
        return total_cost

    def calculate_constrain_cost(self, latency_constrain: ValueConstraint, data_counts):
        """
        计算单个延迟约束的代价，支持多个匹配项
        """
        total_cost = 0
        for variable in latency_constrain.variables:
            matches = re.findall(rf'(\w+)\({variable}\)', latency_constrain.expression)
            if matches:
                for match in matches:
                    algorithm_name = match
                    n = data_counts.get(variable, 0)
                    total_cost += self.calculate_combination_cost(algorithm_name, n)
            else:
                total_cost += 1
        return total_cost

    def evaluate_combinations(self, basic_result, latency_constrain: List[ValueConstraint]):
        """
        根据 basic_result 和 latency_constrain 评估事件组合，并添加未参与约束的状态
        """
        data_counts = {key: len(value) for key, value in basic_result.items()}

        # 根据约束代价排序
        sorted_constraints = sorted(
            latency_constrain,
            key=lambda constrain: self.calculate_constrain_cost(constrain, data_counts)
        )

        # 转换 basic_result 为轻量化数据结构
        simplified_result = self.simplify_basic_result(basic_result)

        # 初始化组合表
        combination_table = defaultdict(list)

        # 遍历约束
        for constrain in sorted_constraints:
            variables = constrain.variables

            # 生成当前约束的组合
            current_combinations = self.generate_combinations(variables, simplified_result)

            # 验证约束，保留满足条件的组合
            valid_combinations = [
                combo for combo in itertools.product(*current_combinations)
                if constrain.validate(dict(zip(variables, combo)))
            ]

            # 如果没有满足条件的组合，直接返回空列表
            if not valid_combinations:
                return []

            # 如果组合表为空，直接初始化
            if not combination_table:
                combination_table = self.create_new_table(variables, valid_combinations)
            else:
                # 根据共享变量连接组合表
                shared_vars = [var for var in variables if var in combination_table]
                combination_table = self.join_with_table(
                    combination_table, variables, valid_combinations, shared_vars
                )

        # 构建最终结果并添加未参与约束的状态
        final_results = self.restore_to_event_wrapper(combination_table, basic_result)
        final_results = self.add_missing_states(final_results, basic_result, latency_constrain)

        return final_results

    def add_missing_states(self, final_results, basic_result, latency_constrain):
        """
        为每个结果添加未参与约束的状态
        """
        # 获取所有未参与约束的状态
        constrained_states = {var for constrain in latency_constrain for var in constrain.variables}
        missing_states = set(basic_result.keys()) - constrained_states

        for result in final_results:
            combination = result[0]  # 获取当前组合的 defaultdict
            for state in missing_states:
                # 如果该状态未在当前组合中，直接从 basic_result 中添加
                if state not in combination:
                    combination[state] = basic_result[state]

        return final_results

    def simplify_basic_result(self, basic_result):
        """
        将 basic_result 转换为轻量化数据结构
        """
        return {
            key: [[event.event.timestamp, event.event.value] for event in events]
            for key, events in basic_result.items()
        }

    def generate_combinations(self, variables, simplified_result):
        """
        生成 simplified_result 中指定变量的所有可能组合
        """
        return [
            [[data[idx] for idx in indices]
             for indices in itertools.chain.from_iterable(
                 itertools.combinations(range(len(data)), r) for r in range(1, len(data) + 1)
             )]
            for var, data in simplified_result.items() if var in variables
        ]

    def create_new_table(self, variables, combinations):
        """
        创建新的组合表，解决嵌套 list 不可哈希问题
        """
        new_table = defaultdict(list)
        for combo in combinations:
            # 将 row 的所有值递归转换为 hashable 类型
            row = {var: value for var, value in zip(variables, combo)}
            hashable_row = {key: self.ensure_hashable(value) for key, value in row.items()}
            new_table[tuple(hashable_row.items())] = row
        return new_table

    def ensure_hashable(self, value):
        """
        递归将 value 的所有 list 转换为 tuple
        """
        if isinstance(value, list):
            # 如果是 list，递归地将其每个元素转换为 tuple
            return tuple(self.ensure_hashable(v) for v in value)
        elif isinstance(value, dict):
            # 如果是 dict，将其键和值都递归转换为 tuple
            return tuple((self.ensure_hashable(k), self.ensure_hashable(v)) for k, v in value.items())
        return value

    def join_with_table(self, combination_table, variables, valid_combinations, shared_vars):
        """
        将新的组合与现有表格连接，当 shared_vars 为空时直接进行笛卡尔积
        """
        if not shared_vars:
            # 如果 shared_vars 为空，直接进行笛卡尔积
            new_table = defaultdict(list)
            for existing_row in combination_table.values():
                for combo in valid_combinations:
                    combo_dict = {var: value for var, value in zip(variables, combo)}
                    merged_row = {**existing_row, **combo_dict}
                    # 确保所有值可哈希
                    hashable_row = {key: self.ensure_hashable(value) for key, value in merged_row.items()}
                    new_table[tuple(hashable_row.items())] = merged_row
            return new_table

        # 否则，基于共享变量构建索引并连接表格
        shared_index = defaultdict(list)
        for key, row in combination_table.items():
            # 构建共享变量索引
            index_key = tuple(self.ensure_hashable(row[var]) for var in shared_vars)
            shared_index[index_key].append(row)

        new_table = defaultdict(list)
        for combo in valid_combinations:
            combo_dict = {var: value for var, value in zip(variables, combo)}
            index_key = tuple(self.ensure_hashable(combo_dict[var]) for var in shared_vars)
            if index_key in shared_index:
                for existing_row in shared_index[index_key]:
                    merged_row = {**existing_row, **combo_dict}
                    # 确保所有值可哈希
                    hashable_row = {key: self.ensure_hashable(value) for key, value in merged_row.items()}
                    new_table[tuple(hashable_row.items())] = merged_row
        return new_table

    def restore_to_event_wrapper(self, combination_table, original_result):
        """
        将组合表恢复为嵌套的列表结构，包含 defaultdict 格式
        """
        restored_results = []

        # 构建反向映射，确保键值格式一致
        reverse_mapping = {
            tuple(self.ensure_hashable([event.event.timestamp, event.event.value])): event
            for key, events in original_result.items()
            for event in events
        }

        for row in combination_table.values():
            restored_combination = defaultdict(list)

            # 遍历组合中的变量和值
            for var, value in row.items():
                # 遍历 value 数组中的每个元素
                for single_value in value:
                    # 确保每个值可哈希
                    hashable_value = tuple(self.ensure_hashable(single_value))
                    event_wrapper = reverse_mapping.get(hashable_value)

                    if event_wrapper is not None:
                        # 将找到的 EventWrapper 添加到对应变量列表
                        restored_combination[var].append(event_wrapper)
                    else:
                        # 如果未找到，记录警告
                        print(f"Warning: Could not find EventWrapper for value {hashable_value}")

            # 将组合和空列表嵌套，添加到最终结果中
            restored_results.append([restored_combination, []])

        return restored_results