import itertools
from collections import defaultdict
from math import comb
from typing import List

from factories.AlgorithmFactory import AlgorithmFactory
from lazy_calculate.TableTool import TableTool
from models.ValueConstraint import ValueConstraint
from shared_calculate.BloomFilterManager import BloomFilterManager


class TableCalculate:
    def __init__(self, cost_threshold=100, event_threshold=20):
        self.AlgorithmFactory = AlgorithmFactory()
        self.cost_threshold = cost_threshold
        self.event_threshold = max(event_threshold, 20)
        self.table_tool = TableTool()

    def evaluate_combinations(self, basic_result, latency_constrain: List[ValueConstraint]):
        """
        根据 basic_result 和 latency_constrain 评估事件组合，并添加未参与约束的状态
        """
        data_counts = {key: len(value) for key, value in basic_result.items()}

        sorted_constraints = sorted(
            latency_constrain,
            key=lambda constrain: self.AlgorithmFactory.calculate_constrain_cost(constrain, data_counts)
        )

        simplified_result = self.table_tool.simplify_basic_result(basic_result)
        combination_table = defaultdict(list)

        for constrain in sorted_constraints:
            variables = constrain.variables
            current_combinations = self.table_tool.generate_combinations(variables, simplified_result)

            valid_combinations = []
            for combo in itertools.product(*current_combinations):
                combo_dict = dict(zip(variables, combo))
                if constrain.validate(combo_dict):
                    valid_combinations.append(combo)

            if not valid_combinations:
                return []

            if not combination_table:
                combination_table = self.table_tool.create_new_table(variables, valid_combinations)
            else:
                shared_vars = [var for var in variables if var in combination_table]
                combination_table = self.table_tool.join_with_table(
                    combination_table, variables, valid_combinations, shared_vars
                )

        final_results = self.table_tool.restore_to_event_wrapper(combination_table, basic_result)
        final_results = self.add_missing_states(final_results, basic_result, latency_constrain)

        return final_results

    # def evaluate_combinations(self, basic_result, latency_constrain: List[ValueConstraint]):
    #     """
    #     根据 basic_result 和 latency_constrain 评估事件组合，并添加未参与约束的状态
    #     """
    #     data_counts = {key: len(value) for key, value in basic_result.items()}
    #
    #     # 根据约束代价排序
    #     sorted_constraints = sorted(
    #         latency_constrain,
    #         key=lambda constrain: self.AlgorithmFactory.calculate_constrain_cost(constrain, data_counts)
    #     )
    #
    #     # 转换 basic_result 为轻量化数据结构
    #     simplified_result = self.simplify_basic_result(basic_result)
    #
    #     # 初始化组合表
    #     combination_table = defaultdict(list)
    #
    #     # 遍历约束
    #     for constrain in sorted_constraints:
    #         variables = constrain.variables
    #
    #         # 生成当前约束的组合
    #         current_combinations = self.generate_combinations(variables, simplified_result)
    #
    #         # 验证约束，保留满足条件的组合
    #         valid_combinations = []
    #         for combo in itertools.product(*current_combinations):
    #             combo_dict = dict(zip(variables, combo))
    #
    #             # bloom过滤器在简单问题上，性能不如直接计算
    #             # 统计组合中事件的总数量
    #             # total_events = sum(len(events) for events in combo_dict.values())
    #             # if total_events > self.event_threshold:
    #                 # 如果成本超过阈值，使用 Bloom Filter
    #                 # expression = constrain.expression
    #                 # hashable_combo = self.ensure_hashable(combo)
    #                 #
    #                 # if bloom_manager.is_calculated(expression, hashable_combo):
    #                 #     # 如果已计算过，直接获取结果
    #                 #     result = bloom_manager.get_result(expression, hashable_combo)
    #                 #     if result:
    #                 #         valid_combinations.append(combo)
    #                 # else:
    #                 #     # 如果未计算过，实际验证
    #                 #     result = constrain.validate(combo_dict)
    #                 #     bloom_manager.add_calculated(expression, hashable_combo)
    #                 #     bloom_manager.add_result(expression, hashable_combo, result)
    #                 #     if result:
    #                 #         valid_combinations.append(combo)
    #
    #                 # 如果成本较低，直接计算
    #             result = constrain.validate(combo_dict)
    #             if result:
    #                 valid_combinations.append(combo)
    #
    #         # 如果没有满足条件的组合，直接返回空列表
    #         if not valid_combinations:
    #             return []
    #
    #         # 如果组合表为空，直接初始化
    #         if not combination_table:
    #             combination_table = self.create_new_table(variables, valid_combinations)
    #         else:
    #             # 根据共享变量连接组合表
    #             shared_vars = [var for var in variables if var in combination_table]
    #             combination_table = self.join_with_table(
    #                 combination_table, variables, valid_combinations, shared_vars
    #             )
    #
    #     # 构建最终结果并添加未参与约束的状态
    #     final_results = self.restore_to_event_wrapper(combination_table, basic_result)
    #     final_results = self.add_missing_states(final_results, basic_result, latency_constrain)
    #
    #     return final_results

    def add_missing_states(self, final_results, basic_result, latency_constrain):
        """
        为每个结果添加未参与约束的状态
        """
        constrained_states = {var for constrain in latency_constrain for var in constrain.variables}
        missing_states = set(basic_result.keys()) - constrained_states

        for result in final_results:
            combination = result[0]  # 获取当前组合的 defaultdict
            for state in missing_states:
                if state not in combination:
                    combination[state] = basic_result[state]

        return final_results

