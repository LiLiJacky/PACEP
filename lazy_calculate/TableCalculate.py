import itertools
from collections import defaultdict
from math import comb
from typing import List

from factories.AlgorithmFactory import AlgorithmFactory
from lazy_calculate.TableTool import TableTool
from models.ValueConstraint import ValueConstraint
from shared_calculate.BloomFilterManager import BloomFilterManager


class TableCalculate:
    def __init__(self, cost_threshold=100, event_threshold=20, selection_strategy = None, min_times = None):
        self.algorithm_factory = AlgorithmFactory()
        self.cost_threshold = cost_threshold
        self.event_threshold = max(event_threshold, 20)
        self.table_tool = TableTool()
        self.selection_strategy = [] if selection_strategy is None else selection_strategy
        self.min_times = min_times

    def evaluate_combinations(self, basic_result, latency_constrain: List[ValueConstraint]):
        """
        根据 basic_result 和 latency_constrain 评估事件组合，并添加未参与约束的状态
        """
        data_counts = {key: len(value) for key, value in basic_result.items()}

        sorted_constraints = sorted(
            latency_constrain,
            key=lambda constrain: self.algorithm_factory.calculate_constrain_cost(constrain, data_counts)
        )

        simplified_result = self.table_tool.simplify_basic_result(basic_result)
        combination_table = {}

        has_combination_state = []
        for constrain in sorted_constraints:
            variables = constrain.variables
            valid_combinations = []

            # 区分是否为可增量计算约束
            algorithm_name = self.algorithm_factory.get_algorithm_name(variables[0], constrain.expression)
            if len(variables) == 1  and self.algorithm_factory.is_dp_algorithm(algorithm_name[0][0]):
                alg = self.algorithm_factory.get_algorithm(algorithm_name[0][0], simplified_result[variables[0]])
                valid_combinations = alg.get_calculate(self.min_times[variables[0]])
                if not valid_combinations:
                    return []

                if not combination_table:
                    combination_table = self.table_tool.create_new_table(variables, valid_combinations)
                else:
                    combination_table = self.table_tool.join_with_table(
                        combination_table, variables, valid_combinations, has_combination_state
                    )
            else:
                new_table = {}
                cannot_key = []
                can_key = []
                # 当前约束涉及变量组合都已经生成时，只需要对当前 combination_table 进行过滤
                if combination_table and set(variables).issubset(set(has_combination_state)):
                    for combo_key, combo_dict in combination_table.items():
                        # 过滤出涉及所有变量的组合
                        combination = {var: combinations for var, combinations in combo_dict.items() if
                                       var in variables}
                        combination_key = self.table_tool.ensure_hashable(combination)
                        if combination_key in can_key:
                            new_table[combo_key] = combo_dict
                        elif combination_key in cannot_key:
                            continue
                        else:
                            if constrain.validate(combination):
                                new_table[combo_key] = combo_dict
                                can_key.append(combination_key)
                            else:
                                cannot_key.append(combination_key)
                    combination_table = new_table
                # 当已求解状体和当前约束状态有交集时，生成已求解部分的组合
                elif combination_table and (set(has_combination_state) & set(variables)):
                    # 记录每个重叠部分和原匹配结果的组合计算成功结果
                    # 生成为求解部分的全部组合
                    current_combinations = self.table_tool.generate_combinations(variables, simplified_result, has_combination_state, self.selection_strategy, self.min_times)
                    unsolved_states =  [var for var in variables if var not in has_combination_state]
                    for combo in itertools.product(*current_combinations):
                        combo_dict = dict(zip(unsolved_states, combo))
                        # 遍历 combination_table 提取涉及当前约束的组合
                        for combo_key, has_calculated_combo_dict in combination_table.items():
                            overlap_combination = {var: combinations for var, combinations in has_calculated_combo_dict.items() if var in variables}
                            overlap_combination.update(combo_dict)
                            new_combo_key = self.table_tool.ensure_hashable(overlap_combination)
                            if new_combo_key in can_key:
                                new_dict = {**has_calculated_combo_dict, **combo_dict}
                                new_full_combo_key = self.table_tool.ensure_hashable(new_dict)
                                new_table[new_full_combo_key] = new_dict
                            elif new_combo_key in cannot_key:
                                continue
                            else:
                                if constrain.validate(overlap_combination):
                                    new_dict = {**has_calculated_combo_dict, **combo_dict}
                                    new_full_combo_key = self.table_tool.ensure_hashable(new_dict)
                                    new_table[new_full_combo_key] = new_dict
                                    can_key.append(new_combo_key)
                                else:
                                    cannot_key.append(new_combo_key)

                    combination_table = new_table
                # 当已求解状体和当前约束状态没有交集时，生成未解部分的组合
                else:
                    # 生成未求解部分的全部组合
                    current_combinations = self.table_tool.generate_combinations(variables, simplified_result, has_combination_state, self.selection_strategy, self.min_times)
                    for combo in itertools.product(*current_combinations):
                        combo_dict = dict(zip(variables, combo))
                        if constrain.validate(combo_dict):
                            valid_combinations.append(combo)
                    if not valid_combinations:
                        return []

                    if not combination_table:
                        combination_table = self.table_tool.create_new_table(variables, valid_combinations)
                    else:
                        combination_table = self.table_tool.join_with_table(
                            combination_table, variables, valid_combinations, has_combination_state
                        )

            if not combination_table:
                return {}
            has_combination_state.extend(variables)

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

