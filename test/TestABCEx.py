import itertools
from collections import defaultdict

import numpy as np

class TestABCEx:
    @staticmethod
    def sum_square_difference(values):
        """计算 sum_square_difference(A)"""
        if not values:  # 避免空列表
            return 0
        data = values[:]
        n = len(data)
        sum_square_diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                sum_square_diff += (data[i] - data[j]) ** 2
        return sum_square_diff

    @staticmethod
    def average(values):
        """计算 average(B)"""
        if not values:  # 避免空列表
            return 0
        return np.mean(values)  # 确保调用 np.mean(values)
    @staticmethod
    def valuate(basic_result):
        return []
        """
        评估所有可能组合并满足给定约束条件的结果
        :param basic_result: 包含 'A', 'B' 的候选解（列表的形式）
        :return: 满足约束条件的解列表
        """
        # 提取候选解
        result = basic_result[0]
        event_A = result['A']
        event_B = result['B']
        event_C = result['C'][0]

        # 提取 'data' 字段作为计算依据
        data_A = [event.event.get_value() for event in event_A]
        data_B = [event.event.get_value() for event in event_B]

        # **筛选满足条件的 B 子集**
        valid_B_combinations = [
            [event_B[idx] for idx in indices] for indices in itertools.chain.from_iterable(
                itertools.combinations(range(len(data_B)), r) for r in range(1, len(data_B) + 1)
            )
            if 10 <= TestABCEx.average([data_B[idx] for idx in indices]) <= 1000
        ]

        # 如果没有满足条件的 B 子集，直接返回空
        if not valid_B_combinations:
            return []

        # **筛选满足条件的 A 子集**
        valid_A_combinations = [
            [event_A[idx] for idx in indices] for indices in itertools.chain.from_iterable(
                itertools.combinations(range(len(data_A)), r) for r in range(1, len(data_A) + 1)
            )
            if 200 <= TestABCEx.sum_square_difference([data_A[idx] for idx in indices]) <= 1000
        ]

        # 如果没有满足条件的 A 子集，也返回空
        if not valid_A_combinations:
            return []

        # 生成格式化的解决方案
        valid_solutions = [
            [
                defaultdict(
                    list,
                    {'A': A, 'B': B, 'C': [event_C]}
                ),
                []
            ]
            for A in valid_A_combinations for B in valid_B_combinations
        ]

        return valid_solutions
