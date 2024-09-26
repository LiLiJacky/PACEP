# 计算工具类
import re
from typing import Dict, Any, List

from factories.AlgorithmFactory import AlgorithmFactory


class ExpressionCalculate:
    def __init__(self, expression: str):
        self.expression = expression
        self.algorithm_factory = AlgorithmFactory()

    def calculate(self, values: Dict[str, Any]) -> bool:
        # 找到比较操作符的位置
        operators = re.findall(r'(<=|>=|<|>)', self.expression)

        if len(operators) != 2:
            raise ValueError("Expression must contain two comparison operators.")

        # 通过操作符分割表达式
        parts = re.split(r'(<=|>=|<|>)', self.expression)

        # 检查表达式的格式是否正确
        if len(parts) != 5:
            raise ValueError(
                "Expression format is not valid. It should have the format 'left_value left_operator middle_expression right_operator right_value'")

        # 提取左侧值、左侧符号、中间表达式、右侧符号、右侧值
        left_value = int(parts[0].strip())
        left_operator = parts[1].strip()
        middle_expression = parts[2].strip()
        right_operator = parts[3].strip()
        right_value = int(parts[4].strip())

        # 计算中间表达式的值
        middle_value = self._evaluate_expression(middle_expression, values)

        if isinstance(middle_value, tuple):
            print("i need to know why")

        # 判断左侧和右侧表达式是否都满足条件
        is_left_satisfied = self._apply_operator(left_value, middle_value, left_operator)
        is_right_satisfied = self._apply_operator(middle_value, right_value, right_operator)

        # 同时满足左侧和右侧条件才返回 True，否则返回 False
        return is_left_satisfied and is_right_satisfied

    def _evaluate_expression(self, expression_part: str, values: Dict[str, Any]) -> float:
        total_sum = 0

        # 拆分通过 + 连接的子计算部分
        sub_expressions = expression_part.split('+')

        for sub_expr in sub_expressions:
            sub_expr = sub_expr.strip()  # 去除可能存在的空格

            if re.match(r'^[A-Z]$', sub_expr) or re.match(r'^[A-Z]\[i\]$', sub_expr):  # 单个字母的普通变量
                base_var = sub_expr
                if re.match(r'^[A-Z]\[i\]$', sub_expr):  # A[i] 格式
                    base_var = sub_expr.split('[')[0]  # 提取基础变量名，如 A
                total_sum += values.get(base_var, 0)[0][1]
            elif re.match(r'\w+\([A-Z]\)\[\d+\]', sub_expr):  # algorithm(K)[num] 格式
                algorithm_name, var, num = re.match(r'(\w+)\(([A-Z])\)\[(\d+)\]', sub_expr).groups()
                algorithm_result = self._apply_algorithm(algorithm_name, values[var], int(num))
                total_sum += algorithm_result
            elif re.match(r'\w+\([A-Z]\)', sub_expr):  # algorithm(K) 格式
                algorithm_name, var = re.match(r'(\w+)\(([A-Z])\)', sub_expr).groups()
                algorithm_result = self._apply_algorithm(algorithm_name, values[var])
                total_sum += algorithm_result
            else:
                raise ValueError(f"Unrecognized sub-expression format: {sub_expr}")

        return total_sum

    def _apply_algorithm(self, func_name: str, values: List[float], *args) -> float:
        full_name = self.algorithm_factory.get_algorithm_sub_map().get(func_name)
        if full_name:
            if func_name == 'nth' or 'sort' in func_name or func_name == 'combinations_square':
                algorithm = self.algorithm_factory.get_algorithm(full_name, values, args)
                result = algorithm.get_calculate(args[0])
            else:
                algorithm = self.algorithm_factory.get_algorithm(full_name, values)
                result = algorithm.get_calculate()

            if isinstance(result, tuple):
                print("i need to know why")
            return result
        else:
            raise ValueError(f"Unknown function '{func_name}' in expression")

    def _apply_operator(self, left_value: float, right_value: float, operator: str) -> bool:
        if isinstance(left_value, tuple):
            print("i need to know why")
        if isinstance(right_value, tuple):
            print("i need to know why")
        if operator == '<':
            return left_value < right_value
        elif operator == '<=':
            return left_value <= right_value
        elif operator == '>':
            return left_value > right_value
        elif operator == '>=':
            return left_value >= right_value
        else:
            raise ValueError(f"Unknown operator '{operator}' in expression")