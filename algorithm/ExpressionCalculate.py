# 计算工具类
import re
from typing import Dict, Any, List

from factories.AlgorithmFactory import AlgorithmFactory


class ExpressionCalculate:
    def __init__(self, expression: str):
        self.expression = expression
        self.algorithm_factory = AlgorithmFactory()

    def calculate(self, values: Dict[str, Any] = None, state_algorithm_results = None) -> bool:
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
        left_value = float(parts[0].strip())
        left_operator = parts[1].strip()
        middle_expression = parts[2].strip()
        right_operator = parts[3].strip()
        right_value = float(parts[4].strip())

        if state_algorithm_results is None:
            # 计算中间表达式的值
            middle_value = self._evaluate_expression(middle_expression, values)
        else:
            # 计算中间表达式的值
            middle_value = self._compose_expression(middle_expression, state_algorithm_results)

        # 判断左侧和右侧表达式是否都满足条件
        is_left_satisfied = self._apply_operator(left_value, middle_value, left_operator)
        is_right_satisfied = self._apply_operator(middle_value, right_value, right_operator)

        # 同时满足左侧和右侧条件才返回 True，否则返回 False
        return is_left_satisfied and is_right_satisfied

    def _compose_expression(self, expression: str, state_algorithm_results) -> float:
        """
                根据状态变量的计算结果，动态计算表达式的值
                :param expression: 表达式字符串
                :param state_algorithm_results: dict，存储各状态变量及其算法计算值
                :return: 表达式计算的结果
                """
        # 初始化新的表达式
        new_expression = expression

        # 替换表达式中的变量名为实际值
        for state, results in state_algorithm_results.items():
            for algorithm, value in results.items():
                new_expression = re.sub(
                    rf'{algorithm}\s*\(\s*{state}\s*\)',  # 匹配类似于 "average(B)"
                    str(value),  # 替换为对应的值
                    new_expression  # 持续更新 new_expression
                )

        # 动态计算表达式值
        try:
            # 使用 `eval` 动态计算替换后的表达式，限制全局变量访问
            return eval(new_expression, {"__builtins__": None}, {})
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {new_expression} -> {e}")

    def _evaluate_expression(self, expression_part: str, values: Dict[str, Any]) -> float:
        # 替换表达式中的变量为其具体数值
        def replace_variables(match):
            sub_expr = match.group(0).strip()  # 匹配到的子表达式

            # 处理普通变量（单个字母或 A[i] 格式）
            if re.match(r'^[A-Z]$', sub_expr) or re.match(r'^[A-Z]\[i\]$', sub_expr):
                #base_var = sub_expr.split('[')[0] if '[' in sub_expr else sub_expr
                var_value = values.get(sub_expr, 0)

                # 检查 var_value 类型
                if isinstance(var_value, list) and len(var_value) > 0 and isinstance(var_value[0], list):
                    return str(var_value[0][1])  # 将变量替换为其值
                elif isinstance(var_value, list):
                    return str(var_value[1])
                else:
                    raise ValueError(f"Invalid value for variable '{sub_expr}': {var_value}")

            # 处理 algorithm(K)[num] 格式
            elif re.match(r'\w+\([A-Z]\)\[\d+\]', sub_expr):
                algorithm_name, var, num = re.match(r'(\w+)\(([A-Z])\)\[(\d+)\]', sub_expr).groups()
                return str(self._apply_algorithm(algorithm_name, values[var], float(num)))

            # 处理 algorithm(K) 格式
            elif re.match(r'\w+\([A-Z]\)', sub_expr):
                algorithm_name, var = re.match(r'(\w+)\(([A-Z])\)', sub_expr).groups()
                return str(self._apply_algorithm(algorithm_name, values[var]))

            raise ValueError(f"Unrecognized sub-expression format: {sub_expr}")

        # 使用正则表达式替换表达式中的变量和算法调用
        expression_with_values = re.sub(r'[A-Z]\[i\]|\w+\([A-Z]\)\[\d+\]|\w+\([A-Z]\)|[A-Z]', replace_variables,
                                        expression_part)

        # 使用 eval 计算最终表达式值
        try:
            result = eval(expression_with_values)
        except ZeroDivisionError:
            raise ValueError("Division by zero encountered in expression.")
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {expression_with_values}. Details: {e}")

        return result


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