import re
from typing import List, Dict, Any

from algorithm.ExpressionCalculate import ExpressionCalculate
from interfaces.Constraint import Constraint

class ValueConstraint(Constraint):
    def __init__(self, variables: List[str], expression: str):
        super().__init__(variables)
        self.expression = expression
        self.expression_calculate = ExpressionCalculate(expression)

    def validate(self, data, context) -> bool:
        try:
            # 如果表达式中只有一个变量
            if len(self.variables) == 1 and ('[' not in self.expression or '[i]' in self.expression):
                single_var = self.variables[0]
                values = {single_var: [[data.event.timestamp, data.event.value]]}
            # 单个Kleene算法求值
            elif len(self.variables) == 1 and '[' in self.expression:
                single_var = self.variables[0]
                var_name = f"{single_var}[i]"
                values = {}
                values_event_list = context.get_events_for_pattern(var_name)
                values_list = []
                for e in values_event_list:
                    values_list.append([e.event.timestamp, e.event.value])

                values[single_var] = values_list
            # 如果表达式中包含多个变量
            else:
                values = {}
                for var in self.variables[:-1]:
                    if re.search(rf'\w+\({var}\)', self.expression) or re.search(rf'\w+\({var}\)\[\d+\]',
                                                                                       self.expression):  # algorithm(K) 或 algorithm(K)[num] 格式
                        var_name = f"{var}[i]"
                        values_event_list = context.get_events_for_pattern(var_name)
                        values_list = []
                        for e in values_event_list:
                            values_list.append([e.event.timestamp, e.event.value])
                        values[var] = values_list
                    elif re.search(rf'\b{var}\b', self.expression):  # 普通变量
                        values[var] = context.get_events_for_pattern(var)[0].event.timestamp, context.get_events_for_pattern(var)[0].event.value
                    else:
                        raise ValueError(f"Unrecognized variable format: {var}")

                values[data.event.variable_name] = [[data.event.timestamp, data.event.value]]

            return self.expression_calculate.calculate(values)

        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def __eq__(self, other):
        if isinstance(other, ValueConstraint):
            return self.variables == other.variables and self.expression == other.expression
        return False

    def __hash__(self):
        return hash((tuple(self.variables), self.expression))

    def __str__(self):
        return (f"ValueConstraint(variables={self.variables}, "
                f"expression='{self.expression}')")