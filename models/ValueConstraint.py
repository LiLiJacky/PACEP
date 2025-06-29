import re
from typing import List, Dict, Any

from algorithm.ExpressionCalculate import ExpressionCalculate
from interfaces.Constraint import Constraint

class ValueConstraint(Constraint):
    def __init__(self, variables: List[str], expression: str):
        super().__init__(variables)
        self.expression = expression
        self.expression_calculate = ExpressionCalculate(expression)

    def validate(self, data = None, context = None, state_algorithm_results = None) -> bool:
        try:
            values = {}
            if context is not None:
                # 如果表达式中只有一个变量
                if len(self.variables) == 1:
                    single_var = self.variables[0]
                    values[single_var] = [[data.event.timestamp, data.event.value]]
                    # 单个Kleene算法求值
                    if '(' in self.expression:
                        single_var = self.variables[0].split('[')[0]
                        values_event_list = context.get_events_for_pattern(single_var)
                        values_list = []
                        for e in values_event_list:
                            values_list.append([e.event.timestamp, e.event.value])

                        # if '[i]' not in self.expression:
                        #     values_list.append([data.event.timestamp, data.event.value])
                        values[single_var] = values_list
                # 如果表达式中包含多个变量
                else:
                    values = {}
                    index = 0
                    for var in self.variables:
                        index += 1
                        if "[i]" in var and var in self.expression: # B[i]
                            values[var] = [data.event.timestamp, data.event.value]
                        elif re.search(rf'\w+\({var}\)', self.expression) or re.search(rf'\w+\({var}\)\[\d+\]',
                                                                                           self.expression):  # algorithm(K) 或 algorithm(K)[num] 格式
                            values_event_list = context.get_events_for_pattern(var)
                            values_list = []
                            for e in values_event_list:
                                values_list.append([e.event.timestamp, e.event.value])
                            values[var] = values_list
                        elif re.search(rf'\b{var}\b', self.expression):  # 普通变量
                            if index != len(self.variables): # 不知道有什么作用
                                values[var] = [context.get_events_for_pattern(var)[0].event.timestamp, context.get_events_for_pattern(var)[0].event.value]
                        else:
                            raise ValueError(f"Unrecognized variable format: {var}")

                    last_var = self.variables[-1].split('[')[0]
                    if "[i]" not in self.expression:
                        if re.search(rf'\w+\({last_var}\)', self.expression) or re.search(rf'\w+\({last_var}\)\[\d+\]',
                                                                                           self.expression):
                            values[last_var].append([data.event.timestamp, data.event.value])
                        else:
                            values[last_var] = [data.event.timestamp, data.event.value]

                if len(values) == 0:
                    return False
            elif state_algorithm_results is not None:
                return self.expression_calculate.calculate(state_algorithm_results=state_algorithm_results)
            else:
                values = data

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
