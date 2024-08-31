from typing import List, Optional, Union

from interfaces.Constraint import Constraint
from models.CountConstraint import CountConstraint
from models.TimeConstarint import TimeConstraint
from models.TypeConstraint import TypeConstraint
from models.ValueConstraint import ValueConstraint


class ConstraintCollection:
    def __init__(self):
        self._value_constrain = []
        self._window_constrain_type = None
        self._time_constrain = []
        self._count_constrain = []
        self._type_constrain = []

    def add_constraint(self, constraint: Constraint):
        if isinstance(constraint, ValueConstraint):
            self._add_or_merge_value_constraint(constraint)
        elif isinstance(constraint, TimeConstraint):
            self._add_or_merge_time_constraint(constraint)
        elif isinstance(constraint, CountConstraint):
            self._add_or_merge_count_constraint(constraint)
        elif isinstance(constraint, TypeConstraint):
            if constraint not in self._type_constrain:
                self._type_constrain.append(constraint)
        else:
            raise ValueError(f"Unknown constraint type: {type(constraint).__name__}")

    def _add_or_merge_value_constraint(self, new_constraint: ValueConstraint):
        for existing_constraint in self._value_constrain:
            if existing_constraint.variables == new_constraint.variables:
                existing_constraint.expression = self._merge_constraints(existing_constraint.expression,
                                                                         new_constraint.expression)
                return
        self._value_constrain.append(new_constraint)

    def _add_or_merge_time_constraint(self, new_constraint: TimeConstraint):
        for existing_constraint in self._time_constrain:
            if existing_constraint.variables == new_constraint.variables:
                existing_constraint.min_time = max(existing_constraint.min_time, new_constraint.min_time)
                existing_constraint.max_time = min(existing_constraint.max_time, new_constraint.max_time)
                return
        self._time_constrain.append(new_constraint)

    def _add_or_merge_count_constraint(self, new_constraint: CountConstraint):
        for existing_constraint in self._count_constrain:
            if existing_constraint.variables == new_constraint.variables:
                existing_constraint.min_count = max(existing_constraint.min_count, new_constraint.min_count)
                existing_constraint.max_count = min(existing_constraint.max_count, new_constraint.max_count)
                return
        self._count_constrain.append(new_constraint)

    def _merge_constraints(self, expr1: str, expr2: str) -> str:
        # 假设表达式格式为 'min <= var - var <= max'
        import re
        pattern = r'(\d+)\s*<=.*<=\s*(\d+)'

        match1 = re.search(pattern, expr1)
        match2 = re.search(pattern, expr2)

        if match1 and match2:
            min1, max1 = int(match1.group(1)), int(match1.group(2))
            min2, max2 = int(match2.group(1)), int(match2.group(2))

            new_min = max(min1, min2)
            new_max = min(max1, max2)

            return f"{new_min} <= {expr1.split('<=')[1].split('<=')[0].strip()} <= {new_max}"

        # 如果表达式不符合预期格式，返回原始表达式（或者根据需要抛出异常）
        return expr1

    @property
    def value_constrain(self) -> List[Constraint]:
        return self._value_constrain

    @property
    def window_constrain_type(self) -> str:
        return self._window_constrain_type

    @window_constrain_type.setter
    def window_constrain_type(self, wt):
        self._window_constrain_type = wt

    @property
    def time_constrain(self) -> List[Constraint]:
        return self._time_constrain

    @property
    def count_constrain(self) -> List[Constraint]:
        return self._count_constrain

    @property
    def type_constrain(self) -> List[Constraint]:
        return self._type_constrain

    def validate(self, event, context: 'ConditionContext'):
        # 验证类型是否满足
        for tp in self._type_constrain:
            if not tp.validate(event, context):
                return False

        # 验证所有time_constrain
        for tc in self.time_constrain:
            if not tc.validate(event, context):
                return False

        # 验证所有count_constrain
        for tc in self._count_constrain:
            if not tc.validate(event, context):
                return False

        # 验证所有value_constrain
        for vc in self.value_constrain:
            if not vc.validate(event, context):
                return False

        # 如果所有验证都通过，返回True
        return True

    def __str__(self):
        value_constraints_str = ', '.join(str(c) for c in self._value_constrain)
        time_constraints_str = ', '.join(str(c) for c in self._time_constrain)
        count_constraints_str = ', '.join(str(c) for c in self._count_constrain)
        type_constraints_str = ', '.join(str(c) for c in self._type_constrain)
        window_constraint_str = str(self._window_constrain_type) if self._window_constrain_type else 'None'

        return (f"ConstraintCollection(\n"
                f"  Value Constraints: [{value_constraints_str}],\n"
                f"  Time Constraints: [{time_constraints_str}],\n"
                f"  Count Constraints: [{count_constraints_str}],\n"
                f"  Type Constraints: [{type_constraints_str}],\n"
                f"  Window Constraint: {window_constraint_str}\n"
                f")")