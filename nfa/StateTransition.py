import hashlib
from typing import TypeVar, Generic
from interfaces.Constraint import Constraint
from models.CountConstraint import CountConstraint
from models.TimeConstarint import TimeConstraint
from models.TypeConstraint import TypeConstraint
from models.ValueConstraint import ValueConstraint
from models.ConstraintCollection import ConstraintCollection

T = TypeVar('T')


class StateTransition(Generic[T]):
    def __init__(self, source_state: 'State[T]', action: 'StateTransitionAction', target_state: 'State[T]', condition: 'ConstraintCollection'):
        self.action = action
        self.source_state = source_state
        self.target_state = target_state
        self.condition = condition

    def add_condition(self, constraint: Constraint):
        """
        Adds a constraint to the appropriate list in the ConstraintCollection,
        ensuring no duplicates are added. For TimeConstraint, if the start and end
        variables are the same as an existing constraint, merge the two constraints
        into a minimal set with the intersection of their time ranges.
        """
        if isinstance(constraint, ValueConstraint):
            if constraint not in self.condition.value_constrain:
                self.condition.value_constrain.append(constraint)
        elif isinstance(constraint, TimeConstraint):
            merged = False
            for existing_constraint in self.condition.time_constrain:
                if (existing_constraint.variables[0] == constraint.variables[0] and
                        existing_constraint.variables[1] == constraint.variables[1]):
                    # 合并两个 TimeConstraint 的时间范围
                    new_min_time = max(existing_constraint.min_time, constraint.min_time)
                    new_max_time = min(existing_constraint.max_time, constraint.max_time)

                    # 更新现有约束的时间范围和表达式
                    existing_constraint.min_time = new_min_time
                    existing_constraint.max_time = new_max_time
                    existing_constraint.expression = (
                        f"{new_min_time} <= {existing_constraint.variables[1]} - {existing_constraint.variables[0]} <= {new_max_time}"
                    )
                    merged = True
                    break

            if not merged:
                self.condition.time_constrain.append(constraint)
        elif isinstance(constraint, CountConstraint):
            merged = False
            for existing_constraint in self.condition.count_constrain:
                if existing_constraint.variables == constraint.variables:
                    # 合并两个 TimesConstraint 的次数范围
                    new_min_times = max(existing_constraint.min_count, constraint.min_count)
                    new_max_times = min(existing_constraint.max_count, constraint.max_count)

                    # 更新现有约束的次数范围
                    existing_constraint.min_count = new_min_times
                    existing_constraint.max_count = new_max_times
                    merged = True
                    break

            if not merged:
                self.condition.count_constrain.append(constraint)
        elif isinstance(constraint, TypeConstraint):
            if constraint not in self.condition.type_constrain:
                self.condition.type_constrain.append(constraint)
        else:
            raise ValueError(f"Unknown constraint type: {type(constraint).__name__}")

    def get_action(self) -> 'StateTransitionAction':
        return self.action

    def get_target_state(self) -> 'State[T]':
        return self.target_state

    def get_source_state(self) -> 'State[T]':
        return self.source_state

    def get_condition(self) -> 'ConstraintCollection':
        return self.condition

    def set_condition(self, condition: 'ConstraintCollection'):
        self.condition = condition

    def __eq__(self, other):
        if isinstance(other, StateTransition):
            return (self.action == other.action and
                    self.source_state.get_name() == other.source_state.get_name() and
                    self.target_state.get_name() == other.target_state.get_name())
        return False

    def __hash__(self):
        return hash((self.action, self.target_state.get_name(), self.source_state.get_name()))

    def __str__(self):
        return (f"StateTransition({self.action}, from {self.source_state.get_name()} to "
                f"{self.target_state.get_name()}"
                f"{', with condition)' if self.condition else ')'}")