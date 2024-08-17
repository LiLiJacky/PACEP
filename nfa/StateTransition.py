import hashlib
from typing import TypeVar, Generic
from interfaces.Constraint import Constraint

T = TypeVar('T')


class StateTransition(Generic[T]):
    def __init__(self, source_state: 'State[T]', action: 'StateTransitionAction', target_state: 'State[T]', condition: 'Constraint'):
        self.action = action
        self.source_state = source_state
        self.target_state = target_state
        self.condition = condition

    def get_action(self) -> 'StateTransitionAction':
        return self.action

    def get_target_state(self) -> 'State[T]':
        return self.target_state

    def get_source_state(self) -> 'State[T]':
        return self.source_state

    def get_condition(self) -> 'Constraint':
        return self.condition

    def set_condition(self, condition: 'Constraint'):
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