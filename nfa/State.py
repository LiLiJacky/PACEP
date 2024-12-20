import hashlib
from typing import Collection, TypeVar, Generic, List

from models.ValueConstraint import ValueConstraint
from nfa.SelectStrategy import SelectStrategy
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction

T = TypeVar('T')

class StateType:
    Start = 'Start'
    Final = 'Final'
    Normal = 'Normal'
    Pending = 'Pending'
    Stop = 'Stop' # 丢弃这次更新的所有中间匹配结果

class State(Generic[T]):
    def __init__(self, name: str, state_type: StateType, select_strategy: SelectStrategy, *lazy_constrains: List[ValueConstraint], min_times = 0):
        self.name = name
        self.state_type = state_type
        self.state_transitions: List['StateTransition[T]'] = []
        self.select_strategy = select_strategy
        self.lazy_value_constraints = lazy_constrains if lazy_constrains else []
        self.min_times = min_times

    def get_state_type(self) -> StateType:
        return self.state_type

    def is_final(self) -> bool:
        return self.state_type == StateType.Final

    def is_start(self) -> bool:
        return self.state_type == StateType.Start

    def get_name(self) -> str:
        return self.name

    def get_state_transitions(self) -> Collection['StateTransition[T]']:
        return self.state_transitions

    def get_select_strategy(self) -> SelectStrategy:
        return self.select_strategy

    def get_lazy_constraints(self) -> List[ValueConstraint]:
        return self.lazy_value_constraints

    def make_start(self):
        self.state_type = StateType.Start

    def add_state_transition(self, action: 'StateTransitionAction', target_state: 'State[T]', condition: 'Constarint[T]'):
        self.state_transitions.append(StateTransition(self, action, target_state, condition))

    def add_ignore(self, condition: 'Constarint[T]'):
        self.add_state_transition(StateTransitionAction.IGNORE, self, condition)

    def add_ignore_with_target(self, target_state: 'State[T]', condition: 'Constarint[T]'):
        self.add_state_transition(StateTransitionAction.IGNORE, target_state, condition)

    def add_take(self, target_state: 'State[T]', condition: 'Constarint[T]'):
        self.add_state_transition(StateTransitionAction.TAKE, target_state, condition)

    def add_proceed(self, target_state: 'State[T]', condition: 'Constarint[T]'):
        self.add_state_transition(StateTransitionAction.PROCEED, target_state, condition)

    def add_take_with_self(self, condition: 'Constarint[T]'):
        self.add_state_transition(StateTransitionAction.TAKE, self, condition)

    def add_lazy_constraints(self, lazy_constraints: List[ValueConstraint]):
        for lazy_constraint in lazy_constraints:
            if lazy_constraint not in self.lazy_value_constraints:
                self.lazy_value_constraints.append(lazy_constraint)

    def clear_lazy_constraints(self):
        self.lazy_value_constraints.clear()

    def __eq__(self, other):
        if isinstance(other, State):
            return (self.name == other.name and
                    self.state_type == other.state_type and
                    self.state_transitions == other.state_transitions)
        return False

    def __hash__(self):
        return hash((self.name, self.state_type, tuple(self.state_transitions)))

    def __str__(self):
        builder = [f"{self.state_type} State {self.name} [\n"]
        for state_transition in self.state_transitions:
            builder.append(f"\t{state_transition},\n")
        builder.append("])")
        return ''.join(builder)

    def is_stop(self) -> bool:
        return self.state_type == StateType.Stop

    def is_pending(self) -> bool:
        return self.state_type == StateType.Pending