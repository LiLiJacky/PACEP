import datetime
import time
from typing import List, Dict, Optional, Collection, Tuple
from dataclasses import dataclass

from interfaces.Constraint import Constraint
from nfa.ComputationState import ComputationState
from nfa.NFAState import NFAState
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from nfa.State import State
from sharedbuffer.NodeId import NodeId
from sharedbuffer.EventId import EventId
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction
from util.DeweyNumber import DeweyNumber


@dataclass
class TimerService:
    def current_processing_time(self) -> int:
        """
        Returns the current system time in milliseconds since the epoch.
        """
        return int(time.time() * 1000)


@dataclass
class ConditionContext:
    shared_buffer_accessor: SharedBufferAccessor
    computation_state: ComputationState
    timer_service: 'TimerService'
    event_timestamp: int
    matched_events: Optional[Dict[str, List]] = None

    def get_events_for_pattern(self, key: str) -> List:
        if self.matched_events is None:
            self.matched_events = self.shared_buffer_accessor.materialize_match(
                NFA.extract_current_matches(self.shared_buffer_accessor, self.computation_state))

        return self.matched_events.get(key, [])

    def timestamp(self) -> int:
        return self.event_timestamp

    def current_processing_time(self) -> int:
        return self.timer_service.current_processing_time()


class EventWrapper:
    def __init__(self, event, timestamp: int, shared_buffer_accessor: SharedBufferAccessor):
        self.event = event
        self.timestamp = timestamp
        self.shared_buffer_accessor = shared_buffer_accessor
        self.event_id = None

    def get_event_id(self):
        if self.event_id is None:
            self.event_id = self.shared_buffer_accessor.register_event(self.event, self.timestamp)
        return self.event_id

    def get_event(self):
        return self.event

    def get_timestamp(self):
        return self.timestamp

    def close(self):
        if self.event_id is not None:
            self.shared_buffer_accessor.release_event(self.event_id)

    def __enter__(self):
        # 在进入上下文管理时，调用 get_event_id 方法以确保事件已注册
        self.get_event_id()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在退出上下文管理时，调用 close 方法以释放事件
        self.close()


@dataclass
class OutgoingEdges:
    edges: List[StateTransition] = None
    current_state: State = None
    total_take_branches: int = 0
    total_ignore_branches: int = 0

    def __init__(self, current_state: State):
        self.current_state = current_state
        self.edges = []

    def add(self, edge: StateTransition):
        if not NFA.is_self_ignore(edge, self.current_state):
            if edge.get_action() == StateTransitionAction.IGNORE:
                self.total_ignore_branches += 1
            elif edge.get_action() == StateTransitionAction.TAKE:
                self.total_take_branches += 1
            self.edges.append(edge)


class NFA:
    def __init__(self, valid_states: Collection[State], window_times: Dict[str, int], window_time: int, handle_timeout: bool):
        self.window_time = window_time
        self.handle_timeout = handle_timeout
        self.states = self.load_states(valid_states)
        self.window_times = window_times

    def load_states(self, valid_states: Collection[State]) -> Dict[str, State]:
        return {state.get_name(): state for state in valid_states}

    def get_window_time(self) -> int:
        return self.window_time

    def get_states(self) -> Collection[State]:
        return self.states.values()

    def create_initial_nfa_state(self) -> NFAState:
        starting_states = [ComputationState.create_start_state(state.get_name()) for state in self.states.values() if state.is_start()]
        return NFAState(starting_states)

    def get_state(self, computation_state: ComputationState) -> State:
        return self.states.get(computation_state.get_current_state_name())

    def is_start_state(self, computation_state: ComputationState) -> bool:
        state = self.get_state(computation_state)
        if state is None:
            raise RuntimeError(f"State {computation_state.get_current_state_name()} does not exist in the NFA.")
        return state.is_start()

    def is_stop_state(self, computation_state: ComputationState) -> bool:
        state = self.get_state(computation_state)
        if state is None:
            raise RuntimeError(f"State {computation_state.get_current_state_name()} does not exist in the NFA.")
        return state.is_stop()

    def is_final_state(self, computation_state: ComputationState) -> bool:
        state = self.get_state(computation_state)
        if state is None:
            raise RuntimeError(f"State {computation_state.get_current_state_name()} does not exist in the NFA.")
        return state.is_final()

    def process(self, shared_buffer_accessor: SharedBufferAccessor, nfa_state: NFAState, event, timestamp: int, after_match_skip_strategy, timer_service) -> Collection[Dict[str, List]]:
        with EventWrapper(event, timestamp, shared_buffer_accessor) as event_wrapper:
            return self.do_process(shared_buffer_accessor, nfa_state, event_wrapper, after_match_skip_strategy, timer_service)

    def advance_time(self, shared_buffer_accessor: SharedBufferAccessor, nfa_state: NFAState, timestamp: int, after_match_skip_strategy) -> Tuple[Collection[Dict[str, List]], Collection[Tuple[Dict[str, List], int]]]:
        result = []
        timeout_result = []
        new_partial_matches = []
        potential_matches = []

        for computation_state in nfa_state.get_partial_matches():
            current_state_name = computation_state.get_current_state_name()
            is_timeout_for_previous_event = (current_state_name in self.window_times and self.is_state_timed_out(computation_state, timestamp, computation_state.get_previous_timestamp(), self.window_times[current_state_name]))
            is_timeout_for_first_event = self.is_state_timed_out(computation_state, timestamp, computation_state.get_start_timestamp(), self.window_time)

            if is_timeout_for_previous_event or is_timeout_for_first_event:
                nfa_state.set_state_changed()

                if self.get_state(computation_state).is_pending():
                    potential_matches.append(computation_state)
                    continue

                if self.handle_timeout:
                    timed_out_pattern = self.extract_current_matches(shared_buffer_accessor, computation_state)
                    timeout_result.append((timed_out_pattern, computation_state.get_previous_timestamp() + self.window_times[current_state_name] if is_timeout_for_previous_event else computation_state.get_start_timestamp() + self.window_time))

                shared_buffer_accessor.release_node(computation_state.get_previous_buffer_entry(), computation_state.get_version())
            else:
                new_partial_matches.append(computation_state)

        # self.process_matches_according_to_skip_strategy(shared_buffer_accessor, nfa_state, after_match_skip_strategy, potential_matches, new_partial_matches, result)

        nfa_state.set_new_partial_matches(new_partial_matches)
        shared_buffer_accessor.advance_time(timestamp)

        return result, timeout_result

    def is_state_timed_out(self, computation_state: ComputationState, timestamp: int, start_timestamp: int, window_time: int) -> bool:
        return not self.is_start_state(computation_state) and 0 < window_time <= timestamp - start_timestamp

    def do_process(self, shared_buffer_accessor: SharedBufferAccessor, nfa_state: NFAState, event_wrapper: EventWrapper, after_match_skip_strategy, timer_service) -> Collection[Dict[str, List]]:
        new_partial_matches = []
        potential_matches = []

        for computation_state in nfa_state.get_partial_matches():
            new_computation_states = self.compute_next_states(shared_buffer_accessor, computation_state, event_wrapper, timer_service)

            if len(new_computation_states) != 1:
                nfa_state.set_state_changed()
            elif not next(iter(new_computation_states)) == computation_state:
                nfa_state.set_state_changed()

            states_to_retain = []
            should_discard_path = False

            for new_computation_state in new_computation_states:
                if self.is_start_state(computation_state) and new_computation_state.get_start_timestamp() > 0:
                    nfa_state.set_new_start_partial_match()

                if self.is_final_state(new_computation_state):
                    potential_matches.append(new_computation_state)
                elif self.is_stop_state(new_computation_state):
                    should_discard_path = True
                    shared_buffer_accessor.release_node(new_computation_state.get_previous_buffer_entry(), new_computation_state.get_version())
                else:
                    states_to_retain.append(new_computation_state)

            if should_discard_path:
                for state in states_to_retain:
                    shared_buffer_accessor.release_node(state.get_previous_buffer_entry(), state.get_version())
            else:
                new_partial_matches.extend(states_to_retain)

        if potential_matches:
            nfa_state.set_state_changed()

        result = []
        self.process_matches_according_to_skip_strategy(shared_buffer_accessor, nfa_state, after_match_skip_strategy, potential_matches, new_partial_matches, result)

        nfa_state.set_new_partial_matches(new_partial_matches)

        return result

    def process_matches_according_to_skip_strategy(self, shared_buffer_accessor: SharedBufferAccessor, nfa_state: NFAState, after_match_skip_strategy, potential_matches, partial_matches, result):
        nfa_state.get_completed_matches().extend(potential_matches)

        while nfa_state.get_completed_matches():
            earliest_match = nfa_state.get_completed_matches()[0]

            if after_match_skip_strategy.is_skip_strategy():
                earliest_partial_match = partial_matches[0] if partial_matches else None
                if earliest_partial_match and not self.is_earlier(earliest_match, earliest_partial_match):
                    break

            nfa_state.set_state_changed()
            nfa_state.get_completed_matches().pop(0)
            matched_result = shared_buffer_accessor.extract_patterns(earliest_match.get_previous_buffer_entry(),
                                                                     earliest_match.get_version())

            after_match_skip_strategy.prune(partial_matches, matched_result, shared_buffer_accessor)
            after_match_skip_strategy.prune(nfa_state.get_completed_matches(), matched_result, shared_buffer_accessor)

            result.append(shared_buffer_accessor.materialize_match(matched_result[0]))
            shared_buffer_accessor.release_node(earliest_match.get_previous_buffer_entry(),
                                                earliest_match.get_version())

        # 获取当前的部分匹配列表
        partial_matches_list = nfa_state.get_partial_matches()

        # 使用列表推导式过滤符合条件的元素，并更新列表
        nfa_state.get_partial_matches()[:] = [
            pm for pm in partial_matches_list
            if pm.get_start_event_id() is None or pm in partial_matches
        ]

    def is_earlier(self, earliest_match, earliest_partial_match) -> bool:
        return earliest_match <= earliest_partial_match

    def compute_next_states(self, shared_buffer_accessor: SharedBufferAccessor, computation_state: ComputationState,
                            event_wrapper: EventWrapper, timer_service) -> Collection[ComputationState]:
        context = ConditionContext(shared_buffer_accessor, computation_state, timer_service,
                                   event_wrapper.get_timestamp())

        outgoing_edges = self.create_decision_graph(context, computation_state, event_wrapper.get_event())

        edges = outgoing_edges.edges
        take_branches_to_visit = max(0, outgoing_edges.total_take_branches - 1)
        ignore_branches_to_visit = outgoing_edges.total_ignore_branches
        total_take_to_skip = max(0, outgoing_edges.total_take_branches - 1)

        resulting_computation_states = []
        for edge in edges:
            if edge.get_action().name == 'IGNORE':
                if not self.is_start_state(computation_state):
                    version = None
                    if self.is_equivalent_state(edge.get_target_state(), self.get_state(computation_state)):
                        to_increase = self.calculate_increasing_self_state(outgoing_edges.total_ignore_branches,
                                                                           outgoing_edges.total_take_branches)
                        version = computation_state.get_version().increase(to_increase)
                    else:
                        version = computation_state.get_version().increase(
                            total_take_to_skip + ignore_branches_to_visit).add_stage()
                        ignore_branches_to_visit -= 1

                    self.add_computation_state(shared_buffer_accessor, resulting_computation_states,
                                               edge.get_target_state(),
                                               computation_state.get_previous_buffer_entry(),
                                               version, computation_state.get_start_timestamp(),
                                               computation_state.get_previous_timestamp(),
                                               computation_state.get_start_event_id())

            elif edge.get_action().name == 'TAKE':
                next_state = edge.get_target_state()
                current_state = edge.get_source_state()

                previous_entry = computation_state.get_previous_buffer_entry()

                current_version = computation_state.get_version().increase(take_branches_to_visit)
                next_version = current_version.add_stage()
                take_branches_to_visit -= 1

                new_entry = shared_buffer_accessor.put(current_state.get_name(), event_wrapper.get_event_id(),
                                                       previous_entry, current_version)

                start_timestamp = event_wrapper.get_timestamp() if self.is_start_state(
                    computation_state) else computation_state.get_start_timestamp()
                if isinstance(start_timestamp, datetime.datetime):
                    start_timestamp = int(start_timestamp.timestamp())
                start_event_id = event_wrapper.get_event_id() if self.is_start_state(
                    computation_state) else computation_state.get_start_event_id()
                previous_timestamp = event_wrapper.get_timestamp()

                self.add_computation_state(shared_buffer_accessor, resulting_computation_states, next_state,
                                           new_entry,
                                           next_version, start_timestamp, previous_timestamp, start_event_id)

                final_state = self.find_final_state_after_proceed(context, next_state, event_wrapper.get_event())
                if final_state:
                    self.add_computation_state(shared_buffer_accessor, resulting_computation_states, final_state,
                                               new_entry, next_version, start_timestamp, previous_timestamp,
                                               start_event_id)

        if self.is_start_state(computation_state):
            total_branches = self.calculate_increasing_self_state(outgoing_edges.total_ignore_branches,
                                                                  outgoing_edges.total_take_branches)
            start_version = computation_state.get_version().increase(total_branches)
            start_state = ComputationState.create_start_state(computation_state.get_current_state_name(),
                                                              start_version)
            resulting_computation_states.append(start_state)

        if computation_state.get_previous_buffer_entry():
            shared_buffer_accessor.release_node(computation_state.get_previous_buffer_entry(),
                                                computation_state.get_version())

        return resulting_computation_states

    def add_computation_state(self, shared_buffer_accessor: SharedBufferAccessor,
                              computation_states: List[ComputationState], current_state: State,
                              previous_entry: NodeId, version: 'DeweyNumber', start_timestamp: int,
                              previous_timestamp: int, start_event_id: EventId):
        computation_state = ComputationState.create_state(current_state.get_name(), previous_entry, version,
                                                          start_timestamp, previous_timestamp, start_event_id)
        computation_states.append(computation_state)
        shared_buffer_accessor.lock_node(previous_entry, computation_state.get_version())

    def find_final_state_after_proceed(self, context: ConditionContext, state: State, event):
        states_to_check = [state]
        try:
            while states_to_check:
                current_state = states_to_check.pop()
                for transition in current_state.get_state_transitions():
                    if transition.get_action() == 'PROCEED' and self.check_filter_condition(context,
                                                                                            transition.get_condition(),
                                                                                            event):
                        if transition.get_target_state().is_final():
                            return transition.get_target_state()
                        else:
                            states_to_check.append(transition.get_target_state())
        except Exception as e:
            raise RuntimeError("Failure happened in filter function.") from e

        return None

    def calculate_increasing_self_state(self, ignore_branches, take_branches):
        return ignore_branches + max(1, take_branches) if take_branches > 0 or ignore_branches > 0 else 0

    def create_decision_graph(self, context: ConditionContext, computation_state: ComputationState, event):
        state: State = self.get_state(computation_state)
        outgoing_edges = OutgoingEdges(state)

        states = [state]

        while states:
            current_state = states.pop()
            state_transitions = current_state.get_state_transitions()

            for state_transition in state_transitions:
                try:
                    if self.check_filter_condition(context, state_transition.get_condition(), event):
                        if state_transition.get_action() == StateTransitionAction.PROCEED:
                            states.append(state_transition.get_target_state())
                        else:
                            outgoing_edges.add(state_transition)
                except Exception as e:
                    raise RuntimeError("Failure happened in filter function.") from e

        return outgoing_edges

    @staticmethod
    def check_filter_condition(context: ConditionContext, condition: Constraint, event):
        return condition is None or condition.validate(event, context)

    @staticmethod
    def extract_current_matches(shared_buffer_accessor: SharedBufferAccessor,
                                computation_state: ComputationState) -> Dict[str, List[EventId]]:
        if computation_state.get_previous_buffer_entry() is None:
            return {}

        paths = shared_buffer_accessor.extract_patterns(computation_state.get_previous_buffer_entry(),
                                                        computation_state.get_version())
        assert len(paths) == 1

        return paths[0]

    @staticmethod
    def is_equivalent_state(s1: State, s2: State) -> bool:
        return s1.get_name() == s2.get_name()

    @staticmethod
    def is_self_ignore(edge: StateTransition, current_state: State) -> bool:
        return NFA.is_equivalent_state(edge.get_target_state(), current_state) and edge.get_action() == 'IGNORE'

