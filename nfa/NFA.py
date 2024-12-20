import datetime
import time
from collections import defaultdict
from typing import List, Dict, Optional, Collection, Tuple
from dataclasses import dataclass

from aftermatch.NoSkipStrategy import NoSkipStrategy
from interfaces.Constraint import Constraint
from lazy_calculate.IncreamentCalculate import IncrementTableCalculate
from lazy_calculate.TableCalculate import TableCalculate
from lazy_calculate.TableTool import TableTool
from models.ValueConstraint import ValueConstraint
from nfa.ComputationState import ComputationState
from nfa.NFAState import NFAState
from nfa.SelectStrategy import SelectStrategy
from shared_calculate.BloomFilterManager import BloomFilterManager
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from nfa.State import State
from sharedbuffer.NodeId import NodeId
from sharedbuffer.EventId import EventId
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction
from test.TestABCEx import TestABCEx
from util.DeweyNumber import DeweyNumber
from lazy_calculate.LazyHandler import LazyHandler


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
        if self.matched_events is None or len(self.matched_events) == 0:
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

    def __str__(self):
        return f"EventWrapper(event={self.event}, timestamp={self.timestamp})"


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
    def __init__(self, valid_states: Collection[State], window_times: Dict[str, int], window_time: int, handle_timeout: bool, lazy_model = False, lazy_calculate_model = None,
                 selection_strategy = None):
        self.window_time = window_time
        self.handle_timeout = handle_timeout
        self.states = self.load_states(valid_states)
        self.window_times = window_times
        self.lazy_model = lazy_model
        self.lazy_calculate_model = lazy_calculate_model
        self.selection_strategy = [] if selection_strategy is None else selection_strategy

        # capacity = min(2**(window_time * 3), 2**20)
        # self.bloom_filter_manager = BloomFilterManager(max(capacity, 8192), 0.001, window_time * 30)

        # 初始化 min_times
        min_times = {}
        for state in valid_states:
            for t in state.state_transitions:
                if t.get_action() == StateTransitionAction.PROCEED:
                    min_times[state.name] = state.min_times
        self.min_times = min_times

    def load_states(self, valid_states: Collection[State]) -> Dict[str, State]:
        return {state.get_name(): state for state in valid_states}

    def get_window_time(self) -> int:
        return self.window_time

    def get_states(self) -> Collection[State]:
        return self.states.values()

    def create_initial_nfa_state(self) -> NFAState:
        starting_states = [ComputationState.create_start_state(state.get_name()) for state in self.states.values() if state.is_start()]
        nfa_state = NFAState()
        nfa_state.set_new_partial_matches(starting_states)
        return nfa_state

    def get_state(self, computation_state: ComputationState) -> State:
        return self.states.get(computation_state.get_current_state_name())

    def is_start_state(self, computation_state: ComputationState) -> bool:
        state = self.get_state(computation_state)
        if state is None:
            raise RuntimeError(f"State {computation_state.get_current_state_name()} does not exist in the NFA.")
        return state.is_start() and computation_state.get_start_timestamp() == -1

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

        self.process_matches_according_to_skip_strategy(shared_buffer_accessor, nfa_state, after_match_skip_strategy, potential_matches, new_partial_matches, result)

        nfa_state.set_new_partial_matches(new_partial_matches)
        shared_buffer_accessor.advance_time(timestamp)

        return result, timeout_result

    def is_state_timed_out(self, computation_state: ComputationState, timestamp: int, start_timestamp: int, window_time: int) -> bool:
        return not (self.is_start_state(computation_state) and computation_state.get_start_timestamp() == -1) and 0 < window_time <= timestamp - start_timestamp

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
        # 由于每个结果都是相互独立，且确定性的，所以可以通过遍历结果来构建共享子图
        self.process_matches_according_to_skip_strategy(shared_buffer_accessor, nfa_state, after_match_skip_strategy, potential_matches, new_partial_matches, result)

        nfa_state.set_new_partial_matches(new_partial_matches)

        return result

    # 判断是否创建新的匹配
    def has_same_state(self, nfa_state: NFAState, computation_state: ComputationState):
        for partial_match in nfa_state.partial_matches:
            if computation_state.get_current_state_name() == partial_match.get_current_state_name() and partial_match.start_timestamp != -1:
                return True

        return False

    def process_matches_according_to_skip_strategy(self, shared_buffer_accessor: SharedBufferAccessor, nfa_state: NFAState, after_match_skip_strategy, potential_matches, partial_matches, result):
        nfa_state.get_completed_matches().extend(potential_matches)

        # lazy_model 的情况下，当第一个事件的是Kleene事件时，当Kleene事件的第一个事件被忽略时，剩下的匹配一定是子集，可以直接跳过
        first_kleene = {}
        son_list = False
        while nfa_state.get_completed_matches():
            earliest_match = nfa_state.get_completed_matches()[0]

            if after_match_skip_strategy.is_skip_strategy():
                earliest_partial_match = partial_matches[0] if partial_matches else None
                if earliest_partial_match and not self.is_earlier(earliest_match, earliest_partial_match):
                    break

            nfa_state.set_state_changed()
            nfa_state.get_completed_matches().pop(0)

            if not self.lazy_model or not son_list:
                matched_result_ex = shared_buffer_accessor.extract_patterns(earliest_match.get_previous_buffer_entry(),
                                                                         earliest_match.get_version())

                if self.lazy_model:
                    matched_result_ex[1].extend(earliest_match.lazy_constraints)

                matched_result = matched_result_ex[0]

                # 获取第一个事件
                first_key = next(iter(matched_result[0]))
                if len(matched_result[0][first_key]) > 1:
                    if first_key not in first_kleene:
                        first_kleene[first_key] = matched_result[0][first_key][0].timestamp
                    else:
                        # 完成去重
                        if matched_result[0][first_key][0].timestamp > first_kleene[first_key]:
                            son_list = True

                if not self.lazy_model or not son_list:
                    after_match_skip_strategy.prune(partial_matches, matched_result, shared_buffer_accessor)
                    after_match_skip_strategy.prune(nfa_state.get_completed_matches(), matched_result, shared_buffer_accessor)

                    result.append([shared_buffer_accessor.materialize_match(matched_result[0]), matched_result_ex[1]])

            shared_buffer_accessor.release_node(earliest_match.get_previous_buffer_entry(),
                                                earliest_match.get_version())

        # 获取当前的部分匹配列表
        partial_matches_list = nfa_state.get_partial_matches()

        # 使用列表推导式过滤符合条件的元素，并更新列表
        nfa_state.get_partial_matches()[:] = [
            pm for pm in partial_matches_list
            if pm.get_start_event_id() is None or pm in partial_matches
        ]

        table_tool = TableTool()

        mix_limit = 5

        # lazy model情况下，当前计算结果需要展开为最终结果
        if len(result) != 0 and self.lazy_model:
            final_result = []
            if self.lazy_calculate_model == 'DIRECT':
                final_result = []
                for r in result:
                    final_results = TestABCEx.valuate(r)
                    if final_results:
                        final_result.extend(final_results)
            elif self.lazy_calculate_model == 'TABLECALCULATE' or (self.lazy_calculate_model == 'MIXTURE' and len(result) <= mix_limit):
            # elif self.lazy_calculate_model == 'TABLECALCULATE':
                table_calculate = TableCalculate(event_threshold=self.window_time/2, selection_strategy = self.selection_strategy, min_times = self.min_times)
                for r in result:
                    #final_results = table_calculate.evaluate_combinations(r[0], r[1], self.bloom_filter_manager)
                    final_results = table_calculate.evaluate_combinations(r[0], r[1])
                    if final_results:
                        final_result.extend(final_results)
            elif self.lazy_calculate_model == 'INCREMENTCALCULATE' or (self.lazy_calculate_model == 'MIXTURE' and len(result) > mix_limit):
                # print("result_time")
                # begin_time = result[0][0]['A'][0].event.timestamp
                # a = datetime.datetime.utcfromtimestamp(begin_time)
                # end_time = result[0][0]['C'][0].event.timestamp
                # b = datetime.datetime.utcfromtimestamp(end_time)
                # print(a)
                # print(b)

                increment_calculate = IncrementTableCalculate(selection_strategy = self.selection_strategy, min_times = self.min_times)

                # Step 1: 按 latency_constrain 的种类对 result[1] 分组
                lazy_constrains_map_keys = []
                latency_group_map = {}
                result_group_map = {}
                for r in result:
                    lazy_constrains = r[1]
                    lazy_constrains_key = table_tool.ensure_hashable(lazy_constrains)
                    if lazy_constrains_key in lazy_constrains_map_keys:
                        result_group = result_group_map[lazy_constrains_key]
                        result_group.append(r)
                        result_group_map[lazy_constrains_key] = result_group
                    else:
                        latency_group_map[lazy_constrains_key] = lazy_constrains
                        result_group_map[lazy_constrains_key] = [r]
                        lazy_constrains_map_keys.append(lazy_constrains_key)

                # Step 2: 遍历分组，分别调用 increment_calculate.evaluate_combinations
                for latency_constrain_key in lazy_constrains_map_keys:
                    # 提取 basic_results 和 constraints
                    basic_results = [r[0] for r in result_group_map[latency_constrain_key]]
                    constraints = latency_group_map[latency_constrain_key]  # 所有分组的 constraints 应该相同

                    # 调用 evaluate_combinations
                    finale_results = increment_calculate.evaluate_incrementally(basic_results, constraints)
                    if finale_results:
                        final_result.extend(finale_results)
            elif self.lazy_calculate_model == 'GRAPH':
                lazy_handler = LazyHandler()
                lazy_handler.create_calculate_graph()
                while result:
                    basic_result = result.pop(0)
                    lazy_calculate_constrains = []
                    for c in basic_result[1]:
                        if c not in lazy_calculate_constrains:
                            lazy_calculate_constrains.append(c)
                    lazy_handler.expand_calculate_graph(basic_result[0], lazy_calculate_constrains)
                lazy_handler.calculate()
                for final_result in lazy_handler.get_final_results():
                    result.append(final_result)
            result.clear()
            if final_result:
                result.extend(final_result)


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
                if not (self.is_start_state(computation_state) and computation_state.get_start_timestamp() == -1):
                    version = None
                    if self.is_equivalent_state(edge.get_target_state(), self.get_state(computation_state)):
                        to_increase = self.calculate_increasing_self_state(outgoing_edges.total_ignore_branches,
                                                                           outgoing_edges.total_take_branches)
                        version = computation_state.get_version().increase(to_increase)
                    else:
                        version = computation_state.get_version().increase(
                            total_take_to_skip + ignore_branches_to_visit).add_stage()
                        ignore_branches_to_visit -= 1

                    # 将IGNORE边上的lazy_constrain 添加到computation_state中，下次take时使用
                    lazy_constraints = []
                    if self.lazy_model:
                        lazy_constraints = edge.get_condition().lazy_calculate_value_constrain

                        for constrain in computation_state.get_lazy_constraints():
                            if constrain not in lazy_constraints:
                                lazy_constraints.append(constrain)
                    self.add_computation_state(shared_buffer_accessor, resulting_computation_states,
                                               edge.get_target_state(),
                                               computation_state.get_previous_buffer_entry(),
                                               version, computation_state.get_start_timestamp(),
                                               computation_state.get_previous_timestamp(),
                                               computation_state.get_start_event_id(),
                                               lazy_constraints)
                    edge.get_condition().clear_lazy_calculate_value_constrain()

            elif edge.get_action().name == 'TAKE':
                next_state = edge.get_target_state()
                current_state = edge.get_source_state()

                previous_entry = computation_state.get_previous_buffer_entry()

                current_version = computation_state.get_version().increase(take_branches_to_visit)
                next_version = current_version.add_stage()
                take_branches_to_visit -= 1

                # lazy_model 下将当前take边的lazy_constrain 和 计算状态下缓存的 lazy_constrain 合并，保存在sharebuffer的边转移上
                lazy_constraints = []
                for lazy_constraint in computation_state.get_lazy_constraints() + edge.get_condition().lazy_calculate_value_constrain:
                    if lazy_constraint not in lazy_constraints:
                        lazy_constraints.append(lazy_constraint)
                new_entry = shared_buffer_accessor.put(current_state.get_name().split(':')[0], event_wrapper.get_event_id(),
                                                       previous_entry, current_version, lazy_constraints)
                # 添加到sharebuffer edge后清楚lazy_constrain
                edge.get_condition().clear_lazy_calculate_value_constrain()
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
                    if self.lazy_model:
                        self.add_computation_state(shared_buffer_accessor, resulting_computation_states, final_state,
                                               new_entry, next_version, start_timestamp, previous_timestamp,
                                               start_event_id, final_state.lazy_value_constraints)
                    else:
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
                              previous_timestamp: int, start_event_id: EventId, lazy_constraints: List[ValueConstraint] = None):
        computation_state = ComputationState.create_state(current_state.get_name(), previous_entry, version,
                                                          start_timestamp, previous_timestamp, start_event_id, lazy_constraints)
        computation_states.append(computation_state)
        shared_buffer_accessor.lock_node(previous_entry, computation_state.get_version())

    def find_final_state_after_proceed(self, context: ConditionContext, state: State, event):
        states_to_check = [state]
        try:
            while states_to_check:
                current_state = states_to_check.pop()
                for transition in current_state.get_state_transitions():
                    if transition.get_action().name == 'PROCEED' and self.check_filter_condition(context,
                                                                                            transition.get_condition(),
                                                                                            event):
                        if transition.get_target_state().is_final():
                            # 处理 final 的lazy proceed
                            target_state = transition.get_target_state()
                            if self.lazy_model:
                                target_state.add_lazy_constraints(transition.condition.lazy_calculate_value_constrain)
                            return target_state
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

            # 因为Ignore 边不可能单独存在
            # 所以通过对edges进行排序，确保Ignore边在Take边后面，可以在边生成时保证Ignore边在Take边后面
            take_transitions = False
            take_transitions_has_changed = False

            # 存储延迟处理的 value_constrain
            for state_transition in state_transitions:
                try:
                    if self.check_filter_condition(context, state_transition.get_condition(), event):
                        if state_transition.get_action() == StateTransitionAction.PROCEED:
                            target_state = state_transition.get_target_state()
                            #lazy model下，需要将延迟处理的 value_constrain 传递给下一个状态
                            if self.lazy_model:
                                lazy_value_constrains = state_transition.get_condition().lazy_calculate_value_constrain
                                if lazy_value_constrains:
                                    target_state.add_lazy_constraints(lazy_value_constrains)

                            states.append(state_transition.get_target_state())
                        else:
                            # 将该状态下被延迟的lazy_constrain 添加到具体的边转移中
                            if self.lazy_model:
                                if current_state.lazy_value_constraints:
                                    #computation_state.add_lazy_constraints(current_state.lazy_value_constraints)
                                    state_transition.condition.lazy_calculate_value_constrain.extend(current_state.lazy_value_constraints)
                            if state_transition.get_action() == StateTransitionAction.TAKE:
                                # 如果转移状态对象是 kleene_dead, 则不需要判断其他边
                                if 'dead' in state_transition.get_target_state().get_name():
                                    break
                                if not take_transitions_has_changed:
                                    take_transitions = True
                                take_transitions_has_changed = True
                                outgoing_edges.add(state_transition)
                            if state_transition.get_action() == StateTransitionAction.IGNORE:
                                # 需要区别不take的类型，如果是STAM类型，允许忽略除了TAKE TO DEAD 边之外所有的类型
                                if current_state.select_strategy == SelectStrategy.STNM and not take_transitions:
                                    outgoing_edges.add(state_transition)
                                elif current_state.select_strategy == SelectStrategy.STAM:
                                    outgoing_edges.add(state_transition)

                except Exception as e:
                    raise RuntimeError("Failure happened in filter function.") from e

            # 清除state状态中的lazy_constrain
            # current_state.lazy_value_constraints.clear()

        return outgoing_edges

    def is_next_kleene(self, context: ConditionContext, computation_state: ComputationState, event):
        state: State = self.get_state(computation_state)

        states = [state]

        while states:
            current_state = states.pop()
            state_transitions = current_state.get_state_transitions()

            for state_transition in state_transitions:
                try:
                    if self.check_filter_condition(context, state_transition.get_condition(), event):
                        if state_transition.get_action() == StateTransitionAction.PROCEED:
                            states.append(state_transition.get_target_state())
                        if state_transition.get_action() == StateTransitionAction.TAKE:
                            return True

                except Exception as e:
                    raise RuntimeError("Failure happened in filter function.") from e

        return False

    @staticmethod
    def check_filter_condition(context: ConditionContext, condition: Constraint, event):
        return condition is None or condition.validate(event, context)

    @staticmethod
    def extract_current_matches(shared_buffer_accessor: SharedBufferAccessor,
                                computation_state: ComputationState) -> Dict[str, List[EventId]]:
        if shared_buffer_accessor.extract_patterns(computation_state.get_previous_buffer_entry(),computation_state.get_version()) is None\
                or computation_state.get_previous_buffer_entry() is None:
                # or computation_state.get_current_state_name() != computation_state.get_previous_buffer_entry().page_name:
            return {}

        paths = shared_buffer_accessor.extract_patterns(computation_state.get_previous_buffer_entry(),
                                                        computation_state.get_version())[0]
        assert len(paths) == 1

        return paths[0]

    @staticmethod
    def is_equivalent_state(s1: State, s2: State) -> bool:
        return s1.get_name() == s2.get_name()

    @staticmethod
    def is_self_ignore(edge: StateTransition, current_state: State) -> bool:
        return NFA.is_equivalent_state(edge.get_target_state(), current_state) and edge.get_action() == 'IGNORE'

