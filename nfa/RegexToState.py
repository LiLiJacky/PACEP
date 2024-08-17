import configparser
import os
from typing import List, Dict, Any
import re
from collections import deque

from generator.RegexPatternGenerator import RegexPatternGenerator
from models.CountConstraint import CountConstraint
from models.TimeConstarint import TimeConstraint
from models.ValueConstraint import ValueConstraint
from nfa.State import State, StateType
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction
from util.NFADrawer import StateDrawer


class RegexToState:
    def __init__(self, regex: str, constraints: Dict[str, Any]):
        self.regex = regex
        self.constraints = constraints
        self.states = []
        self.transitions = []
        self.start_state = None
        self.final_state = None

    def generate_states_and_transitions(self):
        # Parse regex and generate states and transitions
        parsed_states = self._parse_regex(self.regex)
        self._create_states_and_transitions(parsed_states)

        # Apply constraints to transitions
        self._apply_constraints()

    def _parse_regex(self, regex: str) -> List[tuple]:
        pattern = re.compile(r'([A-Z])(\{(\d*)(,(\d+))?\}|\*|\+|\?)?')
        matches = pattern.findall(regex.replace(' ', ''))
        parsed_states = []
        for match in matches:
            state = match[0]
            if match[1] == '+':
                min_repeat = 1
                max_repeat = float('inf')
            elif match[1] == '*':
                min_repeat = 0
                max_repeat = float('inf')
            elif match[1] == '?':
                min_repeat = 0
                max_repeat = 1
            elif '{' in match[1]:
                min_repeat = int(match[2]) if match[2] else 0  # 处理下界为空的情况
                if ',' in match[1]:
                    max_repeat = int(match[4]) if match[4] else float('inf')  # 处理只有上界的情况
                else:
                    max_repeat = min_repeat  # 处理单个数字的情况，如 {3}
            else:
                min_repeat = max_repeat = 1
            parsed_states.append((state, min_repeat, max_repeat))
        return parsed_states

    def _create_states_and_transitions(self, parsed_states: List[tuple]):
        # 创建开始状态
        self.start_state = State("Start", StateType.Start)
        self.states.append(self.start_state)

        # 初始化队列，存储 (current_state, edge_type, previous_state)
        queue = deque([(self.start_state, StateTransitionAction.TAKE)])

        for state, min_repeat, max_repeat in parsed_states:
            next_queue = deque()

            if max_repeat == 1 and min_repeat == 1:  # 普通状态
                current_state = State(f"{state}", StateType.Normal)
                self.states.append(current_state)
                # 更新队列中的所有previous指针，添加边
                for previous_state, edge_type in queue:
                    self._add_transition(previous_state, edge_type, current_state)
                self._add_transition(current_state, StateTransitionAction.IGNORE, current_state)
                next_queue.append((current_state, StateTransitionAction.TAKE))

            elif max_repeat == 1 and min_repeat == 0:  # 选择状态
                current_state = State(f"{state}", StateType.Normal)
                self.states.append(current_state)
                original_queue = list(queue)

                # 原来的previous不变
                for previous_state, edge_type in original_queue:
                    next_queue.append((previous_state, edge_type))

                # 为选择状态创建新的previous并添加边
                for previous_state, edge_type in original_queue:
                    self._add_transition(previous_state, edge_type, current_state)
                    self._add_transition(current_state, StateTransitionAction.IGNORE, current_state)
                    next_queue.append((current_state, StateTransitionAction.TAKE))

            elif max_repeat == float('inf'):  # 处理无上界状态
                if min_repeat == 0:  # K* 展开为 K[i]
                    kleene_state = State(f"{state}[i]", StateType.Normal)
                    self.states.append(kleene_state)
                    for previous_state, edge_type in queue:
                        self._add_transition(previous_state, edge_type, kleene_state)
                    self._add_transition(kleene_state, StateTransitionAction.TAKE, kleene_state)
                    self._add_transition(kleene_state, StateTransitionAction.IGNORE, kleene_state)
                    next_queue = deque([(kleene_state, StateTransitionAction.PROCEED)])

                elif min_repeat >= 1:  # 处理 K+、K{2,} 等无上界状态
                    previous_state = None
                    for i in range(1, min_repeat + 1):
                        current_state = State(f"{state}[{i}]", StateType.Normal)
                        self.states.append(current_state)

                        if i == 1:
                            for prev_state, edge_type in queue:
                                self._add_transition(prev_state, edge_type, current_state)
                        else:
                            self._add_transition(previous_state, StateTransitionAction.TAKE, current_state)
                        self._add_transition(current_state, StateTransitionAction.IGNORE, current_state)
                        previous_state = current_state

                    kleene_state = State(f"{state}[i]", StateType.Normal)
                    self.states.append(kleene_state)
                    self._add_transition(previous_state, StateTransitionAction.TAKE, kleene_state)
                    self._add_transition(kleene_state, StateTransitionAction.TAKE, kleene_state)
                    self._add_transition(kleene_state, StateTransitionAction.IGNORE, kleene_state)
                    next_queue = deque([(kleene_state, StateTransitionAction.PROCEED)])

            elif isinstance(max_repeat, int):  # 处理上下界都存在的状态
                previous_state = None
                for i in range(1, max_repeat + 1):
                    current_state = State(f"{state}[{i}]", StateType.Normal)
                    self.states.append(current_state)

                    if i == 1:
                        for prev_state, edge_type in queue:
                            self._add_transition(prev_state, edge_type, current_state)
                    else:
                        self._add_transition(previous_state, StateTransitionAction.TAKE, current_state)

                    self._add_transition(current_state, StateTransitionAction.IGNORE, current_state)

                    if i >= min_repeat:
                        next_queue.append((current_state, StateTransitionAction.TAKE))

                    previous_state = current_state

                # 如果下界为0，保留原有的previous指针
                if min_repeat == 0:
                    for prev_state, edge_type in queue:
                        next_queue.append((prev_state, edge_type))

            queue = next_queue

        # 创建终止状态
        self.final_state = State("Final", StateType.Final)
        self.states.append(self.final_state)

        # 处理剩余的previous状态指向终止状态的边
        for previous_state, edge_type in queue:
            if edge_type == StateTransitionAction.PROCEED:
                self._add_transition(previous_state, edge_type, self.final_state)
            else:
                self._add_transition(previous_state, StateTransitionAction.TAKE, self.final_state)

    def _add_transition(self, source_state: State, action: StateTransitionAction, target_state: State):
        # 检查是否已经存在相同的边
        for transition in source_state.get_state_transitions():
            if (transition.get_source_state() == source_state and
                    transition.get_action() == action and
                    transition.get_target_state() == target_state):
                return  # 已经存在，不添加

        # 创建并添加状态转换
        transition = StateTransition(source_state, action, target_state, None)
        source_state.get_state_transitions().append(transition)
        self.transitions.append(transition)

    def _apply_constraints(self):
        for value_constraint in self.constraints.get('value_constrain', []):
            constraint = ValueConstraint(value_constraint['variables'], value_constraint['expression'])
            last_var = value_constraint['variables'][-1]
            for transition in self.transitions:
                if last_var in transition.get_source_state().get_name() and transition.get_action() in [
                    StateTransitionAction.TAKE, StateTransitionAction.PROCEED]:
                    transition.set_condition(constraint)

        for time_constraint in self.constraints.get('time_constrain', []):
            constraint = TimeConstraint(time_constraint['variables'], time_constraint['min_time'],
                                        time_constraint['max_time'])
            last_var = time_constraint['variables'][-1]
            for transition in self.transitions:
                if last_var in transition.get_source_state().get_name() and transition.get_action() in [
                    StateTransitionAction.TAKE, StateTransitionAction.PROCEED]:
                    transition.set_condition(constraint)

        for count_constraint in self.constraints.get('count_constrain', []):
            constraint = CountConstraint(count_constraint['variables'], count_constraint['min_count'],
                                         count_constraint['max_count'])
            last_var = count_constraint['variables'][-1]
            for transition in self.transitions:
                if last_var in transition.get_source_state().get_name() and transition.get_action() in [
                    StateTransitionAction.TAKE, StateTransitionAction.PROCEED]:
                    transition.set_condition(constraint)

    def draw(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

        config_path = os.path.join(project_root, 'config.ini')

        config = configparser.ConfigParser()
        config.read(config_path)

        output_dir = config.get('Paths', 'output_dir')
        png_output = config.get('Paths', 'png_output')

        output_dir_path = os.path.join(project_root, output_dir)
        png_output_path = os.path.join(output_dir_path, png_output.strip('/'))

        os.makedirs(output_dir_path, exist_ok=True)
        os.makedirs(png_output_path, exist_ok=True)
        drawer = StateDrawer()
        graph = drawer.draw(self.states, self.transitions)
        graph.render(self.regex + "_state", png_output_path)

    def run_tests(self):
        for transition in self.transitions:
            print(
                f"Transition from {transition.get_source_state().get_name()} to {transition.get_target_state().get_name()} with condition {transition.get_condition()}")


if __name__ == "__main__":
    generator = RegexPatternGenerator()

    num_variables = 5
    num_kleene = 2
    num_value_constrains = 2
    num_time_constrains = 2
    num_count_constrains = 0

    regex = generator.generate_regex(num_variables, num_kleene)
    print(f"Generated Regex: {regex}")

    result = generator.set_constrains(regex, num_value_constrains, num_count_constrains, num_time_constrains)
    variables_domain = result.get("domain")
    variables_value_constrain = result.get("constrains")
    print(f"Generated Constrain: {variables_value_constrain}")

    test = RegexToState(regex, variables_value_constrain)
    test.generate_states_and_transitions()
    test.run_tests()
    test.draw()