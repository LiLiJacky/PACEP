import configparser
import os
from typing import List, Dict, Any
import re
from collections import deque

from generator.RegexPatternGenerator import RegexPatternGenerator
from models.ConstraintCollection import ConstraintCollection
from models.CountConstraint import CountConstraint
from models.TimeConstarint import TimeConstraint
from models.TypeConstraint import TypeConstraint
from nfa.State import State, StateType
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction
from util.NFADrawer import StateDrawer


class RegexToState:
    def __init__(self, regex: str, constraints: ConstraintCollection, contiguity_strategy):
        self.regex = regex
        self.constraints = constraints
        self.states = []
        self.start_state = None
        self.final_state = None
        self.contiguity_strategy = contiguity_strategy

    def generate_states_and_transitions(self):
        # Parse regex and generate states and transitions
        parsed_states = self._parse_regex(self.regex)
        self._create_states_and_transitions(parsed_states)


    def _parse_regex(self, regex: str) -> List[tuple]:
        # 修改正则表达式模式，以正确匹配 {,2}, {2,}, {2,4} 等情况
        pattern = re.compile(r'([A-Z])(\{(\d*)?(,(\d*))?\}|\*|\+|\?)?')
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
        # 创建终止状态提前
        self.final_state = State("Final", StateType.Final)
        self.states.append(self.final_state)

        # 初始化队列，存储 (current_state, edge_type, next_state)
        queue = deque([self.final_state])

        # 从后向前处理 parsed_states
        for state, min_repeat, max_repeat in reversed(parsed_states):
            next_queue = deque()

            current_state_type = StateType.Normal

            isKleene = False if max_repeat == 1 and min_repeat == 1 else True
            current_state = State(f"{state}", current_state_type) if not isKleene else State(f"{state}[i]", current_state_type)
            current_action = StateTransitionAction.TAKE if not isKleene else StateTransitionAction.PROCEED
            self.states.insert(0, current_state)  # 将状态插入到列表的开头
            # 更新队列中的所有 next_state 指针，添加边
            for next_state in queue:
                self._add_transition(current_state, current_action, next_state)
            # 判断当前状态的连续策略
            if not state in self.contiguity_strategy['STRICT']:
                self._add_transition(current_state, StateTransitionAction.IGNORE, current_state)
            # Kleene
            if isKleene:
                # 添加自转移take边
                self._add_transition(current_state, StateTransitionAction.TAKE, current_state)
                # 根据模式符号添加 count_constrain
                times_constraint = CountConstraint([state], min_count=min_repeat, max_count=max_repeat)
                self.constraints.add_constraint(times_constraint)

            next_queue.append(current_state)
            queue = next_queue

        # 处理剩余的 next_state，将其类型设置为 Start
        for next_state in queue:
            next_state.state_type = StateType.Start  # 修改状态类型为 Start

        # Apply constraints to transitions
        self._apply_constraints()

    def _add_transition(self, source_state: State, action: StateTransitionAction, target_state: State):
        # 检查是否已经存在相同的边
        for transition in source_state.get_state_transitions():
            if (transition.get_source_state() == source_state and
                    transition.get_action() == action and
                    transition.get_target_state() == target_state):
                return  # 已经存在，不添加

        # 创建并添加状态转换
        transition = StateTransition(source_state, action, target_state, ConstraintCollection())
        source_state.get_state_transitions().append(transition)

    def _apply_constraints(self):
        # 处理 value_constraints
        for value_constraint in self.constraints.value_constrain:
            last_var = value_constraint.variables[-1]
            base_last_var = last_var.split('[')[0]  # 提取基础变量名
            for state in self.states:
                for transition in state.get_state_transitions():
                    source_name = transition.get_source_state().get_name()
                    base_source_name = source_name.split('[')[0]
                    traget_name = transition.get_target_state().get_name()

                    if source_name == traget_name:
                        # 处理自转移逻辑
                        if (len(value_constraint.variables) == 1 and '[i]' in value_constraint.expression
                                and base_source_name == base_last_var and transition.get_action() == StateTransitionAction.TAKE):
                            transition.add_condition(value_constraint)
                        continue

                    if base_source_name == base_last_var and transition.get_action() in [
                        StateTransitionAction.TAKE, StateTransitionAction.PROCEED]:
                        if len(value_constraint.variables) == 1 and '[i]' in value_constraint.expression:
                            continue
                        transition.add_condition(value_constraint)

        # 处理 time_constraints
        for time_constraint in self.constraints.time_constrain:
            last_var = time_constraint.variables[-1]  # 使用最后一个变量
            max_time = time_constraint.max_time
            first_var = time_constraint.variables[0]  # 使用最后一个变量

            for state in self.states:
                for transition in state.get_state_transitions():
                    source_name = transition.get_source_state().get_name()
                    base_source_name = source_name.split('[')[0]
                    base_first_var = first_var.split('[')[0]
                    base_last_var2 = last_var.split('[')[0]

                    if self.regex.index(base_first_var) < self.regex.index(base_source_name) <= self.regex.index(base_last_var2):
                        current_variables = [first_var, source_name]
                        if transition.get_action() == StateTransitionAction.IGNORE:
                            new_expression = f"0 <= time - {first_var}.time <= {max_time}"
                            current_variables = [first_var, "NOW.EVENT"]
                        else:
                            new_expression = f"0 <= {base_source_name} - {first_var}.time <= {max_time}"

                        new_time_constraint = TimeConstraint(current_variables, 0, max_time)
                        new_time_constraint.expression = new_expression
                        transition.add_condition(new_time_constraint)

        # 忽略 count_constraints（根据最新要求）

        # 处理 type_constraints
        for type_constraint in self.constraints.type_constrain:
            last_var = type_constraint.variables[-1]
            base_last_var = last_var.split('[')[0]  # 提取基础变量名
            for state in self.states:
                for transition in state.get_state_transitions():
                    if transition.get_action == StateTransitionAction.PROCEED:
                        # Proceed 边只需要考虑value constrain
                        continue
                    source_name = transition.get_source_state().get_name()
                    base_source_name = source_name.split('[')[0]

                    if base_source_name == base_last_var and transition.get_action() in [
                        StateTransitionAction.TAKE]:
                        transition.add_condition(type_constraint)

        # 处理 count_constraints
        for count_constraint in self.constraints.count_constrain:
            last_name = count_constraint.variables[0]
            for state in self.states:
                for transition in state.get_state_transitions():
                    source_name = transition.get_source_state().get_name()
                    base_source_name = source_name.split('[')[0]

                    if base_source_name == last_name and transition.get_action() == StateTransitionAction.PROCEED:
                        # 创建一个新的 TimesConstraint 并添加到 transition 中
                        new_times_constraint = CountConstraint([last_name], min_count=count_constraint.min_count,
                                                               max_count=count_constraint.max_count)
                        transition.add_condition(new_times_constraint)

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
        graph = drawer.draw(self.states)
        graph.render(self.regex + "_state", png_output_path)

    def run_tests(self):
        for state in self.states:
            for transition in state.get_state_transitions():
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

    result = generator.set_constraints(regex, num_value_constrains, num_count_constrains, num_time_constrains)
    variables_domain = result.get("domain")
    variables_constraints_collection = result.get("constraints")
    variables_constraints_collection.window_constrain_type = "window_constrain"
    print(f"Generated Constrain: {variables_constraints_collection}")

    test = RegexToState(regex, variables_constraints_collection, contiguity_strategy={'STRICT': []})
    test.generate_states_and_transitions()
    test.draw()
    test.run_tests()