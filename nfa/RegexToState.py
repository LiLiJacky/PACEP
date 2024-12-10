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
from nfa.SelectStrategy import SelectStrategy
from nfa.State import State, StateType
from nfa.StateTransition import StateTransition
from nfa.StateTransitionAction import StateTransitionAction
from util.NFADrawer import StateDrawer


class RegexToState:
    def __init__(self, regex: str, constraints: ConstraintCollection, contiguity_strategy, lazy_model):
        self.regex = regex
        self.constraints = constraints
        self.states = []
        self.start_state = None
        self.final_state = None
        self.contiguity_strategy = contiguity_strategy
        self.kleene_state = []
        self.lazy_model = lazy_model

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
            is_kleene = True
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
                is_kleene = False

            parsed_states.append((state, min_repeat, max_repeat))
            if is_kleene:
                self.kleene_state.append(state)
                if self.lazy_model and state in self.contiguity_strategy['NONDETERMINISTICRELAXED']:
                    self.contiguity_strategy['NONDETERMINISTICRELAXED'].remove(state)
                    self.contiguity_strategy['RELAXED'].add(state)

        return parsed_states

    def _create_states_and_transitions(self, parsed_states: List[tuple]):
        # 创建终止状态提前
        self.final_state = State("Final", StateType.Final, SelectStrategy.SC)
        self.states.append(self.final_state)

        # 初始化队列，存储 (current_state, edge_type, next_state)
        queue = deque([self.final_state])

        # 从后向前处理 parsed_states
        for state, min_repeat, max_repeat in reversed(parsed_states):
            next_queue = deque()

            current_state_type = StateType.Normal

            # 默认STRICT
            is_strict_contiguity = SelectStrategy.SC
            # 选择策略
            if state in self.contiguity_strategy['STRICT']:
                is_strict_contiguity = SelectStrategy.SC
            elif state in self.contiguity_strategy['RELAXED']:
                is_strict_contiguity = SelectStrategy.STNM
            elif state in self.contiguity_strategy['NONDETERMINISTICRELAXED']:
                is_strict_contiguity = SelectStrategy.STAM

            # 判断是否为Kleene状态
            is_infinity_kleene = False if max_repeat == 1 and min_repeat == 1 else True
            current_state = State(f"{state}", current_state_type, is_strict_contiguity)
            current_action = StateTransitionAction.TAKE if not is_infinity_kleene else StateTransitionAction.PROCEED
            self.states.insert(0, current_state)  # 将状态插入到列表的开头
            # 更新队列中的所有 next_state 指针，添加边
            for next_state in queue:
                self._add_transition(current_state, current_action, next_state)
            # 判断当前状态的连续策略
            if is_strict_contiguity != SelectStrategy.SC and not is_infinity_kleene:
                self._add_transition(current_state, StateTransitionAction.IGNORE, current_state)
            # Kleene
            if is_infinity_kleene:
                # 添加自转移take边
                self._add_transition(current_state, StateTransitionAction.TAKE, current_state)

                # Strict 策略下不允许忽略
                if is_strict_contiguity != SelectStrategy.SC:
                    # 添加向0状态转移的Ignore边
                    zero_state = State(f"{state}:0", StateType.Normal, is_strict_contiguity)
                    self.states.insert(0, zero_state)
                    self._add_transition(current_state, StateTransitionAction.IGNORE, zero_state)
                    # 0状态到infinity状态的take边
                    self._add_transition(zero_state, StateTransitionAction.TAKE, current_state)
                    # 添加0状态的子忽略边
                    self._add_transition(zero_state, StateTransitionAction.IGNORE, zero_state)

                # 添加最小次数基础状态
                for i in range(1, min_repeat + 1):
                    index_kleene_state = State(f"{state}:{i}", current_state_type, is_strict_contiguity)
                    self.states.insert(0, index_kleene_state)
                    # 添加kleene基础状态 i -> i - 1 take边
                    self._add_transition(index_kleene_state, StateTransitionAction.TAKE, current_state)
                    if is_strict_contiguity != SelectStrategy.SC:
                        self._add_transition(index_kleene_state, StateTransitionAction.IGNORE, index_kleene_state)

                    current_state = index_kleene_state


                # 根据模式符号添加 count_constrain
                times_constraint = CountConstraint([state], min_count=0, max_count=max_repeat - max_repeat)
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
            base_last_var = last_var.split('[')[0]
            variable_nums = len(value_constraint.variables)
            for state in self.states:
                for transition in state.get_state_transitions():
                    source_name = transition.get_source_state().get_name()

                    # 如果当前边无关，跳过
                    if base_last_var not in source_name or transition.get_action() == StateTransitionAction.IGNORE:
                        continue

                    # 如果是非Kleene事件
                    if '[i]' not in last_var and last_var not in self.kleene_state:
                        transition.add_condition(value_constraint)
                        continue

                    # 如果是 K[i] > last(K) 模式
                    if '[i]' in value_constraint.expression and '(' in value_constraint.expression and transition.get_action() == StateTransitionAction.TAKE:
                        # 有算法第一个状态不加限制, B[i] > last(B)格式
                        index = self.states.index(state)
                        if index == 0 or base_source_name not in self.states[index - 1].get_name():
                            continue
                        # 将其变为 algorithm的形式，添加在proceed边上
                        if self.lazy_model:
                            #TODO: Implement lazy model handling logic here
                            continue
                        transition.add_condition(value_constraint)


                    # 如果是Kleene事件，且包含多个变量，只需要处理infinity_kleene proceed逻辑
                    if (variable_nums != 1 or '[i]' not in value_constraint.expression) and transition.get_action() == StateTransitionAction.PROCEED:
                        transition.add_condition(value_constraint)
                        continue

                    base_source_name = source_name.split(':')[0]
                    # 如果针对单个Kleene事件
                    if variable_nums == 1 and transition.get_action() == StateTransitionAction.TAKE:
                        # B[i] > 10 格式
                        if '[' in last_var:
                            transition.add_condition(value_constraint)
                            continue

        # 处理 time_constraints
        for time_constraint in self.constraints.time_constrain:
            last_var = time_constraint.variables[-1]  # 使用最后一个变量
            max_time = time_constraint.max_time
            first_var = time_constraint.variables[0]  # 使用第一个变量

            for state in self.states:
                for transition in state.get_state_transitions():
                    if transition.get_action() == StateTransitionAction.PROCEED:
                        continue
                    source_name = transition.get_source_state().get_name()
                    base_source_name = source_name.split(':')[0]

                    if self.regex.index(first_var) < self.regex.index(base_source_name) <= self.regex.index(last_var):
                        current_variables = [first_var, source_name]
                        if transition.get_action() == StateTransitionAction.IGNORE:
                            new_expression = f"0 <= time - {first_var}.time <= {max_time}"
                            current_variables = [first_var, "NOW.EVENT"]
                        else:
                            new_expression = f"0 <= {source_name} - {first_var}.time <= {max_time}"

                        new_time_constraint = TimeConstraint(current_variables, 0, max_time)
                        new_time_constraint.expression = new_expression
                        transition.add_condition(new_time_constraint)

        # 忽略 count_constraints（根据最新要求）

        # 处理 type_constraints
        for type_constraint in self.constraints.type_constrain:
            last_var = type_constraint.variables[-1]
            for state in self.states:
                for transition in state.get_state_transitions():
                    if transition.get_action == StateTransitionAction.PROCEED or transition.get_action == StateTransitionAction.IGNORE:
                        # Proceed 边只需要考虑value constrain
                        continue
                    source_name = transition.get_source_state().get_name()

                    if last_var not in source_name:
                        continue

                    # 处理Take边
                    if transition.get_action() == StateTransitionAction.TAKE:
                        transition.add_condition(type_constraint)
                        continue

                    base_source_name = source_name.split(':')[0]
                    # 处理 Ignore边 在识别策略 RELAXED
                    if transition.get_action() == StateTransitionAction.IGNORE:
                        continue
                        # IGNORE 的逻辑修改，只与是否take有关
                        # if base_source_name in self.contiguity_strategy['RELAXED']:
                        #     new_name = []
                        #     for name in type_constraint.variables:
                        #         new_name.append("_not_" + name)
                        #     not_type_constraint = TypeConstraint(new_name, type_constraint.variables_name)
                        #     transition.add_condition(not_type_constraint)

        # 处理 count_constraints
        for count_constraint in self.constraints.count_constrain:
            last_name = count_constraint.variables[0]
            for state in self.states:
                for transition in state.get_state_transitions():
                    if transition.get_action() == StateTransitionAction.PROCEED or transition.get_action() == StateTransitionAction.IGNORE:
                        continue
                    source_name = transition.get_source_state().get_name()
                    base_source_name = source_name.split(':')[0]

                    if base_source_name == last_name and transition.get_action() == StateTransitionAction.PROCEED:
                        # 创建一个新的 TimesConstraint 并添加到 transition 中
                        new_times_constraint = CountConstraint([last_name], min_count=count_constraint.min_count,
                                                               max_count=count_constraint.max_count)
                        transition.add_condition(new_times_constraint)

    def add_lazy_handle(self):
        kleene_dead_state = []
        kleene_dead_state_name = []
        transition_to_dead = []
        for state in self.states:
            state_name = state.name

            # 如果当前事件的take边有包含Kleene事件的算法计算约束，将该约束转移到lazy_calculate_value_constrain
            for edge in state.get_state_transitions():
                if edge.action == StateTransitionAction.TAKE or edge.action == StateTransitionAction.PROCEED:
                    has_kleene_algorithm = False
                    for constraint in edge.condition.value_constrain:
                        if '(' in constraint.expression:
                            edge.condition.value_constrain.remove(constraint)
                            edge.condition.lazy_calculate_value_constrain.append(constraint)
                            has_kleene_algorithm = True
                    if has_kleene_algorithm:
                        continue

            # 如果当前事件是Kleene事件,下一事件不允许跳过当前Kleene事件
            if state_name in self.kleene_state:
                # 获取当前状态的take边的约束
                take_condition = ConstraintCollection()
                for edge in state.get_state_transitions():
                    if edge.action == StateTransitionAction.TAKE:
                        take_condition = edge.condition
                        break

                for edge in state.get_state_transitions():
                    if edge.action == StateTransitionAction.PROCEED and edge.target_state.name != "Final":
                        if len(state.state_transitions[1].condition.value_constrain) != 0:
                            # 当存在一致性约束时，通过添加向dead状态转移的不确定take边实现非类型状态转移
                            dead_state_name = f"{state_name}:dead"
                            after_kleene_state = edge.target_state
                            if dead_state_name not in kleene_dead_state_name:
                                dead_state = State(dead_state_name, StateType.Normal, SelectStrategy.SC)
                                kleene_dead_state.append([dead_state, self.states.index(after_kleene_state) + 1])
                                kleene_dead_state_name.append(dead_state_name)
                            transition_to_dead.append([after_kleene_state, StateTransitionAction.TAKE, dead_state_name, take_condition])
                        else:
                            new_name = []
                            for type in state.state_transitions[1].condition.type_constrain[0].variables:
                                new_name.append("_not_" + type)
                            not_type_constraint = TypeConstraint(new_name,
                                                                 state.state_transitions[1].condition.type_constrain[
                                                                     0].variables_name)
                            for tedge in edge.target_state.get_state_transitions():
                                if tedge.action == StateTransitionAction.IGNORE:
                                    tedge.add_condition(not_type_constraint)



        kleene_dead_state_list = {}
        # 创建 kleene_dead_state
        for dead_state, index in kleene_dead_state:
            self.states.insert(index, dead_state)
            kleene_dead_state_list[dead_state.name] = dead_state

        # 创建 transition_to_dead
        for after_kleene_state, action, dead_state_name, take_condition in transition_to_dead:
            transition = StateTransition(after_kleene_state, action, kleene_dead_state_list[dead_state_name], take_condition)
            # 将dead状态添加到after_kleene_state的转移中的最前面
            after_kleene_state.state_transitions.insert(0, transition)

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