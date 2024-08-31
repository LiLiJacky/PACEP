# util/draw.py
import graphviz
import re
from interfaces.Drawer import Drawer
import configparser
import os

from nfa.StateTransitionAction import StateTransitionAction

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

# 配置文件路径
config_path = os.path.join(project_root, 'config.ini')

# 初始化配置解析器
config = configparser.ConfigParser()
# 读取配置文件
config.read(config_path)

# 获取配置项的值
output_dir = config.get('Paths', 'output_dir')
png_output = config.get('Paths', 'png_output')

# 生成完整的目录路径
output_dir_path = os.path.join(project_root, output_dir)
png_output_path = os.path.join(output_dir_path, png_output.strip('/'))

# 确保路径存在
os.makedirs(output_dir_path, exist_ok=True)
os.makedirs(png_output_path, exist_ok=True)


class NFADrawer(Drawer):
    def __init__(self):
        super().__init__()
        self.graph = graphviz.Digraph(comment='NFA')
        self.state_count = 0

    def _add_state(self):
        """
        添加一个状态到NFA中。
        :return: 新状态的名称
        """
        state_name = f"q{self.state_count}"
        self.graph.node(state_name, state_name)
        self.state_count += 1
        return state_name

    def draw(self, pattern):
        """
        将正则表达式绘制成NFA。
        :param pattern: 输入的正则表达式
        """
        tokens = pattern.split()
        start_state = self._add_state()
        current_state = start_state

        for token in tokens:
            if len(token) == 1:  # Simple variable
                next_state = self._add_state()
                self.graph.edge(current_state, next_state, label=token)
                current_state = next_state
            elif len(token) > 1:  # Variable with quantifier
                var, quantifier = token[0], token[1:]
                if quantifier == '*':
                    next_state = self._add_state()
                    self.graph.edge(current_state, next_state, label=var)
                    self.graph.edge(next_state, next_state, label=var)
                    self.graph.edge(current_state, next_state, label='ε')
                    current_state = next_state
                elif quantifier == '+':
                    next_state = self._add_state()
                    self.graph.edge(current_state, next_state, label=var)
                    self.graph.edge(next_state, next_state, label=var)
                    current_state = next_state
                elif quantifier == '?':
                    next_state = self._add_state()
                    self.graph.edge(current_state, next_state, label=var)
                    self.graph.edge(current_state, next_state, label='ε')
                    current_state = next_state
                elif re.match(r'{\d+}', quantifier):
                    n = int(quantifier[1:-1])
                    for _ in range(n):
                        next_state = self._add_state()
                        self.graph.edge(current_state, next_state, label=var)
                        current_state = next_state
                elif re.match(r'{\d+,}', quantifier):
                    n = int(quantifier[1:-2])
                    for _ in range(n):
                        next_state = self._add_state()
                        self.graph.edge(current_state, next_state, label=var)
                        current_state = next_state
                    self.graph.edge(current_state, current_state, label=var)
                elif re.match(r'{\d+,\d+}', quantifier):
                    n, m = map(int, quantifier[1:-1].split(','))
                    for i in range(n):
                        next_state = self._add_state()
                        self.graph.edge(current_state, next_state, label=var)
                        current_state = next_state
                    for i in range(n, m):
                        next_state = self._add_state()
                        self.graph.edge(current_state, next_state, label=var)
                        self.graph.edge(current_state, next_state, label='ε')
                        current_state = next_state
                elif re.match(r'{,\d+}', quantifier):
                    m = int(quantifier[2:-1])
                    final_state = current_state
                    for i in range(m):
                        next_state = self._add_state()
                        self.graph.edge(current_state, next_state, label=var)
                        current_state = next_state
                        if i == m - 1:
                            self.graph.edge(current_state, final_state, label='ε')

        final_state = self._add_state()
        self.graph.edge(current_state, final_state, label='ε')

        return self.graph

    def render(self, filename='png', format='png'):
        """
        渲染并保存图形。
        :param filename: 输出文件名
        :param format: 输出文件格式
        """
        output_path = os.path.join(png_output_path, filename)
        self.graph.render(filename=output_path, format=format, cleanup=True)


class StateDrawer:
    def __init__(self):
        self.graph = graphviz.Digraph(comment='State Diagram')

    def add_state(self, name, state_type):
        """
           添加一个状态到图中。
           :param name: 状态名
           :param state_type: 状态类型（Start, Normal, Final）
           """
        shape = 'doublecircle' if state_type == 'Final' else 'circle'

        # 特别处理 Start 状态
        # if state_type == 'Start':
        #     self.graph.node(name, name, shape=shape, _attributes={'style': 'filled', 'color': 'lightblue'})
        #     # 添加一个隐式的起始节点，表示小箭头
        #     self.graph.node('start', '', shape='point', width='0.1')
        #     self.graph.edge('start', name)
        # else:
        #     self.graph.node(name, name, shape=shape)

        self.graph.node(name, name, shape=shape)

    def add_edge(self, source, target, label, style='solid', arrowhead='normal'):
        """
        添加一个边到图中。
        :param source: 源状态
        :param target: 目标状态
        :param label: 边的标签
        """
        self.graph.edge(source, target, label=label, _attributes={'style': style, 'arrowhead': arrowhead})

    def draw(self, states):
        """
        根据给定的状态生成图，包括状态及其转换。
        :param states: 状态列表
        """
        for state in states:
            self.add_state(state.get_name(), state.get_state_type())

            # 处理每个状态的转换
            for transition in state.get_state_transitions():
                source = transition.get_source_state().get_name()
                target = transition.get_target_state().get_name()

                # 处理多个条件约束
                label_parts = []
                if transition.get_condition():
                    constraint_collection = transition.get_condition()

                    if constraint_collection.value_constrain:
                        label_parts.append(
                            "Value: " + ", ".join(c.expression for c in constraint_collection.value_constrain))

                    if constraint_collection.time_constrain:
                        label_parts.append("Time: " + ", ".join(
                            c.expression for c in
                            constraint_collection.time_constrain))

                    if constraint_collection.count_constrain:
                        label_parts.append("Count: " + ", ".join(
                            c.expression for c in
                            constraint_collection.count_constrain))

                    if constraint_collection.type_constrain:
                        label_parts.append(
                            "Type: " + ", ".join(f"{c.variables_name}" for c in constraint_collection.type_constrain))

                label = "\n".join(label_parts) if label_parts else ''

                # 根据转换的操作类型设置边的样式
                if transition.get_action() == StateTransitionAction.IGNORE:
                    self.add_edge(source, target, label, style='dashed')
                elif transition.get_action() == StateTransitionAction.PROCEED:
                    self.add_edge(source, target, label, style='dashed', arrowhead='diamond')
                else:
                    self.add_edge(source, target, label)

        return self.graph

    def render(self, name, output_path):
        """
        渲染并保存图形。
        :param output_path: 输出路径
        :param format: 输出格式
        """
        # 设置输出路径
        output_path = os.path.join(output_path, name)
        print("bihao"+output_path)
        self.graph.render(filename=output_path, format='png', cleanup=True)

# 使用示例
if __name__ == "__main__":
    pattern = "A B C+ D* E{2,4} F{3,}"
    drawer = NFADrawer()
    nfa_graph = drawer.draw(pattern)
    drawer.render('nfa_state_diagram')