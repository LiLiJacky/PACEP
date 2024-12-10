import importlib
from collections import defaultdict
import re

class LazyHandlerWithShare:
    def __init__(self):
        self.basic_results = []  # 存放待求解的basic_result
        self.state_event_map = defaultdict(list)  # 每个state对应的最大连续事件list
        self.constraints = []  # 存储需要满足的约束条件
        self.result_cache = {}  # hash表，存储计算结果
        self.final_results = []  # 存储最终计算结果
        self.state_algorithms = defaultdict(list)  # 存储每个state需要计算的算法

    def initialize_graph(self):
        """
        初始化计算图，清空当前存储的待处理数据。
        """
        self.basic_results = []
        self.state_event_map.clear()
        self.constraints = []
        self.final_results = []
        self.state_algorithms.clear()

    def add_to_graph(self, result_data, constraint_list):
        """
        扩展计算图，添加待计算的结果和约束条件，并解析算法。
        """
        state, events = result_data

        # 检查是否为重复的state
        self.basic_results.append(result_data)

        # 添加新的约束并解析算法
        for constraint in constraint_list:
            if constraint not in self.constraints:
                self.constraints.append(constraint)
                self._parse_and_store_algorithms(constraint)

        # 更新state对应的最大事件列表
        for s, e_list in zip(state, events):
            if len(e_list) > len(self.state_event_map[s]):
                self.state_event_map[s] = e_list

    def process_graph(self):
        """
        处理计算图，基于存储的basic_results和constraints计算最终结果。
        """
        while self.basic_results:
            # 按计算复杂度排序，优先处理计算代价较小的结果
            self.basic_results.sort(key=lambda x: self._calculate_total_complexity(x[0]))
            current_result = self.basic_results.pop(0)
            state, event_list = current_result

            # 验证当前state是否满足所有相关约束
            if not all(
                self._validate_constraint(constraint)
                for constraint in self.constraints
                if set(constraint["variables"]).issubset(set(state))
            ):
                continue

            # 计算当前state的结果
            result_hash = hash(tuple(state))
            if result_hash not in self.result_cache:
                computed_result = self._compute(state, event_list)
                self.result_cache[result_hash] = computed_result

            # 将结果添加到最终结果列表中
            self.final_results.append((state, computed_result))

    def get_results(self):
        """
        返回所有处理完成的最终结果。
        """
        return self.final_results

    def _parse_and_store_algorithms(self, constraint):
        """
        解析约束表达式中的算法并按state存储。
        """
        # 提取 xxx(A) 形式的算法
        pattern = r"([a-zA-Z_]+)\(([a-zA-Z_]+)\)"
        matches = re.findall(pattern, constraint["condition"])

        for algorithm, state in matches:
            if algorithm not in self.state_algorithms[state]:
                self.state_algorithms[state].append(algorithm)

    def _calculate_total_complexity(self, state):
        """
        计算当前state需要计算的所有算法的总复杂度。
        """
        total_complexity = 0
        for algorithm in self.state_algorithms.get(state, []):
            total_complexity += self.get_algorithm_calculate_time_complexity(algorithm, len(self.state_event_map[state]))
        return total_complexity


    def _validate_constraint(self, constraint):
        """
        验证某个约束是否被满足。
        """
        variables = constraint["variables"]
        condition = constraint["condition"]
        variable_values = [
            self.result_cache.get(hash(tuple(var))) for var in variables
        ]
        if None in variable_values:
            return True  # 暂时跳过该约束
        return condition(variable_values)

    def _compute(self, state, event_list):
        """
        根据state和事件列表计算具体结果。
        """
        return sum(len(event) for event in event_list)