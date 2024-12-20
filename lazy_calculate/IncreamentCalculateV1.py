from collections import defaultdict
import itertools, re
from sortedcontainers import SortedDict

from factories.AlgorithmFactory import AlgorithmFactory
from lazy_calculate.TableTool import TableTool


class IncrementTableCalculate:
    def __init__(self, cost_threshold=100, event_threshold=20):
        """
        初始化增量表计算类
        """
        self.algorithm_factory = AlgorithmFactory()  # 使用封装的 AlgorithmFactory
        self.cost_threshold = cost_threshold
        self.event_threshold = max(event_threshold, 20)
        self.combination_table = defaultdict(list)  # 全局组合表
        self.time_index = defaultdict(SortedDict)  # 时间范围索引 {state: SortedDict((start_time, end_time): result)}
        self.unsolved_combinations = defaultdict(list)  # 尚未求解的组合
        self.basic_result_index = {}  # 每个 basic_result 的索引表
        self.global_longest_sequences = {}  # 每个状态的全局最长序列
        self.max_lengths = {}  # 存储每个状态的最大数据计数
        self.table_tool = TableTool()  # 使用 TableTool 工具类

    def find_global_longest_sequences_and_lengths(self, basic_results):
        """
        从所有 basic_results 中提取每个状态的全局最长序列和对应的长度
        :param basic_results: List[dict], 每个元素为一个 basic_result
        :return: (global_longest_sequences, max_lengths)
        """
        global_sequences = defaultdict(list)
        max_lengths = defaultdict(int)

        for basic_result in basic_results:
            for state, events in basic_result.items():
                # 如果当前状态的 events 长度更长，则替换
                if len(events) > max_lengths[state]:
                    global_sequences[state] = events  # 替换为当前更长的序列
                    max_lengths[state] = len(events)  # 更新最大长度

        # 对全局序列按照时间戳排序（可选，根据使用场景）
        for state in global_sequences:
            global_sequences[state] = sorted(global_sequences[state], key=lambda x: x.event.timestamp)

        return global_sequences, max_lengths

    def create_basic_result_index(self, basic_results):
        """
        为 basic_results 创建唯一索引
        :param basic_results: List[dict], 每个元素为一个 basic_result
        :return: 索引表 {index: basic_result}
        """
        return {idx: basic_result for idx, basic_result in enumerate(basic_results)}

    def solve_variable_combinations(self, variable, expression):
        """
        对变量的未求解组合执行算法求解，并更新组合表
        :param variable: 当前变量名
        :param expression: 当前约束的表达式
        """
        # 从表达式中解析出算法信息
        matches = re.findall(rf'(\w+)\({variable}(?:\[(\d+)\])?\)', expression)
        if not matches:
            raise ValueError(f"No algorithm found for variable '{variable}' in expression '{expression}'")

        # 确保组合表包含变量和算法名的多级结构
        if variable not in self.combination_table:
            self.combination_table[variable] = defaultdict(list)
        if variable not in self.time_index:
            self.time_index[variable] = {}

        # 遍历匹配到的算法
        for match in matches:
            algorithm_name = match[0]
            index = int(match[1]) if match[1] else None

            # 获取完整算法名
            full_algorithm_name = self.algorithm_factory.algorithm_sub_map.get(algorithm_name)
            if not full_algorithm_name:
                raise ValueError(f"Algorithm '{algorithm_name}' not found in algorithm_sub_map")

            # 初始化存储求解结果的列表
            solved_combinations = []

            # 遍历未求解的组合
            for combination_indices in self.unsolved_combinations[variable]:
                # 根据索引获取事件包裹对象 (EventWrapper)
                combination_wrappers = [
                    self.global_longest_sequences[variable][i] for i in combination_indices
                ]
                # 提取时间和值的数组
                combination_values = [
                    [wrapper.event.timestamp, wrapper.event.value]
                    for wrapper in combination_wrappers
                ]

                # 调用算法工厂计算
                algorithm = self.algorithm_factory.get_algorithm(full_algorithm_name, combination_values)
                solved_value = algorithm.get_calculate(index) if index else algorithm.get_calculate()

                # 获取时间范围
                start_time = combination_values[0][0]  # 起始时间
                end_time = combination_values[-1][0]  # 结束时间

                # 更新时间索引
                self.time_index[variable][(start_time, end_time)] = {
                    "combination": combination_wrappers,  # 存储组合
                    "value": solved_value,  # 存储计算结果
                    "algorithm": full_algorithm_name,  # 存储算法名
                    "valid": True  # 标记组合有效
                }

                # 按算法名组织计算结果
                self.combination_table[variable][algorithm_name].append({
                    "combination": combination_wrappers,
                    "value": solved_value,
                    "time_range": (start_time, end_time)
                })
                solved_combinations.append(solved_value)

            # 删除当前变量的未求解组合（已完成求解）
            del self.unsolved_combinations[variable]

    def evaluate_incrementally(self, basic_results, constraints):
        """
        按伪算法增量求解组合问题
        """
        # 初始化全局最长序列和索引
        self.global_longest_sequences, self.max_lengths = self.find_global_longest_sequences_and_lengths(basic_results)
        self.basic_result_index = self.create_basic_result_index(basic_results)
        processed_constraints = set()

        while len(processed_constraints) < len(constraints):
            # 计算未处理约束的代价
            unprocessed_constraints = [
                constraint for constraint in constraints if constraint not in processed_constraints
            ]
            constraints_cost = [
                (constraint, self.algorithm_factory.calculate_constrain_cost(constraint, self.max_lengths))
                for constraint in unprocessed_constraints
            ]
            min_c, _ = min(constraints_cost, key=lambda x: x[1])  # 找到计算代价最小的约束
            processed_constraints.add(min_c)

            # 动态初始化当前约束所需状态的未求解组合
            for variable in min_c.variables:
                if variable not in self.combination_table:
                    self.unsolved_combinations[variable] = list(itertools.chain.from_iterable(
                        itertools.combinations(range(len(self.global_longest_sequences[variable])), r)
                        for r in range(1, len(self.global_longest_sequences[variable]) + 1)
                    ))
                self.solve_variable_combinations(variable, min_c.expression)

            # 遍历 basic_results，根据当前约束更新组合表
            new_basic_results = []
            for idx, basic_result in self.basic_result_index.items():
                valid_combinations = []
                state_indices = {}

                # 初始化状态与已生成组合的索引
                for state, events in basic_result.items():
                    # 修复问题的关键：确保 entry 是字典，正确访问 "start" 和 "end"
                    state_indices[state] = [
                        i for i, entry in enumerate(self.combination_table.get(state, []))
                        if any(
                            isinstance(entry, dict) and
                            wrapper.event.timestamp >= entry.get("start", float("-inf")) and
                            wrapper.event.timestamp <= entry.get("end", float("inf"))
                            for wrapper in events
                        )
                    ]

                # 根据多个状态的索引生成组合
                for combo_indices in itertools.product(
                        *[state_indices[var] for var in min_c.variables if var in state_indices]
                ):
                    # 构建组合字典
                    combo_dict = {}
                    for var, index in zip(min_c.variables, combo_indices):
                        combo_dict[var] = self.combination_table[var][index]["value"]

                    # 调用约束的 `validate` 方法验证组合
                    if min_c.validate(state_algorithm_results=combo_dict):
                        valid_combinations.append(combo_dict)

                # 如果当前 basic_result 没有满足条件的组合，则跳过
                if not valid_combinations:
                    continue

                # 动态更新组合表
                if not self.combination_table:
                    # 如果组合表为空，直接初始化
                    self.combination_table = self.table_tool.create_new_table(min_c.variables, valid_combinations)
                else:
                    # 否则根据共享变量进行更新
                    shared_vars = [var for var in min_c.variables if var in self.combination_table]
                    self.combination_table = self.table_tool.join_with_table(
                        self.combination_table, min_c.variables, valid_combinations, shared_vars
                    )

                # 保存当前 basic_result 供下一轮求解
                new_basic_results.append((idx, basic_result))

            # 如果 basic_results 数量变化，更新全局最长序列和其长度
            if len(basic_results) != len(new_basic_results):
                self.global_longest_sequences, self.max_lengths = self.find_global_longest_sequences_and_lengths(
                    basic_results
                )
            basic_results = [item[1] for item in new_basic_results]
            self.basic_result_index = {item[0]: item[1] for item in new_basic_results}

        # 构建最终结果
        merged_original_result = defaultdict(list)
        for basic_result in basic_results:
            for key, values in basic_result.items():
                merged_original_result[key].extend(values)

        final_results = self.table_tool.restore_to_event_wrapper(self.combination_table, merged_original_result)

        # 为每条结果补充 missing_states
        for idx, basic_result in self.basic_result_index.items():
            constrained_states = {var for constraint in constraints for var in constraint.variables}
            missing_states = set(basic_result.keys()) - constrained_states
            for restored_combination in final_results:
                combination = restored_combination[0]
                for state in missing_states:
                    if state not in combination:
                        combination[state] = basic_result[state]

        return final_results