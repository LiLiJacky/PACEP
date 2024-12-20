from collections import defaultdict
import itertools
from sortedcontainers import SortedDict
from factories.AlgorithmFactory import AlgorithmFactory
from lazy_calculate.TableTool import TableTool
import re

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
        self.validate_combinations_table = defaultdict(list)  # 用于已存储验证组合的表
        self.basic_result_to_combination_keys = {} # 用于index和hashkey的基本组合

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
            self.combination_table[variable] = defaultdict(dict)
        if variable not in self.time_index:
            self.time_index[variable] = SortedDict()  # 使用 SortedDict 来支持高效区间查询

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

            # 生成哈希键值，确保可哈希
            hash_key = self.table_tool.ensure_hashable(combination_wrappers)

            # 初始化存储求解结果的字典
            solved_values = {}

            # 遍历匹配到的算法
            for match in matches:
                algorithm_name = match[0]
                index = int(match[1]) if match[1] else None

                solved_value = self.algorithm_factory.get_algorithm_solution(algorithm_name, combination_values, index)

                # 将计算结果存入字典中
                solved_values[algorithm_name] = solved_value

            # 获取时间范围
            start_time = combination_values[0][0]  # 起始时间
            end_time = combination_values[-1][0]  # 结束时间

            # 调用 add_to_time_index 方法更新 time_index
            self.add_to_time_index(variable, start_time, end_time, hash_key)

            # 按哈希键值存储计算结果
            if hash_key not in self.combination_table[variable]:
                self.combination_table[variable][hash_key] = {}

            self.combination_table[variable][hash_key]['algorithm'] = solved_values
            self.combination_table[variable][hash_key]['combination'] = combination_wrappers

        # 更新已求解组合表
        # self.unsolved_combinations[variable] = [solved_values]  # 这里可以存储你已计算的解


    def calculate_constraint_costs(self, constraints):
        """
        计算约束的代价
        """
        unprocessed_constraints = [
            constraint for constraint in constraints if constraint not in self.processed_constraints
        ]
        constraints_cost = [
            (constraint, self.algorithm_factory.calculate_constrain_cost(constraint, self.max_lengths))
            for constraint in unprocessed_constraints
        ]
        return constraints_cost

    def select_min_cost_constraint(self, constraints_cost):
        """
        选择代价最小的约束
        """
        min_c, _ = min(constraints_cost, key=lambda x: x[1])  # 找到计算代价最小的约束
        return min_c

    def solve_combinations(self, min_c):
        """
        对当前约束的变量求解组合
        """
        for variable in min_c.variables:
            self.solve_variable_combinations(variable, min_c.expression)

    def update_combination_table(self, valid_combinations, idx):
        """
        更新组合表，若组合表为空则直接创建并添加组合，否则更新已有表
        :param valid_combinations: 当前有效的组合
        :param idx: 当前基本结果的索引
        """
        # 如果 validate_combinations_table 为空，直接初始化并添加组合
        new_validate_combinations_table = {}
        new_basic_result_to_combination_keys = {}
        if not self.validate_combinations_table:
            for combo in valid_combinations:
                # 为每个组合生成一个新的 hash_key
                hash_key = self.table_tool.ensure_hashable(combo)

                # 将该组合添加到 validate_combinations_table 中
                new_validate_combinations_table[hash_key] = combo

                # 将 hash_key 添加到 basic_result_to_combination_keys[idx] 中
                if idx not in new_basic_result_to_combination_keys:
                    new_basic_result_to_combination_keys[idx] = [hash_key]
                else:
                    new_basic_result_to_combination_keys[idx].append(hash_key)

        else:
            # 否则，按照计划更新 validate_combinations_table
            for combo in valid_combinations:
                connect_keys = combo.get("connect_keys", [])

                if not connect_keys:
                    # 如果 connect_keys 为空，进行全连接（笛卡尔积）
                    all_combinations = self.basic_result_to_combination_keys[idx]

                    for combination in all_combinations:
                        # 生成 full_combo_dict 时提取 ('B', 1731323473) 部分，去掉老的 hashkey 1
                        full_combo_dict = {state: value for state, value in combination}

                        # 将 full_combo_dict 和 combo 合并
                        full_combo_dict.update(combo)  # 添加新的 valid_combinations

                        # 为每个新组合生成 hash_key
                        hash_key = self.table_tool.ensure_hashable(full_combo_dict)

                        # 将新生成的组合添加到 new_validate_combinations_table
                        new_validate_combinations_table[hash_key] = full_combo_dict

                        # 将新生成的 hash_key 添加到 new_basic_result_to_combination_keys[idx]
                        if idx not in new_basic_result_to_combination_keys:
                            new_basic_result_to_combination_keys[idx] = [hash_key]
                        else:
                            new_basic_result_to_combination_keys[idx].append(hash_key)

                else:
                    # 如果 connect_keys 不为空时，进行连接处理
                    for connect_key in connect_keys:
                        # 找到与 connect_key 对应的组合并进行全连接
                        matching_combos = self.validate_combinations_table.get(connect_key, [])

                        for match in matching_combos:
                            # 合并现有的组合与新的 valid_combinations
                            full_combo_dict = {**match, **combo}  # 合并组合

                            # 为每个新组合生成 hash_key
                            hash_key = self.table_tool.ensure_hashable(full_combo_dict)

                            # 将新的组合添加到 new_validate_combinations_table
                            new_validate_combinations_table[hash_key] = full_combo_dict

                            # 将 hash_key 添加到 new_basic_result_to_combination_keys[idx]
                            if idx not in new_basic_result_to_combination_keys:
                                new_basic_result_to_combination_keys[idx] = [hash_key]
                            else:
                                new_basic_result_to_combination_keys[idx].append(hash_key)

        return new_validate_combinations_table, new_basic_result_to_combination_keys

    def validate_combinations(self, basic_result, min_c, idx):
        """
        验证组合是否满足约束条件，处理已求解的组合和未求解的组合。
        :param basic_result: 基本结果，包含每个状态的事件
        :param min_c: 最小组合约束
        :param idx: 当前基本结果的索引
        """
        valid_combinations = []  # 存储有效的组合
        state_indices = {}  # 存储每个状态的组合索引
        combination_dict = {}  # 存储所有已求解的组合，每行是 { "A": hashkey, "B": hashkey, ... }

        # 初始化状态与已生成组合的索引
        for state, events in basic_result.items():
            if state not in min_c.variables:
                continue
            state_indices[state] = []

            # 获取该状态所有事件的时间戳
            event_timestamps = [wrapper.event.timestamp for wrapper in events]

            # 获取该状态的第一个和最后一个时间戳
            start_time = event_timestamps[0]  # 第一个时间戳
            end_time = event_timestamps[-1]  # 最后一个时间戯

            # 调用 query_time_range 查询满足的组合
            hash_keys = self.query_time_range(state, start_time, end_time)

            # 将查询结果添加到状态的索引中
            if hash_keys:
                state_indices[state].extend(hash_keys)

        # 确保至少有一个匹配的索引，否则跳过这个状态
        if not any(state_indices.values()):
            return valid_combinations

        # 获取当前basic_result的索引与validate_combinations_table中的key映射关系
        for valid_combinations_hashkey in self.basic_result_to_combination_keys.get(idx, []):
            comb = self.validate_combinations_table.get(valid_combinations_hashkey, {})

            # 如果没有对应的组合，跳过
            if not comb:
                continue

            # 用于存储每个变量对应的算法解决方案
            combo_dict = {}  # 在这里初始化 combo_dict

            # 遍历connect_keys，获取validate_combinations_table中对应的组合
            for var in min_c.variables:
                state_key = comb.get(var, {})
                if state_key:
                    state_comb = self.combination_table[var].get(state_key, {})
                    algorithms = self.algorithm_factory.get_algorithm_name(var, min_c.expression)

                    for algorithm in algorithms:
                        # 如果该组合已经求解，直接获取其值
                        if algorithm in state_comb["algorithm"]:
                            if var not in combo_dict:
                                combo_dict[var] = {"algorithm": []}
                            combo_dict[var]["algorithm"].append({algorithm: state_comb["algorithm"][algorithm]})
                        else:
                            value = self.algorithm_factory.get_algorithm_solution(algorithm, state_comb["combination"])
                            state_comb["algorithm"].append({algorithm: value})
                            if var not in combo_dict:
                                combo_dict[var] = {"algorithm": []}
                            combo_dict[var]["algorithm"].append({algorithm: value})

            if not combo_dict:
                continue
            # 使用组合的变量值来去重
            combo_signature = self.table_tool.ensure_hashable(combo_dict)

            # 添加到combination_dict
            if combo_signature in combination_dict:
                combination_dict[combo_signature]["connect_keys"].add(state_key)
            else:
                # 假设 combo_signature 是唯一的标识符，combo_dict 存储了每个变量和它的 state_key
                combination_dict[combo_signature] = {**{var: state_key for var, state_key in combo_dict.items()},
                                                     "connect_keys": {state_key}}  # 使用 set 存储 unique state_key

        # 如果没有有效的组合（即combination_dict为空），则直接进行全连接组合
        validation_combos = []  # 存储最终的有效组合

        if combination_dict:
            # 在此步骤之前，我们已经获得了所有关于此次constraint的所有state的组合
            # 接下来需要将combo_dict拓展state_indices部分，全连接即可
            for combo_dict_entry in combination_dict.values():
                # 将组合的变量和连接key整合到一个完整的组合字典中
                full_combo_dict = {}

                # 合并不同key的组合
                for var, val in combo_dict_entry.items():
                    full_combo_dict[var] = val

                # 将重复的key加到connect_keys
                full_combo_dict["connect_keys"] = [combo_dict_entry["connect_keys"]]  # 将多个hashkey作为connect_keys

                all_combinations = itertools.product(*state_indices.values())  # 使用笛卡尔积组合hashkey

                for combination in all_combinations:
                    full_combo = {state: key for state, key in zip(state_indices.keys(), combination)}
                    full_combo_dict.extend({state: self.combination_table[state][key]["algorithm"] for state, key in
                                    full_combo.items()})

                # 调用约束验证方法
                if min_c.validate(state_algorithm_results=full_combo_dict):
                    validation_combos.append(full_combo_dict)
        else:
            # 如果combination_dict为空，进行笛卡尔积（全连接）生成组合
            # 将所有状态的hashkey组合成笛卡尔积
            all_combinations = itertools.product(*state_indices.values())  # 使用笛卡尔积组合hashkey

            for combination in all_combinations:
                full_combo_dict = {state: key for state, key in zip(state_indices.keys(), combination)}
                current_comb = {state: self.combination_table[state][key]["algorithm"] for state, key in full_combo_dict.items()}

                if min_c.validate(state_algorithm_results = current_comb):
                    validation_combos.append(full_combo_dict)

        return validation_combos

    def process_basic_results(self, min_c):
        """
        遍历 basic_results，根据当前约束更新组合表
        """
        new_basic_results = []
        # 临时存储更新后的表
        new_validate_combinations_table = {}
        new_basic_result_to_combination_keys = {}

        for idx, basic_result in self.basic_result_index.items():
            # 获取当前basic_result与约束的有效组合
            valid_combinations = self.validate_combinations(basic_result, min_c, idx)

            if not valid_combinations:
                continue

            # 更新组合表并返回新生成的表
            updated_validate_combinations_table, updated_basic_result_to_combination_keys = self.update_combination_table(
                valid_combinations, idx)

            if updated_validate_combinations_table:
                # 将新结果合并到临时变量中
                new_validate_combinations_table.update(updated_validate_combinations_table)
                new_basic_result_to_combination_keys.update(updated_basic_result_to_combination_keys)

                # 将更新后的基本结果添加到new_basic_results
                new_basic_results.append((idx, basic_result))

        # 更新原始数据表和索引
        self.basic_result_to_combination_keys = new_basic_result_to_combination_keys
        self.validate_combinations_table = new_validate_combinations_table

        return new_basic_results

    def merge_results(self, basic_results):
        """
        合并最终结果
        """
        merged_original_result = defaultdict(list)
        for basic_result in basic_results:
            for key, values in basic_result.items():
                merged_original_result[key].extend(values)

        return merged_original_result

    def fill_missing_states(self, final_results, constraints):
        """
        填充缺失的状态
        """
        for idx, basic_result in self.basic_result_index.items():
            constrained_states = {var for constraint in constraints for var in constraint.variables}
            missing_states = set(basic_result.keys()) - constrained_states
            for restored_combination in final_results:
                combination = restored_combination[0]
                for state in missing_states:
                    if state not in combination:
                        combination[state] = basic_result[state]

    def initialize_unsolved_combinations(self, min_c):
        """
        初始化当前约束所需状态的未求解组合
        """
        for variable in min_c.variables:
            if variable not in self.combination_table:
                # 初始化未求解组合
                self.unsolved_combinations[variable] = list(itertools.chain.from_iterable(
                    itertools.combinations(range(len(self.global_longest_sequences[variable])), r)
                    for r in range(1, len(self.global_longest_sequences[variable]) + 1)
                ))

    def add_to_time_index(self, variable, start_time, end_time, hash_key):
        """
        将组合的哈希键存入时间索引中
        """
        if variable not in self.time_index:
            self.time_index[variable] = SortedDict()

        # 插入一个新的时间范围
        time_range = (start_time, end_time)
        if time_range not in self.time_index[variable]:
            self.time_index[variable][time_range] = []

        # 将哈希键添加到对应的时间范围中
        self.time_index[variable][time_range].append(hash_key)

    def query_time_range(self, variable, start_time, end_time):
        """
        查询指定时间范围内的所有组合哈希键
        """
        if variable not in self.time_index:
            return []

        # 查询范围 [start_time, end_time] 内的所有条目
        keys_within_range = []

        # 使用 bisect_left 和 bisect_right 查找时间范围内的区间
        start_pos = self.time_index[variable].bisect_left((start_time, -float('inf')))
        end_pos = self.time_index[variable].bisect_right((end_time, float('inf')))

        # 通过区间查询时间范围
        for time_range in list(self.time_index[variable].items())[start_pos:end_pos]:
            # 获取每个时间范围的哈希键
            keys_within_range.extend(time_range[1])

        return keys_within_range

    def evaluate_incrementally(self, basic_results, constraints):
        """
        按伪算法增量求解组合问题
        """
        # 初始化全局最长序列和索引
        self.global_longest_sequences, self.max_lengths = self.find_global_longest_sequences_and_lengths(basic_results)
        self.basic_result_index = self.create_basic_result_index(basic_results)
        self.processed_constraints = set()

        while len(self.processed_constraints) < len(constraints):
            # 计算约束的代价并选择最优约束
            constraints_cost = self.calculate_constraint_costs(constraints)
            min_c = self.select_min_cost_constraint(constraints_cost)
            self.processed_constraints.add(min_c)

            # 初始化当前约束所需状态的未求解组合
            self.initialize_unsolved_combinations(min_c)

            # 求解组合
            self.solve_combinations(min_c)

            # 根据约束更新组合表
            new_basic_results = self.process_basic_results(min_c)

            # 更新全局最长序列
            if len(basic_results) != len(new_basic_results):
                basic_results = [item[1] for item in new_basic_results]
                self.basic_result_index = {item[0]: item[1] for item in new_basic_results}
                self.global_longest_sequences, self.max_lengths = self.find_global_longest_sequences_and_lengths(
                    basic_results
                )

        # 合并最终结果
        final_results = []
        final_hashkyes = []
        for idx, combos in self.basic_result_to_combination_keys.items():
            for combo in combos:
                result_comb = {}
                for state, key in combo:
                    result_comb[state] = self.combination_table[state][key]["combination"]
                # 添加丢失状态
                for state, wrappers in self.basic_result_index.get(idx).items():
                    if state not in result_comb:
                        result_comb[state] = wrappers
                result_hashkey = self.table_tool.ensure_hashable(result_comb)
                if result_hashkey not in final_hashkyes:
                    result = [result_comb, []]
                    final_results.append(result)
                    final_hashkyes.append(result_hashkey)

        return final_results