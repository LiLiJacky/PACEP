import importlib, re
from math import comb

# 定义算法缩写到系统中全名的映射
ALGORITHM_SUB_MAP = {
    "first": "first_value",
    "last": "last_value",
    "random": "random_value",
    "nth": "nth_value",
    "average": "average_value",
    "sum": "sum_value",
    "min": "min_value",
    "max": "max_value",
    "merge_sort": "merge_sort",
    #"binary_search": "binary_search_timestamp",
    "bubble_sort": "bubble_sort",
    "triplet_sum": "triplet_product_sum",
    "permutations": "permutations",
    "combinations_square": "combinations_square_average_sum",
    "subset_sum": "subset_sum",
    "sum_square_difference": "sum_square_difference",
    "least_squares": "least_squares",
    "increasing": "increasing",
    "decreasing": "decreasing",
    "non_increasing": "non_increasing",
    "non_decreasing": "non_decreasing",
    "standard_deviation": "standard_deviation",
    "increasing_dp": "increasing_dp",
    "decreasing_dp": "decreasing_dp",
    "non_increasing_dp": "non_increasing_dp",
    "non_decreasing_dp": "non_decreasing_dp",
}

ALGORITHM_FACTORY_MAP = {
    "first_value": "factories.O1Factory.O1Factory",
    "last_value": "factories.O1Factory.O1Factory",
    "random_value": "factories.O1Factory.O1Factory",
    "nth_value": "factories.O1Factory.O1Factory",
    "average_value": "factories.ONFactory.ONFactory",
    "sum_value": "factories.ONFactory.ONFactory",
    "min_value": "factories.ONFactory.ONFactory",
    "max_value": "factories.ONFactory.ONFactory",
    "merge_sort": "factories.ONLogNFactory.ONLogNFactory",
    "binary_search_timestamp": "factories.OLogNFactory.OLogNFactory",
    "bubble_sort": "factories.ON2Factory.ON2Factory",
    "triplet_product_sum": "factories.ON3Factory.ON3Factory",
    "permutations": "factories.ONFactorialFactory.ONFactorialFactory",
    "combinations_square_average_sum": "factories.ONkFactory.ONkFactory",
    "subset_sum": "factories.O2NFactory.O2NFactory",
    "sum_square_difference": "factories.ON2Factory.ON2Factory",
    "least_squares": "factories.ONFactory.ONFactory",
    "increasing": "factories.ONFactory.ONFactory",
    "decreasing": "factories.ONFactory.ONFactory",
    "non_increasing": "factories.ONFactory.ONFactory",
    "non_decreasing": "factories.ONFactory.ONFactory",
    "standard_deviation": "factories.ONFactory.ONFactory",
    "increasing_dp": "factories.ON2Factory.ON2Factory",
    "decreasing_dp": "factories.ON2Factory.ON2Factory",
    "non_increasing_dp": "factories.ON2Factory.ON2Factory",
    "non_decreasing_dp": "factories.ON2Factory.ON2Factory",
}

ALGORITHM_DP = [
    "increasing_dp",
    "decreasing_dp",
    "non_increasing_dp",
    "non_decreasing_dp"
]

class AlgorithmFactory:
    def __init__(self, *args):
        self.algorithm_sub_map = ALGORITHM_SUB_MAP
        self.algorithm_factory_map = ALGORITHM_FACTORY_MAP

    def get_factory(self, complexity):
        factories = {
            "O(1)": "factories.O1Factory.O1Factory",
            "O(n)": "factories.ONFactory.ONFactory",
            "O(log n)": "factories.OLogNFactory.OLogNFactory",
            "O(n^2)": "factories.ON2Factory.ON2Factory",
            "O(n^3)": "factories.ON3Factory.ON3Factory",
            "O(2^n)": "factories.O2NFactory.O2NFactory",
            "O(n!)": "factories.ONFactorialFactory.ONFactorialFactory",
            "O(N^k)": "factories.ONkFactory.ONkFactory"
        }
        if complexity in factories:
            module_path, class_name = factories[complexity].rsplit('.', 1)
            module = importlib.import_module(module_path)
            factory_class = getattr(module, class_name)
            return factory_class()
        else:
            raise ValueError(f"Unknown complexity: {complexity}")

    def get_algorithm_range(self, full_name, range):
        factory_path = self.algorithm_factory_map.get(full_name)
        if not factory_path:
            raise ValueError(f"Factory for algorithm '{full_name}' not found.")

        module_path, class_name = factory_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        factory_class = getattr(module, class_name)
        factory = factory_class()
        return factory.get_algorithm(full_name, data)

    def get_algorithm(self, full_name, data, *args):
        factory_path = self.algorithm_factory_map.get(full_name)
        if not factory_path:
            raise ValueError(f"Factory for algorithm '{full_name}' not found.")

        module_path, class_name = factory_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        factory_class = getattr(module, class_name)
        factory = factory_class()

        # 判断是否超过范围
        if args and args[0][0] >= len(data):
            return factory.get_algorithm(full_name, data, 0)
        return factory.get_algorithm(full_name, data, args)

    def get_algorithm_sub_map(self):
        return self.algorithm_sub_map

    def get_algorithm_factory_map(self):
        return self.algorithm_factory_map

    def get_algorithm_list(self):
        # 提取算法缩写列表
        return list(self.algorithm_sub_map.keys())

    def get_algorithm_calculate_time_complexity(self, name, n):
        """
        根据算法全名调用其 get_time_complexity 方法，返回计算复杂度的具体值。
        :param name: 算法全名（如 "sum_value"）
        :param n: 输入数据规模（如 1000）
        :return: 计算复杂度的具体值
        """
        # 获取算法全名
        full_name = self.algorithm_sub_map.get(name)
        if not full_name:
            raise ValueError(f"Full name for algorithm '{name}' not found.")

        # 获取工厂路径
        factory_path = self.algorithm_factory_map.get(full_name)
        if not factory_path:
            raise ValueError(f"Factory for algorithm '{full_name}' not found.")

        # 动态加载模块并获取工厂类
        module_path, class_name = factory_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        factory_class = getattr(module, class_name)

        # 初始化工厂实例并获取算法类
        factory = factory_class()
        algorithm = factory.get_algorithm(full_name, None)  # 此处无需传入数据实例

        # 检查算法是否具有 get_time_complexity 方法
        if not hasattr(algorithm, "get_time_complexity"):
            if not hasattr(factory, "get_time_complexity"):
                raise AttributeError(f"Algorithm '{full_name}' does not implement 'get_time_complexity' or 'get_calculate' method.")
            else:
                return factory.get_time_complexity(n)

        # 调用 get_time_complexity 方法并返回计算复杂度
        return algorithm.get_time_complexity(n)

    def calculate_combination_cost(self, algorithm_name, n):
        """
        计算从 C(n, 1) 到 C(n, n) 的总复杂度
        """
        total_cost = 0

        if algorithm_name in ALGORITHM_DP:
            total_cost = self.get_algorithm_calculate_time_complexity(algorithm_name, n)
        else:
            for k in range(1, n + 1):
                combination_count = comb(n, k)
                complexity = self.get_algorithm_calculate_time_complexity(algorithm_name, k)
                total_cost += combination_count * complexity
        return total_cost

    def calculate_constrain_cost(self, latency_constrain: 'ValueConstraint', data_counts):
        """
        计算单个延迟约束的代价，支持多个匹配项
        """
        # 如果dp，则直接返回完整计算代价
        if len(latency_constrain.variables) == 1:
            variable = latency_constrain.variables[0]
            alg =  re.findall(rf'(\w+)\({variable}\)', latency_constrain.expression)
            if len(alg) == 1 and alg[0] in ALGORITHM_DP:
                return self.get_algorithm_calculate_time_complexity(alg[0], data_counts.get(variable, 0))

        total_cost = 0
        for variable in latency_constrain.variables:
            matches = re.findall(rf'(\w+)\({variable}\)', latency_constrain.expression)
            if matches:
                for match in matches:
                    algorithm_name = match
                    n = data_counts.get(variable, 0)
                    total_cost += self.calculate_combination_cost(algorithm_name, n)
            else:
                total_cost += 1

        # 计算变量组合总数（组合乘积的代价）
        if len(latency_constrain.variables) > 1:
            variable_combinations = 1
            for variable in latency_constrain.variables:
                variable_combinations *= sum(
                    comb(data_counts[variable], k) for k in range(1, data_counts[variable] + 1))

            total_cost += variable_combinations  # 加入变量组合的乘积代价

        return total_cost

    def get_algorithm_solution(self, algorithm_name, combination_values, index=None):
        """
        获取算法计算的结果，如果算法名称无效或无法找到对应算法，则抛出异常。
        :param algorithm_name: 算法名称
        :param combination_values: 组合值，传递给算法进行计算
        :param index: 算法所需的索引（可选）
        :return: 算法计算的结果
        """
        # 获取完整算法名
        full_algorithm_name = self.algorithm_sub_map.get(algorithm_name)
        if not full_algorithm_name:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in algorithm_sub_map")

        # 调用算法工厂计算
        algorithm = self.get_algorithm(full_algorithm_name, combination_values)
        return algorithm.get_calculate(index) if index else algorithm.get_calculate()

    def get_algorithm_name(self, var, expression):
        """
        获取表达式中的算法名称（变量部分）和算法名，以及可能的索引（如 algorithm(A)[idx]）。

        :param var: 变量，例如 'A'
        :param expression: 包含算法表达式的字符串，例如 '10<=average(B)<=1000' 或 'algorithm(A)[idx]'
        :return: 算法名称列表
        """
        # 正则匹配 "algorithm(A)" 或 "algorithm(A)[idx]" 中的算法名称和索引
        pattern = re.compile(r'(\w+)\(' + re.escape(var) + r'\)(?:\[(\w+)\])?')

        # 查找匹配的部分
        matches = re.findall(pattern, expression)

        return matches

    def is_dp_algorithm(self, algorithm_name):
        return algorithm_name in ALGORITHM_DP

# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 20), (3, 30), (4, 40), (5, 10)]

    factory = AlgorithmFactory().get_factory("O(1)")
    algorithm = factory.get_algorithm("first_value", data)
    print(algorithm.get_calculate())  # 输出: 10

    factory = AlgorithmFactory().get_factory("O(n)")
    algorithm = factory.get_algorithm("sum_value", data)
    print(algorithm.get_calculate())  # 输出: 150
