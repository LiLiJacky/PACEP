import importlib

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
    "combinations_square": "combinations_square_average_sum"
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
    "merge_sort": "factories.OLogNFactory.OLogNFactory",
    "binary_search_timestamp": "factories.OLogNFactory.OLogNFactory",
    "bubble_sort": "factories.ON2Factory.ON2Factory",
    "triplet_product_sum": "factories.ON3Factory.ON3Factory",
    "permutations": "factories.ONFactorialFactory.ONFactorialFactory",
    "combinations_square_average_sum": "factories.ONkFactory.ONkFactory"
}


class AlgorithmFactory:
    def __init__(self):
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
        return factory.get_algorithm(full_name, data, *args)

    def get_algorithm_sub_map(self):
        return self.algorithm_sub_map

    def get_algorithm_factory_map(self):
        return self.algorithm_factory_map

    def get_algorithm_list(self):
        # 提取算法缩写列表
        return list(self.algorithm_sub_map.keys())


# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 20), (3, 30), (4, 40), (5, 10)]

    factory = AlgorithmFactory().get_factory("O(1)")
    algorithm = factory.get_algorithm("first_value", data)
    print(algorithm.get_calculate())  # 输出: 10

    factory = AlgorithmFactory().get_factory("O(n)")
    algorithm = factory.get_algorithm("sum_value", data)
    print(algorithm.get_calculate())  # 输出: 150
