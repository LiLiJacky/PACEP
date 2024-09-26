# algorithm_factory/O2nFactory.py
from interfaces.Algorithm import Algorithm
from itertools import combinations


class SubsetSumAlgorithm(Algorithm):
    @property
    def name(self):
        return "Subset Sum"

    @property
    def complexity_time(self):
        return "O(2^n)"

    @property
    def complexity_space(self):
        return "O(2^n)"

    def get_calculate(self):
        n = len(self.data)
        all_subsets = []
        for i in range(n + 1):
            for subset in combinations(self.data, i):
                all_subsets.append(subset)

        result = [sum(value for _, value in subset) for subset in all_subsets]
        return result[len(result) // 2]

    def get_calculate_range(self, range):
        return range


class O2NFactory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "subset_sum": SubsetSumAlgorithm,
        }
        if algorithm_name in algorithms:
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 20), (3, 30), (4, 40), (5, 10)]

    factory = O2NFactory()

    algorithm = factory.get_algorithm("subset_sum", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Subset Sum
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(2^n)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(2^n)
    result = algorithm.get_calculate()
    print(f"Result: {result}")  # 输出: 所有子集的和，如 [0, 50, 20, 70, 30, 80, 50, 100, 40, 90, 60, 110, 70, 120, 100, 150]
