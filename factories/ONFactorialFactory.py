# algorithm_factory/ONFactorialFactory.py
from interfaces.Algorithm import Algorithm
from itertools import permutations


class PermutationsAlgorithm(Algorithm):
    @property
    def name(self):
        return "Permutations"

    @property
    def complexity_time(self):
        return "O(n!)"

    @property
    def complexity_space(self):
        return "O(n!)"

    def get_calculate(self):
        permuts = list(permutations(self.data))
        return permuts[0][0][1]

    def get_calculate_range(self, range):
        return range


class ONFactorialFactory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "permutations": PermutationsAlgorithm,
        }
        if algorithm_name in algorithms:
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 20), (3, 30), (4, 40), (5, 10)]

    factory = ONFactorialFactory()

    algorithm = factory.get_algorithm("permutations", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Permutations
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n!)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(n!)
    result = algorithm.get_calculate()
    print(f"Result: {result}")  # 输出: 所有排列
