# algorithm_factory/ONFactory.py
from interfaces.Algorithm import Algorithm


class AverageValueAlgorithm(Algorithm):
    @property
    def name(self):
        return "Average Value"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        total = sum(value for timestamp, value in self.data)
        return total / len(self.data)

    def get_calculate_range(self, range):
        return range

class SumValueAlgorithm(Algorithm):
    @property
    def name(self):
        return "Sum Value"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        return sum(value for timestamp, value in self.data)

    def get_calculate_range(self, range):
        return range

class MinValueAlgorithm(Algorithm):
    @property
    def name(self):
        return "Min Value"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        return min(value for timestamp, value in self.data)

    def get_calculate_range(self, range):
        return range

class MaxValueAlgorithm(Algorithm):
    @property
    def name(self):
        return "Max Value"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        return max(value for timestamp, value in self.data)

    def get_calculate_range(self, range):
        return range

class ONFactory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "average_value": AverageValueAlgorithm,
            "sum_value": SumValueAlgorithm,
            "min_value": MinValueAlgorithm,
            "max_value": MaxValueAlgorithm,
        }
        if algorithm_name in algorithms:
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


# 使用示例
if __name__ == "__main__":
    data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]

    factory = ONFactory()

    algorithm = factory.get_algorithm("average_value", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Average Value
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(1)
    print(f"Result: {algorithm.get_calculate()}")  # 输出: 30.0

    algorithm = factory.get_algorithm("sum_value", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Sum Value
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(1)
    print(f"Result: {algorithm.get_calculate()}")  # 输出: 150

    algorithm = factory.get_algorithm("min_value", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Min Value
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(1)
    print(f"Result: {algorithm.get_calculate()}")  # 输出: 10

    algorithm = factory.get_algorithm("max_value", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Max Value
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(1)
    print(f"Result: {algorithm.get_calculate()}")  # 输出: 50
