# algorithm_factory/ONkFactory.py
from interfaces.Algorithm import Algorithm
from itertools import combinations


class CombinationsSquareAverageSumAlgorithm(Algorithm):
    def __init__(self, data, k):
        super().__init__(data)
        self.k = k

    @property
    def name(self):
        return "Combinations Square Average Sum"

    @property
    def complexity_time(self):
        return f"O(n^{self.k})"

    @property
    def complexity_space(self):
        return f"O(n^{self.k})"

    def get_calculate(self):
        combs = list(combinations(self.data, self.k))
        square_avg_sum = sum(sum(value ** 2 for _, value in comb) / self.k for comb in combs)
        return square_avg_sum

    def get_calculate_range(self, range):
        lower_bound = range[0]
        high_bound = range[1]
        if lower_bound > high_bound:
            raise ValueError("Lower bound must be less than or equal to high bound.")

            # 计算平方和的最小值
        if lower_bound <= 0 <= high_bound:
            # 如果 0 在范围内，最小的平方和是 0
            min_sum = 0
        else:
            # 否则，选择范围内绝对值最小的数的平方和
            min_value = min(abs(lower_bound), abs(high_bound))
            min_sum = min_value ** 2 * 5

            # 计算平方和的最大值
        max_value = max(abs(lower_bound), abs(high_bound))
        max_sum = max_value ** 2 * 5

        return [min_sum, max_sum]

class ONkFactory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "combinations_square_average_sum": CombinationsSquareAverageSumAlgorithm,
        }
        if algorithm_name in algorithms:
            if len(args) == 1:
                k = args[0]
                return algorithms[algorithm_name](data, k)
            else:
                raise ValueError(f"{algorithm_name} algorithm requires one argument for k")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 20), (3, 30), (4, 40), (5, 10)]

    factory = ONkFactory()

    k = 3  # 假设我们要计算所有长度为 3 的组合的平方值的平均值之和
    algorithm = factory.get_algorithm("combinations_square_average_sum", data, k)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Combinations Square Average Sum
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n^3)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(n^3)
    result = algorithm.get_calculate()
    print(f"Result: {result}")  # 输出: 所有长度为 3 的组合的平方值的平均值之和
