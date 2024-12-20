# algorithm_factory/ONFactory.py
import numpy as np
import pandas as pd

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
        if len(self.data) == 0:
            return 0
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


class LeastSquaresAlgorithm(Algorithm):
    @property
    def name(self):
        return "Least Squares"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        data = np.array(self.data)
        # Convert Unix timestamps to ordinal dates
        time_stamp = np.array([pd.to_datetime(ts, unit='s').toordinal() for ts in data[:, 0]])
        values = data[:, 1].astype(float)
        n = len(time_stamp)
        sum_timestamp = np.sum(time_stamp)
        sum_value = np.sum(values)
        sum_tv = np.sum(values * time_stamp)
        sum_timestamp_squared = np.sum(time_stamp ** 2)

        numerator = n * sum_tv - sum_value * sum_timestamp
        denominator = n * sum_timestamp_squared - sum_timestamp ** 2

        return numerator / denominator if denominator != 0 else 0

    def get_calculate_range(self, range):
        return range

class IncreasingAlgorithm(Algorithm):
    @property
    def name(self):
        return "Increasing"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        data = np.array(self.data)
        
        values = data[:, 1].astype(float)  # 获取 value 列
        return 1 if np.all(np.diff(values) > 0) else -1  # 检查是否递增

    def get_calculate_range(self, range):
        return range

class DecreasingAlgorithm(Algorithm):
    @property
    def name(self):
        return "Decreasing"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        data = np.array(self.data)
        
        values = data[:, 1].astype(float)  # 获取 value 列
        return 1 if np.all(np.diff(values) < 0) else -1  # 检查是否递减

    def get_calculate_range(self, range):
        return range


class NonIncreasingAlgorithm(Algorithm):
    @property
    def name(self):
        return "Non-Increasing"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        data = np.array(self.data)
        
        values = data[:, 1].astype(float)  # 获取 value 列
        return 1 if  np.all(np.diff(values) <= 0) else -1  # 检查是否非递增

    def get_calculate_range(self, range):
        return range


class NonDecreasingAlgorithm(Algorithm):
    @property
    def name(self):
        return "Non-Decreasing"

    @property
    def complexity_time(self):
        return "O(n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        data = np.array(self.data)
        
        values = data[:, 1].astype(float)  # 获取 value 列
        return 1 if np.all(np.diff(values) >= 0) else -1 # 检查是否非递减

    def get_calculate_range(self, range):
        return range


class ONFactory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "average_value": AverageValueAlgorithm,
            "sum_value": SumValueAlgorithm,
            "min_value": MinValueAlgorithm,
            "max_value": MaxValueAlgorithm,
            "least_squares": LeastSquaresAlgorithm,
            "increasing": IncreasingAlgorithm,
            "decreasing": DecreasingAlgorithm,
            "non_increasing": NonIncreasingAlgorithm,
            "non_decreasing": NonDecreasingAlgorithm,
        }
        if algorithm_name in algorithms:
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def get_time_complexity(self, n: int):
        return n


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
