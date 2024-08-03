# algorithm_factory/OLogNFactory.py
from interfaces.Algorithm import Algorithm


class BinarySearchTimestampAlgorithm(Algorithm):
    def __init__(self, data, target_timestamp):
        super().__init__(data)
        self.target_timestamp = target_timestamp

    @property
    def name(self):
        return "Binary Search by Timestamp"

    @property
    def complexity_time(self):
        return "O(log n)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        left, right = 0, len(self.data) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_timestamp, mid_value = self.data[mid]
            if mid_timestamp == self.target_timestamp:
                return mid_value
            elif mid_timestamp < self.target_timestamp:
                left = mid + 1
            else:
                right = mid - 1
        return None  # 如果未找到 target_timestamp

    def get_calculate_range(self, range):
        return range


class OLogNFactory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "binary_search_timestamp": BinarySearchTimestampAlgorithm,
        }
        if algorithm_name in algorithms:
            if algorithm_name == "binary_search_timestamp":
                if len(args) == 1:
                    return algorithms[algorithm_name](data, args[0])
                else:
                    raise ValueError("binary_search_timestamp algorithm requires one argument for the target timestamp")
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 40), (3, 30), (4, 20), (5, 10)]

    factory = OLogNFactory()

    algorithm = factory.get_algorithm("binary_search_timestamp", data, 3)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Binary Search by Timestamp
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(log n)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(1)
    print(f"Result: {algorithm.get_calculate()}")  # 输出: 30 (3 对应的 value)
