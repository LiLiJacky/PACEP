# algorithm_factory/ON2Factory.py
from interfaces.Algorithm import Algorithm


class BubbleSortAlgorithm(Algorithm):
    @property
    def name(self):
        return "Bubble Sort"

    @property
    def complexity_time(self):
        return "O(n^2)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self, index):
        data = self.data[:]
        n = len(data)
        if index >= n:
            return -1
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j][1] > data[j + 1][1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data[index][1]

    def get_calculate_range(self, range):
        return range

class SumSquareDifferenceAlgorithm(Algorithm):
    @property
    def name(self):
        return "Sum Square Difference"

    @property
    def complexity_time(self):
        return "O(n^2)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        data = self.data[:]
        n = len(data)
        sum_square_diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                sum_square_diff += (data[i][1] - data[j][1]) ** 2
        return sum_square_diff

    def get_calculate_range(self, range):
        return range

class ON2Factory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "bubble_sort": BubbleSortAlgorithm,
            "sum_square_difference": SumSquareDifferenceAlgorithm
        }
        if algorithm_name in algorithms:
            if algorithm_name == "bubble_sort":
                if len(args) == 1:
                    return algorithms[algorithm_name](data)
                else:
                    raise ValueError("binary_search_timestamp algorithm requires one argument for the target timestamp")
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def get_time_complexity(self, n):
        return n**2

# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 20), (3, 30), (4, 40), (5, 10)]

    factory = ON2Factory()

    algorithm = factory.get_algorithm("bubble_sort", data, 2)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Bubble Sort
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n^2)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(1)
    sorted_data = algorithm.get_calculate(2)
    print(f"Sorted Data: {sorted_data}")  # 输出: [(5, 10), (2, 20), (3, 30), (4, 40), (1, 50)]
