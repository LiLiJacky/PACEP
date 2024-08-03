# algorithm_factory/ON3Factory.py
from interfaces.Algorithm import Algorithm


class TripletProductSumAlgorithm(Algorithm):
    @property
    def name(self):
        return "Triplet Product Sum"

    @property
    def complexity_time(self):
        return "O(n^3)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        n = len(self.data)
        total_sum = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    total_sum += self.data[i][1] * self.data[j][1] * self.data[k][1]
        return total_sum

    def get_calculate_range(self, range):
        return [range[0] * 3, range[1] * 3]

class ON3Factory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "triplet_product_sum": TripletProductSumAlgorithm,
        }
        if algorithm_name in algorithms:
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


# 使用示例
if __name__ == "__main__":
    data = [(1, 1), (2, 2), (3, 3)]

    factory = ON3Factory()

    algorithm = factory.get_algorithm("triplet_product_sum", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Triplet Product Sum
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n^3)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(1)
    result = algorithm.get_calculate()
    print(f"Result: {result}")  # 输出: 216 (所有三元组 (a, b, c) 乘积的和：1*1*1 + 1*1*2 + ... + 3*3*3)
