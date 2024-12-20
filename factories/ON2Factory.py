# algorithm_factory/ON2Factory.py
import numpy as np

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
        data = [row[1] for row in self.data]
        n = len(data)
        if n <= 1:
            return  -1

        sum_square_diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                sum_square_diff += (data[i] - data[j]) ** 2

        num = n * (n-1) / 2

        # 计算均值
        mean_value = sum(data) / len(data)

        # 计算总体方差 (除以 n)
        variance = sum((x - mean_value) ** 2 for x in data) / len(data)
        return sum_square_diff / num - 4 * variance**2

    def get_calculate_range(self, range):
        return range


class IncreasingSubsequencesDPAlgorithm(Algorithm):
    @property
    def name(self):
        return "Increasing Subsequences"

    @property
    def complexity_time(self):
        return "O(n^2 * k)"

    @property
    def complexity_space(self):
        return "O(n^2)"

    def get_calculate(self, min_times):
        data = np.array(self.data)
        if len(data) < min_times:
            return []

        # 转换数据为 numpy 数组并获取时间戳和对应的值列
        data = np.array(data)
        timestamps = data[:, 0].astype(int)  # 获取时间戳列
        values = data[:, 1].astype(float)  # 获取值列

        # dp[i] 存储以 values[i] 为结尾的所有递增子序列
        dp = [[] for _ in range(len(values))]

        # 每个元素本身是一个递增子序列
        for i in range(len(values)):
            dp[i].append([[timestamps[i], values[i]]])

        # 动态规划填充 dp 数组，查找非递减子序列
        for i in range(1, len(values)):
            for j in range(i):
                if values[j] < values[i]:  # 递增
                    # 直接扩展已有的递增子序列
                    for subseq in dp[j]:
                        dp[i].append(subseq + [[timestamps[i], values[i]]])

        # 收集所有递增子序列，排除单个元素的子序列
        result = []
        for subseqs in dp:
            for subseq in subseqs:
                if len(subseq) >= min_times:  # 排除单个元素子序列
                    result.append((subseq,))  # 包装为元组

        return result

    def get_calculate_range(self, range):
        # 假设 get_calculate_range 的功能是处理范围参数
        # 在这里返回的只是输入的范围，实际应用可以根据需求来调整
        return range

    def get_time_complexity(self, n):
        """
        预估最坏情况下的时间开销，基于 O(n^2) 时间复杂度。

        :param n: 输入数据的大小
        :return: 预估的时间开销
        """
        # 直接根据 O(n^2) 估算时间开销
        return n ** 2

class NonDecreasingSubsequencesDPAlgorithm(Algorithm):
    @property
    def name(self):
        return "NonDecreasing Subsequences"

    @property
    def complexity_time(self):
        return "O(n^2 * k)"

    @property
    def complexity_space(self):
        return "O(n^2)"

    def get_calculate(self, min_times):
        data = np.array(self.data)
        if len(data) < min_times:
            return []

        # 转换数据为 numpy 数组并获取时间戳和对应的值列
        data = np.array(data)
        timestamps = data[:, 0].astype(int)  # 获取时间戳列
        values = data[:, 1].astype(float)  # 获取值列

        # dp[i] 存储以 values[i] 为结尾的所有递增子序列
        dp = [[] for _ in range(len(values))]

        # 每个元素本身是一个递增子序列
        for i in range(len(values)):
            dp[i].append([[timestamps[i], values[i]]])

        # 动态规划填充 dp 数组，查找非递减子序列
        for i in range(1, len(values)):
            for j in range(i):
                if values[j] <= values[i]:  # 非递减条件
                    # 直接扩展已有的递增子序列
                    for subseq in dp[j]:
                        dp[i].append(subseq + [[timestamps[i], values[i]]])

        # 收集所有递增子序列，排除单个元素的子序列
        result = []
        for subseqs in dp:
            for subseq in subseqs:
                if len(subseq) >= min_times:  # 排除单个元素子序列
                    result.append((subseq,))  # 包装为元组

        return result

    def get_calculate_range(self, range):
        # 假设 get_calculate_range 的功能是处理范围参数
        # 在这里返回的只是输入的范围，实际应用可以根据需求来调整
        return range

    def get_time_complexity(self, n):
        """
        预估最坏情况下的时间开销，基于 O(n^2) 时间复杂度。

        :param n: 输入数据的大小
        :return: 预估的时间开销
        """
        # 直接根据 O(n^2) 估算时间开销
        return n ** 2

class NonIncreasingSubsequencesDPAlgorithm(Algorithm):
    @property
    def name(self):
        return "NonDecreasing Subsequences"

    @property
    def complexity_time(self):
        return "O(n^2 * k)"

    @property
    def complexity_space(self):
        return "O(n^2)"

    def get_calculate(self, min_times):
        data = np.array(self.data)
        if len(data) < min_times:
            return []

        # 转换数据为 numpy 数组并获取时间戳和对应的值列
        data = np.array(data)
        timestamps = data[:, 0].astype(int)  # 获取时间戳列
        values = data[:, 1].astype(float)  # 获取值列

        # dp[i] 存储以 values[i] 为结尾的所有递增子序列
        dp = [[] for _ in range(len(values))]

        # 每个元素本身是一个递增子序列
        for i in range(len(values)):
            dp[i].append([[timestamps[i], values[i]]])

        # 动态规划填充 dp 数组，查找非递减子序列
        for i in range(1, len(values)):
            for j in range(i):
                if values[j] >= values[i]:  # 非递增条件
                    # 直接扩展已有的递增子序列
                    for subseq in dp[j]:
                        dp[i].append(subseq + [[timestamps[i], values[i]]])

        # 收集所有递增子序列，排除单个元素的子序列
        result = []
        for subseqs in dp:
            for subseq in subseqs:
                if len(subseq) >= min_times:  # 排除单个元素子序列
                    result.append((subseq,))  # 包装为元组

        return result

    def get_calculate_range(self, range):
        # 假设 get_calculate_range 的功能是处理范围参数
        # 在这里返回的只是输入的范围，实际应用可以根据需求来调整
        return range

    def get_time_complexity(self, n):
        """
        预估最坏情况下的时间开销，基于 O(n^2) 时间复杂度。

        :param n: 输入数据的大小
        :return: 预估的时间开销
        """
        # 直接根据 O(n^2) 估算时间开销
        return n ** 2


class DecreasingSubsequencesDPAlgorithm(Algorithm):
    @property
    def name(self):
        return "NonDecreasing Subsequences"

    @property
    def complexity_time(self):
        return "O(n^2 * k)"

    @property
    def complexity_space(self):
        return "O(n^2)"

    def get_calculate(self, min_times):
        data = np.array(self.data)
        if len(data) < min_times:
            return []

        # 转换数据为 numpy 数组并获取时间戳和对应的值列
        data = np.array(data)
        timestamps = data[:, 0].astype(int)  # 获取时间戳列
        values = data[:, 1].astype(float)  # 获取值列

        # dp[i] 存储以 values[i] 为结尾的所有递增子序列
        dp = [[] for _ in range(len(values))]

        # 每个元素本身是一个递增子序列
        for i in range(len(values)):
            dp[i].append([[timestamps[i], values[i]]])

        # 动态规划填充 dp 数组，查找非递减子序列
        for i in range(1, len(values)):
            for j in range(i):
                if values[j] > values[i]:  # 非递减条件
                    # 直接扩展已有的递增子序列
                    for subseq in dp[j]:
                        dp[i].append(subseq + [[timestamps[i], values[i]]])

        # 收集所有递增子序列，排除单个元素的子序列
        result = []
        for subseqs in dp:
            for subseq in subseqs:
                if len(subseq) >= min_times:  # 排除单个元素子序列
                    result.append((subseq,))  # 包装为元组

        return result

    def get_calculate_range(self, range):
        # 假设 get_calculate_range 的功能是处理范围参数
        # 在这里返回的只是输入的范围，实际应用可以根据需求来调整
        return range

    def get_time_complexity(self, n):
        """
        预估最坏情况下的时间开销，基于 O(n^2) 时间复杂度。

        :param n: 输入数据的大小
        :return: 预估的时间开销
        """
        # 直接根据 O(n^2) 估算时间开销
        return n ** 2

class ON2Factory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "bubble_sort": BubbleSortAlgorithm,
            "sum_square_difference": SumSquareDifferenceAlgorithm,
            "increasing_dp": IncreasingSubsequencesDPAlgorithm,
            "decreasing_dp": DecreasingSubsequencesDPAlgorithm,
            "non_increasing_dp": NonIncreasingSubsequencesDPAlgorithm,
            "non_decreasing_dp": NonDecreasingSubsequencesDPAlgorithm,
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
