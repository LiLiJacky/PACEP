# algorithm_factory/NLogNFactory.py
from interfaces.Algorithm import Algorithm


class MergeSortAlgorithm(Algorithm):
    @property
    def name(self):
        return "Merge Sort"

    @property
    def complexity_time(self):
        return "O(n log n)"

    @property
    def complexity_space(self):
        return "O(n)"

    def get_calculate(self):
        return self.merge_sort(self.data)

    def merge_sort(self, data):
        if len(data) <= 1:
            return data

        mid = len(data) // 2
        left = self.merge_sort(data[:mid])
        right = self.merge_sort(data[mid:])

        return self.merge(left, right)

    def merge(self, left, right):
        sorted_list = []
        while left and right:
            if left[0][1] <= right[0][1]:
                sorted_list.append(left.pop(0))
            else:
                sorted_list.append(right.pop(0))

        sorted_list.extend(left if left else right)
        return sorted_list

    def get_calculate_range(self, range):
        return range

class NLogNFactory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "merge_sort": MergeSortAlgorithm,
        }
        if algorithm_name in algorithms:
            if algorithm_name == "merge_sort":
                if len(args) == 1:
                    return algorithms[algorithm_name](data, args[0])
                else:
                    raise ValueError("merge_sort algorithm requires one argument for the index")
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


# 使用示例
if __name__ == "__main__":
    data = [(1, 50), (2, 20), (3, 30), (4, 40), (5, 10)]

    factory = NLogNFactory()

    algorithm = factory.get_algorithm("merge_sort", data)
    print(f"Algorithm Name: {algorithm.name}")  # 输出: Merge Sort
    print(f"Time Complexity: {algorithm.complexity_time}")  # 输出: O(n log n)
    print(f"Space Complexity: {algorithm.complexity_space}")  # 输出: O(n)
    sorted_data = algorithm.get_calculate()
    print(f"Sorted Data: {sorted_data}")  # 输出: [(5, 10), (2, 20), (3, 30), (4, 40), (1, 50)]
