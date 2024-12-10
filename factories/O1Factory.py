# algorithm_factory/O1Factory.py
import random
from interfaces.Algorithm import Algorithm


class FirstValueAlgorithm(Algorithm):
    @property
    def name(self):
        return "First Value"

    @property
    def complexity_time(self):
        return "O(1)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        return self.data[0][1]  # 返回第一个值的 value

    def get_calculate_range(self, range):
        return range


class LastValueAlgorithm(Algorithm):
    @property
    def name(self):
        return "Last Value"

    @property
    def complexity_time(self):
        return "O(1)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        return self.data[-1][1]  # 返回最后一个值的 value

    def get_calculate_range(self, range):
        return range


class RandomValueAlgorithm(Algorithm):
    @property
    def name(self):
        return "Random Value"

    @property
    def complexity_time(self):
        return "O(1)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        return random.choice(self.data)[1]  # 返回随机一个值的 value

    def get_calculate_range(self, range):
        return range


class NthValueAlgorithm(Algorithm):
    def __init__(self, data, n):
        super().__init__(data)
        self.n = n

    @property
    def name(self):
        return f"{self.n}th Value"

    @property
    def complexity_time(self):
        return "O(1)"

    @property
    def complexity_space(self):
        return "O(1)"

    def get_calculate(self):
        if 0 <= self.n < len(self.data):
            return self.data[self.n][1]  # 返回第 n 个值的 value
        else:
            raise IndexError("Index out of range")

    def get_calculate_range(self, range):
        return range


class O1Factory:
    def get_algorithm(self, algorithm_name, data, *args):
        algorithms = {
            "first_value": FirstValueAlgorithm,
            "last_value": LastValueAlgorithm,
            "random_value": RandomValueAlgorithm,
            "nth_value": NthValueAlgorithm,
        }
        if algorithm_name in algorithms:
            if algorithm_name == "nth_value":
                if len(args) == 1:
                    return algorithms[algorithm_name](data, args[0])
                else:
                    raise ValueError("nth_value algorithm requires one argument for the index")
            return algorithms[algorithm_name](data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def get_time_complexity(self, n):
        return n
