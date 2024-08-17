import copy
from typing import List

class DeweyNumber:
    def __init__(self, start):
        if isinstance(start, int):
            self.dewey_number = [start]
        elif isinstance(start, DeweyNumber):
            self.dewey_number = copy.deepcopy(start.dewey_number)
        elif isinstance(start, list):
            self.dewey_number = start
        else:
            raise TypeError("Unsupported type for initialization")

    @staticmethod
    def from_string(dewey_number_string: str) -> 'DeweyNumber':
        splits = dewey_number_string.split(".")
        if len(splits) == 1:
            return DeweyNumber(int(dewey_number_string))
        elif len(splits) > 0:
            dewey_number = [int(part) for part in splits]
            return DeweyNumber(dewey_number)
        else:
            raise ValueError(f"Failed to parse {dewey_number_string} as a Dewey number")

    def is_compatible_with(self, other: 'DeweyNumber') -> bool:
        if self.length() > other.length():
            # prefix case
            return all(self.dewey_number[i] == other.dewey_number[i] for i in range(other.length()))
        elif self.length() == other.length():
            # check init digits for equality
            last_index = self.length() - 1
            return (all(self.dewey_number[i] == other.dewey_number[i] for i in range(last_index)) and
                    self.dewey_number[last_index] >= other.dewey_number[last_index])
        else:
            return False

    def get_run(self) -> int:
        return self.dewey_number[0]

    def length(self) -> int:
        return len(self.dewey_number)

    def increase(self, times: int = 1) -> 'DeweyNumber':
        new_dewey_number = copy.deepcopy(self.dewey_number)
        new_dewey_number[-1] += times
        return DeweyNumber(new_dewey_number)

    def add_stage(self) -> 'DeweyNumber':
        new_dewey_number = copy.deepcopy(self.dewey_number)
        new_dewey_number.append(0)
        return DeweyNumber(new_dewey_number)

    def __eq__(self, other) -> bool:
        if isinstance(other, DeweyNumber):
            return self.dewey_number == other.dewey_number
        return False

    def __hash__(self) -> int:
        return hash(tuple(self.dewey_number))

    def __str__(self) -> str:
        return ".".join(map(str, self.dewey_number))


if __name__ == "__main__":
    # 示例使用
    dn1 = DeweyNumber(1)
    dn2 = DeweyNumber(dn1)
    dn3 = dn1.increase()
    dn4 = dn3.add_stage()
    dn5 = DeweyNumber.from_string("1.2.3")

    print(dn1)  # 输出: 1
    print(dn2)  # 输出: 1
    print(dn3)  # 输出: 2
    print(dn4)  # 输出: 2.0
    print(dn5)  # 输出: 1.2.3
    print(dn5.is_compatible_with(dn3))  # 输出: False