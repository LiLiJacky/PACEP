from typing import List

from interfaces.Constraint import Constraint
from interfaces.DataItem import DataItem


class TimesConstraint(Constraint):
    def __init__(self, variables: List[str], min_times: int, max_times: int):
        super().__init__(variables)
        self.min_times = min_times
        self.max_times = max_times

    def validate(self, data: List[DataItem]) -> bool:
        try:
            count = data.__sizeof__()
            return self.min_times <= count <= self.max_times
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def __eq__(self, other):
        if isinstance(other, TimesConstraint):
            return (self.variables == other.variables and
                    self.min_times == other.min_times and
                    self.max_times == other.max_times)
        return False

    def __hash__(self):
        return hash((tuple(self.variables), self.min_times, self.max_times))

    def __str__(self):
        return (f"TimesConstraint(variables={self.variables}, "
                f"min_times={self.min_times}, max_times={self.max_times})")