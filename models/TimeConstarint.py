from typing import List, Any, Dict

from interfaces.Constraint import Constraint


class TimeConstraint(Constraint):
    def __init__(self, variables: List[str], min_time: int, max_time: int):
        super().__init__(variables)
        self.min_time = min_time
        self.max_time = max_time
        self.expression = str(min_time) + " <= " + str(variables[len(variables) - 1]) + " - " + str(variables[0]) + " <= " + str(max_time)

    def validate(self, data: Dict[str, Any]) -> bool:
        try:
            times = [data[var]['timestamp'] for var in self.variables]
            time_span = max(times) - min(times)
            return self.min_time <= time_span <= self.max_time
        except Exception as e:
            print(f"Validation error: {e}")
            return False