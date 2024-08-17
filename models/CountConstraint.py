from typing import List, Dict, Any

from interfaces.Constraint import Constraint


class CountConstraint(Constraint):
    def __init__(self, variables: List[str], min_count: int, max_count: int):
        super().__init__(variables)
        self.min_count = min_count
        self.max_count = max_count
        self.expression = str(min_count) + " <= " + str(variables[len(variables) - 1]) + " - " + str(
            variables[0]) + " <= " + str(max_count)

    def validate(self, data: Dict[str, Any]) -> bool:
        try:
            count = len([var for var in self.variables if var in data])
            return self.min_count <= count <= self.max_count
        except Exception as e:
            print(f"Validation error: {e}")
            return False