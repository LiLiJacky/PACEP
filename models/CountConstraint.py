from typing import List, Dict, Any

from interfaces.Constraint import Constraint

class CountConstraint(Constraint):
    def __init__(self, variables: List[str], min_count: int, max_count: int):
        super().__init__(variables)
        self.min_count = min_count
        self.max_count = max_count
        if variables[-1] != variables[0]:
            self.expression = f"{min_count} <= {variables[-1]} - {variables[0]} <= {max_count}"
        else:
            self.expression = f"{min_count} <= size({variables[0]}) <= {max_count}"

    def validate(self, data, context) -> bool:
        kleene_name = self.variables[0] + "[i]"
        kleene_events = context.get_events_for_pattern(kleene_name)
        try:
            count = len(kleene_events)
            return self.min_count <= count <= self.max_count
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def __eq__(self, other):
        if isinstance(other, CountConstraint):
            return (self.variables == other.variables and
                    self.min_count == other.min_count and
                    self.max_count == other.max_count)
        return False

    def __hash__(self):
        return hash((tuple(self.variables), self.min_count, self.max_count))

    def __str__(self):
        return (f"CountConstraint(variables={self.variables}, "
                f"min_count={self.min_count}, max_count={self.max_count}, "
                f"expression='{self.expression}')")