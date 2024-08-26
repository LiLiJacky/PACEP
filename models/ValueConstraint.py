from typing import List, Dict, Any

from interfaces.Constraint import Constraint

class ValueConstraint(Constraint):
    def __init__(self, variables: List[str], expression: str):
        super().__init__(variables)
        self.expression = expression

    def validate(self, data: Dict[str, Any]) -> bool:
        try:
            values = {var: data[var] for var in self.variables}
            return eval(self.expression, {}, values)
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def __eq__(self, other):
        if isinstance(other, ValueConstraint):
            return self.variables == other.variables and self.expression == other.expression
        return False

    def __hash__(self):
        return hash((tuple(self.variables), self.expression))

    def __str__(self):
        return (f"ValueConstraint(variables={self.variables}, "
                f"expression='{self.expression}')")