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