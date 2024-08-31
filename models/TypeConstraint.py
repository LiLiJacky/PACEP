from typing import List

from interfaces.Constraint import Constraint
from interfaces.DataItem import DataItem


class TypeConstraint(Constraint):
    def __init__(self, variables: List[str], variables_name: List[str]):
        super().__init__(variables)
        self.variables_name = variables_name

    def validate(self, data, context) -> bool:
        try:
            return data.event.variable_name in self.variables_name
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def __eq__(self, other):
        if isinstance(other, TypeConstraint):
            return self.variables == other.variables and self.variables_name == other.variables_name
        return False

    def __hash__(self):
        return hash((tuple(self.variables), tuple(self.variables_name)))

    def __str__(self):
        return (f"TypeConstraint(variables={self.variables}, "
                f"variables_name={self.variables_name})")