from abc import ABC, abstractmethod
from typing import List, Any, Dict

class Constraint(ABC):
    def __init__(self, variables: List[str]):
        self.variables = variables

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        pass

    def __eq__(self, other):
        if isinstance(other, Constraint):
            return self.variables == other.variables
        return False

    def __hash__(self):
        return hash(tuple(self.variables))

    def __str__(self):
        return f"Constraint(variables={self.variables})"