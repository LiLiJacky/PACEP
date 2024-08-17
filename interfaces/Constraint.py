from abc import ABC, abstractmethod
from typing import List, Any, Dict

class Constraint(ABC):
    def __init__(self, variables: List[str]):
        self.variables = variables

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        pass