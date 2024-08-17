# interfaces/Algorithm.py
from abc import ABC, abstractmethod

class Algorithm(ABC):
    def __init__(self, data, *args):
        self.data = data

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def complexity_time(self):
        pass

    @property
    @abstractmethod
    def complexity_space(self):
        pass

    @abstractmethod
    def get_calculate(self):
        pass

    @abstractmethod
    def get_calculate_range(self, data):
        pass