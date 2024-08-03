from abc import ABC, abstractmethod
from datetime import datetime

class DataItem(ABC):
    def __init__(self, variable_name, value, timestamp=None):
        self.variable_name = variable_name
        self.value = value
        self.timestamp = timestamp if timestamp else datetime.now()

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def get_timestamp(self):
        pass

    def __repr__(self):
        return f"DataItem(event_type={self.variable_name}, timestamp={self.timestamp}, data={self.value})"