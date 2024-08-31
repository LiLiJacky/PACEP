from abc import ABC, abstractmethod
from datetime import datetime

class DataItem(ABC):
    def __init__(self, variable_name, value, timestamp=None):
        self.variable_name = variable_name
        self.value = value
        if isinstance(timestamp, datetime):
            self.timestamp = int(timestamp.timestamp())  # 将 datetime 转换为时间戳
        else:
            self.timestamp = int(timestamp) if isinstance(timestamp, (int, float)) else int(
                datetime.now().timestamp())

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def get_timestamp(self):
        pass

    def __repr__(self):
        return f"DataItem(event_type={self.variable_name}, timestamp={self.timestamp}, data={self.value})"