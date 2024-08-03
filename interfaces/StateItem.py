from abc import ABC, abstractmethod

class StateItem(ABC):
    def __init__(self, name: str):
        self.name = name
        self.buffer = []

    @abstractmethod
    def add_data(self, data):
        """
        添加数据到状态的缓冲区。
        """
        pass

    @abstractmethod
    def get_buffer(self):
        """
        获取状态的缓冲区数据。
        """
        pass

    @abstractmethod
    def clear_buffer(self):
        """
        清空状态的缓冲区。
        """
        pass

    def __repr__(self):
        return f"State(name={self.name}, buffer={self.buffer})"