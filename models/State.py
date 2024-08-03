from interfaces.StateItem import StateItem


class State(StateItem):
    def __init__(self, name: str):
        super().__init__(name)

    def add_data(self, data):
        """
        添加数据到状态的缓冲区。
        """
        self.buffer.append(data)

    def get_buffer(self):
        """
        获取状态的缓冲区数据。
        """
        return self.buffer

    def clear_buffer(self):
        """
        清空状态的缓冲区。
        """
        self.buffer = []