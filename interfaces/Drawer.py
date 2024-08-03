# interfaces/drawer_interface.py
from abc import ABC, abstractmethod

class Drawer(ABC):
    def __init__(self):
        self.graph = None

    @abstractmethod
    def draw(self, pattern: str):
        """
        将模式绘制成图形。
        :param pattern: 输入的模式
        """
        pass

    @abstractmethod
    def render(self, filename: str, format: str):
        """
        渲染并保存图形。
        :param filename: 输出文件名
        :param format: 输出文件格式
        """
        pass
