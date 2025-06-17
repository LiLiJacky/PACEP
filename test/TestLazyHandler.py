from collections import defaultdict
from lazy_calculate.LazyHandler import LazyHandler  # 假设 LazyHandler 已经实现
from models.Data import Data
from models.ValueConstraint import ValueConstraint
from nfa.NFA import EventWrapper


class TestLazyHandler:
    def __init__(self):
        self.handler = LazyHandler()  # 初始化 LazyHandler
        self.shared_buffer_accessor = self._mock_shared_buffer_accessor()  # 模拟 SharedBufferAccessor
        self.test_data = self._prepare_test_data()  # 准备测试数据

    def _mock_shared_buffer_accessor(self):
        """
        模拟 SharedBufferAccessor，用于测试。
        """
        class MockSharedBufferAccessor:
            def __init__(self):
                self.event_registry = {}
                self.event_counter = 0

            def register_event(self, event, timestamp):
                self.event_counter += 1
                event_id = f"event_{self.event_counter}"
                self.event_registry[event_id] = (event, timestamp)
                return event_id

            def release_event(self, event_id):
                if event_id in self.event_registry:
                    del self.event_registry[event_id]

        return MockSharedBufferAccessor()

    def _prepare_test_data(self):
        """
        根据提供的数据结构创建测试用例
        """
        # 创建 Data 数据
        data_items_A = [
            Data(variable_name="a", value=10, timestamp=1731323471),
            Data(variable_name="a", value=25, timestamp=1731323472),
            Data(variable_name="a", value=15, timestamp=1731323475),
            Data(variable_name="a", value=20, timestamp=1731323476),
        ]
        data_items_B = [
            Data(variable_name="b", value=16, timestamp=1731323473),
            Data(variable_name="b", value=9, timestamp=1731323474),
            Data(variable_name="b", value=9, timestamp=1731323477),
        ]
        data_items_C = [
            Data(variable_name="c", value=10, timestamp=1731323479),
        ]

        # EventWrapper 的列表
        events_A = [EventWrapper(event=item, timestamp=item.get_timestamp(), shared_buffer_accessor=self.shared_buffer_accessor) for item in data_items_A]
        events_B = [EventWrapper(event=item, timestamp=item.get_timestamp(), shared_buffer_accessor=self.shared_buffer_accessor) for item in data_items_B]
        events_C = [EventWrapper(event=item, timestamp=item.get_timestamp(), shared_buffer_accessor=self.shared_buffer_accessor) for item in data_items_C]

        # 约束
        constraints = [
            ValueConstraint(['B'], '10<=average(B)<=1000'),
            ValueConstraint(['A'], '200<=sum_square_difference(A)<=1000'),
        ]

        return [
            [defaultdict(list, {'A': events_A, 'B': [events_B[2]], 'C': events_C}), constraints],
            [defaultdict(list, {'A': events_A[:2], 'B': events_B, 'C': events_C}), constraints],
            [defaultdict(list, {'A': events_A[1:], 'B': [events_B[2]], 'C': events_C}), constraints],
            [defaultdict(list, {'A': events_A[1:2], 'B': events_B, 'C': events_C}), constraints],
            [defaultdict(list, {'A': events_A[2:], 'B': [events_B[2]], 'C': events_C}), constraints],
            [defaultdict(list, {'A': [events_A[3]], 'B': [events_B[2]], 'C': events_C}), constraints],
        ]

    def test_expand_calculate_graph(self):
        result = self.test_data
        lazy_handler = LazyHandler()
        lazy_handler.create_calculate_graph()
        while result:
            basic_result = result.pop(0)
            lazy_calculate_constrains = []
            for c in basic_result[1]:
                if c not in lazy_calculate_constrains:
                    lazy_calculate_constrains.append(c)
            lazy_handler.expand_calculate_graph(basic_result[0], lazy_calculate_constrains)
        lazy_handler.calculate()
        for final_result in lazy_handler.get_final_results():
            result.append(final_result)

    def _print_blocks(self):
        """
        打印 LazyHandler 当前的块信息
        """
        print("Current Blocks:")
        for block in self.handler.get_blocks():
            print(block)


if __name__ == "__main__":
    # 初始化测试类并运行测试
    tester = TestLazyHandler()
    tester.test_expand_calculate_graph()