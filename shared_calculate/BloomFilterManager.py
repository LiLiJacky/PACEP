from pybloom_live import BloomFilter
import time


class BloomFilterManager:
    def __init__(self, capacity, error_rate=0.01, reset_interval=3600):
        """
        初始化 Bloom Filter 管理器
        """
        self.capacity = capacity
        self.error_rate = error_rate
        self.reset_interval = reset_interval

        self.calculated_filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)  # 第一层
        self.true_filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)  # 第二层 True
        self.false_filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)  # 第二层 False

        self.last_reset_time = time.time()

    def add_calculated(self, expression, combination):
        """添加到第一层 Bloom Filter"""
        self._reset_or_expand_if_needed()
        key = self._generate_key(expression, combination)
        self.calculated_filter.add(key)

    def is_calculated(self, expression, combination):
        """检查是否已计算过"""
        self._reset_or_expand_if_needed()
        key = self._generate_key(expression, combination)
        return key in self.calculated_filter

    def add_result(self, expression, combination, result):
        """根据结果添加到第二层"""
        self._reset_or_expand_if_needed()
        key = self._generate_key(expression, combination)
        if result:
            self.true_filter.add(key)
        else:
            self.false_filter.add(key)

    def get_result(self, expression, combination):
        """查询第二层，返回 True/False 或 None"""
        self._reset_or_expand_if_needed()
        key = self._generate_key(expression, combination)
        if key in self.true_filter:
            return True
        if key in self.false_filter:
            return False
        return None

    def _reset_or_expand_if_needed(self):
        """
        检查是否需要重置或扩展 Bloom Filter
        """
        # 检查时间间隔是否超出重置周期
        current_time = time.time()
        if current_time - self.last_reset_time >= self.reset_interval:
            print("Resetting Bloom Filters due to time interval.")
            self._reset_filters()
            return

        # 检查容量是否达到限制
        if len(self.calculated_filter) >= self.capacity:
            print("Bloom Filter is at capacity, resetting filters.")
            self._reset_filters()

    def _reset_filters(self):
        """
        重置所有 Bloom Filters
        """
        self.calculated_filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)
        self.true_filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)
        self.false_filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)
        self.last_reset_time = time.time()

    def _generate_key(self, expression, combination):
        """
        生成唯一键，用于标识 expression 和 combination
        """
        return f"{expression}-{hash(combination)}"