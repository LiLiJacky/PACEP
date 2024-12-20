from pybloom_live import BloomFilter
import time


class SimpleBloomFilterManager:
    def __init__(self, capacity, error_rate=0.01, reset_interval=36000):
        """
        初始化简单 Bloom Filter 管理器
        """
        self.capacity = capacity
        self.error_rate = error_rate
        self.reset_interval = reset_interval

        self.filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)
        self.last_reset_time = time.time()

    def add(self, key):
        """添加到 Bloom Filter"""
        self._reset_if_needed()
        self.filter.add(key)

    def check(self, key):
        """检查是否在 Bloom Filter 中"""
        self._reset_if_needed()
        return key in self.filter

    def _reset_if_needed(self):
        """
        检查是否需要重置 Bloom Filter
        """
        # 检查时间间隔是否超出重置周期
        current_time = time.time()
        if current_time - self.last_reset_time >= self.reset_interval:
            print("Resetting Bloom Filter due to time interval.")
            self._reset_filter()

        # 检查容量是否达到限制
        if len(self.filter) >= self.capacity:
            print("Bloom Filter is at capacity, resetting filter.")
            self._reset_filter()

    def _reset_filter(self):
        """
        重置 Bloom Filter
        """
        self.filter = BloomFilter(capacity=self.capacity, error_rate=self.error_rate)
        self.last_reset_time = time.time()