from collections import deque
from typing import List

from lazy_calculate.DataBlock import DataBlock
from lazy_calculate.DataEdge import Edge
from nfa.ComputationState import ComputationState
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor


class LazyHandler:
    def __init__(self):
        self.blocks = []  # 初始化存储所有的 DataBlock 实例
        self.final_results = []  # 存储最终计算结果
        self.start_block = []  # 存储起始块
        self.lazy_constraints = {}  # 存储延迟处理的约束


    def create_calculate_graph(self):
        pass

    def expand_calculate_graph(self, basic_result, lazy_constraints):
        print(basic_result)
        # TODO 这里假设所有延迟处理的约束都一样
        if len(lazy_constraints) == 0:
            self.lazy_constraints = lazy_constraints

        for state, events in basic_result.items():
            # 将当前状态数据与已有块对比，处理重叠和新增
            self._process_state_blocks(state, events)

        pass

    def calculate(self):
        pass

    def get_final_results(self):
        return self.final_results

    def _process_state_blocks(self, state, events):
        """
        处理某个状态的块，检查子集关系并拆分数据块。
        :param state: 当前状态名称
        :param events: 当前状态的数据列表
        """
        new_blocks = []
        event_set = set(events)  # 将事件转换为集合方便操作

        for block in self.blocks:
            if block.state_name != state:
                continue

            block_content = set(block.content)

            if event_set.issubset(block_content):
                # 如果事件完全属于块，跳过
                print(f"Skipping block {block.unique_id}, as events are a subset.")
                return

            if block_content & event_set:
                # 如果存在重叠，拆分为三个部分
                overlap = block_content & event_set
                left_content = block_content - overlap
                right_content = event_set - overlap

                print(
                    f"Splitting block {block.unique_id}: Left={left_content}, Overlap={overlap}, Right={right_content}")

                # 创建新块
                if left_content:
                    left_block = DataBlock(state_name=state, content=list(left_content))
                    new_blocks.append(left_block)
                    Edge(left_block, block, label="LeftOverlap")

                if right_content:
                    right_block = DataBlock(state_name=state, content=list(right_content))
                    new_blocks.append(right_block)
                    Edge(block, right_block, label="RightOverlap")

                # 原块只保留重叠部分
                block.content = list(overlap)

        # 如果没有重叠，直接添加新块
        if not any(event_set & set(block.content) for block in self.blocks):
            new_block = DataBlock(state_name=state, content=events)
            new_blocks.append(new_block)
            print(f"Created new block for state {state}: {new_block}")

        # 更新块集合
        self.blocks.extend(new_blocks)


    def get_blocks(self):
        """获取当前所有块的列表"""
        return self.blocks