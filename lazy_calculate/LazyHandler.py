import importlib
from collections import deque
from typing import List

from lazy_calculate.DataBlock import DataBlock
from lazy_calculate.DataEdge import Edge
from nfa.ComputationState import ComputationState
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor


class LazyHandler:
    def __init__(self):
        pass