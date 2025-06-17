from lazy_calculate.TableTool import TableTool
from nfa.NFA import EventWrapper  # 假设 EventWrapper 来自于该模块
from datetime import datetime

if __name__ == '__main__':
    table_tool = TableTool()
    print(table_tool.ensure_hashable({'a': ['2024-11-11 11:20:25', '2024-11-11 11:20:30', '2024-11-11 11:20:42', '2024-11-11 11:20:48', '2024-11-11 11:21:09'], 'b': ['2024-11-11 11:21:13', '2024-11-11 11:21:15', '2024-11-11 11:21:35', '2024-11-11 11:21:55', '2024-11-11 11:21:59']}))
    s = {'a': ['2024-11-11 11:20:25', '2024-11-11 11:20:30', '2024-11-11 11:20:42', '2024-11-11 11:20:48', '2024-11-11 11:21:09'], 'b': ['2024-11-11 11:21:13', '2024-11-11 11:21:15', '2024-11-11 11:21:35', '2024-11-11 11:21:55', '2024-11-11 11:21:59']}
    s_q = table_tool.ensure_hashable(s)
    s_t = table_tool.ensure_hashable(s)
    print(s_q == s_t)