import os
import time
from multiprocessing import Queue

from aftermatch.NoSkipStrategy import NoSkipStrategy
from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from generator.RegexPatternGenerator import RegexPatternGenerator
from nfa.NFA import NFA, TimerService, EventWrapper
from nfa.ABRegexToStateAppend import RegexToState
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from simulated_data.SimulateData2 import DataSource
from util.NFADrawer import NFADrawer, png_output_path

if __name__ == "__main__":
    # Initialize the pattern generator
    generator = RegexPatternGenerator()

    # Generate regex and variable distributions
    num_variables = 0  # Number of regular variables
    num_kleene = 2  # Number of Kleene closure variables
    num_value_constrains = 2  # Number of constrains between variables
    num_time_constrains = 1
    num_count_constrains = 0

    regex = generator.generate_regex(num_variables, num_kleene)
    print(f"Generated Regex: {regex}")

    result = generator.set_constraints(regex, num_value_constrains, num_count_constrains, num_time_constrains)
    variables_domain = result.get("domain")
    variables_constrain = result.get("constraints")
    window_time = generator.window_time

    print(f"Generated Regex: {regex}")
    print(f"Variables Domain: {variables_domain}")
    print(f"Variables Value Constrain: {variables_constrain}")

    # Draw NFA from the generated regex
    # nfa_drawer = NFADrawer()
    # nfa_drawer.draw(regex)
    # # 设置输出路径
    # output_path = os.path.join(png_output_path, regex, "_according_regex")
    # # 渲染图形并保存为 PNG 文件
    # nfa_drawer.render(output_path, format='png')

    # Initialize the data queue
    data_queue = Queue()

    # Initialize and start data generation
    data_source = DataSource(regex, variables_domain)
    processes = data_source.start_data_generation(data_queue)

    # 使用 RegexToState 类生成状态和转换
    regex_to_state = RegexToState(regex, variables_constrain)
    regex_to_state.generate_states_and_transitions()
    valid_states = regex_to_state.states
    regex_to_state.draw()

    # 使用生成的状态和转换创建 NFA
    nfa = NFA(valid_states, {}, window_time, handle_timeout=True)

    # 配置 SharedBuffer
    config = SharedBufferCacheConfig.from_config('config.ini')
    shared_buffer = SharedBuffer(config)
    accessor = SharedBufferAccessor(shared_buffer)

    # 创建 TimerService 实例
    timer_service = TimerService()

    # 创建初始 NFA 状态
    nfa_state = nfa.create_initial_nfa_state()

    # 模拟数据流处理
    try:
        while True:
            data_item = data_queue.get()
            print(data_item)
            if data_item:
                # 封装事件并处理
                event_wrapper = EventWrapper(data_item, data_item.timestamp, accessor)
                # 处理事件并更新 NFA 状态
                matches = nfa.process(accessor, nfa_state, event_wrapper, data_item.timestamp, NoSkipStrategy.INSTANCE,  TimerService())
                # 修剪超过时间窗口限制的部分匹配结果
                nfa.advance_time(accessor, nfa_state, event_wrapper.timestamp, NoSkipStrategy.INSTANCE)
                print(f"Now PartialMatches: {len(nfa_state.partial_matches)}")

                if matches:
                    print(f"Match found: {matches}")
                    break  # 如果匹配成功，可以选择终止处理，或者继续处理其他事件
            time.sleep(0.001)  # To prevent the main loop from consuming too much CPU
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()


    print("Pattern recognition complete.")
