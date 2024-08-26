import os
import time
from multiprocessing import Queue, Process
from generator.RegexPatternGenerator import RegexPatternGenerator
from sharedbuffer.EventId import EventId
from simulated_data.SimulateData2 import DataSource, data_consumer
from util.NFADrawer import NFADrawer, png_output_path
from nfa.ABRegexToStateAppend import RegexToState
from nfa.NFA import NFA, TimerService
from sharedbuffer.SharedBuffer import SharedBuffer, SharedBufferCacheConfig
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from models.Data import Data

if __name__ == "__main__":
    # Initialize the pattern generator
    generator = RegexPatternGenerator()

    # Generate regex and variable distributions
    num_variables = 5  # Number of regular variables
    num_kleene = 2  # Number of Kleene closure variables
    num_value_constrains = 2  # Number of constrains between variables
    num_time_constrains = 2
    num_count_constrains = 0

    regex = generator.generate_regex(num_variables, num_kleene)
    print(f"Generated Regex: {regex}")

    result = generator.set_constrains(regex, num_value_constrains, num_count_constrains, num_time_constrains)
    variables_domain = result.get("domain")
    variables_value_constrain = result.get("constrains")

    print(f"Generated Regex: {regex}")
    print(f"Variables Domain: {variables_domain}")
    print(f"Variables Value Constrain: {variables_value_constrain}")

    # Draw NFA from the generated regex
    nfa_drawer = NFADrawer()
    nfa_drawer.draw(regex)
    # 设置输出路径
    output_path = os.path.join(png_output_path, regex)
    # 渲染图形并保存为 PNG 文件
    nfa_drawer.render(output_path, format='png')

    # Initialize the data queue
    data_queue = Queue()

    # Initialize and start data generation
    data_source = DataSource(regex, variables_domain)
    processes = data_source.start_data_generation(data_queue)

    # 使用 RegexToState 类生成状态和转换
    regex_to_state = RegexToState(regex, variables_value_constrain)
    regex_to_state.generate_states_and_transitions()

    # 获取生成的状态和转换
    states = regex_to_state.states
    transitions = regex_to_state.transitions

    # 使用生成的状态和转换创建 NFA
    nfa = NFA(states, {}, 0, handle_timeout=False)

    # 配置 SharedBuffer
    config = SharedBufferCacheConfig.from_config('config.ini')
    shared_buffer = SharedBuffer(config)
    accessor = SharedBufferAccessor(shared_buffer)

    # 创建 TimerService 实例
    timer_service = TimerService()

    # 创建初始的 NFA 状态
    nfa_state = nfa.create_initial_nfa_state()

    try:
        while True:
            data_item = data_queue.get()
            print(data_item)
            if data_item:
                event_id = EventId(data_item.variable_name, int(data_item.timestamp.timestamp() * 1000))
                matches = nfa.process(accessor, nfa_state, event_id, int(data_item.timestamp.timestamp() * 1000), None, timer_service)
                if matches:
                    print(f"Matches found: {matches}")
            time.sleep(0.1)  # To prevent the main loop from consuming too much CPU
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()