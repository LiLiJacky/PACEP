import os

from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from generator.RegexPatternGenerator import RegexPatternGenerator
from nfa.NFA import NFA, TimerService
from nfa.ABRegexToStateAppend import RegexToState
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from simulated_data.SimulateData2 import DataSource
from util.NFADrawer import NFADrawer, png_output_path

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
    output_path = os.path.join(png_output_path, regex, "_according_regex")
    # 渲染图形并保存为 PNG 文件
    nfa_drawer.render(output_path, format='png')

    # Initialize and start data generation
    data_source = DataSource(regex, variables_domain)
    data_source.start_data_generation()

    # 使用 RegexToState 类生成状态和转换
    regex_to_state = RegexToState(regex, variables_value_constrain)
    regex_to_state.generate_states_and_transitions()

    # 根据状态和边约束画图
    regex_to_state.draw()

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

    # 开始数据生成并进行模式识别
    data_stream = data_source.start_data_generation()  # 假设这个方法返回生成的数据流
    nfa_state = nfa.create_initial_nfa_state()

    for event in data_stream:
        event_id, timestamp, _ = event
        matches = nfa.process(accessor, nfa_state, event_id, timestamp, None, timer_service)
        if matches:
            print(f"Matches found: {matches}")

    print("Pattern recognition complete.")
