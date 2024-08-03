import os

from generator.RegexPatternGenerator import RegexPatternGenerator
from simulated_data.SimulateData import DataSource
from util.NFADrawer import NFADrawer, png_output_path

if __name__ == "__main__":
    # Initialize the pattern generator
    generator = RegexPatternGenerator()

    # Generate regex and variable distributions
    num_variables = 5  # Number of regular variables
    num_kleene = 2  # Number of Kleene closure variables
    num_constrains = 2  # Number of constrains between variables

    regex = generator.generate_regex(num_variables, num_kleene)
    print(f"Generated Regex: {regex}")

    result = generator.set_constrains(regex, num_constrains)
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

    # Initialize and start data generation
    data_source = DataSource(regex, variables_domain)
    data_source.start_data_generation()
