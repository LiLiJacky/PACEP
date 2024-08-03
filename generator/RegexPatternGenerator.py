import random
import re
import string

from factories.AlgorithmFactory import AlgorithmFactory
from generator.DataGenerator import DataGenerator


class RegexPatternGenerator:
    def __init__(self):
        self.variables = []
        self.variable_names = []
        self.variable_domains = {}
        self.algorithm_factory = AlgorithmFactory()
        self.regex = ""

    def generate_variable(self, name, quantifier=None):
        """
        生成变量的正则表达式表示。
        :param name: 变量名
        :param quantifier: 量词类型 (None, '*', '+', '?', '{n}', '{n,}', '{n,m}', '{,m}')
        :return: 变量的正则表达式
        """
        if quantifier:
            return f"{name}{quantifier}"
        else:
            return name

    def generate_quantifier(self):
        """
        随机生成一个量词。
        :return: 量词字符串
        """
        quantifiers = ['*', '+', '?', '{1}', '{2,}', '{3,5}', '{,4}']
        return random.choice(quantifiers)

    def generate_regex(self, num_variables, num_kleene):
        """
        生成一个带有指定数量常规变量和量词变量的正则表达式。
        :param num_variables: 常规变量数量
        :param num_kleene: 量词变量数量
        :return: 正则表达式
        """
        available_letters = list(string.ascii_uppercase)
        self.variables = []

        all_vars = [self.generate_variable(letter) for letter in available_letters[:num_variables]]
        kleene_vars = [self.generate_variable(letter, self.generate_quantifier()) for letter in
                       available_letters[num_variables:num_variables + num_kleene]]

        # 为每个变量生成定义域，并存储在字典中
        for var in all_vars:
            variable_info = DataGenerator(var)
            self.variable_domains[var] = variable_info
        for var_kleene in kleene_vars:
            letters = re.findall(r'[A-Za-z]', var_kleene)
            var = ''.join(letters)
            variable_info = DataGenerator(var)
            self.variable_domains[var] = variable_info

        all_vars.extend(kleene_vars)
        random.shuffle(all_vars)

        self.regex = ' '.join(all_vars)
        self.variable_names = [var[0] for var in all_vars]
        return self.regex

    def parse_regex_variables(self, regex):
        """
        从正则表达式中提取变量名。
        :param regex: 正则表达式
        :return: 变量名列表
        """
        pattern = re.compile(r"\b([A-Z])\b")
        return pattern.findall(regex)

    def select_algorithm(self, variable, quantifier):
        """
        为变量选择合适的求值算法。
        :param variable: 变量名
        :param quantifier: 量词
        :return: 算法名称和附加参数
        """
        algorithms = self.algorithm_factory.get_algorithm_list()
        range = self.variable_domains.get(variable)["value_range"]
        algorithms_description = variable

        if quantifier == "":
            value_range = range
        else:
            selected_algorithm = random.choice(algorithms)
            algorithms_factory = AlgorithmFactory()
            full_name = algorithms_factory.get_algorithm_sub_map().get(selected_algorithm)
            algorithms_description = f"{selected_algorithm}({variable})"
            value_range = range
            if full_name:
                if selected_algorithm == 'nth' or 'sort' in selected_algorithm:
                    index = random.randint(1, 10)  # 示例索引
                    algorithms_description = f"{selected_algorithm}({variable})[{index}]"
                    algorithm = self.algorithm_factory.get_algorithm(full_name, [], 0)
                elif selected_algorithm == 'combinations_square':
                    length = random.randint(2, 5)  # 示例长度
                    algorithms_description = f"combinations_square({variable})[{length}]"
                    algorithm = self.algorithm_factory.get_algorithm(full_name, [], 1)
                else:
                    algorithm = self.algorithm_factory.get_algorithm(full_name, [])
                value_range = algorithm.get_calculate_range(range)

        result = {
            "algorithm_description": algorithms_description,
            "value_range": value_range
        }

        return result

    def set_constrains(self, regex, num_constrains):
        """
        设置变量之间的约束条件，并返回变量和求值方法的结构。
        :param regex: 生成的正则表达式
        :param num_constrains: 约束条件数量
        :return: 包含变量和求值方法的结构
        """
        variables = self.parse_regex_variables(regex)
        constrains = []

        for _ in range(num_constrains):
            num_vars_in_constrain = random.randint(2, len(variables))
            selected_vars = variables[:num_vars_in_constrain]

            constrain_expr_parts = []
            for var in selected_vars:
                quantifier_match = re.search(rf"{var}([*+?{{\d+,?\d*}}]*)", regex)
                quantifier = quantifier_match.group(1) if quantifier_match else ''
                algo_expression = self.select_algorithm(var, quantifier)
                constrain_expr_parts.append(algo_expression)

            # 为每个变量生成一个在其范围内的随机值
            random_values = [random.randint(int(var.get("value_range")[0]), int(var.get("value_range")[1])) for var in constrain_expr_parts]
            # 计算这些随机值的总和
            total_sum = sum(random_values)
            lef = total_sum - 10

            constrain_expr_par = [item['algorithm_description'] for item in constrain_expr_parts]
            constrain_expr = str(lef) + " < " + ' + '.join(constrain_expr_par) + " < " + str(total_sum)
            constrains.append({
                "variables": selected_vars,
                "constrain": constrain_expr
            })

        return {
            "variables": variables,
            "regex": regex,
            "constrains": constrains,
            "domain": self.variable_domains
        }


# 使用示例
if __name__ == "__main__":
    generator = RegexPatternGenerator()

    num_variables = 5
    num_kleene = 2
    num_constrains = 2

    regex = generator.generate_regex(num_variables, num_kleene)
    print(f"Generated Regex: {regex}")

    result = generator.set_constrains(regex, num_constrains)
    print(f"Variables: {result['variables']}")
    print(f"Regex: {result['regex']}")
    print(f"Constrains: {result['constrains']}")
    print(f"Variables_Domain: {result['domain']}")
