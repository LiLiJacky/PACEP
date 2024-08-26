import random
import re
import string
from datetime import datetime

from factories.AlgorithmFactory import AlgorithmFactory
from generator.DataGenerator import DataGenerator
from models.ConstraintCollection import ConstraintCollection
from models.CountConstraint import CountConstraint
from models.TimeConstarint import TimeConstraint
from models.TypeConstraint import TypeConstraint
from models.ValueConstraint import ValueConstraint


class RegexPatternGenerator:
    def __init__(self):
        self.variables = []
        self.variable_names = []
        self.variable_domains = {}
        self.algorithm_factory = AlgorithmFactory()
        self.regex = ""
        self.constrains_collection = ConstraintCollection()

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
                    value_range = algorithm.get_calculate_range(range)
                else:
                    algorithm = self.algorithm_factory.get_algorithm(full_name, [])
                    value_range = algorithm.get_calculate_range(range)

        result = {
            "algorithm_description": algorithms_description,
            "value_range": value_range
        }

        return result

    def set_constraints(self, regex, num_value_constraints, num_count_constraints, num_time_constraints):
        """
        设置变量之间的约束条件，并返回变量和求值方法的结构。
        :param regex: 生成的正则表达式
        :param num_value_constraints: 数值约束条件数量
        :param num_count_constraints: 计数约束条件数量
        :param num_time_constraints: 时间约束条件数量
        :return: 包含变量和求值方法的结构
        """
        variables = self.parse_regex_variables(regex)
        constraintCollection = ConstraintCollection()

        # 保持变量顺序与正则表达式中的一致
        all_vars = sorted(variables, key=lambda var: regex.index(var))

        # 生成 value_constraint
        for _ in range(num_value_constraints):
            num_vars_in_constraint = random.randint(2, len(all_vars))
            selected_vars = random.sample(all_vars, num_vars_in_constraint)
            selected_vars = sorted(selected_vars, key=lambda var: regex.index(var))

            value_constraint_expr_parts = []
            for var in selected_vars:
                quantifier_match = re.search(rf"{var}([*+?{{\d+,?\d*}}]*)", regex)
                quantifier = quantifier_match.group(1) if quantifier_match else ''
                algo_expression = self.select_algorithm(var, quantifier)
                value_constraint_expr_parts.append(algo_expression)

            random_values = [random.randint(int(var.get("value_range")[0]), int(var.get("value_range")[1])) for var in
                             value_constraint_expr_parts]
            total_sum = sum(random_values)
            left = total_sum - 10

            value_constraint_expr_par = [item['algorithm_description'] for item in value_constraint_expr_parts]
            value_constraint_expr = str(left) + " < " + ' + '.join(value_constraint_expr_par) + " < " + str(total_sum)
            constraintCollection.add_constraint(ValueConstraint(
                variables=selected_vars,
                expression=value_constraint_expr
            ))

        # 生成上下界约束
        for _ in range(num_value_constraints):
            var = random.choice(all_vars)
            lower_bound = random.randint(1, 10)
            upper_bound = lower_bound + random.randint(1, 10)
            include_lower = random.choice([True, False])
            include_upper = random.choice([True, False])
            lower_symbol = "<=" if include_lower else "<"
            upper_symbol = "<=" if include_upper else "<"
            constraint_expr = f"{lower_bound} {lower_symbol} {var} {upper_symbol} {upper_bound}"
            constraintCollection.add_constraint(ValueConstraint(
                variables=[var],
                expression=constraint_expr
            ))

        # 生成全局 count_constraint
        if num_count_constraints > 0:
            min_count = sum([1 if '{' not in var else int(re.search(r'{(\d+),?', var).group(1)) for var in variables])
            max_count = min_count + random.randint(1, 5)
            constraintCollection.add_constraint(CountConstraint(
                variables=all_vars,
                min_count=min_count,
                max_count=max_count
            ))
            num_count_constraints -= 1

        # 生成其他 count_constraints
        for _ in range(num_count_constraints):
            num_vars_in_constraint = random.randint(2, len(all_vars))
            selected_vars = random.sample(all_vars, num_vars_in_constraint)
            kleene_vars = [var for var in selected_vars if any(quant in self.regex for quant in ['*', '+', '{', '?'])]
            if kleene_vars:
                min_count = sum(
                    [1 if '{' not in var else int(re.search(r'{(\d+),?', var).group(1)) for var in selected_vars])
                max_count = min_count + random.randint(1, 5)
                constraintCollection.add_constraint(CountConstraint(
                    variables=selected_vars,
                    min_count=min_count,
                    max_count=max_count
                ))

        # 生成全局 time_constraint
        if num_time_constraints > 0:
            total_time = 0
            for var in all_vars:
                if any(quant in var for quant in ['*', '+', '{', '?']):
                    total_time += random.randint(40, 60)
                else:
                    total_time += random.randint(20, 30)
            min_time = 0  # 默认情况下，设置min_time为0
            max_time = total_time
            constraintCollection.add_constraint(TimeConstraint(
                variables=all_vars,
                min_time=min_time,
                max_time=max_time
            ))
            num_time_constraints -= 1

        # 生成其他 time_constraints
        for _ in range(num_time_constraints):
            num_vars_in_constraint = random.randint(2, len(all_vars))
            selected_vars = random.sample(all_vars, num_vars_in_constraint)
            selected_vars = sorted(selected_vars, key=lambda var: regex.index(var))
            total_time = 0
            for var in selected_vars:
                if any(quant in var for quant in ['*', '+', '{', '?']):
                    total_time += random.randint(40, 60)
                else:
                    total_time += random.randint(20, 30)
            min_time = 0  # 默认情况下，设置min_time为0
            max_time = total_time
            constraintCollection.add_constraint(TimeConstraint(
                variables=selected_vars,
                min_time=min_time,
                max_time=max_time
            ))

        # 生成 type_constraint 基于正则表达式字母，保持顺序
        for letter in sorted(set(re.findall(r'[A-Z]', regex)), key=regex.index):
            constraintCollection.add_constraint(TypeConstraint(
                variables=[letter],
                variables_name=[letter]
            ))

        constraintCollection.window_constrain_type = "time_constrains"
        self.constrains_collection = constraintCollection

        return {
            "variables": all_vars,
            "regex": regex,
            "constraints": constraintCollection,
            "domain": self.variable_domains
        }


# 使用示例
if __name__ == "__main__":
    generator = RegexPatternGenerator()

    num_variables = 5
    num_kleene = 2
    num_value_constrains = 2
    num_count_constrains = 2
    num_time_constrains = 2

    regex = generator.generate_regex(num_variables, num_kleene)
    result = generator.set_constraints(regex, num_value_constrains, num_count_constrains, num_time_constrains)

    print(f"Variables: {result['variables']}")
    print(f"Regex: {result['regex']}")
    print(f"Constrains: {result['constraints']}")
    print(f"Variables_Domain: {result['domain']}")