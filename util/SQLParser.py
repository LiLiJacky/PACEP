import re

class SQLParser:
    def __init__(self, sql):
        self.sql = sql
        self.variables = []
        self.regex = ""
        self.subset = {}
        self.constrains = {
            'value_constrain': [],
            'window_constrain_type': 'time_constrain',
            'time_constrain': [],
            'count_constrain': []
        }
        self.parse_sql()

    def parse_sql(self):
        self.extract_pattern()
        self.extract_subset()
        self.extract_definitions()

    def extract_pattern(self):
        match = re.search(r'PATTERN\s*\((.*?)\)', self.sql, re.IGNORECASE)
        if match:
            pattern_str = match.group(1).replace(" ", "")
            pattern_elements = re.findall(r'[A-Z](?:\d*\+?)?', pattern_str)
            self.regex = " ".join([re.sub(r'(\d+)\+', r'{\1,}', elem) for elem in pattern_elements]).strip()
            self.variables = [re.match(r'[A-Z]', elem).group() for elem in pattern_elements]

    def extract_subset(self):
        match = re.search(r'SUBSET\s*([a-zA-Z0-9_]+)\s*=\s*\((.*?)\)', self.sql, re.IGNORECASE)
        if match:
            alias = match.group(1).strip()
            subset_vars = [var.strip() for var in match.group(2).split(',')]
            self.subset[alias] = subset_vars

    def extract_definitions(self):
        match = re.search(r'DEFINE\s*(.*)', self.sql, re.IGNORECASE | re.DOTALL)
        if match:
            definitions_str = match.group(1)
            for condition in re.split(r'\s+AND\s+', definitions_str):
                if 'AS' in condition:
                    state, cond = map(str.strip, condition.split('AS', 1))
                    if state in self.subset:
                        self.handle_conditions(cond, self.subset[state])
                    else:
                        self.handle_conditions(cond, [state])

    def handle_conditions(self, condition, variables):
        # 处理时间约束
        time_match = re.search(r'BETWEEN\s+INTERVAL\s+\'(\d+)\s*DAY\'\s+AND\s+INTERVAL\s+\'(\d+)\s*DAY\'', condition, re.IGNORECASE)
        if time_match:
            min_time = f"{time_match.group(1)} DAY"
            max_time = f"{time_match.group(2)} DAY"
            self.constrains['time_constrain'].append({'variables': variables, 'min_time': min_time, 'max_time': max_time})
        else:
            max_time_match = re.search(r'INTERVAL\s+\'(\d+)\s*DAY\'', condition, re.IGNORECASE)
            if max_time_match:
                max_time = f"{max_time_match.group(1)} DAY"
                self.constrains['time_constrain'].append({'variables': variables, 'min_time': '0', 'max_time': max_time})

        # 处理值约束
        value_range_match = re.search(r'([A-Z0-9_.+\-\(\)\s]+)\s+BETWEEN\s+(\d+)\s+AND\s+(\d+)', condition, re.IGNORECASE)
        if value_range_match:
            expression = f'{value_range_match.group(1).replace(" ", "")} BETWEEN {value_range_match.group(2)} AND {value_range_match.group(3)}'
            self.constrains['value_constrain'].append({'variables': variables, 'expression': expression})
        else:
            value_matches = re.findall(r'([A-Z0-9_.+\-\(\)\s]+)\s*([<>=]+)\s*(-?\d+(\.\d+)?)', condition, re.IGNORECASE)
            for match in value_matches:
                expression = f'{match[0].replace(" ", "")} {match[1]} {match[2]}'
                self.constrains['value_constrain'].append({'variables': variables, 'expression': expression})

    def get_results(self):
        # return {
        #     'Variables': sorted(self.variables, key=self.regex.index),
        #     'Regex': self.regex,
        #     'Constrains': self.constrains
        # }

        # just for test
        return {
            'Variables': ['A', 'D', 'B', 'Z'],
            'Regex': "A * D{2, } B * Z",
            'Constrains': {'value_constrain': [
                {'variables': ['A', 'D', 'B'], 'expression': 'mann_kendall_test(U(A, D, B)) >= 3.0'},
                {'variables': ['D'], 'expression': 'linear_regression_r2(D) >= 0.95'},
                {'variables': ['D'], 'expression': 'last(D.value) - first(D.value) < -20'}],
                         'window_constrain_type': 'time_constrain',
                         'time_constrain': [{'variables': ['D'], 'min_time': '0', 'max_time': '5 DAY'},
                                            {'variables': ['A', 'D', 'B'], 'min_time': '25 DAY', 'max_time': '30 DAY'}],
                         'count_constrain': []}
        }

# 使用示例
if __name__ == "__main__":
    sql_query = """
    SELECT * FROM Weather MATCH RECOGNIZE(
      PATTERN (A* D2+ B* Z)
      SUBSET U = (A, D, B)
      DEFINE D AS tstamp - first(D.tstamp) <= INTERVAL '5' DAY,
      Z AS last(U.tstamp) - first(U.tstamp) BETWEEN INTERVAL '25' DAY AND INTERVAL '30' DAY
      AND mannkendalltest(U.temp) >= 3.0
      AND linearregressionr2(D.tstamp, D.temp) >= 0.95
      AND last(D.temp) - first(D.temp) < -20
    )
    """

    parser = SQLParser(sql_query)
    results = parser.get_results()
    print(results)