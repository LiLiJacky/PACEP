from datetime import datetime, timedelta
import re

from interfaces.StateItem import State
from models.Automaton import Automaton
from models.Data import Data

class SequenceMatcher:
    def __init__(self, regex, constraints, variable_info):
        self.regex = regex
        self.constraints = constraints
        self.variable_info = variable_info
        self.automaton = self._create_automaton()

    def _create_automaton(self):
        tokens = self.regex.split()
        states = []
        transitions = {}
        final_states = []

        initial_state = State("q0")
        states.append(initial_state)
        current_state = initial_state

        for i, token in enumerate(tokens):
            if len(token) == 1:
                next_state = State(f"q{i+1}")
                states.append(next_state)
                if current_state.name not in transitions:
                    transitions[current_state.name] = []
                transitions[current_state.name].append((token, next_state.name))
                current_state = next_state
            elif len(token) > 1:
                var, quantifier = token[0], token[1:]
                if quantifier == '*':
                    next_state = State(f"q{i+1}")
                    states.append(next_state)
                    if current_state.name not in transitions:
                        transitions[current_state.name] = []
                    transitions[current_state.name].append((var, next_state.name))
                    transitions[next_state.name] = [(var, next_state.name)]
                    current_state = next_state
                elif quantifier == '+':
                    next_state = State(f"q{i+1}")
                    states.append(next_state)
                    if current_state.name not in transitions:
                        transitions[current_state.name] = []
                    transitions[current_state.name].append((var, next_state.name))
                    transitions[next_state.name] = [(var, next_state.name)]
                    current_state = next_state
                elif quantifier == '?':
                    next_state = State(f"q{i+1}")
                    states.append(next_state)
                    if current_state.name not in transitions:
                        transitions[current_state.name] = []
                    transitions[current_state.name].append((var, next_state.name))
                    transitions[current_state.name].append(('ε', next_state.name))
                    current_state = next_state
                elif re.match(r'{\d+}', quantifier):
                    n = int(quantifier[1:-1])
                    for j in range(n):
                        next_state = State(f"q{i+1}_{j}")
                        states.append(next_state)
                        if current_state.name not in transitions:
                            transitions[current_state.name] = []
                        transitions[current_state.name].append((var, next_state.name))
                        current_state = next_state
                elif re.match(r'{,\d+}', quantifier):
                    m = int(quantifier[2:-1])
                    for j in range(m):
                        next_state = State(f"q{i+1}_{j}")
                        states.append(next_state)
                        if current_state.name not in transitions:
                            transitions[current_state.name] = []
                        transitions[current_state.name].append((var, next_state.name))
                        current_state = next_state
                        if j == m - 1:
                            if current_state.name not in transitions:
                                transitions[current_state.name] = []
                            transitions[current_state.name].append(('ε', next_state.name))

        final_states.append(current_state.name)
        return Automaton(states, transitions, initial_state.name, final_states, self.constraints)

    def process_stream(self, data_stream):
        results = []
        for event in data_stream:
            if self.automaton.process_event(event):
                if self.automaton.is_final_state():
                    if self.automaton.check_constraints():
                        results.append(self.automaton.outTime())
                    self.automaton = self.automaton.clone()
        return results

# Example usage
if __name__ == "__main__":
    # Example data stream
    data_stream = [
        Data("A", 100, datetime.now()),
        Data("E", 50, datetime.now() + timedelta(seconds=10)),
        Data("B", 20, datetime.now() + timedelta(seconds=20)),
        Data("G", 10, datetime.now() + timedelta(seconds=30)),
        Data("C", 30, datetime.now() + timedelta(seconds=40)),
        Data("F", 5, datetime.now() + timedelta(seconds=50)),
        Data("D", 1, datetime.now() + timedelta(seconds=60)),
        # Add more Data instances...
    ]

    regex = "A E B G{1} C F* D"
    constraints = [
        "162 < A + E + B + sum(G) < 172",
        "189 < A + E + B + permutations(G) + C < 199"
    ]
    variable_info = {"A": None, "E": None, "B": None, "G": None, "C": None, "F": None, "D": None}

    matcher = SequenceMatcher(regex, constraints, variable_info)
    results = matcher.process_stream(data_stream)
    for result in results:
        print(result)