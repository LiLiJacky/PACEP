from copy import deepcopy

class Automaton:
    def __init__(self, states, transitions, initial_state, final_states, constraints):
        self.states = states
        self.transitions = transitions
        self.current_state = initial_state
        self.final_states = final_states
        self.constraints = constraints
        self.data_buffer = []

    def clone(self):
        return deepcopy(self)

    def outTime(self):
        return self.data_buffer

    def process_event(self, event):
        if self.current_state in self.transitions:
            for (event_type, next_state) in self.transitions[self.current_state]:
                if event.variable_name == event_type:
                    self.current_state = next_state
                    self.states[next_state].add_data(event)
                    self.data_buffer.append(event)
                    return True
        return False

    def is_final_state(self):
        return self.current_state in self.final_states

    def check_constraints(self):
        for constraint in self.constraints:
            if not eval(constraint):
                return False
        return True