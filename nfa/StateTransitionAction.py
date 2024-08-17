from enum import Enum

class StateTransitionAction(Enum):
    TAKE = 'TAKE'  # take the current event and assign it to the current state
    IGNORE = 'IGNORE'  # ignore the current event
    PROCEED = 'PROCEED'  # do the state transition and keep the current event for further processing (epsilon transition)