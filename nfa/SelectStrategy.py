from enum import Enum

class SelectStrategy(Enum):
    SC = 'STRICT'  # do not skip any event
    STNM = 'SKIP_TILL_NEXT_MATCH'  # skip all events till the next match
    STAM = 'SKIP_TILL_ANY_MATCH'  # skip all events till any match