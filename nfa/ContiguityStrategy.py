from enum import Enum

class ContiguityStrategy(Enum):
    STRICT = 'STRICT'  # Expects all matching events to appear strictly one after the other, without any non-matching events in-between
    RELAXED = 'RELAXED'  # Ignores non-matching events appearing in-between the matching ones
    NONDETERMINISTICRELAXED = 'NONDETERMINISTICRELAXED' #  Further relaxes contiguity, allowing additional matches that ignore some matching events.