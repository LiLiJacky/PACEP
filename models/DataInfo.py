# models/DataInfo.py
class DataInfo:
    def __init__(self, names, normal, matched, frequency, probability):
        self.names = names
        self.normal = normal
        self.matched = matched
        self.frequency = frequency
        self.probability = probability

    def __str__(self):
        return (f"DataInfo(names={self.names}, normal={self.normal}, "
                f"matched={self.matched}, frequency={self.frequency}, "
                f"probability={self.probability})")
