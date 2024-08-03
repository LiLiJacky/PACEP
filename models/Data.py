from datetime import datetime

from interfaces.DataItem import DataItem

class Data(DataItem):
    def __init__(self, variable_name, value, timestamp=None):
        super().__init__(variable_name, value, timestamp)

    def get_value(self):
        """
        Returns the value of the data item.
        """
        return self.value

    def get_timestamp(self):
        """
        Returns the timestamp of the data item.
        """
        return self.timestamp
