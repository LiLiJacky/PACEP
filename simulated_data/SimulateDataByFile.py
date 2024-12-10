import time
import random
import pandas as pd
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from models.Data import Data


def data_producer_from_nasdaq(csv_file, rate, output_queue):
    """
    Function to read data from a CSV and put it in the output queue according to the interval_range.
    """
    data = pd.read_csv(csv_file)  # Read data from CSV
    for _, row in data.iterrows():
        # Create a data item with the CSV row
        variable_name = row['Name']
        low = row['Low']
        high = row['High']
        close = row['Close']
        date = row['Date']

        date_object = datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        # 将 datetime 对象转换为 UNIX 时间戳
        timestamp = int(date_object.timestamp())

        # Create Data object
        data_item_low = Data('LOW', low, timestamp)
        data_item_high = Data('HIGH', high, timestamp)
        data_item_close = Data('CLOSE', close, timestamp)

        # Put the generated data item in the queue
        output_queue.put(data_item_low)
        output_queue.put(data_item_high)
        output_queue.put(data_item_close)

        # Sleep for a random interval within the specified range to control the generation rate
        time.sleep(3 / rate)

def data_producer_from_test(csv_file, rate, output_queue):
    """
    Function to read data from a CSV and put it in the output queue according to the interval_range.
    """
    data = pd.read_csv(csv_file)  # Read data from CSV
    for _, row in data.iterrows():
        # Create a data item with the CSV row
        variable_name = row['type']
        value = row['value']
        ts = row['timestamp']

        date_object = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

        # 将 datetime 对象转换为 UNIX 时间戳
        timestamp = int(date_object.timestamp())

        # Create Data object
        data_item = Data(variable_name, value, timestamp)

        # Put the generated data item in the queue
        output_queue.put(data_item)

        # Sleep for a random interval within the specified range to control the generation rate
        time.sleep(3 / rate)


def data_producer_from_gm(csv_file, rate, output_queue):
    """
    Function to read data from a CSV and put it in the output queue according to the interval_range.
    """
    data = pd.read_csv(csv_file)  # Read data from CSV
    for _, row in data.iterrows():
        # Create a data item with the CSV row
        variable_name = row['Type']
        value = row['Close']
        ts = row['Date']

        date_object = datetime.strptime(ts, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        # 将 datetime 对象转换为 UNIX 时间戳
        timestamp = int(date_object.timestamp())

        # Create Data object
        data_item = Data(variable_name, value, timestamp)

        # Put the generated data item in the queue
        output_queue.put(data_item)

        # Sleep for a random interval within the specified range to control the generation rate
        time.sleep(3 / rate)

def data_producer_from_stream(stream_file, rate, output_queue):
    """
    Function to read data from a stream and put it in the output queue according to the interval_range.
    """
    with open(stream_file, 'r') as stream:
        for line in stream:
            # Create a data item with the stream line
            l = line.replace(" ", "")
            variables = l.strip().split(',')

            begin_time = int(
                datetime.strptime('2025-11-11 11:11:11', '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp())

            # Create Data object
            data_item = Data(variables[2], variables[4], int(variables[0]) + begin_time)

            # Put the generated data item in the queue
            output_queue.put(data_item)

            # Sleep for a random interval within the specified range to control the generation rate
            time.sleep(3 / rate)

def data_consumer(input_queue):
    """
    Function to consume data from the input queue and process it.
    """
    while True:
        try:
            data_item = input_queue.get()  # Blocking call, waits for data
            if data_item:
                # Process the data item (e.g., print it, store it, send it)
                print(data_item)
        except KeyboardInterrupt:
            break


class DataSource:
    def __init__(self, csv_file, rate):
        self.csv_file = csv_file
        self.rate = rate

    def start_data_generation(self, data_queue):
        """
        Starts data generation process by reading from a CSV file.
        """
        processes = []

        # Start producer process to read from CSV
        process = Process(target=data_producer_from_nasdaq,
                          args=(self.csv_file, self.rate, data_queue))
        processes.append(process)
        process.start()

        return processes

    def start_data_generation_test(self, data_queue):
        """
        Starts data generation process by reading from a CSV file.
        """
        processes = []

        # Start producer process to read from CSV
        process = Process(target=data_producer_from_test,
                          args=(self.csv_file, self.rate, data_queue))
        processes.append(process)
        process.start()

        return processes

    def start_data_generation_stock(self, data_queue):
        """
        Starts data generation process by reading from a CSV file.
        """
        processes = []

        # Start producer process to read from CSV
        process = Process(target=data_producer_from_stream,
                          args=(self.csv_file, self.rate, data_queue))
        processes.append(process)
        process.start()

        return processes

    def start_data_generation_stock_gm(self, data_queue):
        """
        Starts data generation process by reading from a CSV file.
        """
        processes = []

        # Start producer process to read from CSV
        process = Process(target=data_producer_from_gm,
                          args=(self.csv_file, self.rate, data_queue))
        processes.append(process)
        process.start()

        return processes


# Example usage
if __name__ == '__main__':
    # Create a DataSource object with the CSV file and interval range (seconds)
    data_source = DataSource(csv_file='../data/simulate_data/stock.stream', rate=1000)

    # Start data generation
    data_source.start_data_generation_stock()