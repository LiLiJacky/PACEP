import time
import random
from datetime import datetime
from multiprocessing import Process, Queue
from models.Data import Data
from models.DataInfo import DataInfo


def data_producer(data_info, output_queue):
    """
    Function to produce data for a given variable and put it in the output queue.
    """
    while True:
        # Calculate the number of data items to generate in 60 seconds
        num_items = int(60 / data_info.frequency)

        for _ in range(num_items):
            # Decide whether to generate normal or matched data based on probability
            if random.random() < data_info.probability:
                for matched_value in data_info.matched:
                    # Create a data item with the matched data
                    timestamp = datetime.now()
                    data_item = Data(data_info.names, matched_value, timestamp)
                    # Put the generated data item in the queue
                    output_queue.put(data_item)
                    # Sleep for the specified frequency
                    time.sleep(data_info.frequency)
            else:
                data_value = random.choice(data_info.normal)
                # Create a data item with the normal data
                timestamp = datetime.now()
                data_item = Data(data_info.names, data_value, timestamp)
                # Put the generated data item in the queue
                output_queue.put(data_item)

                # Sleep for the specified frequency
                time.sleep(data_info.frequency)


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
    def __init__(self, data_info_list):
        self.data_info_list = data_info_list

    def start_data_generation(self, data_queue):
        """
        Starts data generation processes for each variable based on the provided data info list.
        """
        processes = []

        # Start producer processes for each data info
        for data_info in self.data_info_list:
            process = Process(target=data_producer, args=(data_info, data_queue))
            processes.append(process)
            process.start()

        return processes
