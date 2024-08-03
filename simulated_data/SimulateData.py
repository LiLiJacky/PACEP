import time
import random
from datetime import datetime
from multiprocessing import Process, Queue
from scipy.stats import norm, uniform, expon

from models.Data import Data


def data_producer(variable_name, distribution, interval_range, output_queue):
    """
    Function to produce data for a given variable and put it in the output queue.
    """
    while True:
        # Generate data item
        data_value = int(distribution.rvs())
        timestamp = datetime.now()
        data_item = Data(variable_name, data_value, timestamp)

        # Put the generated data item in the queue
        output_queue.put(data_item)

        # Sleep for a random interval within the specified range
        time.sleep(random.uniform(*interval_range))


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
    def __init__(self, regex, variable_info):
        self.regex = regex
        self.variable_info = variable_info

    def start_data_generation(self):
        """
        Starts data generation processes for each variable based on the provided regex and variable info.
        """
        processes = []
        data_queue = Queue()
        variables = self.regex.split()

        kleene_closures = {var[0] for var in variables if var[-1] in '*+?' or '{' in var}
        simple_vars = set(self.variable_info.keys()) - kleene_closures

        # Start producer processes for simple variables
        for var in simple_vars:
            interval_range = (10, 30)
            process = Process(target=data_producer,
                              args=(var, self.variable_info[var]['distribution'], interval_range, data_queue))
            processes.append(process)
            process.start()

        # Start producer processes for Kleene closure variables
        for var in kleene_closures:
            interval_range = (1, 2)
            process = Process(target=data_producer,
                              args=(var, self.variable_info[var]['distribution'], interval_range, data_queue))
            processes.append(process)
            process.start()

        # Start the consumer process
        consumer_process = Process(target=data_consumer, args=(data_queue,))
        consumer_process.start()
        processes.append(consumer_process)

        try:
            while True:
                time.sleep(1)  # Keep the main process alive
        except KeyboardInterrupt:
            for process in processes:
                process.terminate()
                process.join()