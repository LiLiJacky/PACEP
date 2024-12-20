import os

import numpy as np
from scipy.stats import norm, uniform, expon

def DataGenerator(variable_name):
    """
    Generates a random distribution and its value range for a given variable name.
    """
    # Randomly select a distribution type
    distribution_types = ['uniform', 'normal', 'exponential']

    selected_distribution = np.random.choice(distribution_types)

    distribution_info = {}

    # Parameters and ranges for the selected distribution
    if selected_distribution == 'uniform':
        # Uniform distribution from 0 to a random upper limit
        lower_bound = np.random.randint(0, 50)
        upper_bound = np.random.randint(lower_bound + 1, lower_bound + 150)
        distribution = uniform(loc=lower_bound, scale=upper_bound - lower_bound)
        value_range = [lower_bound, upper_bound]

        # Store information to reconstruct the range
        distribution_info = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    elif selected_distribution == 'normal':
        # Normal distribution with random mean and standard deviation
        mean = np.random.randint(50, 150)
        std_dev = np.random.randint(10, 50)
        distribution = norm(loc=mean, scale=std_dev)
        value_range = [mean - 3 * std_dev, mean + 3 * std_dev]  # Approx. 99.7% of the data

        # Store information to reconstruct the range
        distribution_info = {
            "mean": mean,
            "std_dev": std_dev
        }

    elif selected_distribution == 'exponential':
        # Exponential distribution with a random scale
        scale = np.random.randint(1, 50)
        distribution = expon(scale=scale)
        value_range = [0, scale * 5]  # Approx. 99th percentile

        # Store information to reconstruct the range
        distribution_info = {
            "scale": scale
        }

    return {
        "variable": variable_name,
        "distribution_type": selected_distribution,
        "distribution": distribution,
        "value_range": value_range,
        "distribution_info": distribution_info  # Store parameters for reconstruction
    }


import pandas as pd
import random
from datetime import datetime, timedelta


def generate_data_with_pandas_sorted(event_dict, num_records, output_name, start_time=None, time_interval_seconds=1):
    """
    Generate a sorted dataset based on event type frequencies and save it as a CSV using pandas.
    Data is sorted by timestamp in ascending order.

    Parameters:
    event_dict (dict): A dictionary mapping event types to their frequencies (e.g., {"a": 0.2, "b": 0.8}).
    num_records (int): The number of records to generate.
    output_name (str): The name of the output CSV file (without extension).
    start_time (datetime): The start time for the first event. Defaults to current time if None.
    time_interval_seconds (int): The time interval (in seconds) between consecutive records.

    Returns:
    str: The name of the generated CSV file.
    """
    # Set the start time for the first event, if not provided
    if start_time is None:
        start_time = datetime.now()

    # Prepare the event types and their probabilities
    event_types = list(event_dict.keys())
    probabilities = list(event_dict.values())
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    # List to hold the data
    data = []

    # Generate data
    current_time = start_time
    for _ in range(num_records):
        rand = random.random()
        event_type = None
        for i, cp in enumerate(cumulative_probabilities):
            if rand <= cp:
                event_type = event_types[i]
                break
        value = random.randint(1, 100)  # Generate a random integer value between 1 and 100
        data.append([event_type, value, current_time.strftime("%Y-%m-%d %H:%M:%S")])

        # Increment the time for the next record
        current_time += timedelta(seconds=time_interval_seconds)

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=["type", "value", "timestamp"])

    # Save to CSV
    df.to_csv(f"../../data/minimal_test/{output_name}.csv", index=False)

    return f"{output_name}.csv"

def generate_lng_data_with_pandas_sorted(event_dict, num_records, output_name, start_time=None, time_interval_seconds=1):
    """
    Generate a sorted dataset based on event type frequencies and save it as a CSV using pandas.
    Data is sorted by timestamp in ascending order.

    Parameters:
    event_dict (dict): A dictionary mapping event types to their frequencies (e.g., {"a": 0.2, "b": 0.8}).
    num_records (int): The number of records to generate.
    output_name (str): The name of the output CSV file (without extension).
    start_time (datetime): The start time for the first event. Defaults to current time if None.
    time_interval_seconds (int): The time interval (in seconds) between consecutive records.

    Returns:
    str: The name of the generated CSV file.
    """
    # Set the start time for the first event, if not provided
    if start_time is None:
        start_time = datetime.now()

    # Prepare the event types and their probabilities
    event_types = list(event_dict.keys())
    probabilities = list(event_dict.values())
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    # List to hold the data
    data = []

    # Generate data
    current_time = start_time
    for _ in range(num_records):
        rand = random.random()
        event_type = None
        for i, cp in enumerate(cumulative_probabilities):
            if rand <= cp:
                event_type = event_types[i]
                break
        if event_type == 'a':
            value = random.randint(1, 150)  # Generate a random integer value between 1 and 100
        elif event_type == 'b':
            value = random.randint(1, 10)
        elif event_type == 'c':
            value = random.randint(100, 300)
        data.append([event_type, value, current_time.strftime("%Y-%m-%d %H:%M:%S")])

        # Increment the time for the next record
        current_time += timedelta(seconds=time_interval_seconds)

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=["type", "value", "timestamp"])

    # Save to CSV
    df.to_csv(f"../../data/minimal_test/{output_name}.csv", index=False)

    return f"{output_name}.csv"


def generate_flash_data_with_pandas_sorted(event_dict, num_records, output_name, start_time=None, time_interval_seconds=1):
    """
    Generate a sorted dataset based on event type frequencies and save it as a CSV using pandas.
    Data is sorted by timestamp in ascending order.

    Parameters:
    event_dict (dict): A dictionary mapping event types to their frequencies (e.g., {"close_price": 0.2, "low_price": 0.3, "high_price": 0.5}).
    num_records (int): The number of records to generate.
    output_name (str): The name of the output CSV file (without extension).
    start_time (datetime): The start time for the first event. Defaults to current time if None.
    time_interval_seconds (int): The time interval (in seconds) between consecutive records.

    Returns:
    str: The name of the generated CSV file.
    """
    # Set the start time for the first event, if not provided
    if start_time is None:
        start_time = datetime.now()

    # Prepare the event types and their probabilities
    event_types = list(event_dict.keys())
    probabilities = list(event_dict.values())
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    # List to hold the data
    data = []

    # Generate data
    current_time = start_time
    for _ in range(num_records):
        rand = random.random()
        event_type = None
        for i, cp in enumerate(cumulative_probabilities):
            if rand <= cp:
                event_type = event_types[i]
                break

        # Generate a random floating point value for the event type
        if event_type == "close_price":
            value = round(random.uniform(15.0, 25.0), 2)  # Example range for close_price
        elif event_type == "low_price":
            value = round(random.uniform(10.0, 20.0), 2)  # Example range for low_price
        elif event_type == "high_price":
            value = round(random.uniform(20.0, 30.0), 2)  # Example range for high_price
        else:
            value = round(random.uniform(10.0, 30.0), 2)  # Default case

        # Append data with event type, value, and timestamp
        data.append([event_type, value, current_time.strftime("%Y-%m-%d %H:%M:%S")])

        # Increment the time for the next record
        current_time += timedelta(seconds=time_interval_seconds)

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=["type", "value", "timestamp"])

    # Save to CSV
    df.to_csv(f"../../data/minimal_test/{output_name}.csv", index=False)

    return f"{output_name}.csv"


# Example usage
if __name__ == "__main__":
    variable_info = DataGenerator("SampleVariable")
    print("Variable Info:")
    print("Variable Name:", variable_info['variable'])
    print("Distribution Type:", variable_info['distribution_type'])
    print("Value Range:", variable_info['value_range'])
    print("Distribution Info:", variable_info['distribution_info'])