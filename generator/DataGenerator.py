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

# Example usage
if __name__ == "__main__":
    variable_info = DataGenerator("SampleVariable")
    print("Variable Info:")
    print("Variable Name:", variable_info['variable'])
    print("Distribution Type:", variable_info['distribution_type'])
    print("Value Range:", variable_info['value_range'])
    print("Distribution Info:", variable_info['distribution_info'])