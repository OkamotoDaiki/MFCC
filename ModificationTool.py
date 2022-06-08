import numpy as np

def assign_zero2mean(array):
    """
    If antilogarithm is 0, assign mean with other data

    Args:
        array: Data array containing zeros.

    Returns:
        array: Data array with zeros converted to the mean of the data.
    """
    zero_array_numbers = [i for i in range(len(array)) if array[i] == 0]
    not_zero_array_numbers = [i for i in range(len(array)) if array[i] != 0]
    mean = np.mean(not_zero_array_numbers)
    for i in zero_array_numbers:
        array[i] = mean
    return array