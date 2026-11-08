"""
Split the data into training/validation/test data splits for part1
"""

import numpy as np

# Split the data into sequential training/validation/test datasets
def temporal_split(data):
    #sort ratings by timestamp to order data into sequence
    timestamps = data.timestamp.to_numpy()
    timestamps = np.sort(timestamps)

    # training, test, validation = 80%, 10%, 10%
    #define the scale factors for the training and validation splits, test split includes everything after the end of the validation split
    train_scale_factor = 0.8
    validate_scale_factor = 0.9
    #split the data based on the scale factors
    train_end = timestamps[int(len(timestamps) * train_scale_factor)]
    validate_end = timestamps[int(len(timestamps) * validate_scale_factor)]
    train = data[data.timestamp <= train_end].copy()
    validate = data[(data.timestamp > train_end) & (data.timestamp <= validate_end)].copy()
    test = data[data.timestamp > validate_end].copy()

    #sort to keep the sequence structure
    train = (train.sort_values(by=["timestamp"], ascending=False))
    validate = (validate.sort_values(by=["timestamp"], ascending=False))
    test = (test.sort_values(by=["timestamp"], ascending=False))
    return train, test, validate

