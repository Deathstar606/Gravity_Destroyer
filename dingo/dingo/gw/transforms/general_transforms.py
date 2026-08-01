import numpy as np

class UnpackDict(object):
    """
    Unpacks the dictionary to prepare it for final output of the dataloader.
    Only returns elements specified in selected_keys.
    """
    def __init__(self, selected_keys):
        self.selected_keys = selected_keys

    def __call__(self, input_sample):
        selected = []
        for key in self.selected_keys:
            value = input_sample[key]
            if isinstance(value, np.ndarray):
                value = value.copy()
            selected.append(value)
        return selected