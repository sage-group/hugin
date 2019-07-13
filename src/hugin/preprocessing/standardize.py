import pickle
import numpy as np

class SkLearnStandardizer(object):
    def __init__(self, path, standardize_output=False):
        self.path = path
        self.standardize_output = standardize_output
        with open(self.path, "rb") as f:
            self.model = pickle.load(f)

    def __call__(self, input_data, output_data=None):
        if isinstance(input_data, dict):
            new_input = {}
            new_output = {}
            for k,v in input_data.items():
                old_shape = v.shape[:]
                new_input[k] = self.model.transform(v.reshape(-1, 1)).reshape(old_shape)
            if self.standardize_output:
                for k,v in input_data.items():
                    old_shape = v.shape[:]
                    new_output[k] = self.model.transform(v.reshape(-1, 1)).reshape(old_shape)

            return (new_input, new_output if new_output else output_data)

        elif isinstance(input_data, (np.ndarray, np.generic)) and output_data is None:
            old_shape = input_data.shape[:]
            return self.model.transform(input_data.reshape(-1, 1)).reshape(old_shape)
        else:
            raise NotImplementedError("Unsupported transform scenario")
