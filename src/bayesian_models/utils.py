import torch
import numpy as np

def to_string(array):
    string = ""
    for element in array:
        string += str(element)

    return string

def set_random_seed(seed):
    """
    Sets all random seeds
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

