import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=1, keepdims=True)