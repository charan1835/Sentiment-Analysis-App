import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    was_1d = False
    if x.ndim == 1:
        was_1d = True
        x = x.reshape(1, -1)
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    result = e_x / e_x.sum(axis=1, keepdims=True)
    return result.flatten() if was_1d else result