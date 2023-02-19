import numpy as np
from collections import deque

def add_to_slidewindow(one_dire, window_size):
    window = deque(maxlen=window_size)
    window.append(one_dire)
    return 
