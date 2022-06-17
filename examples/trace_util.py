import numpy as np
import os

def load_trace_file(name):
    trace_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", name)
    data = np.loadtxt(trace_path)
    y = data[:, 0].copy()-1
    x = data[:, 1:].copy()
    return x, y
    
    