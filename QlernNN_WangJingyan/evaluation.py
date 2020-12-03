import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pickle 




def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()

def load_variable(filename):
    f = open(filename, 'rb')
    v = pickle.load(f)
    f.close()
    return v
