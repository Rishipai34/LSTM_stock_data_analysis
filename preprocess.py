import numpy as np

def preprocess(data, step_size):
    x,y = [],[]
    for i in range(len(data) - step_size -1):
        a = data[i:(i + step_size)]
        x.append(a)
        y.append(data[i+ step_size])

    return np.array(x), np.array(y)