import numpy as np

colors = ['#000000', '#E24A33', '#34b9bd', '#FBC15E', '#348ABD', '#FFB5B8', '#777777', '#8EBA42', '#988ED5', ]

def moving_average(a, *, window_size):
    return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]
