import numpy as np

class Test:
    def __init__(self, tries):
        self.tries = tries


class Try:
    def __init__(self, v1, v2):
        self.v = np.array([v1, v2])

t = np.array([Try(1, 2), Try(3, 4), Try(5, 6)])
T = Test(t)
# print all the v values in T without using a loop