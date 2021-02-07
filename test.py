import numpy as np

class class_a:
    def __init__(self):
        self.a = np.array([1,1])

class_a = class_a()
b = class_a.a
print(class_a.a)
b[1]=0
print(b)
print(class_a.a)