import numpy as np
import torch

class class_a:
    def __init__(self):
        self.a = np.array([1,1])

def test_1():
    a = torch.nn.functional.one_hot(torch.LongTensor([1,2]), num_classes=7)
    print(a)

def test_2():
    e_decay = 0.05
    get_epsilon = lambda episode: np.exp(-episode * e_decay)

    print(get_epsilon(1))

test_2()