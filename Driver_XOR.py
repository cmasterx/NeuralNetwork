from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example
import numpy as np
from random import randint

xorAI = NeuralNetwork([2, 20, 20, 1])

MAX_LEARN = 1000
example_list = []
for i in range(MAX_LEARN):
    a = randint(0, 1)
    b = randint(0, 1)
    example_list.append(Example(np.array([[a],[b]]), np.array([[ int(a == b) ]])))

xorAI.learn(example_list)

for i in range(10):
    a = randint(0, 1)
    b = randint(0, 1)
    c = xorAI.output((np.array([[a],[b]])))

    print('Set [{},{}] -> {} | AI: {}'.format(a, b, int(a==b), c))