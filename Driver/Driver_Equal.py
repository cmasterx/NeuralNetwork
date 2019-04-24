from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example
import numpy as np
from random import randint

xorAI = NeuralNetwork([2, 2, 2, 1], 0.1)

MAX_LEARN = 10000
example_list = []
for i in range(MAX_LEARN):
    a = randint(0, 1)
    b = randint(0, 1)
    example_list.append(Example(np.array([[a],[b]]), np.array([[ int(a == b) ]])))

xorAI.learn(example_list)

total = 500
correct = 0

for i in range(total):
    a = randint(0, 1)
    b = randint(0, 1)
    c = xorAI.output((np.array([[a], [b]])))

    if int(a == b) == int(c[0][0] + 0.5):
        correct = correct + 1

    print('Set [{},{}] -> {} | AI: {}'.format(a, b, int(a == b), c))

print('Correct: {}%'.format(correct / total * 100))