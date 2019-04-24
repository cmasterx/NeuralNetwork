from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example
import numpy as np
from random import randint

xorAI = NeuralNetwork([2, 4, 4, 1], 0.1)

e1 = Example(np.array([[-1], [-1]]), np.array([[1]]))
e2 = Example(np.array([[-1], [1]]), np.array([[1]]))
e3 = Example(np.array([[1], [-1]]), np.array([[-1]]))
e4 = Example(np.array([[1], [1]]), np.array([[1]]))

e = [e1, e2, e3, e4]

MAX_LEARN = 10000
example_list = []
for i in range(MAX_LEARN):
    r = randint(0, 3)
    example_list.append(e[r])
    # example_list.append(Example(np.array([[a],[b]]), np.array([[ int(a == 1 or b == 1) ]])))

xorAI.learn(example_list)

total = 500
correct = 0

for i in range(total):
    a = randint(0, 1)
    b = randint(0, 1)
    c = xorAI.output((np.array([[a],[b]])))

    # if int(a == 1 or b == 1) == int(c[0][0] + 0.5):
    #     correct = correct + 1

    print('Set [{},{}] -> {} | AI: {}'.format(a, b, int(not(a == 1 and b == 0)), c))

print('Correct: {}%'.format(correct / total * 100))