from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example
import numpy as np
from random import randint

AND2 = NeuralNetwork([2, 1], 0.05)

e1 = Example(np.array([[0], [0]]), np.array([[0]]))
e2 = Example(np.array([[0], [1]]), np.array([[0]]))
e3 = Example(np.array([[1], [0]]), np.array([[0]]))
e4 = Example(np.array([[1], [1]]), np.array([[1]]))

e = [e1, e2, e3, e4]

MAX_LEARN = 10000
example_list = []
for i in range(MAX_LEARN):
    r = randint(0, 3)
    example_list.append(e[r])

AND2.learn(example_list)

total = 500
correct = 0

for i in range(total):
    r = randint(0, 3)
    c = AND2.output(e[r].input)

    if int(round(c[0][0]) == e[r].result[0][0]):
        correct = correct + 1


    print('Set [{},{}] -> {} | AI: {}'.format(e[r].input[0][0], e[r].input[1][0], e[r].result[0], c))

print('\nCorrect: {}%'.format(correct / total * 100))