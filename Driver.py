import matplotlib.pyplot as plt
import numpy as np
from random import randint
from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example

# jacket = NeuralNetwork([10, 120, 120, 120, 1])
# jacket()

# nw = NeuralNetwork([2,2], 0.1)
#
# example = []
# for i in range(4000):
#
#     r = randint(0,1)
#
#     if r == 0:
#         e = Example(np.array([[0],[1]]),np.array([[0.3],[1]]))
#     elif r == 1:
#         e = Example(np.array([[1],[0]]),np.array([[1],[0.3]]))
#
#     example.append( e )


mj_example = []
majorityFunction = NeuralNetwork([1, 1])
for i in range(40000):

    r = randint(0, 1)

    mj_example.append( Example( np.array([[r]]) , np.array([[r]]) ) )

majorityFunction.learn(mj_example)

for i in range(30):
    print('------')
    print(i)
    r = randint(0,1)
    solution = majorityFunction.output(np.array([[r]]))
    print(r)
    print(solution)

print(mj_example[0].result)
val = majorityFunction.output(mj_example[0].result)
print(val)
print(val)
# plt.plot(x, y)
# plt.show()
