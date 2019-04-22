import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class Example:

    def __init__(self, input_layer=None, result=None):
        self.input = input_layer
        self.result = result


class NeuralNetwork:

    # initialize NueralNetwork class
    def __init__(self, net_info, alpha=0.1):
        self.weight_list = []
        self.bias_list = []
        self.alpha = alpha
        self.pass_num = 1

        # generates an array of weight matrix for hidden layer and output layer neurons
        for i in range(1, len(net_info) - 1):
            self.weight_list.append(np.random.rand(net_info[i], net_info[i - 1]) * 0.2 - 0.1)

        self.weight_list.append(np.random.rand(net_info[-1], net_info[-2]) * 0.1)

        # generates an array of bias matrix for hidden layer and output layer neurons
        for i in net_info[1:]:
            self.bias_list.append((np.zeros((i, 1))))

    # produces output based on neural network
    def output(self, junction_layer):

        # tanh sigmoid function for hidden layers
        for l in range(len(self.weight_list) - 1):
            junction_layer = np.tanh(self.weight_list[l].dot(junction_layer) + self.bias_list[l])

        # sigmoid function for last layer
        junction_layer = sigmoid(self.weight_list[-1].dot(junction_layer) + self.bias_list[-1])

        return junction_layer

    # produces output for every single node
    def output_node(self, junction_layer):

        res = [junction_layer]

        for l in range(len(self.weight_list) - 1):
            res.append(np.tanh(self.weight_list[l].dot(res[-1]) + self.bias_list[l]))

        # sigmoid function for last layer
        res.append(sigmoid(self.weight_list[-1].dot(res[-1]) + self.bias_list[-1]))

        return res

    # back propagation learning
    def learn(self, example_list):
        for example in example_list:
            # outputs of each neuron
            a = self.output_node(example.input)

            # print(example.result)
            # if a[-1].all(example.result):
            #     self.pass_num = self.pass_num + 1

            # delta of last layer neurons
            delta = [a[-1] * (1 - a[-1]) * (example.result - a[-1])]

            # computes delta for the hidden layers
            for i in reversed(range(len(self.weight_list) - 1)):
                # print('------')
                sum_data = self.weight_list[i + 1].transpose().dot(delta[0])
                # sum_data = delta[0].transpose().dot(self.weight_list[i + 1])
                delta = [(1 - a[i + 1]**2) * sum_data] + delta

            for i in range(len(self.weight_list)):
                self.weight_list[i] = self.weight_list[i] + self.alpha / self.pass_num * a[i + 1] * delta[i]
