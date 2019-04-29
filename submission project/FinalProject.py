##### ------ FLAGS ------ #####
### The following booleans are flags to enable in the program

## Enable to build a new neural network and train it
# False to load neural network data previously trained
TRAIN = False

## Enables graphing if TRAIN is set to True
# NOTE: If set to true, this requires matplotlib installed because the training algorithm also generates
# a graph AI accuracy over time
GRAPHING = False

## If TRAIN is set to False, Neural Network model/data will be loaded from following file name
# - Comes with pre-trained neural network called 'nw.npy'
NW_FILENAME = 'nw.npy'

## Generates a graph of neural network accuracy as noise increases
# NOTE: If set to true, this requires matplotlib installed because the training algorithm also generates
# a graph AI over a number of noise
TEST_NOISE_ACCURACY = False

## Test each letter by adding noise until it fails
TEST_NOISE_LETTER = True
## Graph Noise Letter Testing

## Majority Function
# Enable to test neural network with majority function
TEST_MAJORITY = False

# ----------- END OF FLAGS ----------- #

import sys
sys.path.insert(0, './packages')

# imports
import numpy as np
from PIL import ImageFont
from random import randint, shuffle, sample


# from csce420_final_project.NeuralNetwork import NeuralNetwork
# from csce420_final_project.NeuralNetwork import Example

########################################################################################################################

# sigmoid function
def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


# example class
# input: a 2D array of the inputs of the neural network
#        this 2D array has one column and a row for each input
# result: a 2D array of the expected outputs of the neural network
#         this 2D array has one column and a row for each output
class Example:

    def __init__(self, input_layer=None, result=None):
        self.input = input_layer  # input
        self.result = result  # output


# Neural netork class
# net_info: each element in the array represents a layer and the value represents the number of neurons for that layer
#           the first index is the number of inputs and the last index is the number of outputs
# alpha: learning rate of algorithm
class NeuralNetwork:

    # initialize NueralNetwork class
    def __init__(self, net_info=[1, 1], alpha=0.1):
        self.weight_list = []
        self.bias_list = []
        self.alpha = alpha
        self.pass_num = 1

        net_info[0] = net_info[0] + 1

        # generates an array of weight matrix for hidden layer and output layer neurons
        for i in range(1, len(net_info) - 1):
            self.weight_list.append(np.random.rand(net_info[i], net_info[i - 1]) * 0.2 - 0.1)

        self.weight_list.append(np.random.rand(net_info[-1], net_info[-2]) * 0.1)

        # generates an array of bias matrix for hidden layer and output layer neurons
        for i in net_info[1:]:
            self.bias_list.append((np.zeros((i, 1))))

    # produces output based on neural network
    # junction_layer: a 2D array of the inputs of the neural network
    #                 this 2D array has one column and a row for each input
    # @return a 2D array of the outputs of the neural network
    #         this 2D array has one column and a row for each output
    def output(self, junction_layer):
        junction_layer = np.vstack([[[1]], junction_layer])

        # tanh sigmoid function for hidden layers
        for l in range(len(self.weight_list)):
            junction_layer = np.tanh(self.weight_list[l].dot(junction_layer) + self.bias_list[l])

        # sigmoid function for last layer
        # junction_layer = sigmoid(self.weight_list[-1].dot(junction_layer) + self.bias_list[-1])

        return junction_layer

    # produces output for every single node
    # helper function for learning algorith, do not use
    def output_node(self, junction_layer):
        junction_layer = np.vstack([[[1]], junction_layer])

        res = [junction_layer]

        for l in range(len(self.weight_list)):
            res.append(np.tanh(self.weight_list[l].dot(res[-1]) + self.bias_list[l]))

        # sigmoid function for last layer
        # res.append(sigmoid(self.weight_list[-1].dot(res[-1]) + self.bias_list[-1]))

        return res

    # back propagation learning
    # example_list: an Example class containing the input and the desired output of the neural network
    def learn(self, example_list):
        for example in example_list:
            # outputs of each neuron
            a = self.output_node(example.input)

            # delta of last layer neurons
            delta = [(1 - a[-1] * a[-1]) * (example.result - a[-1])]

            # computes delta for the hidden layers
            for i in reversed(range(len(self.weight_list) - 1)):
                sum_data = self.weight_list[i + 1].transpose().dot(delta[0])
                delta = [(1 - a[i + 1] ** 2) * sum_data] + delta

            for i in range(len(self.weight_list)):
                self.weight_list[i] = self.weight_list[i] + self.alpha / self.pass_num * delta[i].dot(a[i].transpose())

    # saves neural network data to file
    # file_name: file name
    def save(self, file_name):
        data = [self.weight_list, self.bias_list, self.alpha, self.pass_num]
        np.save(file_name, data, allow_pickle=True, fix_imports=True)

    # loads neural network data to file
    # file_name: file name
    def load(self, file_name):
        data = np.load(file_name, allow_pickle=True, fix_imports=True)
        self.weight_list = data[0]
        self.bias_list = data[1]
        self.alpha = data[2]
        self.pass_num = data[3]

    # returns a neural network from the loaded file
    # file_name: file name
    @staticmethod
    def new_load(file_name):
        nw = NeuralNetwork()
        nw.load(file_name)

        return nw


# END of Neural Network Algorithm #####################################################################

# Load font
font = ImageFont.load("ie9x14u.pil")


# Letter class stores information about letter and padding for randomizing position
class Letter:

    def __init__(self, image=None, x_delta=None, y_delta=None):
        self.image = image
        self.x_delta = x_delta
        self.y_delta = y_delta


# Returns a 14 by 9 matrix of a character from the loaded font
def get_char_matrix(char):
    mask = font.getmask(char)
    return (np.reshape(np.array(mask), (14, 9))) / 255


# Returns a list of Letters - each letter is shifted to the top left and is assigned
# deltas on how much the letter can be shifted on the x and y axis
def generate_char_list():
    font_list = []

    for i in range(ord('A'), ord('Z') + 1):
        font_foo = get_char_matrix(chr(i))

        # rolls over font to very top pixels
        while sum(font_foo[0, :]) == 0:
            font_foo = np.roll(font_foo, -1, axis=0)

        # rolls over font to very left pixels
        while sum(font_foo[:, 0]) == 0:
            font_foo = np.roll(font_foo, -1, axis=1)

        # calculates deltas of font
        x_delta = 0
        for i in reversed(range(len(font_foo[0, :]))):
            if sum(font_foo[:, i]) == 0:
                x_delta = x_delta + 1
            else:
                break

        y_delta = 0
        for i in reversed(range(len(font_foo[:, 0]))):
            if sum(font_foo[i, :]) == 0:
                y_delta = y_delta + 1
            else:
                break

        l_class = Letter(font_foo, x_delta, y_delta)

        font_list.append(l_class)

    return font_list


# Add noise to image
# img - 2d matrix of image
# noise - amount of noise to add
def add_noise_image(img, noise):
    noise_list = sample(range(126), noise)

    for n in noise_list:
        r = n // 9
        c = n % 9

        img[r][c] = 1 - img[r][c]


# Add noise to image of example
# img - 2d matrix of image
# noise - amount of noise to add
def add_noise_example(example, noise):
    img = example.input
    img = img.reshape(14, 9)
    add_noise_image(img, noise)

    img = img.reshape(126, 1)
    example.input = img


# generates an image from a character input
# char_list an array of all the possible characters
def generate_char_img(char_list, char=-1, noise=0):
    # -1 for random char
    if char == -1:
        char = randint(0, len(char_list) - 1)

    x_delta = randint(0, char_list[char].x_delta)
    y_delta = randint(0, char_list[char].y_delta)

    img = char_list[char].image
    img = np.roll(img, x_delta, axis=1)
    img = np.roll(img, y_delta, axis=0)

    # adds noise to image
    add_noise_image(img, noise)

    return img


# generates image example with input and expected results when input is fed to the Neural Network
def generate_example(char_list, char=-1, noise=0):
    if char == -1:
        char = randint(0, 25)

    res = np.zeros((26, 1))
    res[char][0] = 1 - res[char][0]

    # generates random noise from 0 to 3 if noise == -1
    if noise == -1:
        noise = randint(0, 3)

    img = generate_char_img(char_list, char, noise)
    img = img.reshape(126, 1)

    return Example(img, res)


# generates a num number of examples to a list. This calls generate_example
def generate_example_list(char_list, num, char=-1, noise=0):
    gen_list = []

    for i in range(num):
        gen_list.append(generate_example(char_list, char, noise))

    return gen_list


# returns a character based on the results from the neural network or example output
def result_to_char(arr):
    arr = arr.reshape(1, 26)[0]

    max_val = arr[0]
    idx = 0

    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            idx = i

    return chr(ord('A') + idx)


# returns an epoch list of Examples
def generate_epoch(char_list, noise=-1, shuffle_list=True):
    epoch_list = []

    for i in range(len(char_list)):
        epoch_list.append(generate_example(char_list, i, noise))

    if shuffle_list:
        shuffle(epoch_list)

    return epoch_list


# Returns a percentage of the examples the ai is able to complete correctly
def test_ai(ai, examples):
    correct = 0

    for ex in examples:
        ai_out = ai.output(ex.input)
        ai_char = result_to_char(ai_out)
        correct_ans = result_to_char(ex.result)

        if ai_char == correct_ans:
            correct = correct + 1

    return correct / len(examples) * 100


########################################################################################################################

# Majority Function test
if TEST_MAJORITY:
    AND_AI = NeuralNetwork([2, 10, 1], 0.1)

    MAX_LEARN = 10000
    example_list = []
    for i in range(MAX_LEARN):
        a = randint(0, 1)
        b = randint(0, 1)
        example_list.append(Example(np.array([[a], [b]]), np.array([[int(a == 1 and b == 1)]])))

    AND_AI.learn(example_list)

    total = 500
    correct = 0

    for i in range(total):
        a = randint(0, 1)
        b = randint(0, 1)
        c = AND_AI.output((np.array([[a], [b]])))

        if int(a == 1 and b == 1) == int(c[0][0] + 0.5):
            correct = correct + 1

        print('Set [{},{}] -> {} | AI: {} -> {}'.format(a, b, int(a == 1 and b == 1), c, round(c[0, 0])))

    print('Correct: {}%'.format(correct / total * 100))

# Main Program

# Neural Network variables
gcl = generate_char_list()
test = generate_char_img(gcl)
img_recog_ai = NeuralNetwork([126, 126, 60, 26], 0.06)

# if TRAIN is true, retrains new neural network. If false, load from file
if TRAIN:
    num_epoch = 1500
    sample_set = generate_example_list(gcl, 300, noise=-1)

    y = [test_ai(img_recog_ai, sample_set)]
    for i in range(num_epoch):
        img_recog_ai.learn(generate_epoch(gcl, 0))
        y.append(test_ai(img_recog_ai, sample_set))

    print('Final Accuracy: {}%'.format(y[num_epoch]))

    if GRAPHING:
        import matplotlib.pyplot as plt

        plt.plot(range(0, num_epoch + 1), y)
        plt.title('Neural Network Accuracy vs EPOCH w Shifted Letters and No Noise')

        # ticks
        plt.xticks(np.arange(0, 1601, 200))
        # plt.yticks(np.arange(0, 1601, 100))

        # labels
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        plt.grid()
        plt.savefig('Neural Network Accuracy vs EPOCH w Shifting and No Noise.png')
        plt.show()

else:
    img_recog_ai.load(NW_FILENAME)

# Tests accuracy of program for added noise
if TEST_NOISE_ACCURACY:
    num_to_test = 20
    example_list = generate_example_list(gcl, 1000, noise=0)

    import matplotlib.pyplot as plt

    noise_results = []
    for i in range(num_to_test + 1):

        example_list_cpy = example_list

        for j in example_list_cpy:
            add_noise_example(j, i)

        noise_results.append(test_ai(img_recog_ai, example_list_cpy))

    tna_title = 'Neural Network Accuracy vs Noise Added of 1000 Samples'

    if GRAPHING:
	    plt.plot(range(num_to_test + 1), noise_results)
	    plt.title(tna_title)
	    plt.xlabel('# of Noise')
	    plt.ylabel('Accuracy (%)')
	    plt.xticks(np.arange(0, 21, 2))
	    plt.yticks(np.arange(0, 101, 20))
	    plt.grid()
	    plt.savefig('{}.png'.format(tna_title))
	    plt.show()

if TEST_NOISE_LETTER:

    # generates test images with no noise for each letter
    noise_letter_array = []

    # generates examples from A - Z with no noise
    for i in range(26):
        noise_letter_array.append(generate_example(gcl, i, 0))

    failure_list = np.zeros(26)

    # tests up to 20 random noise
    for i in range(21):

        # coppies array
        cpy_noise_letter_array = noise_letter_array

        # add noise to each list
        for j in cpy_noise_letter_array:
            add_noise_example(j, i)

        for ex in noise_letter_array[:]:
            ai_res = img_recog_ai.output(ex.input)
            ai_res = result_to_char(ai_res)
            ex_res = result_to_char(ex.result)

            idx = ord(ex_res) - ord('A')

            if ai_res != ex_res and failure_list[idx] == 0:
                failure_list[idx] = i

    failures = []
    for i in range(len(failure_list)):
        failures.append([chr(i + ord('A')), failure_list[i]])

    print('Number of noises for image recognition to fail:')
    print(failures)
