import numpy as np
from PIL import ImageFont
from random import randint
from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example

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


# generates an image from a character input
def generate_char_img(char_list, char=-1, noise=0):
    # -1 for random char
    if char == -1:
        char = randint(0, 25)

    x_delta = randint(0, char_list[char].x_delta)
    y_delta = randint(0, char_list[char].y_delta)

    img = char_list[char].image
    img = np.roll(img, x_delta, axis=1)
    img = np.roll(img, y_delta, axis=0)

    for i in range(noise):
        r = randint(0, 13)
        c = randint(0, 8)

        img[r][c] = 1- img[r][c]

    return img


# generates image example with input and expected results when input is fed to the Neural Network
def generate_example(char_list, char=-1, noise=-1):
    if char == -1:
        char = randint(0, 25)

    res = np.zeros((26, 1))
    res[char][0] = 1 - res[char][0]

    # generates random noise from 0 to 3 if noise == -1
    if noise == -1:
        noise = randint(0,3)

    img = generate_char_img(char_list, char, noise)
    img = img.reshape(126, 1)

    return Example(img, res)


def result_to_char(arr):
    arr = arr.reshape(1, 26)[0]

    max_val = arr[0]
    idx = 0

    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            idx = i

    return chr(ord('A') + idx)


########################################################################################################################


gcl = generate_char_list()
test = generate_char_img(gcl)

f = generate_example(gcl)

# img_recog_ai = NeuralNetwork([126, 200, 200, 200, 26], 0.06)
img_recog_ai = NeuralNetwork([126, 126, 60, 26], 0.06)

ex_list = []
for i in range(100000):
    ex_list.append(generate_example(gcl))

img_recog_ai.learn(ex_list)

# img_recog_ai.load('test_good.npy')

# validates predication
total = 1000
correct = 0
for i in range(total):
    e = generate_example(gcl)

    ai_out = img_recog_ai.output(e.input)
    ai_char = result_to_char(ai_out)
    correct_ans = result_to_char(e.result)

    print('Comparing Result: [{}] | AI: [{}]'.format(correct_ans, ai_char))

    if ai_char == correct_ans:
        correct = correct + 1
    else:
        print('Wrong {} : {}'.format(correct_ans, ai_char))


print('\n{} Correct'.format(correct / total * 100))

img_recog_ai.save('v.npy')
