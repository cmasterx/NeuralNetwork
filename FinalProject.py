import numpy as np
from PIL import ImageFont
from random import randint
from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example

font = ImageFont.load("ie9x14u.pil")


# Letter class stores information about letter and padding for randomizing position
class Letter:

    def __init__(self, image=None, x_delta=None, y_delta=None):
        self.image = image
        self.x_delta = x_delta
        self.y_delta = y_delta


def get_char_matrix(char):
    mask = font.getmask(char)
    return (np.reshape(np.array(mask), (14, 9))) / 255


def generate_char_list():
    font_list = []

    for i in range(ord('A'), ord('Z') + 1):
        font_foo = get_char_matrix(chr(i))

        x_delta = 0;
        y_delta = 0;

        while sum(font_foo[0,:]) == 0:
            x_delta = x_delta + 1

        l_class = Letter(font_foo)

        font_list.append(font_foo)

    return font_list


# generates a random char image example
# TODO do noise
def generate_char_img(char_list, char=-1, noise=0):
    # -1 for random char
    if char == -1:
        char = randint(0, 25)

    x_delta = randint(0, 1)
    y_delta = randint(0, 5)

    img = char_list[char]
    img = np.roll(img, x_delta, axis=1)
    img = np.roll(img, y_delta, axis=0)

    return img


def generate_example(char_list, char=-1):
    if char == -1:
        char = randint(0, 25)

    res = np.zeros((26, 1))
    res[char][0] = 1 - res[char][0]

    img = generate_char_img(char_list, char)
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

print('\n{} Correct'.format(correct / total * 100))
