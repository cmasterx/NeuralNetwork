import numpy as np
from PIL import ImageFont
from random import randint
from csce420_final_project.NeuralNetwork import NeuralNetwork
from csce420_final_project.NeuralNetwork import Example

font = ImageFont.load("ie9x14u.pil")


def get_char_matrix(char):
    mask = font.getmask(char)
    return (np.reshape(np.array(mask), (14, 9))) / 255


def generate_char_list():
    font_list = []

    for i in range(ord('A'), ord('Z') + 1):
        font_foo = get_char_matrix(chr(i))
        font_list.append(np.roll(font_foo, -2, axis=0))

    return font_list

# generates a random char image example
# TODO do noise
def generate_char_img(char_list, char=-1,noise=0):

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

    res = np.zeros((26,1))
    res[char][0] = 1 - res[char][0]

    img = generate_char_img(char_list, char)
    img = img.reshape(126,1)

    return Example(img, res)



gcl = generate_char_list()
test = generate_char_img(gcl)

f = generate_example(gcl)

img_recog_ai = NeuralNetwork([126, 200, 200, 200, 26], 0.01)

ex_list = []
for i in range(10000):
    ex_list.append(generate_example(gcl))

img_recog_ai.learn(ex_list)