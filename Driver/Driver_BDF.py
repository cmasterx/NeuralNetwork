from PIL import ImageFont, ImageDraw, Image, BdfFontFile
import numpy as np

# image = Image.new('RGB', (60, 30), color = 'red')
# draw = ImageDraw.Draw(image)

font = ImageFont.load("ie9x14u.pil")


def get_char_matrix(char):
    mask = font.getmask(char)
    return (np.reshape(np.array(mask), (14,9))) / 255


def matrix_or(m1, m2):
    m1 = m1.reshape(1, 126)
    m2 = m2.reshape(1, 126)
    m1 = m1[0]
    m2 = m2[0]

    return np.array([ [m1[i] or m2[i] for i in range(126)] ]).reshape(14,9)
# use a bitmap font
# char = 'L'
# c_arr = np.reshape(np.array(mask), (14,9))

a = get_char_matrix('A')

for i in range(ord('B'), ord('Z') + 1):
    a = matrix_or(a, get_char_matrix(chr(i)))

a = np.roll(a, -2, axis=0)

# for i in range(ord('A'), ord('Z') + 1)

# for i in range(ord('A'), ord('Z') + 1):
#

# draw.text((10, 10), "hello", font=font)
#
# # use a truetype font
# font = ImageFont.truetype("arial.ttf", 15)
#
# draw.text((10, 25), "world", font=font)