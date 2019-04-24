from PIL import ImageFont, ImageDraw, Image, BdfFontFile
import numpy as np

# image = Image.new('RGB', (60, 30), color = 'red')
# draw = ImageDraw.Draw(image)

font = ImageFont.load("ie9x14u.pil")


def get_char_matrix(char):
    mask = font.getmask(char)
    return np.reshape(np.array(mask), (14,9))

# use a bitmap font
# char = 'L'
# c_arr = np.reshape(np.array(mask), (14,9))

sum = get_char_matrix('A')
# for i in range(ord('A'), ord('Z') + 1)


# draw.text((10, 10), "hello", font=font)
#
# # use a truetype font
# font = ImageFont.truetype("arial.ttf", 15)
#
# draw.text((10, 25), "world", font=font)