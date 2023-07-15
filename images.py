import sys

import cv2
import numpy as np
from numpy import uint8
from pdf2image import convert_from_path

DEFAULT_TOP_RANGE = (5, 27)
DEFAULT_BODY_RANGE_1 = (34, 80)
DEFAULT_BODY_RANGE_2 = (37, 80)
DEFAULT_BOTTOM_RANGE = (82, 100)
DEFAULT_BODY_SCALE_1 = 0.77
DEFAULT_BODY_SCALE_2 = 0.77
DEFAULT_BOTTOM_Y_FRAC = 0.7


class ImageInfo:
    def __init__(self, path, body_range, body_scale, top_range=DEFAULT_TOP_RANGE, bottom_range=DEFAULT_BOTTOM_RANGE):
        self.path = path
        self.top_range = top_range
        self.body_range = body_range
        self.bottom_range = bottom_range
        self.body_scale = body_scale


def increase_brightness(img, value=-100):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value > 0:
        v[v > 255 - value] = 255
        v[v <= 255 - value] += value
    else:
        v[v < -value] = 0
        v[v >= -value] -= uint8(-value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


class ImageBuilder:
    def __init__(self, size):
        height, width = size
        self.image = np.zeros((height, width, 3), np.uint8)
        self.image[:] = (255, 255, 255)

    def save_image(self, path):
        cv2.imwrite(path, self.image)

    def addd(self, src, offset, scale_x, scale_y):
        y, x = offset
        h, w = src.shape[0], src.shape[1]
        src = cv2.resize(src, (int(w * scale_x), int(h * scale_y)))
        h, w = src.shape[0], src.shape[1]
        self.image[y: y + h, x: x + w] = src

    def add(self, src, offset, scale):
        self.addd(src, offset, scale, scale)


def combine(info1, info2, result_path):
    image1 = cv2.imread(info1.path)
    image2 = cv2.imread(info2.path)

    h, w = image1.shape[0], image1.shape[1]
    builder = ImageBuilder((h, w))

    def H(frac):
        return int(h * frac / 100)

    def W(frac):
        return int(w * frac / 100)

    def left_offset(scale):
        return W(100 * (1.0 - scale)) // 2

    def range_height(range):
        return H(range[1] - range[0])

    def crop(image, range):
        return image[H(range[0]):H(range[1]), 0:w]

    offset = H(1)

    body_height_1 = range_height(info1.body_range)
    body_height_2 = range_height(info2.body_range)
    body_height_scaled_1 = int(info1.body_scale * body_height_1)
    body_height_scaled_2 = int(info2.body_scale * body_height_2)
    body_1_offset_top = range_height(info1.top_range) + offset
    body_2_offset_top = body_1_offset_top + body_height_scaled_1 + offset

    bottom_offset_top = body_2_offset_top + body_height_scaled_2 + offset
    bottom_height = h - offset - bottom_offset_top
    bottom_scale_y = min(1.0, bottom_height / H(info1.bottom_range[1] - info1.bottom_range[0]))
    bottom_scale_x = DEFAULT_BOTTOM_Y_FRAC

    body_offset_left_1 = left_offset(info1.body_scale)
    body_offset_left_2 = left_offset(info2.body_scale)
    bottom_offset_left = left_offset(bottom_scale_x)

    builder.add(crop(image1, info1.top_range), (0, 0), 1.0)
    builder.addd(crop(image1, info1.bottom_range), (bottom_offset_top, bottom_offset_left), bottom_scale_x,
                 bottom_scale_y)
    builder.add(crop(image1, info1.body_range), (body_1_offset_top, body_offset_left_1), info1.body_scale)
    builder.add(crop(image2, info2.body_range), (body_2_offset_top, body_offset_left_2), info2.body_scale)

    builder.save_image(result_path)


def pdf_to_png(input_file, output_file):
    images = convert_from_path(input_file)
    images[0].save(output_file, 'PNG')


def convert_to_png(input_file):
    if input_file.endswith(".png"):
        return input_file
    output_file = input_file.replace(".pdf", ".png")
    pdf_to_png(input_file, output_file)
    return output_file


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Please pass 3 arguments:\n1) Image 1 path\n2) Image 2 path\n3) Result path.\n")
        exit(1)

    first = convert_to_png(sys.argv[1])
    second = convert_to_png(sys.argv[2])

    combine(ImageInfo(first, DEFAULT_BODY_RANGE_1, DEFAULT_BODY_SCALE_1),
            ImageInfo(second, DEFAULT_BODY_RANGE_2, DEFAULT_BODY_SCALE_2),
            result_path=sys.argv[3])
