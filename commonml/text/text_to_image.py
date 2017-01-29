# coding: utf-8

from PIL import Image
from commonml.utils import text2bitarray, bitarray2text

import numpy as np


def text2bitimage(text,
                  filename,
                  text_len=100,
                  bit_size=24,
                  mode="RGB",
                  converter=lambda x: x):
    bitarray = text2bitarray(text=text,
                             text_len=text_len,
                             bit_size=bit_size,
                             dtype=np.uint8,
                             converter=converter)
    bitarray *= 255
    im = Image.fromarray(bitarray)
    im = im.convert(mode)
    im.save(filename, 'png')


def bitimage2text(filename,
                  mode="L",
                  converter=lambda x: x):
    with Image.open(filename) as im:
        im = im.convert(mode)
        bitarray = np.array(np.asarray(im) / 255, dtype=np.uint8)
        return bitarray2text(bitarray=bitarray,
                             converter=converter)
