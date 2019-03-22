# coding: utf-8

from logging import getLogger
import os
import time

import six

import numpy as np


logger = getLogger(__name__)
_range = six.moves.range


def get_nested_value(doc, field, default_value=None):
    field_names = field.split('.')
    current_doc = doc
    for name in field_names[:-1]:
        if name in current_doc:
            current_doc = current_doc[name]
        else:
            current_doc = None
            break
    last_name = field_names[-1]
    if current_doc is not None and last_name in current_doc:
        return current_doc[last_name] if current_doc[last_name] is not None else default_value
    return default_value


def text2bitarray(text,
                  text_len=100,
                  bit_size=24,
                  dtype=np.float32,
                  converter=lambda x: x):
    result = []
    for c in list(text):
        bits = bin(ord(c))[2:].zfill(bit_size)[0:bit_size]
        result.append(converter([int(b) for b in bits]))
        text_len -= 1
        if text_len <= 0:
            break
    for _ in _range(0, text_len):
        result.append(converter([0 for _ in _range(0, bit_size)]))
    if dtype is None:
        return result
    return np.array(result, dtype=dtype)


def bitarray2text(bitarray,
                  converter=lambda x: x):
    if isinstance(bitarray, list):
        bitarray = np.array(bitarray)
    text_len = bitarray.shape[0]
    chars = []
    for i in _range(0, text_len):
        bits = converter(bitarray[i].tolist())
        c = chr(int(''.join(map(str, bits)), 2))
        if c == chr(0):
            break
        chars.append(c)
    return ''.join(chars)
