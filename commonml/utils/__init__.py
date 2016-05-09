# coding: utf-8

from logging import getLogger
import os
import time

import six

import numpy as np


logger = getLogger('commonml.utils')


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
        if isinstance(current_doc[last_name], list):
            return ' '.join(current_doc[last_name])
        return current_doc[last_name] if current_doc[last_name] is not None else default_value
    return default_value


def text2bitarray(s, l=100, dtype=np.float32):
    result = []
    for c in list(s):
        bits = bin(ord(c))[2:]
        bits = '000000000000000000000000'[len(bits):] + bits
        result.append([int(b) for b in bits][0:24])
        l -= 1
        if l <= 0:
            break
    for _ in six.moves.range(0, l):
        result.append(np.zeros(24))
    return np.array(result, dtype=dtype)
