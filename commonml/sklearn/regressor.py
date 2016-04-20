# coding: utf-8

from logging import getLogger

from chainer import Link, Chain, ChainList
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import chainer

import chainer.functions as F
import chainer.links as L
import numpy as np


logger = getLogger('commonml.sklearn.regressor')


class Regressor(Chain):
    def __init__(self, predictor, lossfun):
        super(Regressor, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.loss = None

    def __call__(self, x, t, train=True):
        y = self.predictor(x, train=train)
        self.loss = self.lossfun(y, t)
        return self.loss

def mean_squared_error_regressor(predictor):
    return Regressor(predictor=predictor, lossfun=F.mean_squared_error)