# coding: utf-8

import inspect
from logging import getLogger

from chainer import Chain

import chainer.functions as F


logger = getLogger('commonml.sklearn.regressor')


class Regressor(Chain):
    def __init__(self, predictor, lossfun):
        super(Regressor, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.loss = None
        self.has_train = 'train' in inspect.getargspec(self.predictor.__call__).args

    def __call__(self, x, t, train=True):
        if self.has_train:
            y = self.predictor(x, train=train)
        else:
            y = self.predictor(x)
        self.loss = self.lossfun(y, t)
        return self.loss


def mean_squared_error_regressor(predictor):
    return Regressor(predictor=predictor, lossfun=F.mean_squared_error)
