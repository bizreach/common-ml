# coding: utf-8

import inspect
from logging import getLogger

from chainer import Chain

import chainer.functions as F
import numpy as np


logger = getLogger('commonml.sklearn.regressor')


class Regressor(Chain):
    def __init__(self, predictor, lossfun,
                 prefit_y=lambda y: y.reshape((y.shape[0], 1)) if y.ndim == 1 else y,
                 astype_y=lambda y: y.astype(np.float32) if y.dtype != np.float32 else y,
                 postpredict_y=lambda y: y):
        super(Regressor, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.loss = None
        self.has_train = 'train' in inspect.getargspec(self.predictor.__call__).args
        self.prefit_y = prefit_y
        self.astype_y = astype_y
        self.postpredict_y = postpredict_y

    def __call__(self, x, t, train=True):
        if self.has_train:
            y = self.predictor(x, train=train)
        else:
            y = self.predictor(x)
        self.loss = self.lossfun(y, t)
        return self.loss


def mean_squared_error_regressor(predictor):
    return Regressor(predictor=predictor, lossfun=F.mean_squared_error)
