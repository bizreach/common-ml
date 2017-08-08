# coding: utf-8

from logging import getLogger

import chainer
from chainer import Chain
from chainer import reporter

import chainer.functions as F
import numpy as np


logger = getLogger(__name__)


class Regressor(Chain):
    def __init__(self, predictor, lossfun, accfun,
                 prefit_y=lambda y: y.reshape((y.shape[0], 1)) if y.ndim == 1 else y,
                 astype_y=lambda y: y.astype(np.float32) if y.dtype != np.float32 else y,
                 postpredict_y=lambda y: y):
        super(Regressor, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.prefit_y = prefit_y
        self.astype_y = astype_y
        self.postpredict_y = postpredict_y

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = self.lossfun(y, t)
        reporter.report({'loss': loss}, self)
        if self.accfun is not None:
            accuracy = self.accfun(y, t)
            reporter.report({'accuracy': accuracy}, self)
        return loss


def mean_squared_error_regressor(predictor, accfun=None):
    return Regressor(predictor=predictor,
                     lossfun=F.mean_squared_error,
                     accfun=accfun)
