# coding: utf-8

import inspect
from logging import getLogger

from chainer import Chain
from chainer import reporter

import chainer.functions as F
import numpy as np


logger = getLogger(__name__)


class Classifier(Chain):
    def __init__(self, predictor, lossfun, accfun,
                 prefit_y=lambda y: y.reshape((y.shape[0], 1)) if y.ndim == 1 else y,
                 astype_y=lambda y: y.astype(np.float32) if y.dtype != np.float32 else y,
                 postpredict_y=lambda y: y):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.has_train = 'train' in inspect.getargspec(self.predictor.__call__).args
        self.prefit_y = prefit_y
        self.astype_y = astype_y
        self.postpredict_y = postpredict_y

    def __call__(self, x, t, train=True):
        self.y = None
        self.loss = None
        self.accuracy = None
        if self.has_train:
            y = self.predictor(x, train=train)
        else:
            y = self.predictor(x)
        self.loss = self.lossfun(y, t)
        reporter.report({'loss': self.loss}, self)
        if self.accfun is not None:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


def softmax_classifier(predictor, accfun=None):
    return Classifier(predictor=predictor,
                      lossfun=F.softmax,
                      accfun=accfun)


def softmax_cross_entropy_classifier(predictor, accfun=None):
    return Classifier(predictor=predictor,
                      lossfun=F.softmax_cross_entropy,
                      accfun=accfun,
                      prefit_y=lambda y: y,
                      astype_y=lambda y: y.astype(np.int32) if y.dtype != np.int32 else y,
                      postpredict_y=lambda y: y.argmax(axis=1))


def hinge_classifier(predictor, accfun=None):
    return Classifier(predictor=predictor,
                      lossfun=F.hinge,
                      accfun=accfun,
                      prefit_y=lambda y: y,
                      astype_y=lambda y: y.astype(np.int32) if y.dtype != np.int32 else y,
                      postpredict_y=lambda y: y.argmax(axis=1))


def sigmoid_classifier(predictor, accfun=None):
    return Classifier(predictor=predictor,
                      lossfun=F.sigmoid,
                      accfun=accfun)


def sigmoid_cross_entropy_classifier(predictor, accfun=None):
    return Classifier(predictor=predictor,
                      lossfun=F.sigmoid_cross_entropy,
                      accfun=accfun,
                      prefit_y=lambda y: y,
                      astype_y=lambda y: y.astype(np.int32) if y.dtype != np.int32 else y)
