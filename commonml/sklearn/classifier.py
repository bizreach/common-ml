# coding: utf-8

import inspect
from logging import getLogger

from chainer import Chain

import chainer.functions as F
import numpy as np


logger = getLogger('commonml.sklearn.classifier')


class Classifier(Chain):
    def __init__(self, predictor, lossfun,
                 prefit_y=lambda y: y.reshape((y.shape[0], 1)) if y.ndim == 1 else y,
                 astype_y=lambda y: y.astype(np.float32) if y.dtype != np.float32 else y,
                 postpredict_y=lambda y: y):
        super(Classifier, self).__init__(predictor=predictor)
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


def softmax_classifier(predictor):
    return Classifier(predictor=predictor, lossfun=F.softmax)


def softmax_cross_entropy_classifier(predictor):
    return Classifier(predictor=predictor,
                      lossfun=F.softmax_cross_entropy,
                      prefit_y=lambda y: y,
                      astype_y=lambda y: y.astype(np.int32) if y.dtype != np.int32 else y,
                      postpredict_y=lambda y: y.argmax(axis=1))


def hinge_classifier(predictor):
    return Classifier(predictor=predictor,
                      lossfun=F.hinge,
                      prefit_y=lambda y: y,
                      astype_y=lambda y: y.astype(np.int32) if y.dtype != np.int32 else y,
                      postpredict_y=lambda y: y.argmax(axis=1))


def sigmoid_classifier(predictor):
    return Classifier(predictor=predictor, lossfun=F.sigmoid)


def sigmoid_cross_entropy_classifier(predictor):
    return Classifier(predictor=predictor,
                      lossfun=F.sigmoid_cross_entropy,
                      prefit_y=lambda y: y,
                      astype_y=lambda y: y.astype(np.int32) if y.dtype != np.int32 else y,
                      postpredict_y=lambda y: y.argmax(axis=1))
