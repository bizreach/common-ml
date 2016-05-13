# coding: utf-8

import inspect
from logging import getLogger

from chainer import Chain

import chainer.functions as F


logger = getLogger('commonml.sklearn.classifier')


class Classifier(Chain):
    def __init__(self, predictor, lossfun):
        super(Classifier, self).__init__(predictor=predictor)
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


def softmax_classifier(predictor):
    return Classifier(predictor=predictor, lossfun=F.softmax)


def softmax_cross_entropy_classifier(predictor):
    return Classifier(predictor=predictor, lossfun=F.softmax_cross_entropy)


def hinge_classifier(predictor):
    return Classifier(predictor=predictor, lossfun=F.hinge)


def sigmoid_classifier(predictor):
    return Classifier(predictor=predictor, lossfun=F.sigmoid)


def sigmoid_cross_entropy_classifier(predictor):
    return Classifier(predictor=predictor, lossfun=F.sigmoid_cross_entropy)
