# coding: utf-8

from logging import getLogger

from chainer import Link, Chain, ChainList
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import chainer

import chainer.functions as F
import chainer.links as L
import numpy as np


logger = getLogger('commonml.sklearn.classifier')


class Classifier(Chain):
    def __init__(self, predictor, lossfun):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.loss = None

    def __call__(self, x, t, train=True):
        y = self.predictor(x, train=train)
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
