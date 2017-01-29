# coding: utf-8

from logging import getLogger

from chainer import Variable
from scipy.sparse.base import spmatrix
import six

from commonml.sklearn.estimator import ChainerEstimator
import numpy as np


logger = getLogger(__name__)


# experimental
class RnnEstimator(ChainerEstimator):

    def __init__(self, **params):
        super(RnnEstimator, self).__init__(**params)

    def var_x(self, x, xp):
        if isinstance(x, spmatrix):
            x = x.toarray()
        if x.dtype != np.int32:
            x = x.astype(np.int32)
        return Variable(xp.asarray(x))

    def update(self, X, y):
        size = X.data.shape[1]
        accum_loss = None
        for pos in six.moves.range(size):
            X_new = Variable(X.data[:, pos])
            accum_loss = self.model(X_new, y)

        self.model.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()
        self.optimizer.update()
        self.model.predictor.reset_state()

    def evaluate_loss(self, X, y, has_train):
        self.model.predictor.reset_state()
        size = X.data.shape[1]
        accum_loss = None
        for pos in six.moves.range(size):
            X_new = Variable(X.data[:, pos])
            accum_loss = super(RnnEstimator, self).evaluate_loss(X_new, y, has_train)
        return accum_loss

    def predict_on_predictor(self, X, has_train):
        self.model.predictor.reset_state()
        size = X.data.shape[1]
        result = None
        for pos in six.moves.range(size):
            X_new = Variable(X.data[:, pos])
            result = super(RnnEstimator, self).predict_on_predictor(X_new, has_train)
        return result
