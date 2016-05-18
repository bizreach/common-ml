# coding: utf-8

import inspect
from logging import getLogger

from chainer import cuda, Variable, optimizers
from scipy.sparse.base import spmatrix
import six
from sklearn.base import BaseEstimator

import numpy as np


logger = getLogger('commonml.sklearn.estimator')


class ChainerEstimator(BaseEstimator):

    def __init__(self,
                 model,
                 optimizer=optimizers.SGD(),
                 batch_size=100,
                 n_epoch=20,
                 report=10,
                 gpu=-1):
        self.model = model
        if gpu >= 0:
            cuda.get_device(0).use()
            self.model.to_gpu()
        self.optimizer = optimizer
        self.optimizer.setup(self.model)
        self.n_epoch = n_epoch
        self.report = report
        self.batch_size = batch_size
        self.gpu = gpu

    def fit(self, X, y=None):
        if y is None:
            raise ValueError('y is None.')

        xp = np if self.gpu < 0 else cuda.cupy

        y = self.model.prefit_y(y)

        is_spmatrix = isinstance(X, spmatrix)
        data_size = X.shape[0] if is_spmatrix else len(X)

        for epoch in six.moves.range(self.n_epoch):
            logger.info(u'epoch %d', epoch)
            indexes = np.random.permutation(data_size)
            for i in six.moves.range(0, data_size, self.batch_size):
                x1 = X[indexes[i: i + self.batch_size]]
                y1 = y[indexes[i: i + self.batch_size]]
                if is_spmatrix:
                    x1 = x1.toarray()
                if x1.dtype != np.float32:
                    x1 = x1.astype(np.float32)
                if isinstance(y1, spmatrix):
                    y1 = y1.toarray()
                y1 = self.model.astype_y(y1)
                x2 = Variable(xp.asarray(x1))
                y2 = Variable(xp.asarray(y1))
                self.optimizer.update(self.model, x2, y2)

            if self.report > 0 and epoch % self.report == 0:
                sum_loss = 0
                for i in six.moves.range(0, data_size, self.batch_size):
                    x1 = X[indexes[i: i + self.batch_size]]
                    y1 = y[indexes[i: i + self.batch_size]]
                    if is_spmatrix:
                        x1 = x1.toarray()
                    if x1.dtype != np.float32:
                        x1 = x1.astype(np.float32)
                    if isinstance(y1, spmatrix):
                        y1 = y1.toarray()
                    y1 = self.model.astype_y(y1)
                    x2 = Variable(xp.asarray(x1))
                    y2 = Variable(xp.asarray(y1))
                    loss = self.model(x2, y2, train=False)
                    sum_loss += loss.data * len(x1)
                mean_loss = sum_loss / data_size
                logger.info(' -> loss %f', mean_loss)

    def predict(self, X):
        xp = np if self.gpu < 0 else cuda.cupy

        is_spmatrix = isinstance(X, spmatrix)
        data_size = X.shape[0] if is_spmatrix else len(X)

        has_train = 'train' in inspect.getargspec(self.model.predictor.__call__).args
        results = None
        for i in six.moves.range(0, data_size, self.batch_size):
            end = i + self.batch_size
            x1 = X[i: end if end < data_size else data_size]
            if is_spmatrix:
                x1 = x1.toarray()
            if x1.dtype != np.float32:
                x1 = x1.astype(np.float32)
            x2 = Variable(xp.asarray(x1))
            if has_train:
                pred = self.model.predictor(x2, train=False)
            else:
                pred = self.model.predictor(x2)
            if results is None:
                results = cuda.to_cpu(pred.data)
            else:
                results = np.concatenate((results, cuda.to_cpu(pred.data)),
                                         axis=0)

        return self.model.postpredict_y(results)

    def score(self, X, y, sample_weight=None):
        from commonml.sklearn.classifier import Classifier
        from commonml.sklearn.regressor import Regressor
        if isinstance(self.model, Classifier):
            from sklearn.metrics.classification import accuracy_score
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        elif isinstance(self.model, Regressor):
            from sklearn.metrics.regression import r2_score
            return r2_score(y, self.predict(X), sample_weight=sample_weight,
                            multioutput='variance_weighted')
        else:
            raise ValueError('Unsupported model.')
