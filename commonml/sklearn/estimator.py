# coding: utf-8

import inspect
from logging import getLogger

from chainer import cuda, Variable, optimizers
from scipy.sparse.base import spmatrix
import six
from sklearn.base import BaseEstimator

import numpy as np


logger = getLogger(__name__)


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

    def update(self, X, y):
        self.optimizer.update(self.model, X, y)

    def var_x(self, x, xp):
        if isinstance(x, spmatrix):
            x = x.toarray()
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return Variable(xp.asarray(x))

    def var_y(self, y, xp):
        if isinstance(y, spmatrix):
            y = y.toarray()
        y = self.model.astype_y(y)
        return Variable(xp.asarray(y))

    def evaluate_loss(self, X, y, has_train):
        if has_train:
            return self.model(X, y, train=False)
        else:
            return self.model(X, y)

    def evaluate(self, X, y, indexes, batch_size):
        xp = np if self.gpu < 0 else cuda.cupy
        is_spmatrix = isinstance(X, spmatrix)
        data_size = X.shape[0] if is_spmatrix else len(X)
        sum_loss = 0
        has_train = 'train' in inspect.getargspec(self.model.predictor.__call__).args
        for i in six.moves.range(0, data_size, batch_size):
            end = i + batch_size
            ids = indexes[i: end if end < data_size else data_size]
            x2 = self.var_x(X[ids], xp)
            y2 = self.var_y(y[ids], xp)
            loss = self.evaluate_loss(x2, y2, has_train)
            sum_loss += loss.data * len(ids)
        mean_loss = sum_loss / data_size
        logger.info(' -> loss %f', mean_loss)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError('y is None.')

        xp = np if self.gpu < 0 else cuda.cupy

        y = self.model.prefit_y(y)

        data_size = X.shape[0] if isinstance(X, spmatrix) else len(X)

        batch_size = self.batch_size
        epoch = 0
        while epoch < self.n_epoch:
            logger.info(u'epoch %d', epoch)
            indexes = np.random.permutation(data_size)
            try:
                for i in six.moves.range(0, data_size, batch_size):
                    end = i + batch_size
                    ids = indexes[i: end if end < data_size else data_size]
                    x2 = self.var_x(X[ids], xp)
                    y2 = self.var_y(y[ids], xp)
                    self.update(x2, y2)
            except RuntimeError as e:
                if 'out of memory' not in e.message:
                    raise e
                # TODO clear GPU memory
                if x2 is not None:
                    x2.to_cpu()
                if y2 is not None:
                    y2.to_cpu()
                batch_size = int(batch_size * 0.8)
                logger.warn(u'Memory shortage. batch_size is changed to %d', batch_size)
                continue

            if self.report > 0 and epoch % self.report == 0:
                self.evaluate(X, y, indexes, batch_size)

            epoch += 1

    def predict_on_predictor(self, X, has_train):
        if has_train:
            return self.model.predictor(X, train=False)
        else:
            return self.model.predictor(X)

    def predict(self, X):
        xp = np if self.gpu < 0 else cuda.cupy

        is_spmatrix = isinstance(X, spmatrix)
        data_size = X.shape[0] if is_spmatrix else len(X)

        has_train = 'train' in inspect.getargspec(self.model.predictor.__call__).args
        batch_size = self.batch_size
        results = None
        while True:
            try:
                for i in six.moves.range(0, data_size, batch_size):
                    end = i + batch_size
                    x2 = self.var_x(X[i: end if end < data_size else data_size], xp)
                    pred = self.predict_on_predictor(x2, has_train)
                    if results is None:
                        results = cuda.to_cpu(pred.data)
                    else:
                        results = np.concatenate((results, cuda.to_cpu(pred.data)),
                                                 axis=0)
            except RuntimeError as e:
                if 'out of memory' not in e.message:
                    raise e
                # TODO clear GPU memory
                if x2 is not None:
                    x2.to_cpu()
                results = None
                batch_size = int(batch_size * 0.8)
                logger.warn(u'Memory shortage. batch_size is changed to %d', batch_size)
                continue
            break

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
