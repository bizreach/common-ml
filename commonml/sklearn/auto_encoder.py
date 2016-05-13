# coding: utf-8

from logging import getLogger

from chainer import cuda, Chain, optimizers, Variable
from scipy.sparse.base import spmatrix
import six
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin

import chainer.functions as F
import chainer.links as L
from commonml.sklearn.estimator import ChainerEstimator
import numpy as np


logger = getLogger('commonml.sklearn.auto_encoder')


class AutoEncoder(BaseEstimator, VectorizerMixin):

    def __init__(self,
                 in_size,
                 out_size,
                 regressor,
                 batch_size=100,
                 n_epoch=1,
                 dropout_ratio=.5,
                 optimizer=optimizers.SGD(),
                 report=10,
                 gpu=-1):
        self.model = AutoEncoderModel(in_size=in_size,
                                      out_size=out_size,
                                      dropout_ratio=dropout_ratio)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gpu = gpu
        self.n_epoch = n_epoch
        self.report = report
        self.regressor = regressor

    def get_estimator(self):
        return ChainerEstimator(model=self.regressor(self.model),
                                optimizer=self.optimizer,
                                batch_size=self.batch_size,
                                gpu=self.gpu,
                                report=self.report,
                                n_epoch=self.n_epoch)

    def fit(self, X, y=None):
        estimator = self.get_estimator()
        estimator.fit(X, X)

    def transform(self, X):
        xp = np if self.gpu < 0 else cuda.cupy
        if isinstance(X, spmatrix):
            data_size = X.shape[0]
            indexes = np.random.permutation(data_size)
            Y = None
            for i in six.moves.range(0, data_size, self.batch_size):
                X_sub = Variable(xp.asarray(X[indexes[i: i + self.batch_size]].toarray(), dtype=np.float32))
                Y_sub = self.model.predict(X_sub)
                if self.gpu >= 0:
                    Y_sub.to_cpu()
                Y_sub = Y_sub.data
                if Y is None:
                    Y = Y_sub
                else:
                    Y = np.r_[Y, Y_sub]
            return Y
        else:
            Y = self.model.predict(Variable(xp.asarray(X)))
            if self.gpu >= 0:
                Y.to_cpu()
            return Y.data

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class AutoEncoderModel(Chain):

    def __init__(self, in_size, out_size, dropout_ratio=.5):
        super(AutoEncoderModel, self).__init__(l_enc=L.Linear(in_size, out_size),
                                               l_dec=L.Linear(out_size, in_size),
                                               )
        self.dropout_ratio = dropout_ratio

    def __call__(self, x, train=True):
        h1 = F.dropout(F.relu(self.l_enc(x)), ratio=self.dropout_ratio, train=train)
        h2 = F.dropout(F.relu(self.l_dec(h1)), ratio=self.dropout_ratio, train=train)
        return h2

    def predict(self, x):
        return F.relu(self.l_enc(x))
