# coding: utf-8

from logging import getLogger
import time

from chainer import Link, Chain, ChainList, optimizer, cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import chainer
import six
from sklearn.base import BaseEstimator

import chainer.functions as F
import chainer.links as L
from commonml.sklearn import ChainerEstimator
import numpy as np
import math

logger = getLogger('commonml.sklearn.rnn_estimator')


class RnnEstimator(ChainerEstimator):

    def __init__(self, bprop_len=35, **params):
        super(RnnEstimator, self).__init__(**params)
        self.bprop_len = bprop_len

    def fit(self, X, y, report_interval=1000):
        if X is None or y is None:
            raise ValueError('X and/or y is None.')

        xp = np if self.gpu < 0 else cuda.cupy

        data_size = len(X)
        jump = data_size // self.batch_size
        cur_log_perp = xp.zeros(())
        start_at = time.time()
        cur_at = start_at
        accum_loss = 0
        batch_idxs = list(range(self.batch_size))
        self.model.predictor.reset_state()
        self.model.zerograds()
        for i in six.moves.range(jump * self.n_epoch):
            x = Variable(xp.asarray([X[(jump * j + i) % data_size] for j in batch_idxs]))
            t = Variable(xp.asarray([y[(jump * j + i) % data_size] for j in batch_idxs]))

            loss_i = self.model(x, t)
            accum_loss += loss_i
            cur_log_perp += loss_i.data

            if (i + 1) % self.bprop_len == 0:
                #logger.info('Updating parameters')
                accum_loss.backward()
                accum_loss.unchain_backward()
                accum_loss = 0
                self.optimizer.update()
                self.model.predictor.reset_state()
                self.model.zerograds()

            if (i + 1) % report_interval == 0:
                now = time.time()
                throuput = float(report_interval) / (now - cur_at)
                perp = math.exp(float(cur_log_perp) / report_interval)
                logger.info('iter {}/{} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(i + 1, jump * self.n_epoch, perp, throuput))
                cur_at = now
                cur_log_perp.fill(0)

    def predict(self, X):
        xp = np if self.gpu < 0 else cuda.cupy

        data_size = len(X)

        results = None
        for i in six.moves.range(0, data_size, self.batch_size):
            end = i + self.batch_size
            x1 = X[i: end if end < data_size else data_size]
            x2 = Variable(xp.asarray(x1))
            pred = F.softmax(self.model.predictor(x2, train=False))
            if results is None:
                results = cuda.to_cpu(pred.data)
            else:
                results = xp.concatenate((results, cuda.to_cpu(pred.data)), axis=0)

        return results
