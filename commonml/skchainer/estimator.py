# coding: utf-8

import inspect
from logging import getLogger

from chainer import cuda, optimizers, training, iterators
from chainer.dataset import convert
from chainer.training.extensions import ProgressBar
from sklearn.base import BaseEstimator

import numpy as np


logger = getLogger('commonml.sklearn.estimator')


class ChainerEstimator(BaseEstimator):

    def __init__(self,
                 model,
                 optimizer=optimizers.SGD(),
                 batch_size=100,
                 stop_trigger=(20, 'epoch'),
                 out='result',
                 device=-1):
        if device >= 0:
            cuda.get_device(device).use()
            model.to_gpu()
        optimizer.setup(model)
        self.model = model
        self.optimizer = optimizer
        self.stop_trigger = stop_trigger
        self.batch_size = batch_size
        self.device = device
        self.out = out

    def fit(self, X, y=None, extensions=None):
        if y is None:
            raise ValueError('y is None.')

        from commonml.skchainer import XyDataset

        batch_size = self.batch_size
        dataset = XyDataset(X=X, y=y, astype_y=self.model.astype_y)
        while True:
            try:
                dataset_iter = iterators.SerialIterator(dataset, self.batch_size)
                updater = training.StandardUpdater(dataset_iter, self.optimizer, device=self.device)  # ParallelUpdater
                trainer = training.Trainer(updater, self.stop_trigger, out='result')

                if extensions is None:
                    extensions = [ProgressBar()]
                for extension in extensions:
                    trainer.extend(extension)
                trainer.run()
                break
            except RuntimeError as e:
                if 'out of memory' not in e.message:
                    raise e
                batch_size = int(batch_size * 0.8)
                logger.warn(u'Memory shortage. batch_size is changed to %d', batch_size)
                continue

    def predict(self, X, converter=convert.concat_examples):

        from commonml.skchainer import XyDataset

        has_train = 'train' in inspect.getargspec(self.model.predictor.__call__).args

        def predict_on_predictor(X):
            if has_train:
                return self.model.predictor(X, train=False)
            else:
                return self.model.predictor(X)

        results = None
        batch_size = self.batch_size
        dataset = XyDataset(X=X)
        while True:
            try:
                dataset_iter = iterators.SerialIterator(dataset, self.batch_size, repeat=False, shuffle=False)
                for batch in dataset_iter:
                    in_arrays = converter(batch, self.device)
                    pred = predict_on_predictor(in_arrays[0])
                    if results is None:
                        results = cuda.to_cpu(pred.data)
                    else:
                        results = np.concatenate((results, cuda.to_cpu(pred.data)),
                                                 axis=0)
            except RuntimeError as e:
                if 'out of memory' not in e.message:
                    raise e
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
