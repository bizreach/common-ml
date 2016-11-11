from scipy.sparse.base import spmatrix
import six

import numpy as np


class XyDataset(object):

    def __init__(self, model, X, y=None, X_dtype=np.float32):
        if X is None:
            raise ValueError('no X are given')
        length = X.shape[0] if isinstance(X, spmatrix) else len(X)
        self.model = model
        self.X = X
        self.y = model.prefit_y(y) if y is not None else None
        self.X_dtype = X_dtype
        self._length = length

    def __getitem__(self, index):
        is_slice = isinstance(index, slice)
        if is_slice:
            size = self.X.shape[0]
            if size < index.stop:
                index = slice(index.start, size, index.step)
        X_sub = self.X[index]
        if isinstance(X_sub, spmatrix):
            X_sub = X_sub.toarray()
        if X_sub.dtype != self.X_dtype:
            X_sub = X_sub.astype(self.X_dtype)

        length = len(X_sub) if is_slice else 1

        if self.y is not None:
            y_sub = self.y[index]
            if isinstance(y_sub, spmatrix):
                y_sub = y_sub.toarray()
            if self.model.astype_y is not None:
                y_sub = self.model.astype_y(y_sub)

            return [tuple([X_sub[i], y_sub[i]])
                    for i in six.moves.range(length)] if is_slice else tuple([X_sub, y_sub])
        else:
            return [tuple([X_sub[i]])
                    for i in six.moves.range(length)] if is_slice else tuple([X_sub])

    def __len__(self):
        return self._length
