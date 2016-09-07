from scipy.sparse.base import spmatrix
import six

import numpy as np


class XyDataset(object):

    def __init__(self, X, y=None, astype_y=None):
        if X is None:
            raise ValueError('no X are given')
        length = X.shape[0] if isinstance(X, spmatrix) else len(X)
        self.X = X
        self.y = y
        self._length = length
        self.astype_y = astype_y

    def __getitem__(self, index):
        X_sub = self.X[index]
        if isinstance(X_sub, spmatrix):
            X_sub = X_sub.toarray()
        if X_sub.dtype != np.float32:
            X_sub = X_sub.astype(np.float32)

        is_slice = isinstance(index, slice)
        length = len(X_sub) if is_slice else 1

        if self.y is not None:
            y_sub = self.y[index]
            if isinstance(y_sub, spmatrix):
                y_sub = y_sub.toarray()
            if self.astype_y is not None:
                y_sub = self.astype_y(y_sub)

            return [tuple([X_sub[i], y_sub[i]])
                    for i in six.moves.range(length)] if is_slice else tuple([X_sub, y_sub])
        else:
            return [tuple([X_sub[i]])
                    for i in six.moves.range(length)] if is_slice else tuple([X_sub])

    def __len__(self):
        return self._length
