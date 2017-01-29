# coding: utf-8

from logging import getLogger
import re
import six

from sklearn.base import BaseEstimator

import numpy as np


logger = getLogger(__name__)

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)


class VocabularyTransformer(BaseEstimator):

    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None,
                 analyzer=None):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        if analyzer:
            self.analyzer = analyzer
        else:
            def tokenizer(x):
                return TOKENIZER_RE.findall(x)
            self.analyzer = tokenizer
        self.word2index = None
        self.index2word = None
        if vocabulary is not None:
            self.make_index(vocabulary)

    def make_index(self, vocabulary):
        self.word2index = {}
        self.index2word = {}
        self.word2index[''] = 0
        self.index2word[0] = ''
        for token, count in sorted(six.iteritems(vocabulary), key=lambda x: x[1], reverse=True):
            if count > self.min_frequency:
                idx = len(self.word2index)
                self.word2index[token] = idx
                self.index2word[idx] = token

    def fit(self, raw_documents):
        vocabulary = {}
        for tokens in raw_documents:
            for token in self.analyzer(tokens):
                if token in vocabulary:
                    vocabulary[token] += 1
                else:
                    vocabulary[token] = 1
        self.make_index(vocabulary)

    def transform(self, raw_documents):
        for tokens in raw_documents:
            word_ids = np.zeros(self.max_document_length, np.int64)
            idx = 0
            for token in self.analyzer(tokens):
                if idx >= self.max_document_length:
                    break
                if token in self.word2index:
                    word_ids[idx] = self.word2index.get(token)
                    idx += 1
            yield word_ids

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def get_feature_names(self):
        return list(six.iterkeys(self.word2index))

    def inverse_transform(self, X):
        return [[self.index2word.get(idx) for idx in x] for x in X]
