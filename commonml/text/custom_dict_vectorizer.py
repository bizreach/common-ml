# coding: utf-8

from logging import getLogger

from commonml import es
from commonml.utils import get_nested_value
from scipy.sparse.construct import hstack
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin, TfidfVectorizer, \
    CountVectorizer
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import json


logger = getLogger('commonml.text.custom_dict_vectorizer')


class NumberPaththroughVectorizer(object):

    def __init__(self, dtype):
        self.dtype_text = dtype
        self.vocabulary_ = ['number']

    def fit(self, raw_documents):
        pass

    def transform(self, raw_documents):
        if self.dtype_text == 'float32':
            dtype = np.float32
        elif self.dtype_text == 'int32':
            dtype = np.int32
        output = [[number] for number in raw_documents]
        return csr_matrix(output, dtype=dtype)

    def fit_transform(self, raw_documents):
        return self.transform(self, raw_documents)

    def get_feature_names(self):
        # TODO what do i return
        return self.vocabulary_


class ExtendedLabelBinarizer(LabelBinarizer):

    def __init__(self, neg_label=0, pos_label=1,
                 sparse_output=False, labelindex_path=None):
        super(ExtendedLabelBinarizer, self) \
            .__init__(neg_label, pos_label, sparse_output)
        self.labelindex_path = labelindex_path
        if self.labelindex_path is not None:
            with open(self.labelindex_path, 'r') as f:
                self.labelindex = json.load(f)

    def fit(self, y):
        if self.labelindex_path is not None:
            super(ExtendedLabelBinarizer, self).fit(self.labelindex)
        else:
            super(ExtendedLabelBinarizer, self).fit(y)

    def get_feature_names(self):
        return self.classes_


def build_custom_vectorizer(config):
    vect_rules = []
    for vect_config in config.values():
        vect_rule = {}
        vect_rule['name'] = vect_config.get('name')
        vect_type = vect_config.pop('type')
        vect_args = vect_config.get('vectorizer')
        analyzer_url = vect_config.get('analyzer')
        if analyzer_url is not None:
            vect_args['tokenizer'] = es.build_analyzer(analyzer_url)
        vectorizer = None
        if vect_type == 'count':
            vectorizer = CountVectorizer(**vect_args)
        elif vect_type == 'tfidf':
            vectorizer = TfidfVectorizer(**vect_args)
        elif vect_type == 'number':
            vectorizer = NumberPaththroughVectorizer(**vect_args)
        elif vect_type == 'label':
            vectorizer = ExtendedLabelBinarizer(**vect_args)
        if vectorizer is not None:
            vect_rule['vectorizer'] = vectorizer
            vect_rules.append(vect_rule)
    return CustomDictVectorizer(vect_rules=vect_rules)


def get_nested_str_value(doc, field, default_value=None):
    value = get_nested_value(doc, field, None)
    if value is None:
        return default_value
    if isinstance(value, list):
        return ' '.join(value)
    return value


class CustomDictVectorizer(BaseEstimator, VectorizerMixin):

    def __init__(self, vect_rules):
        self.vect_rules = vect_rules

    def fit(self, raw_documents):
        for vect_rule in self.vect_rules:
            name = vect_rule.get('name')
            vect = vect_rule.get('vectorizer')
            if not hasattr(vect, '__call__'):
                vect.fit([get_nested_str_value(x, name, '') for x in raw_documents])

    def transform(self, raw_documents):
        results = []
        for vect_rule in self.vect_rules:
            name = vect_rule.get('name')
            vect = vect_rule.get('vectorizer')
            if hasattr(vect, '__call__'):
                data = vect([get_nested_str_value(x, name, '') for x in raw_documents])
            else:
                data = vect.transform([get_nested_str_value(x, name, '') for x in raw_documents])
            results.append(data)
        return hstack(results, format='csr', dtype=np.float32)

    def fit_transform(self, raw_documents, y=None):
        results = []
        for vect_rule in self.vect_rules:
            name = vect_rule.get('name')
            vect = vect_rule.get('vectorizer')
            if hasattr(vect, '__call__'):
                data = vect([get_nested_str_value(x, name, '') for x in raw_documents])
            else:
                data = vect.fit_transform([get_nested_str_value(x, name, '') for x in raw_documents])
            results.append(data)
        return hstack(results, format='csr', dtype=np.float32)

    def get_feature_names(self, append_name=True):
        results = []
        for vect_rule in self.vect_rules:
            vect = vect_rule.get('vectorizer')
            if hasattr(vect, '__call__'):
                results.append(vect_rule.get('name'))
            else:
                if append_name:
                    name = vect_rule.get('name')
                    names = [u'{0}={1}'.format(name, x) for x in vect.get_feature_names()]
                else:
                    names = vect.get_feature_names()
                results.extend(names)
        return results

    def get_feature_size(self):
        size = 0
        for vect_rule in self.vect_rules:
            vect = vect_rule.get('vectorizer')
            size += len(vect.vocabulary_)
        return size

    def inverse_transform(self, X):
        names = np.array(self.get_feature_names())

        def get_names(x):
            indices = np.argwhere(x.toarray().flatten() > 0).flatten()
            if len(indices) == 0:
                return []
            else:
                return names[indices]
        return [get_names(x) for x in X]
