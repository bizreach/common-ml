# coding: utf-8

from commonml.sklearn import estimator
from commonml.sklearn import rnn_estimator
from commonml.sklearn import regressor
from commonml.sklearn import classifier
from commonml.sklearn import auto_encoder

ChainerEstimator = estimator.ChainerEstimator

RnnEstimator = rnn_estimator.RnnEstimator

MeanSquaredErrorRegressor = regressor.mean_squared_error_regressor

SoftmaxCrossEntropyClassifier = classifier.softmax_cross_entropy_classifier
SoftmaxClassifier = classifier.softmax_classifier
HingeClassifier = classifier.hinge_classifier
SigmoidClassifier = classifier.sigmoid_classifier
SigmoidCrossEntropyClassifier = classifier.sigmoid_cross_entropy_classifier

AutoEncoder = auto_encoder.AutoEncoder
