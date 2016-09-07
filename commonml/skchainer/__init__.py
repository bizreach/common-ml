# coding: utf-8

from commonml.skchainer import xy_dataset
from commonml.skchainer import estimator
from commonml.skchainer import regressor
from commonml.skchainer import classifier

ChainerEstimator = estimator.ChainerEstimator

MeanSquaredErrorRegressor = regressor.mean_squared_error_regressor

SoftmaxCrossEntropyClassifier = classifier.softmax_cross_entropy_classifier
SoftmaxClassifier = classifier.softmax_classifier
HingeClassifier = classifier.hinge_classifier
SigmoidClassifier = classifier.sigmoid_classifier
SigmoidCrossEntropyClassifier = classifier.sigmoid_cross_entropy_classifier

XyDataset = xy_dataset.XyDataset
