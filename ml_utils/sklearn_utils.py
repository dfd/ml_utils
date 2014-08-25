#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
from numpy import array, dot


class ModelAverage(BaseEstimator):
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.weight_a = weight_a
        self.model_a = model_a
        self.model_b = model_b
        
    def fit(self, X, y=None):
        self.model_a.fit(X,y)
        self.model_b.fit(X,y)        
        return self

    def predict_proba(self, X):
        proba_a = self.model_a.predict_proba(X)
        proba_b = self.model_b.predict_proba(X)
        
        proba_a0 = proba_a[:,0]
        proba_a0.shape = (1, proba_a0.shape[0])
        proba_a1 = proba_a[:,1]
        proba_a1.shape = (1, proba_a1.shape[0])
        proba_b0 = proba_b[:,0]
        proba_b0.shape = (1, proba_b0.shape[0])
        proba_b1 = proba_b[:,1]
        proba_b1.shape = (1, proba_b1.shape[0])
        
        left_side = dot(array([self.weight_a, 1-self.weight_a]), 
                         np.vstack((proba_a0, proba_b0)))
        right_side = dot(array([self.weight_a, 1-self.weight_a]), 
                         np.vstack((proba_a1, proba_b1)))
        
        left_side.shape = (left_side.shape[0],1)
        right_side.shape = (right_side.shape[0],1)
        
        return np.hstack((left_side, right_side))
        
    def predict(self, X):
        return self.predict_proba(X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T




def prepare_df_for_scoring(ground_truth, predictions):
    """return DataFrame with common processing for custom scoring functions
    ground_truth: a numpy array of actual values
    predictions: a numpy array of probabilities for assignment to class 1
    """
    if len(predictions.shape) > 1:
        probs = predictions[:,-1]
    else:
        probs = predictions
    labels = ground_truth
    probs = np.ravel(probs)
    probs.shape = (probs.shape[0],1)
    labels = np.ravel(labels)
    labels.shape = (labels.shape[0],1)
    print labels.shape
    print probs.shape
    df = pd.DataFrame(np.concatenate((probs, labels), axis=1),
            columns=['score','label'])

    df.sort('score', ascending=False, inplace=True)
    total_true = df.label.sum()
    total_false = df.label.count() - total_true
    df['offlabel'] = abs(1-df['label'])
    return df, total_true, total_false


def ks(ground_truth, predictions):
    """
    return a dataframe with data necessary to calculate and plot ks
    ground_truth: a numpy array of actual values
    predictions: a numpy array of probabilities for assignment to class 1
    """

    (df, total_true, total_false) = prepare_df_for_scoring(ground_truth,
            predictions)

    df['true_pct'] = np.round(((df.label / total_true).cumsum()), 4) * 100
    df['false_pct'] = np.round(((df.offlabel / total_false).cumsum()), 4) * 100
    df['random'] = np.round(np.arange(start=0, stop=1, 
        step = 1./(total_true+total_false)), 4) * 100

    df['ks'] = df.true_pct - df.false_pct
    df['quantile'] = np.round(np.arange(start=0, stop=1, 
        step = 1./(total_true+total_false)), 4) * 100

    return df


def ks_score(ground_truth, predictions):
    """
    returns the ks_score given actual labels and probabilities
    ground_truth: a numpy array of actual values
    predictions: a numpy array of probabilities for assignment to class 1
    """
    df = ks(ground_truth, predictions)
    return df.ks.max()


# create scorer for use in sklearn models
ks_scorer = make_scorer(ks_score, greater_is_better=True)


def profit(ground_truth, predictions, cost, margin):
    """return a DataFrame with relevant calculations for profit-based
        model scores based on cost per target, and an expected margin per
        conversion
    ground_truth: numpy array of correct binary labels
    predictions: numpy array of probabilities of class 1
    cost: a penalty per target
    margin: reward for a conversion
    """
    
    (df, total_true, total_false) = prepare_df_for_scoring(ground_truth,
            predictions)

    df['margin'] = np.round(((df.label).cumsum()) * margin, 4)
    df['ones'] = np.repeat(1, df.count()[0])
    df['cost'] = df['ones'].cumsum() * cost
    df['profit'] = df['margin'] - df['cost']
    df['quantile'] = np.round(np.arange(start=0, stop=1, 
        step = 1./(total_true+total_false)), 4) * 100
    start = (total_true*margin-df.count()[0]*cost)/(df.count()[0])

    stop = total_true*margin-df.count()[0]*cost
    print stop
    df['random'] = np.round(np.linspace(start=start,
        stop=stop, num=df.count()[0] ), 4)

    return df


def profit_score(ground_truth, predictions, cost, margin):
    """returns the max profit delivered by a model, based on cost per target,
        and an expected margin per conversion, and an ordering of the target
        list according to scored propensity
    ground_truth: numpy array of correct binary labels
    predictions: numpy array of probabilities of class 1
    cost: a penalty per target
    margin: reward for a conversion
    """
    df = profit(ground_truth, predictions, cost, margin)
    return df.profit.max()


def make_profit_scorer(cost, margin):
    """return a scoring function based on profit_score for use in sklearn models
    cost: a penalty per target
    margin: reward for a conversion
    """
    return make_scorer(profit_score, greater_is_better=True, cost=cost,
            margin=margin)
