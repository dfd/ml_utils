#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer

def ks(ground_truth, predictions):
    """
    return a dataframe with data necessary to calculate and plot ks
    ground_truth: a numpy array of actual values
    predictions: a numpy array of probabilities for assignment to class 1
    """

    probs = predictions
    labels = ground_truth
    probs.shape = (probs.shape[0],1)
    labels = np.ravel(labels)
    labels.shape = (labels.shape[0],1)
    df = pd.DataFrame(np.concatenate((probs, labels), axis=1),
            columns=['score','label'])

    df.sort('score', ascending=False, inplace=True)
    total_true = df.label.sum()
    total_false = df.label.count() - total_true
    df['offlabel'] = abs(1-df['label'])
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


ks_scorer = make_scorer(ks_score, greater_is_better=True)


def profit(ground_truth, predictions, cost, revenue):
    """


    """
    
    probs = predictions
    labels = ground_truth
    probs.shape = (probs.shape[0],1)
    labels = np.ravel(labels)
    #labels = np.ravel(labels)
    labels.shape = (labels.shape[0],1)
    df = pd.DataFrame(np.concatenate((probs, labels), axis=1),
            columns=['score','label'])

    df.sort('score', ascending=False, inplace=True)
    total_true = df.label.sum()
    total_false = df.label.count() - total_true

    df['revenue'] = np.round(((df.label).cumsum()) * revenue, 4)
    df['ones'] = np.repeat(1, df.count()[0])
    df['cost'] = df['ones'].cumsum() * cost
    df['profit'] = df['revenue'] - df['cost']
    df['quantile'] = np.round(np.arange(start=0, stop=1, 
        step = 1./(total_true+total_false)), 4) * 100
    start = (total_true*revenue-df.count()[0]*cost)/(df.count()[0])
    #stop = df['profit'][df['score']==df['score'].max()].values[0]
    stop = total_true*revenue-df.count()[0]*cost
    print stop
    df['random'] = np.round(np.linspace(start=start,
        stop=stop,
        num=df.count()[0] ), 4)

    return df


def profit_score(ground_truth, predictions, cost, revenue):
    df = profit(ground_truth, predictions, cost, revenue)
    return df.profit.max()


def make_profit_scorer(cost, revenue):
    return make_scorer(profit_score, cost=cost, revenue=revenue)
