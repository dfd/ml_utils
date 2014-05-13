#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer

def ks(ground_truth, predictions):
    """
    ks calculates the ks score for a probabilitic prediction of a binary outcome
        against its true values
    ground_truth: a numpy array of actual values
    predictions: a numpy array of probabilities for assignment to class 1
    """
    probs = predictions
    labels = ground_truth
    probs.shape = (probs.shape[0],1)
    labels.shape = (labels.shape[0],1)
    #print np.hstack((probs, labels))
    df = pd.DataFrame(np.concatenate((probs, labels), axis=1),
            columns=['score','label'])
    df.sort('score', ascending=False, inplace=True)
    #df.reset_index(inplace=True)
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
    #df.set_index('quantile', drop=True, inplace=True)
    return df


def ks_score(ground_truth, predictions):
    df = ks(ground_truth, predictions)
    return df.ks.max()


ks_scorer = make_scorer(ks_score, greater_is_better=True)
