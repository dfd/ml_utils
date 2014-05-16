#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pylab
from numpy import arange, array
from itertools import cycle
from matplotlib.colors import hex2color
from  matplotlib.colors import colorConverter as cc
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from sklearn_utils import ks, profit
import pandas as pd

single_rgb_to_hsv=lambda rgb: rgb_to_hsv( array(rgb).reshape(1,1,3) ).reshape(3)
single_hsv_to_rgb=lambda hsv: hsv_to_rgb( array(hsv).reshape(1,1,3) ).reshape(3)

def desaturate(color):

    hsv = single_rgb_to_hsv(color)
    hsv[1] = 0.5
    hsv[2] = 0.7
    return single_hsv_to_rgb(hsv)


def desaturize(ax=None):

    if ax is None: ax=plt.gca()
    ax.set_axisbelow(True)
    ax.set_axis_bgcolor([0.9]*3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('outward',10))
    ax.spines['left'].set_edgecolor('gray')
    ax.spines['bottom'].set_edgecolor('gray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True,color='w',linestyle='-',linewidth=2)
    for line in ax.lines:
        col = line.get_color()
        line.set_color(desaturate(cc.to_rgb(col)))

    for patch in ax.patches:
        col = patch.get_facecolor()
        patch.set_facecolor(desaturate(cc.to_rgb(col)))
        patch.set_edgecolor(patch.get_facecolor())

    return ax


def univariate_plots(data, group_bys, y_var, ylim=None, metric_type='mean'):
    """Create univariate plots of numeric variables by each categorical
        variables
    data: a pandas DataFrame
    group_bys: a list of column names for categorical variables
    y_var: a numeric variable
    ylim: the range of values covered on the y axis (tuple)
    metric_type: valid string, mean or median
    """

    for i, group_by in enumerate(group_bys):
        fig, axes = plt.subplots()
        ax = data[group_by].value_counts().sort_index().plot(kind="bar"); 
        _ = plt.axhline(0, color='k')
        ax2 = ax.twinx()
        ax.get_axes()
        ax.set_title(group_by)
        left, width = -.125, .5
        bottom, height = .25, .5
        right = 1.125
        top = bottom + height
        ax.text(left, 0.5*(bottom+top), 'Counts',
                horizontalalignment='right',
                verticalalignment='center',
                rotation='vertical',
                transform=ax.transAxes)

        ax.text(right, 0.5*(bottom+top), y_var,
                horizontalalignment='right',
                verticalalignment='center',
                rotation='vertical',
                transform=ax.transAxes)

        labels = [label.get_text() for label in
                ax.get_axes().get_xaxis().get_majorticklabels()]

        if metric_type=='mean':
            metric_values = data.groupby(group_by).mean().ix[labels,
                         y_var].values

        else:
            metric_values = data.groupby(group_by).median().ix[labels,
                         y_var].values

        ax2.plot(ax.get_xticks(), metric_values, linestyle='-', marker='o',
                         linewidth=2.0, color=".3")

        if ylim:
            ax2.set_ylim(ylim)

        desaturize(ax)

    plt.show()


def make_boxplot(data, cols, byvar, figsize=(6,4), ylim=None):
    """Create a boxplot of one variable vs one or more group by variables
    data: a pandas DataFrame
    cols: the columns of interest
    byvar: the names of columns to group the data by
        (string or list of strings)
    figsize is the size of the plot (tuple)
    ylim: the range of values covered on the y axis (tuple)
    """

    #plt.figure()
    plt.rcParams['figure.figsize'] = figsize
    bp = data[cols].boxplot(by=byvar)
    bp.set_axis_bgcolor([0.9]*3)
    if ylim:
        bp.set_ylim(ylim)

    plt.show()


def plot_ks(model, X, y):
    """Score and plot the lift of a model on a dataset
    model: an sklearn model that give probability scores
    X: a numpy array of input data for model
    y: a numpy array of labels for input data
    """
    # model needs predict_proba()
    df = ks(y, model.predict_proba(X)[:,1])
    df.set_index('quantile', drop=True, inplace=True)
    with pd.plot_params.use('x_compat', True):
        df.true_pct.plot(color='r')
        df.false_pct.plot(color='g')
        df.random.plot()
        plt.vlines(df['ks'].argmax(), 0, 100)

    plt.show()


def plot_profit(model, X, y, cost, margin):
    """Score and plot the profitability of a model on a dataset
    model: an sklearn model that give probability scores
    X: a numpy array of input data for model
    y: a numpy array of labels for input data
    """
    # model needs predict_proba()
    print model.predict_proba(X)[:,1]
    df = profit(y, model.predict_proba(X)[:,1], cost, margin)
    df.set_index('quantile', drop=True, inplace=True)
    with pd.plot_params.use('x_compat', True):
        df.profit.plot(color='r')
        df.random.plot()
        plt.vlines(df['profit'].argmax(), df['random'][df['profit'].argmax()],
                df['profit'].max())

    plt.show()
