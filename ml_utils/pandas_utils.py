#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats

def get_num_char_vars(data, missing=False, limit=None):
    """ Return lists of column names for numeric and character columns
    data: a pandas DataFrame
    missing: boolean for whether to return only names of columns with
    missing values
    limit: the upper bound of non-missing values to be included
    """

    num_vars = []
    char_vars = []
    if missing == True:
        if not limit:
            limit = data.shape[0]

    else:
        limit = data.shape[0]+1

    for col in data.columns:
        if data[col].count() < limit:
            if data[col].dtypes == "object":
                char_vars.append(col)

            else:
                num_vars.append(col)
            
    return num_vars, char_vars


def drop_under_threshold(data, threshold=None):
    """drop all columns that have fewer non-missing values than the threshold
    data: a pandas DataFrame
    threshold: the lowest number of non-missing values to keep a column,
        if omitted, function will drop any columns with missing values
    """

    if not threshold:
        threshold = data.shape[0]

    num_vars, char_vars = get_num_char_vars(data, missing=True, limit=threshold)
    data.drop(num_vars + char_vars, inplace=True, axis=1)


def print_unique_values(data, var_list=None):
    """Print the unique values for either a list of variables or all character
    variables
    data: a pandas DataFrame
    var_list: a list of column names; if None, then use character variables
    """

    if not var_list:
        num_vars, var_list = get_num_char_vars(data)

    if isinstance(var_list, basestring):
        var_list = [var_list]

    for col in var_list:
        print col
        print data[col].unique()


def to_str(s, integer=True):
    """Convert a column to strings
    usage: new_var = data.apply(lambda f : to_str(f['COLNAME']) , axis = 1)
    integer: boolean for whether to convert to int first
    """

    try:
        if integer:
            s1 = str(int(s))

        else:
            s1 = str(s)

        return s1

    except ValueError:
        return s


def to_int(s):
    """Convert a column to ints
    usage: new_var = data.apply(lambda f : to_int(f['COLNAME']) , axis = 1)
    """

    try:
        s1 = int(s)
        return s1

    except ValueError:
        return s


def to_float(s):
    """Convert a column to ints
    usage: new_var = data.apply(lambda f : to_int(f['COLNAME']) , axis = 1)
    """

    try:
        s1 = float(s)
        return s1

    except ValueError:
        return s


def convert_col_type(data, cols, new_type, integer=True):
    """Convert specified columns into specified type or types in place
    data: a pandas DataFrame
    cols: a list of columns to convert
    new_type: the new type for the columns ('int', 'float', 'str')
        list of same length as cols or string
    integer: boolean for whether to convert to int before str, only used when
        new_type is 'str'
    """

    if isinstance(cols, basestring):
        cols = [cols]

    if isinstance(new_type, basestring):
        new_type = [new_type] * len(cols)

    for idx, col in enumerate(cols):
        try:
            if new_type[idx] == "int":
                data[col] = data.apply(lambda f : to_int(f[col]), axis = 1)

            elif new_type[idx] == "float":
                data[col] = data.apply(lambda f : to_float(f[col]), axis = 1)

            elif new_type[idx] == "str":
                data[col] = data.apply(lambda f : to_str(f[col], integer),
                        axis = 1)

            else:
                print(str(new_type[idx]) + " is not a valid option.  Skipping.")

        except IndexError:
            print("'cols' and 'new_type' must be lists of the same length"
                    " or new_type must be one valid option")


def impute_by_groups(data, cols, groupbys, impute_type):
    """Fill in missing values with the mean or median by another variable
    data: a pandas DataFrame
    cols: a list of names of columns to be imputed
    groupbys: an ordered list of columns from which to base the imputation
    impute_type: "mean" or "median"
    dataframe
    """

    if isinstance(cols, basestring):
        cols = [cols]

    if isinstance(groupbys, basestring):
        groupbys = [groupbys]

    for col in cols:
        for gb in groupbys:
            data[col].fillna(data.groupby(gb)[col].transform(impute_type),
                    inplace=True)

        #for remaining missing values, impute based on all non-missing values
        if impute_type=="median":
            data[col].fillna(data[col].median(), inplace=True)

        elif impute_type=="mean":
            data[col].fillna(data[col].mean(), inplace=True)


def standardize_dataframe(data, sd=1, drop_last_dummy=True):
    """Return new DataFrame with categorical variables converted to dummies
    and numeric variables standardized to 0 mean, 1 standard deviation
    data: a pandas DataFrame
    sd: the resulting standard deviation of variables it is applied to
    drop_last_dummy: boolean for whether or not to drop the last dummy
    resulting from the conversion of categorical variables
    """

    num_vars, char_vars = get_num_char_vars(data)
    rich_features = data.ix[:,num_vars]
    rich_features = (rich_features - 
            rich_features.mean()) / ((1/sd)*rich_features.std())

    for col in char_vars:
        temp_df = pd.get_dummies(data[col], prefix=col)
        if drop_last_dummy:
            temp_df = temp_df.ix[:,0:(len(temp_df.columns)-1)]

        rich_features = pd.concat([rich_features, pd.DataFrame(temp_df)],
                axis=1)

    return rich_features


def count_missing(data, cols=None, pct=False):
    """Return a series representing the number of missing values for each
    column.  Optionally limit which columns are evaluated
    data: a pandas DataFrame
    cols: a list of column names
    """

    if cols:
        if isinstance(cols, basestring):
            cols = [cols]
    else:
        cols = list(data.columns)

    df = data[cols].shape[0] - data[cols].count()
    if pct:
        df = df/(data.shape[0] + 0.0)

    return df


def mean_by_group(data, col, groupby):
    """Return a DataFrame containing the mean of a variable, and counts,
    grouped by a second variable
    data: a pandas DataFrame
    col: the name of the column to be averaged
    groupby: the name of the column to group the records by
    """

    df = pd.DataFrame({groupby:data[groupby].unique(),
        col: data.groupby(groupby).mean().ix[data[groupby].unique(),
            col].values, 'count':
        data.groupby(groupby).count().ix[data[groupby].unique(),
            groupby].values})
    df.set_index(groupby, inplace=True)
    df = df[[col, 'count']]
    return df


def keep_only_values(data, col, value):
    """drop any rows for which the column does not have a value in a list
    data: a pandas DataFrame
    col: a column to check
    value: a list of values to keep; delete the rest
    """
   
    total = list(data[col].unique())
    invalid = set(total) - set(value)
    for val in invalid:
        data = data[data[col] != val]

    return data



def drop_correlated(data, cols, threshold):
    #corrs = data[num_vars].corr()
    #cols = corrs.columns
    #rows = corrs.index
    #corrs_true = abs(corrs.values) > 0.7
    #corr_df = pd.DataFrame(corrs_true, index=rows, columns = cols)
    pass


def append_boxcox(data, cols, drop_old=False):
    """Apply boxcox transformations to a list of columns
    data: a pandas DataFrame
    cols: a list of column names for which to perform boxcox transformations
    """
    if isinstance(cols, basestring):
        cols = [cols]

    for col in cols:
        data[col + '_boxcox'] = stats.boxcox(data[col])[0]
        if drop_old:
            data.drop(col, axis=1, inplace=True)
