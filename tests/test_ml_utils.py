#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_ml_utils
----------------------------------

Tests for `ml_utils` module.
"""

import unittest

import env

from ml_utils import pandas_utils as pu

import pandas as pd
import math
import numpy as np


class TestPandasUtilsWithDataFrame(unittest.TestCase):

    def setUp(self):
        d = {'one' : pd.Series([1., 2., 3., 4., 5., 6., 7.],
            index=['a', 'b', 'c', 'd', 'e', 'f', 'g']),
         'two' : pd.Series([1.1, 2.2, 3.3, 4.4],
             index=['a', 'b', 'c', 'd']),
         'three' : pd.Series(['z','z','x','y','x','y','u','w'],
             index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),
         'four' : pd.Series(['a','a','b','b','c','c','c','d'],
             index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),
         'five' : pd.Series(['1.1', '2.2', '3.3', '4.4'],
             index=['a', 'b', 'c', 'd']),
         'six' : pd.Series(['one','one','two','two','two'],
             index=['a', 'b', 'c', 'd', 'e'])
        }

        self.df = pd.DataFrame(d)


    def test_get_num_char_vars(self):
        (num_vars, char_vars) = pu.get_num_char_vars(self.df)
        self.assertEqual(set(num_vars), set(['one','two']),
            'num_vars are incorrect')
        self.assertEqual(set(char_vars), set(['three','four','five','six']),
            'char_vars are incorrect')

    def test_get_num_char_vars_missing(self):
        (num_vars, char_vars) = pu.get_num_char_vars(self.df, missing=True)
        self.assertEqual(set(num_vars), set(['one','two']),
            'num_vars are incorrect - missing values not detected')
        self.assertEqual(set(char_vars), set(['five', 'six']),
            'char_vars are incorrect - non-missing values not excluded')

    def test_get_num_char_vars_limit(self):
        (num_vars, char_vars) = pu.get_num_char_vars(self.df, missing=True,
                limit=self.df.shape[0]-2)
        self.assertEqual(set(num_vars), set(['two']),
            'num_vars are incorrect')
        self.assertEqual(set(char_vars), set(['five','six']),
            'char_vars are incorrect - non-missing values not excluded')

    def test_get_unique_values(self):
        vals = {}
        vals['four'] = ['a', 'b', 'c', 'd']
        vals['three'] = ['z', 'x', 'y', 'u', 'w']
        print self.df
        # drop five and six because assert equality of NaNs mixed with strings
        # is tough
        self.df.drop(['five','six'], axis=1, inplace=True)
        test_vals = pu.get_unique_values(self.df)
        self.assertEqual(set(vals.keys()), set(test_vals.keys()),
                'dictionary keys do not match')
        for key in vals.keys():
            print set(test_vals[key])
            print set(vals[key])
            self.assertEqual(set(vals[key]), set(test_vals[key]),
                'the elements in %s differ' % key)


    def test_get_unique_values_column(self):
        vals = {}
        vals['three'] = ['z', 'x', 'y', 'u', 'w']
        test_vals = pu.get_unique_values(self.df, 'three')
        self.assertEqual(set(vals.keys()), set(test_vals.keys()),
                'dictionary keys do not match')
        self.assertEqual(set(vals['three']), set(test_vals['three']),
                'the elements in three differ')

    def test_get_unique_values_list(self):
        vals = {}
        vals['three'] = ['z', 'x', 'y', 'u', 'w']
        test_vals = pu.get_unique_values(self.df, ['three'])
        self.assertEqual(set(vals.keys()), set(test_vals.keys()),
                'dictionary keys do not match')
        self.assertEqual(set(vals['three']), set(test_vals['three']),
                'the elements in three differ')

    def test_convert_col_type_int(self):
        pu.convert_col_type(self.df, ['five'], 'int', True)
        for i in range(4):
            self.assertEqual(self.df.five[i], self.df.one[i])
        for i in range(4,8):
            self.assertTrue(math.isnan(self.df.five[i]))

    def test_convert_col_type_float(self):
        pu.convert_col_type(self.df, ['five'], 'float', False)
        for i in range(8):
            self.assertTrue(isinstance(self.df.five[i], float))
        self.assertTrue(math.isnan(self.df.one[7]))


    def test_convert_col_type_str(self):
        pu.convert_col_type(self.df, ['one', 'two'], 'str')
        for i in range(7):
            self.assertTrue(isinstance(self.df.one[i], basestring),
                    'problem in column one, index %i' % i)
        self.assertTrue(math.isnan(self.df.one[7]))
        for i in range(4):
            self.assertTrue(isinstance(self.df.two[i], basestring),
                    'problem in column two number, index %i' % i)
        for i in range(4,8):
            self.assertTrue(math.isnan(self.df.two[i]),
                    'problem in column two nan, index %i' % i)


    def test_impute_by_group_median(self):
        pu.impute_by_groups(self.df, 'one', 'three', 'median')
        self.assertEqual(self.df.one[7], 4)

    def test_impute_list_by_groups_median(self):
        pu.impute_by_groups(self.df, ['one','two'], 'three', 'median')
        self.assertEqual(self.df.one[7], 4)
        self.assertEqual(self.df.two[4], 3.3)
        self.assertEqual(self.df.two[5], 4.4)
        self.assertEqual(self.df.two[6], 3.3)
        self.assertEqual(self.df.two[7], 3.3)

    def test_impute_list_by_groups_median(self):
        pu.impute_by_groups(self.df, ['one','two'], ['three','four'], 'median')
        self.assertEqual(self.df.one[7], 4)
        self.assertEqual(self.df.two[4], 3.3)
        self.assertEqual(self.df.two[5], 4.4)
        self.assertEqual(self.df.two[6], 3.85)
        self.assertEqual(self.df.two[7], 3.3)

    def test_impute_list_by_groups_mean(self):
        values = [1.1, 2.2, 3.3, 4.4, 3.3, 4.4, 3.85]
        pu.impute_by_groups(self.df, ['one','two'], ['three','four'], 'mean')
        self.assertEqual(self.df.one[7], 4)
        self.assertEqual(self.df.two[4], 3.3)
        self.assertEqual(self.df.two[5], 4.4)
        self.assertEqual(self.df.two[6], 3.85)
        self.assertEqual(round(self.df.two[7],5), round(sum(values)/float(len(values)),5))


    def test_impute_list_by_groups_mode(self):
        self.df['three']['g'] = 'z'
        pu.impute_by_groups(self.df, 'six', 'three', 'mode')
        self.assertEqual(self.df.six[5], 'two')
        self.assertEqual(self.df.six[6], 'one')
        self.assertEqual(self.df.six[7], 'two')

    def tearDown(self):
        pass


class TestPandasUtilsWithoutDataFrame(unittest.TestCase):

    def test_to_str_int(self):
        str_val = pu.to_str(2)
        self.assertTrue(isinstance(str_val, basestring))
        self.assertEqual(float(str_val),2) 

    def test_to_str_float(self):
        str_val = pu.to_str(2.1, integer=False)
        self.assertTrue(isinstance(str_val, basestring))
        self.assertEqual(float(str_val),2.1) 

    def test_to_str_float_int(self):
        str_val = pu.to_str(2.1, integer=True)
        self.assertTrue(isinstance(str_val, basestring))
        self.assertEqual(float(str_val),2) 

    def test_to_int(self):
        int_val = pu.to_int("2")
        self.assertTrue(isinstance(int_val, int))
        self.assertEqual(int_val, 2)

    def test_to_int_float(self):
        int_val = pu.to_int("2.1")
        self.assertTrue(isinstance(int_val, int))
        self.assertEqual(int_val, 2)

    def test_to_int_str(self):
        val = pu.to_int("word")
        self.assertTrue(val, "word")

    def test_to_float(self):
        val = pu.to_float("2.1")
        self.assertTrue(val, "2.1")

    def test_to_float(self):
        val = pu.to_float("word")
        self.assertTrue(val, "word")





if __name__ == '__main__':
    unittest.main()
