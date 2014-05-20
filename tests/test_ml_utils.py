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


class TestPandas_utils(unittest.TestCase):

    def setUp(self):
        d = {'one' : pd.Series([1., 2., 3., 4., 5., 6., 7.], index=['a', 'b', 'c', 'd', 'e', 'f', 'g']),
         'two' : pd.Series([1.1, 2.2, 3.3, 4.4], index=['a', 'b', 'c', 'd']),
         'three' : pd.Series(['z','z','x','y','x','y','u','w'], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),
         'four' : pd.Series(['a','a','b','b','c','c','c','d'], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

        }

        self.df = pd.DataFrame(d)


    def test_get_num_char_vars(self):
        (num_vars, char_vars) = pu.get_num_char_vars(self.df)
        self.assertEqual(set(num_vars), set(['one','two']),
            'num_vars are incorrect')
        self.assertEqual(set(char_vars), set(['three','four']),
            'char_vars are incorrect')

    def test_get_num_char_vars_missing(self):
        (num_vars, char_vars) = pu.get_num_char_vars(self.df, missing=True)
        self.assertEqual(set(num_vars), set(['one','two']),
            'num_vars are incorrect - missing values not detected')
        self.assertEqual(set(char_vars), set([]),
            'char_vars are incorrect - non-missing values not excluded')

    def test_get_num_char_vars_limit(self):
        (num_vars, char_vars) = pu.get_num_char_vars(self.df, missing=True,
                limit=self.df.shape[0]-2)
        self.assertEqual(set(num_vars), set(['two']),
            'num_vars are incorrect')
        self.assertEqual(set(char_vars), set([]),
            'char_vars are incorrect - non-missing values not excluded')

    def test_get_unique_values(self):
        vals = {}
        vals['four'] = ['a', 'b', 'c', 'd']
        vals['three'] = ['z', 'x', 'y', 'u', 'w']
        test_vals = pu.get_unique_values(self.df)
        self.assertEqual(set(vals.keys()), set(test_vals.keys()),
                'dictionary keys do not match')
        self.assertEqual(set(vals['four']), set(test_vals['four']),
                'the elements in four differ')
        self.assertEqual(set(vals['three']), set(test_vals['three']),
                'the elements in three differ')

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

    def test_to_str_and_to_int(self):
        new_var = self.df.apply(lambda f: pu.to_str(f['one']), axis = 1)
        for i in range(new_var.count()):
            self.assertTrue(isinstance(new_var[i],basestring),
                    'problem converting int to string')

        self.df['new_var'] = new_var
        new_var = self.df.apply(lambda f: pu.to_int(f['new_var'],), axis = 1)
        for i in range(new_var.count()):
            self.assertEqual(new_var[i],round(new_var[i]),
                    'probelm converting string to int')

        new_var = self.df.apply(lambda f: pu.to_str(f['two'], False), axis = 1)
        for i in range(new_var.count()):
            self.assertTrue(isinstance(new_var[i],basestring),
                    'problem converting float to string')




    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
