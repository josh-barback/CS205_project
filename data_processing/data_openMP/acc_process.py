#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 08:31:33 2017

@author: josh
"""

# suppress RutimeWarnings:
import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

# process data
import os

func_dir = '.../data_openMP/' # set path to the directory that contains acc_functions
data_dir = '.../data_test/'   # set path to the directory that contains test cases

os.chdir(func_dir)
import acc_functions as acc

import time
folder_format = 'output_%m-%d-%Y_%H:%M:%S/'
output_dir = data_dir + time.strftime(folder_format, time.localtime())

acc.proc(data_dir, output_dir, run = 'OpenMP', plot = False)


