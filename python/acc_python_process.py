#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 03:18:11 2017

@author: josh
"""

# suppress RutimeWarnings:
import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

# process data
import os

#func_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/week of 4-10 work/'
#data_dir  = '/home/josh/Desktop/Dropbox/_research 2017/work/week of 4-10 work/new_directory_structure/'

func_dir = '/n/home07/cs205u1703/CS_205/Project/code/'
data_dir = '/n/home07/cs205u1703/CS_205/Project/Python_version/'

os.chdir(func_dir)
import acc_python_functions as acc

acc.proc(data_dir)