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

func_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/parallel/'
data_dir  = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/parallel_data/'

#func_dir = '/n/home07/cs205u1703/CS_205/Project/code/parallel/'
#data_dir = '/n/home07/cs205u1703/CS_205/Project/OpenMP_version/'

os.chdir(func_dir)
import acc_parallel_functions as acc_parallel

acc_parallel.proc(data_dir)
