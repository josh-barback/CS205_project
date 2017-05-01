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

func_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/final_process_code/'
data_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/final_directory_structure/'

#func_dir = '/n/home07/cs205u1703/CS_205/Project/code/parallel/'
#data_dir = '/n/home07/cs205u1703/CS_205/Project/OpenMP_version/'

os.chdir(func_dir)
import acc_functions as acc

import time
folder_format = 'output_%m-%d-%Y_%H:%M:%S/'

#for i in range(5):
#    output_dir = data_dir + time.strftime(folder_format, time.localtime())
#    acc.proc(data_dir, output_dir, run = 'Python', plot = False)

for i in range(5):
    output_dir = data_dir + time.strftime(folder_format, time.localtime())
    acc.proc(data_dir, output_dir, run = 'Cython', plot = False)

for i in range(5):
    output_dir = data_dir + time.strftime(folder_format, time.localtime())
    acc.proc(data_dir, output_dir, run = 'OpenMP', plot = False)


