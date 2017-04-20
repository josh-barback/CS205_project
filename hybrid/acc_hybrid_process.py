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

#func_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/mpi_code/'
#data_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/final_directory_structure/'

func_dir = '/n/home07/cs205u1703/CS_205/Project/final/hybrid_code/'
data_dir = '/n/home07/cs205u1703/CS_205/Project/final/data/'

os.chdir(func_dir)
import acc_hybrid_functions as acc

#for i in range(5):
#    acc.proc(data_dir, run = 'OpenMP', plot = False)

acc.proc(data_dir, run = 'OpenMP', plot = False)

