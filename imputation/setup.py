#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 08:37:06 2017

@author: josh
"""


from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("impute_helpers_parallel",
              ["impute_helpers_parallel.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]

setup( 
  name = "impute_helpers_parallel",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)


#setup(
#    ext_modules = cythonize("acc_helpers.pyx")
#)


    
    
    
