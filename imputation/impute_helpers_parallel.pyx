
# to run this file
# at commandline:
# python setup.py build_ext --inplace
# then:
# import impute_helpers_parallel

cimport cython
cimport openmp
from cython.parallel import prange, parallel
from cython import boundscheck, wraparound

import  numpy as np
cimport numpy as np

import pandas as pd


#################################################
# Downsample epochs according to pattern
# Cython version
#################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef downsample_Cython(
                        np.ndarray[double, ndim = 1] activity, 
                        int epoch_length, 
                        int on_seconds, 
                        int off_seconds):

    cdef long i
    cdef int  on_epochs =    on_seconds / epoch_length
    cdef int  off_epochs =   off_seconds / epoch_length
    cdef int  cycle_length = on_epochs + off_epochs
    cdef long n_cycles = len(activity) / cycle_length
    cdef np.ndarray[double, ndim = 1] activity_ds
    cdef np.ndarray[long, ndim = 1]  cycle_starts = cycle_length * np.arange(n_cycles)
    
    activity_ds = np.copy(activity)

    for i in cycle_starts:
        activity_ds[i + on_epochs : i + cycle_length] = np.repeat(np.nan, off_epochs)
                
    return activity_ds

    
#################################################
# Downsample epochs according to pattern
# OpenMP version
#################################################
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef downsample_OpenMP(
                        np.ndarray[double, ndim = 1] activity, 
                        int epoch_length, 
                        int on_seconds, 
                        int off_seconds):

    cdef long i, j
    cdef int  on_epochs =    on_seconds / epoch_length
    cdef int  off_epochs =   off_seconds / epoch_length
    cdef int  cycle_length = on_epochs + off_epochs
    cdef long n_cycles = len(activity) / cycle_length
    #cdef np.ndarray[double, ndim = 1] activity_ds
    cdef double[:] activity_ds = np.copy(activity)
    cdef np.ndarray[long, ndim = 1]   cycle_starts = cycle_length * np.arange(n_cycles)
    #cdef np.ndarray[double, ndim = 1] empty = np.repeat(np.nan, off_epochs)
    cdef double[:] empty = np.repeat(np.nan, off_epochs)
    cdef long n_starts = len(cycle_starts)
    
    # activity_ds = np.copy(activity)

    # for i in cycle_starts:
    for j in prange(n_starts, nogil = True, schedule = dynamic, num_threads = 32):
        i = cycle_starts[j]
        activity_ds[i + on_epochs : i + cycle_length] = empty
                
    return np.asarray(activity_ds)
    
#################################################
# Extract active bursts and sedentary periods
# Cython version
# (no OpenMP version due to dependence of loop iterations.)
#################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_times_Cython(np.ndarray[double, ndim = 1] activity, 
                       np.ndarray[long, ndim = 1] start, 
                       np.ndarray[long, ndim = 1] end):

    cdef long i
    cdef long count = 0
    
    cdef np.ndarray[long, ndim = 1] bursts         = np.asarray([], dtype = long)
    cdef np.ndarray[object, ndim = 1]  classification = np.asarray([], dtype = object)
    cdef np.ndarray[object, ndim = 1]  complete       = np.asarray([], dtype = object)
    cdef np.ndarray[long, ndim = 1] burst_start = np.asarray([], dtype = long)
    cdef np.ndarray[long, ndim = 1] burst_end   = np.asarray([], dtype = long)

    cdef np.str state = None
    cdef np.str complete_state = None

            
    for i in range(len(activity)):
        
        if i == 0:
            if activity[i] == 1:
                count += 1
                state = 'Active'
                complete_state = 'left-cutoff'
                burst_start = np.append(burst_start, start[i])
            elif activity[i] == 0:
                count += 1
                state = 'Sedentary'
                complete_state = 'left-cutoff'
                burst_start = np.append(burst_start, start[i])
            elif np.isnan(activity[i]):
                state = 'Missing'
                
        if i > 0 and i < len(activity) - 1:
            
            if state == 'Active':
                if activity[i] == 1:
                    count += 1
                elif activity[i] == 0:
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Active')
                    complete = np.append(complete, complete_state)
                    burst_end = np.append(burst_end, end[i-1])
                    burst_start = np.append(burst_start, start[i])
                    count = 1
                    state = 'Sedentary'                        
                    complete_state = 'complete'
                elif np.isnan(activity[i]):
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Active')
                    if complete_state == 'left-cutoff':   
                        complete = np.append(complete, 'both-cutoff')
                    elif complete_state == 'complete':
                        complete = np.append(complete, 'right-cutoff')
                    burst_end = np.append(burst_end, end[i-1])
                    count = 0
                    state = 'Missing'                        
                    complete_state = None
                    
            elif state == 'Sedentary':
                if activity[i] == 1:
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Sedentary')
                    complete = np.append(complete, complete_state)
                    burst_end = np.append(burst_end, end[i-1])
                    burst_start = np.append(burst_start, start[i])
                    count = 1
                    state = 'Active'
                    complete_state = 'complete'
                elif activity[i] == 0:
                    count += 1
                elif np.isnan(activity[i]):
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Sedentary')
                    if complete_state == 'left-cutoff':
                        complete = np.append(complete, 'both-cutoff')
                    elif complete_state == 'complete':
                        complete = np.append(complete, 'right-cutoff')
                    burst_end = np.append(burst_end, end[i-1])
                    count = 0
                    state = 'Missing'
                    complete_state = None
                                                        
            elif state == 'Missing':
                if activity[i] == 1:
                    count = 1
                    state = 'Active'
                    complete_state = 'left-cutoff'
                    burst_start = np.append(burst_start, start[i])
                elif activity[i] == 0:
                    count = 1
                    state = 'Sedentary'
                    complete_state = 'left-cutoff'                    
                    burst_start = np.append(burst_start, start[i])
                    
        if i == len(activity) - 1:
                
            if state == 'Active':
                if activity[i] == 1:
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Active')
                    if complete_state == 'complete':
                        complete = np.append(complete, 'right-cutoff')
                    if complete_state == 'left-cutoff':
                        complete = np.append(complete, 'both-cutoff')
                    burst_end = np.append(burst_end, end[i])
                elif activity[i] == 0:
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Active')
                    complete = np.append(complete, complete_state)
                    burst_end = np.append(burst_end, end[i-1])
                    bursts = np.append(bursts, 1)
                    classification = np.append(classification, 'Sedentary')
                    complete = np.append(complete, 'right-cutoff')
                    burst_start = np.append(burst_start, start[i])
                    burst_end = np.append(burst_end, end[i])
                elif np.isnan(activity[i]):
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Active')
                    if complete_state == 'complete':
                        complete = np.append(complete, 'right-cutoff')
                    if complete_state == 'left-cutoff':
                        complete = np.append(complete, 'both-cutoff')
                    burst_end = np.append(burst_end, end[i-1])
                                                            
            elif state == 'Sedentary':
                if activity[i] == 1:
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Sedentary')
                    complete = np.append(complete, complete_state)
                    burst_end = np.append(burst_end, end[i-1])
                    bursts = np.append(bursts, 1)
                    classification = np.append(classification, 'Active')
                    complete = np.append(complete, 'right-cutoff')
                    burst_start = np.append(burst_start, start[i])
                    burst_end = np.append(burst_end, end[i])
                elif activity[i] == 0:
                    bursts = np.append(bursts, count + 1)
                    classification = np.append(classification, 'Sedentary')
                    if complete_state == 'complete':
                        complete = np.append(complete, 'right-cutoff')
                    if complete_state == 'left-cutoff':
                        complete = np.append(complete, 'both-cutoff')
                    burst_end = np.append(burst_end, end[i])
                elif np.isnan(activity[i]):
                    bursts = np.append(bursts, count)
                    classification = np.append(classification, 'Sedentary')
                    if complete_state == 'complete':
                        complete = np.append(complete, 'right-cutoff')
                    if complete_state == 'left-cutoff':
                        complete = np.append(complete, 'both-cutoff')
                    burst_end = np.append(burst_end, end[i-1])
  
            elif state == 'Missing':
                if activity[i] == 1:
                    bursts = np.append(bursts, 1)
                    classification = np.append(classification, 'Active')
                    complete = np.append(complete, 'both-cutoff')
                    burst_start = np.append(burst_start, start[i])
                    burst_end = np.append(burst_end, end[i])
                elif activity[i] == 0:
                    bursts = np.append(bursts, 1)
                    classification = np.append(classification, 'Sedentary')
                    complete = np.append(complete, 'both-cutoff')
                    burst_start = np.append(burst_start, start[i])
                    burst_end = np.append(burst_end, end[i])

    d = {'bursts'  : bursts, 
         'class'   : classification, 
         'complete': complete, 
         'start'   : burst_start, 
         'end'     : burst_end}

    result = pd.DataFrame.from_dict(d)
        
    return result
    
    
#################################################
# Fit imputation model
# Cython version
# (No OpenMP version.)
#################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fit_Cython(np.ndarray[double, ndim = 1] activity_ds, 
                 int on_seconds, 
                 int epoch_length, 
                 double depth):
                
    cdef long  i, end, k, key
    cdef double y
    cdef int   on_epochs = on_seconds / epoch_length
    cdef int   n_predictors = int(np.floor(on_epochs * depth))
    cdef long  n_epochs = len(activity_ds)

    cdef double  n_0 = np.nansum(activity_ds)        
    cdef double  n_1 = np.nansum(1 - activity_ds)
    cdef double prop_ones = (n_1 + 0.0) / (n_0 + n_1)
        
    cdef np.ndarray[double, ndim = 2] data, d, store
    cdef np.ndarray[long, ndim = 1] find
    cdef np.ndarray[long, ndim = 1] keep, keys, counts
    cdef np.ndarray[double, ndim = 1] unique, block
    cdef dict model, temp
        
    # stack data
    data = None        
    
    for i in range(n_predictors + 1):
        block = activity_ds[i : n_epochs]
        end = len(block) - len(block)%(n_predictors + 1)
        d = np.array(block[0:end]).reshape((end/(n_predictors + 1), n_predictors + 1))       
        if i == 0:
            data = d
        else:
            data = np.vstack( (data, d) )

    # drop rows with missing data
    find = np.repeat(1, len(data))
    
    for i in range(len(data)):            
        if any(list(np.isnan(data[i]))):
            find[i] = 0
    keep = np.asarray([i for i in range(len(data)) if find[i] == 1])
    data = data[keep, :]

    # build dictionary        
    keys = np.arange(int('1'*n_predictors, 2) + 1)
    store = np.repeat(np.nan, len(data) * len(keys)).reshape(len(data), len(keys))
    
    for i in range(len(data)):            
        key = int(str(data[i][0:n_predictors]).replace('[', '').replace(']', '').replace('.', '').replace(' ', ''), 2)
        y = data[i][n_predictors]
        store[i, key] = y

    model = {}
    for k in keys:
        unique, counts = np.unique(store[:, k], return_counts = True)
        temp = dict(zip(unique, counts))
        if 0 in temp and 1 in temp:
            model[k] = (temp[1] + 0.0) / (temp[1] + temp[0])
        elif 0 in temp and not 1 in temp:
            model[k] = 0
        elif 1 in temp and not 0 in temp:
            model[k] = 1
        else:
            model[k] = prop_ones

    return model  

    
#################################################
# Run imputation model on downsampled data set
# Cython version
# (No OpenMP version.)
#################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef impute_Cython(np.ndarray[double, ndim = 1] activity_ds, 
                    dict model, 
                    int on_seconds, 
                    int epoch_length):

    cdef long i, t, on_epochs, depth, cycle_length, key
    cdef np.ndarray[long, ndim = 1] starts
    cdef np.ndarray[double, ndim = 1] imputed
    cdef str bin_key
    
    imputed = np.copy(activity_ds)        
    depth = int(np.log2(len(model)))
    on_epochs = on_seconds / epoch_length
    cycle_length = on_epochs * 2
    starts = np.arange(on_epochs, len(activity_ds), cycle_length)
    
    for t in starts:            
        for i in np.arange(on_epochs):
            if t + i < len(imputed):                 
                bin_key = str(list(imputed[t + i - depth : t + i])).replace('.0', '').replace('[', '').replace(']', '').replace('.', '').replace(' ', '').replace(',', '')
                key = int(bin_key, 2)
                imputed[t + i] = np.random.binomial(1, model[key])
            
    return imputed


