

# to run this file
# at commandline:
# python setup.py build_ext --inplace
# then:
# import acc_c_helpers

cimport cython

import  numpy as np
cimport numpy as np


#########################################
# Miscellaneous constants
#########################################

gee = 9.80665
second = 1000 # UTC = milliseconds from epoch
minute = 60*second
hour   = 60*minute
day    = 24*hour

#####################################
# Helper for get_epochs
#   - input signal magnitude data as np.array([times, samples])
#   - calculate mean absolute deviation for each epoch
#   - count samples for each epoch
#####################################

cpdef proc_epochs(long[:] times, double[:] samples, 
                  long[:] epoch_start, long[:] epoch_end):
    
    cdef long i, j, e0, e1
    cdef long n_epochs = len(epoch_start)

    mean_dev  = np.repeat(np.nan, n_epochs)
    n_samples = np.repeat(np.nan, n_epochs)
    
    for i in range(n_epochs):

        e0 = epoch_start[i]
        e1 = epoch_end[i]

        index = [j for j in range(len(times)) if (times[j] >= e0 and times[j] <= e1)]
        s = np.asarray([samples[j] for j in index])

        if len(s) > 0:             
            mean_dev[i] = np.mean(abs(s - gee))
        
        n_samples[i] = len(s)             
        
    return mean_dev, n_samples


#####################################
# Another helper for get_epochs
#   - input signal magnitude data as np.array([times, samples])
#   - calculate mean absolute deviation for each epoch
#   - count samples for each epoch
#####################################

cpdef double[:] proc_locf(double[:] epoch_dev, double[:] epoch_samples):
    
    cpdef long i
    cpdef double[:] epoch_locf = np.repeat(np.nan, len(epoch_dev))
    
    for i in range(len(epoch_samples)):    

        if epoch_samples[i] != 0:
            epoch_locf[i] = epoch_dev[i]

        else:
            epoch_locf[i] = epoch_locf[i - 1]

    return epoch_locf


#####################################
# Replacement for get_times
#   - Get burst lengths and sedentary times
####################################

cpdef get_times(long[:] activity):

    burst_lengths = []
    ie_times      = []
    
    cdef int i
    cdef int act_count = 0
    cdef int sed_count = 0
    state = None
    
    for i in range(len(activity)):
            
        if i==0:
    
            if activity[0] == 0:
                burst_lengths.append(None)
                sed_count += 1
                state = 'Sedent'
            
            if activity[0] == 1:
                act_count += 1
                state = 'Active'
    
        if i > 0 and i < len(activity) - 1:
    
            if activity[i] == 0:
                if state == 'Sedent':
                    sed_count += 1
                else: # if state == 'Active':
                    burst_lengths.append(act_count)
                    act_count = 0
                    state = 'Sedent'
                    sed_count += 1
            
            if activity[i] == 1:
                if state == 'Sedent':
                    ie_times.append(sed_count)
                    sed_count = 0
                    state = 'Active'
                    act_count += 1
                else: # if state == 'Active':
                    act_count += 1
                
        if i == len(activity) - 1: 
            if activity[i] == 0:
                if state == 'Active':
                    burst_lengths.append(act_count)
                    ie_times.append(None)
                if state == 'Sedent':
                    ie_times.append(None)
                    
            if activity[i] == 1:
                if state == 'Sedent':
                    ie_times.append(sed_count)    
                    
    return [burst_lengths, ie_times]





