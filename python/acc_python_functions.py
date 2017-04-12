#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:55:50 2017

@author: josh
"""

import pandas as pd
import numpy as np
import time
import os
import zipfile
import powerlaw

from statsmodels.tsa.stattools import acf



#########################################
# Miscellaneous constants
#########################################

gee = 9.80665
second = 1000 # UTC = milliseconds from epoch
minute = 60*second
hour   = 60*minute
day    = 24*hour


####################################
# Convert date = 'mm/dd/yyyy', time = 'hh:mm'
# to GM UTC timestamp
####################################

def to_gmtime(date, t):
    
    temp = time.mktime(time.strptime( date + ' ' + t, '%m/%d/%Y %H:%M' ))

    return temp * 1000

    
####################################
# Convert timestamp to local time
####################################
    
def to_localtime(timestamp, time_format = '%H:%M'):
    
    return time.strftime(time_format, time.localtime(timestamp / 1000))

####################################
# Unzip data
# Input data directory (not zip directory)
####################################

def unzip(directory):
    files = os.listdir(directory + 'raw_zips/')
    
    if not os.path.exists(directory + 'raw_unzipped/'):
        os.makedirs(directory + 'raw_unzipped/')
    
    zips = [f for f in files if f[-4:] == '.zip'] 
    for z in zips:
        temp_ref = zipfile.ZipFile(directory + 'raw_zips/' + z)
        temp_ref.extractall(directory + 'raw_unzipped/')
        temp_ref.close()
        
    print 'Unzipped', len(zips), 'directories.'


####################################
# Get researcher records
####################################

def get_records(directory):
    
    records = open(directory + 'researcher_records.csv', 'r').readlines()
     
    temp = []    
    for i in range(1, len(records)):    
        temp.append(records[i].split(',')[0:4])
        
    print 'Read', len(records)-1, 'records.'        
    return temp

    
#########################################
# Stack selected files from raw data directory
# Return stack as dataframe
#########################################

def stack_files(files, source):
   
    stack = []
    
    for f in files:    
        temp = pd.read_csv(source + f, sep = ',')
        if len(stack) == 0:
            stack = temp
        else:
            stack = stack.append(temp, ignore_index = True)
                
    stack = stack.sort_values(by = 'timestamp', na_position = 'first')
    stack.index = range(len(stack))

    stack.drop('UTC time', 1, inplace = True)
    stack.drop('accuracy', 1, inplace = True)
    
    return stack    
    

#########################################
# From stacked raw dataframe, 
# generate signal magnitude and return
#########################################

def sigmag(raw, phone_type):

    timestamp = raw['timestamp']

    if phone_type == 'Android':
        x = raw['x']
        y = raw['y']
        z = raw['z']

    if phone_type == 'iPhone':
        x = raw['x'] * gee
        y = raw['y'] * gee
        z = raw['z'] * gee
        
    return (x**2 + y**2 + z**2) ** (0.5)
    
    
#########################################
# Given sigmag data frame and start/end times:
# Return data frame with MAD from 1g per 5-second epoch,
# also return LOCF imputation by default for
# epochs with no samples.
#########################################

def get_epochs(sigmag, start, end, locf = True):
        
    times = sigmag['timestamp'] 
    t0 = times[0] - ( times[0] % (5*second) )
    t1 = times[len(times)-1]
    
    # set epoch start/end times
    epoch_start = np.arange(t0, t1, 5*second)
    n_epochs = len(epoch_start)
    epoch_end = epoch_start + (5*1000) - 1

    # calculate mean absolute deviation for each epoch
    # count samples for each epoch
    epoch_dev = []
    epoch_samples = []
    
    for i in range(n_epochs):
        
        e0 = epoch_start[i]
        e1 = epoch_end[i]
        temp = sigmag.loc[ (sigmag['timestamp'] >= e0) & (sigmag['timestamp'] <= e1) ]
        samples = np.asarray(temp['sig_mag'])
        
        if len(samples) > 0:             
            mean_dev = np.mean(abs(samples - gee))
        if len(samples) == 0:
            mean_dev = None
             
        epoch_dev.append(mean_dev)
        epoch_samples.append(len(samples))

    # do LOCF imputation
    if locf:
        epoch_locf = []
        for i in range(n_epochs):

            if epoch_samples[i] > 0:
                epoch_locf.append(epoch_dev[i])
        
            if epoch_samples[i] == 0:
                epoch_locf.append(epoch_locf[i - 1])
                
    # create data frame and write to csv
    temp_dict = {'start'    : epoch_start,
                 'end'      : epoch_end,
                 'mean_dev' : epoch_dev,
                 'n_samples': epoch_samples,
                 'locf'     : epoch_locf}

    epochs = pd.DataFrame.from_dict(temp_dict)
    epochs.index = range(len(epochs))    
    
    # print n_epochs, 'epochs have been processed.'
    return epochs
    
####################################
# Process data for a record:
# [id, date, time_start, time_end]
####################################

def process_data(record, directory):
            
    idd, date, t0, t1 = record
    
    # get phone type
    id_dir = directory +  'raw_unzipped/' + idd + '/identifiers/'
    identifier = os.listdir(id_dir)[0]    
    temp = open(id_dir + identifier, 'r').readlines()
    if temp[1].find('iPhone') > -1:
        phone_type = 'iPhone'
    if temp[1].find('Android') > -1:
        phone_type = 'Android'
        
    # get start/end times for each raw file
    raw_dir = directory + 'raw_unzipped/' + idd + '/accelerometer/'
    raw_files = os.listdir(raw_dir)
    file_info = []    
    for f in raw_files:
        temp = open(raw_dir + f, 'r').readlines()        
        first = int(temp[1]. split(',')[0])       
        last =  int(temp[-1].split(',')[0])
        file_info.append([first, last, f])
        
    # convert record times to GM times    
    tt0 = to_gmtime(date, t0)
    tt1 = to_gmtime(date, t1)
    
    # drop files outside of relevant times, stack remaining files
    to_stack = [ i[2] for i in file_info if 
                (i[0] > tt0 and i[0] < tt1) or (i[1] > tt0 and i[1] < tt1) ]
    day_stack = stack_files(to_stack, raw_dir)

    # get signal magnitudes
    day_stack['sig_mag'] = sigmag(day_stack, phone_type)
    
    # convert iPhone measurements to m/s^2
    if phone_type == 'iPhone':
        day_stack['x'] = gee * day_stack['x']
        day_stack['y'] = gee * day_stack['y']
        day_stack['z'] = gee * day_stack['z']

    # get epochs
    epochs = get_epochs(day_stack, tt0, tt1)
    
    # write to file
    destination = directory + 'proc_data/' + idd + '/'
    if not os.path.exists(destination):
        os.makedirs(destination)

    fix_date = date.replace('/', '-')
    day_stack.to_csv(destination + idd + '_' + 'data' + '_' + fix_date + '_' + t0 + '-' + t1 + '.csv',
                     sep = ',', header = True, index = False)
    epochs.   to_csv(destination + idd + '_' + 'epochs' + '_' + fix_date + '_' + t0 + '-' + t1 + '.csv',
                     sep = ',', header = True, index = False)
    
    print 'Processed', len(to_stack), 'files for', idd, 'from', date + '.'
            
        
####################################
# Find 54.9%ile cutoffs for each
# participant in records
####################################

def get_cutoffs(records, directory):
    
    idds = list(set([r[0] for r in records]))

    cutoff_dict = {}
            
    for idd in idds:
        idd_dir = directory + 'proc_data/' + idd + '/'
        files = [f for f in os.listdir(idd_dir) if f.find('epochs') > -1]
        mads = []
        for f in files:
            mads = mads + list(pd.read_csv(idd_dir + f)['mean_dev'])
        mads = [i for i in mads if not np.isnan(i)]
        cutoff_dict[idd] = np.percentile(mads, 54.9)
    
    print 'Calculated cutoffs for', len(idds), 'participants.'                
    return cutoff_dict

    
####################################
# Append activity classifications to
# epoch data
####################################

def activity_class(cutoffs, directory):

    proc_dir = directory + 'proc_data/'
    idds = os.listdir(proc_dir)

    file_count = 0
    
    for idd in idds:
        work_dir = proc_dir + idd + '/'
        files = [f for f in os.listdir(work_dir) if f.find('epochs') > -1]
        
        for f in files:
            file_count += 1
            temp = pd.read_csv(work_dir + f)            
            c = cutoffs[idd]   
            mad = temp['mean_dev']
            b = mad > c
            temp['activity'] = b.astype(int)
            temp.to_csv(work_dir + f,
                        sep = ',', header = True, index = False)

    print 'Appended activity classification to', file_count, 'files.'

    

#########################################
# Generate summary plots.
# Optionally, save a copy to directory.
#########################################

def plot_summary(t, sigmag, t_epochs, mad, activity, record, cutoff, directory = None):

    import matplotlib.pyplot as plt

    idd, date, t0, t1 = record

    tt0 = to_gmtime(date, t0)
    tt1 = to_gmtime(date, t1)    
    
    # set up x-axis ticks    
    xticks = np.arange(tt0-tt0%(60*60*1000) , tt1, 60*60*1000)
    xlabels = [to_localtime(i) for i in xticks]
    
    plt.close('all')

    f, (a0, a1, a2) = plt.subplots(3, 1, 
                                   gridspec_kw = {'height_ratios': [3, 2, 1]},
                                    figsize = (16, 12))

    # plot signal magnitude
    a0.plot(t, sigmag, ',', color = 'b')
    a0.hlines(np.arange(4) * gee, xmin = tt0, xmax = tt1, color = '0.65')

    a0.set_yticks(np.arange(4) * gee)
    a0.set_yticklabels([str(i) + 'g' for i in range(4)], rotation = 'vertical')
    a0.set_ylabel('Signal Magnitude')
            
    a0.set_xticks(xticks)
    a0.set_xticklabels(xlabels)

    a0.axis([tt0, tt1, -1, 3*gee + 1])        
    
    # plot mean absolute deviations
    a1.plot(t_epochs, mad, color = 'k')

    a1.set_yticks([gee/4])
    a1.set_yticklabels(['0.25g'], rotation = 'vertical')
    a1.hlines(gee/4, xmin = tt0, xmax = tt1, color = '0.65')
    a1.hlines(cutoff, xmin = tt0, xmax = tt1, 
              linestyle = '--', color = 'r', linewidth = 2)
    a1.set_ylabel('Mean Abs. Dev. from 1g')
            
    a1.set_xticks(xticks)
    a1.set_xticklabels(xlabels)

    a1.axis([tt0, tt1, -0.2, 0.5*gee])  
    
    # plot activity classification
    act_class = [t_epochs[i] for i in range(len(t_epochs)) 
                 if activity[i] == 1]
    a2.vlines(act_class, ymin = 0, ymax = 1, color = 'g', alpha = 0.4)

    a2.set_yticks([])
    a2.set_ylabel('Activity')
    
    a2.set_xticks(xticks)
    a2.set_xticklabels(xlabels)
    a2.set_xlabel('Clock Time')
    
    a2.axis([tt0, tt1, 0, 1])  
    
    # set title    
    a0.set_title('Data summary for ' + idd + ' on ' + 
                 date + ', ' + t0 + '-' + t1)
    
    # save a copy
    if directory !=None:
        
        fix_date = date.replace('/', '-')
        destination = directory + 'proc_data/' + idd + '/' + idd + '_' + 'summary-plots' + '_' + fix_date + '.png'
        f.savefig(destination, bbox_inches='tight')
        plt.close('all')

    
####################################
# Generate summaries and plots from 
# processed data.
# Write summaries to output file.
####################################
    
def summarize(records, cutoffs, directory, plot = True):    
    
    for r in records:
        
        idd, date, t0, t1 = r
        cutoff = cutoffs[idd]
        fix_date = date.replace('/', '-')

        data_name = 'proc_data/' + idd + '/' + idd + '_' + 'data' + '_' + fix_date + '_' + t0 + '-' + t1 + '.csv'
        epochs_name = 'proc_data/' + idd + '/' + idd + '_' + 'epochs' + '_' + fix_date + '_' + t0 + '-' + t1 + '.csv'
        
        temp_data   = pd.read_csv(directory + data_name)
        temp_epochs = pd.read_csv(directory + epochs_name)
                
        # generate plots        
        if plot:
            t      = temp_data['timestamp']
            sigmag = temp_data['sig_mag']        
            
            t_epochs = temp_epochs['start'] + (2.5 * second)
            mad      = temp_epochs['mean_dev']
            activity = temp_epochs['activity']
    
            plot_summary(t, sigmag, t_epochs, mad, activity, r, cutoff, directory)

        # generate some summaries        
        mean_Hz = np.mean(temp_epochs['n_samples'])/5
        prop_zero = (0.0 + len([i for i in temp_epochs['n_samples'] if i == 0])) / len(temp_epochs)
        summary = [len(temp_data), len(temp_epochs), mean_Hz, prop_zero, cutoff]
        
        # write to output summary        
        output = directory + 'output_summary.csv'
        if os.path.exists(output):
            temp = open(output, 'a')
            to_write = ','.join(r) + ',' + ','.join(str(x) for x in summary) + ',,,'
            temp.write(to_write + '\n')
            temp.close()
            
        else:
            summary_labels = ['user_id', 'date', 'time_start', 'time_end',
                              'n_obs', 'n_epochs', 'mean_Hz', 'prop_zero',
                              'cutoff', 'median_act', 'alpha_act', 'median_sed', 'alpha_sed']
            temp = pd.DataFrame(index = np.asarray([0]), 
                                columns = summary_labels)
            temp.loc[0] = r + summary + list(np.repeat(np.nan, 4))
            temp.to_csv(output, sep = ',', header = True, index = False)
    
    print 'Generated summaries for', len(records), 'records.'


####################################
# Get burst lengths and sedentary times,
# return as data frame.
####################################

def get_times(activity):

    burst_lengths = []
    ie_times      = []
    
    act_count = 0
    sed_count = 0
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


####################################
# Plot summaries of active and 
# sedentary periods.
# Optionally, save plots using
# path prefix.
####################################

def activity_plots(activity, times, title, path_prefix = None):
    
    import matplotlib.pyplot as plt
        
    plt.close('all')
    
    f1, (a0, a1, a2) = plt.subplots(1, 3, figsize = (18, 5))
    #plt.rc('text', usetex = True)
    #plt.rc('font', family = 'serif')
    plt.suptitle(title)

    # autocorrelation, 20 minutes of lags
    autocorr = acf(activity, nlags = 60*60 / 15, alpha = .05)
    
    a0. axhline(y = 0, color = '0.75')
    a0.plot(range(  (60*60 / 15) + 1  ), np.log10(autocorr[0]), color = 'b', linewidth = 2, label = 'Est. autocorrelation')
    a0.plot(range(  (60*60 / 15) + 1  ), np.log10(autocorr[1][:, 0]), linestyle = ':', linewidth = 2, color = 'g', label = '95% Conf. interval')
    a0.plot(range(  (60*60 / 15) + 1  ), np.log10(autocorr[1][:, 1]), linestyle = ':', linewidth = 2, color = 'g')

    a0.set_xticks(np.arange(60, 75*60/15, 5*60 / 5))
    a0.set_xticklabels(np.arange(5, 5*60/5, 5))    
    a0.set_xlabel('Lag (Minutes)')
    
    a0.set_yticks([-2, -1, 0])
    a0.set_yticklabels(['10^-2', '10^-1', '10^0'], rotation = 'vertical')    
    #a0.set_ylabel(r'Autocorrelation (log scale)')
    
    a0.set_title('Autocorrelation of Active Epochs')
    a0.legend(loc = 3, prop = {'size': 12})
    a0.axis([0, 240, -2, 0])

    # histogram of active bursts
    xticks = np.asarray([0.5, 2, 10, 30]) * (60/5)
    xticklabels = ['0.5', '2', '10', '30']

    powerlaw.Fit(times[0], xmin = 1, discrete = True).power_law.plot_pdf(ax = a1, color = 'g', linewidth = 2, label = 'Power Law fit')
    powerlaw.Fit(times[0], xmin = 1, discrete = True).lognormal.plot_pdf(ax = a1, color = 'r', linewidth = 2, label = 'Lognormal fit')
    #powerlaw.Fit(times[0], xmin = 1, discrete = True).exponential.plot_pdf(ax = a1, color = 'r', label = 'Exponential fit')
    #powerlaw.Fit(times[0], xmin = 1, discrete = True).stretched_exponential.plot_pdf(ax = a1, color = 'b', label = 'Stretched Expo fit')
    #powerlaw.Fit(times[0], xmin = 1, discrete = True).truncated_power_law.plot_pdf(ax = a1, color = 'b', label = 'Trunc. Power Law fit')
    powerlaw.plot_pdf(times[0], ax = a1, color = 'b', linestyle = 'None', marker = 'o', alpha = 0.75, label = 'Empirical PDF')
    alpha_0 = powerlaw.Fit(times[0], xmin = 1, discrete = True).alpha
    
    a1.set_xticks(xticks)
    a1.set_xticklabels(xticklabels)
    a1.set_xlabel('Minutes')
    
    a1.legend(loc = 1, prop = {'size': 12})
    a1.axis([10**0, 10**3, 10**(-5), 10**0])
    a1.set_title('Density of Active Bursts, ' + 'alpha=' + str(alpha_0)[0:4])
    
    # histogram of sedentary times
    powerlaw.Fit(times[1], xmin = 1, discrete = True).power_law.plot_pdf(ax = a2, color = 'g', linewidth = 2, label = 'Power Law fit')
    powerlaw.Fit(times[1], xmin = 1, discrete = True).lognormal.plot_pdf(ax = a2, color = 'r', linewidth = 2, label = 'Lognormal fit')
    #powerlaw.Fit(times[1], xmin = 1, discrete = True).exponential.plot_pdf(ax = a2, color = 'r', label = 'Exponential fit')
    #powerlaw.Fit(times[1], xmin = 1, discrete = True).stretched_exponential.plot_pdf(ax = a2, color = 'b', label = 'Stretched Expo fit')
    #powerlaw.Fit(times[1], xmin = 1, discrete = True).truncated_power_law.plot_pdf(ax = a2, color = 'b', label = 'Trunc. Power Law fit')
    powerlaw.plot_pdf(times[1], ax = a2, color = 'b', linestyle = 'None', marker = 'o', alpha = 0.75, label = 'Empirical PDF')
    alpha_1 = powerlaw.Fit(times[1], xmin = 1, discrete = True).alpha

    a2.set_xticks(xticks)
    a2.set_xticklabels(xticklabels)
    a2.set_xlabel('Minutes')

    a2.legend(loc = 1, prop = {'size': 12})
    a2.axis([10**0, 10**3, 10**(-5), 10**0])
    a2.set_title('Density of Sedentary Times, ' + 'alpha=' + str(alpha_1)[0:4])

    # plot autocorrelations of burst and ie time series
    autocorr_0 = acf(times[0], nlags = len(times[0])/2, alpha = .05)
    autocorr_1 = acf(times[1], nlags = len(times[1])/2, alpha = .05)

    f2, (b0, b1) = plt.subplots(1, 2, figsize = (18, 8))
    #plt.rc('text', usetex = True)
    #plt.rc('font', family = 'serif')
    plt.suptitle(title)
        
    b0.plot(range(  len(times[0])/2 + 1  ), (autocorr_0[0]), color = 'b', label = 'Est. autocorrelation')
    b0.plot(range(  len(times[0])/2 + 1  ), (autocorr_0[1][:, 0]), linestyle = ':', color = 'g', label = '95% Conf. interval')
    b0.plot(range(  len(times[0])/2 + 1  ), (autocorr_0[1][:, 1]), linestyle = ':', color = 'g')
    b0.axhline(y = 0, color = '0.75')
    b0.axvline(x = 4, color = 'r', alpha = 0.5, label = '4 Events')
    b0.set_xlabel('Lag (Bursts)')
    b0.legend(loc = 1)
    b0.axis([0, len(times[0])/2, -0.5, 1])
    b0.set_title('Autocorrelation of Active Bursts')
    
    b1.plot(range(  len(times[1])/2 + 1  ), (autocorr_1[0]), color = 'b', label = 'Est. autocorrelation')
    b1.plot(range(  len(times[1])/2 + 1  ), (autocorr_1[1][:, 0]), linestyle = ':', color = 'g', label = '95% Conf. interval')
    b1.plot(range(  len(times[1])/2 + 1  ), (autocorr_1[1][:, 1]), linestyle = ':', color = 'g')
    b1.axhline(y = 0, color = '0.75')
    b1.axvline(x = 4, color = 'r', alpha = 0.5, label = '4 Events')
    b1.set_xlabel('Lag (Sedentary Times)')
    b1.legend(loc = 1)
    b1.axis([0, len(times[1])/2, -0.5, 1])
    b1.set_title('Autocorrelation of Sedentary Times')
    
    if path_prefix !=None:
        f1.savefig(path_prefix + '_activity.png', bbox_inches='tight')
        f2.savefig(path_prefix + '_autocorr.png', bbox_inches='tight')
        plt.close('all')


####################################
# Get activity date,
# plot summaries, 
# and write results to file.
####################################

def analyze_activity(directory, plot = True):

    output = pd.read_csv(directory + 'output_summary.csv')
    update = pd.DataFrame(index = np.arange(len(output)), columns = output.columns)
    
    activity_dir = directory + 'activity/'
    if not os.path.exists(activity_dir):
        os.makedirs(activity_dir)

    for i in range(len(output)):
        idd, date, t0, t1 = list(output.iloc[i])[0:4]
        fix_date = date.replace('/', '-')
        epoch_file = directory + 'proc_data/' + idd + '/' + idd + '_epochs_' + fix_date + '_' + t0 + '-' + t1 + '.csv'
        activity = pd.read_csv(epoch_file)['activity']
        times = get_times(activity)

        # save record of activity        
        temp = {'active_bursts': times[0], 'sedent_periods': times[1]}
        write_to = activity_dir + idd + '_' + fix_date + '_activity.csv'
        pd.DataFrame.from_dict(temp).to_csv(write_to, sep = ',', header = True, index = False)

        # drop leading 'None' from times[0] if necessary
        times[0] = [t for t in times[0] if t != None]        

        # drop trailing 'None' from times[1] if necessary
        times[1] = [t for t in times[1] if t != None]        

        # generate plots
        if plot:        
            title = 'Activity for ' + idd + ' on ' + date
            destination_prefix = activity_dir + idd + '_' + fix_date
            activity_plots(activity, times, title, destination_prefix)
        
        # update output summary file        
        median_act = np.median(times[0])
        alpha_act  = powerlaw.Fit(times[0], xmin = 1, discrete = True).alpha
        median_sed = np.median(times[1])
        alpha_sed  = powerlaw.Fit(times[1], xmin = 1, discrete = True).alpha

        update.iloc[i] = list(output.iloc[i])[0:9] + [median_act, alpha_act, median_sed, alpha_sed]

    update.to_csv(directory + 'output_summary.csv', sep = ',', header = True, index = False)
    print 'Activity analysis for', len(output), 'records is finished.'
    
    
####################################
# Process data
####################################

def proc(data_dir = None):
    
    t0 = time.time()
    
    # unzip data, assume each participant's data is in a single zip
    unzip(data_dir)
    
    # get researcher records
    records = get_records(data_dir)
    
    # Process data into days, write signal magnitudes to file    
    for r in records:
        process_data(r, data_dir)
    
    # Find cutoffs for each participant
    cutoffs = get_cutoffs(records, data_dir)
    
    # Append activity classification to sigmag files    
    activity_class(cutoffs, data_dir)
    
    # Generate plots and write summary file
    summarize(records, cutoffs, data_dir, plot = False)
    
    # Get burst lengths and sedentary
    # times, estimate paramterers,
    # write everything to output file.
    analyze_activity(data_dir, plot = False)
    
    # Time results    
    t = time.time() - t0 # seconds
    
    if not os.path.exists(data_dir + 'time.txt'):
        os.mknod(data_dir + 'time.txt')
    
    temp = open(data_dir + 'time.txt', 'a')
    temp.write(' ' + str(t) + ',')
    temp.close()    




####################################
#
#
####################################


####################################
#
#
####################################


####################################
#
#
####################################
        