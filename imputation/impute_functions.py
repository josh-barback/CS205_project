
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import powerlaw

#from statsmodels.tsa.stattools import acf

import impute_helpers_parallel as helpers_parallel

    
#################################################
# Read epochs file for a length of iPhone data
#################################################

def read_data(filename):
    
    data = pd.read_csv(filename)
    activity = np.asarray(data['activity'])
    start    = np.asarray(data['start'])
    end      = np.asarray(data['end'])
    
    return activity.astype(float), start.astype(long), end.astype(long)


#################################################
# Downsample epochs according to pattern
#################################################

def downsample(activity, epoch_length = 5, on_seconds = 30, off_seconds = 30, run = 'OpenMP'):

    if on_seconds == 0 or off_seconds == 0:
        activity_ds = np.copy(activity)
    
    else:
        if run == 'Python':
            activity_ds = np.copy(activity)
    
            on_epochs = on_seconds / epoch_length
            off_epochs = off_seconds / epoch_length
            
            cycle_length = on_epochs + off_epochs
            n_cycles = len(activity) / cycle_length
            
            cycle_starts = cycle_length * np.arange(n_cycles)            

            for i in cycle_starts:
                activity_ds[i + on_epochs : i + cycle_length] = np.repeat(np.nan, off_epochs)
    
        if run == 'Cython':
            activity_ds = helpers_parallel.downsample_Cython(activity, epoch_length, on_seconds, off_seconds)
            
        if run == 'OpenMP':
            activity_ds = helpers_parallel.downsample_OpenMP(activity, epoch_length, on_seconds, off_seconds)
    
    return activity_ds

    
#################################################
# Extract active bursts and sedentary periods
#################################################

def get_times(activity, start, end, run = 'OpenMP'):

    if run == 'Python':
        
        bursts      = []
        classification = []
        complete    = []
        burst_start = []
        burst_end  = []

        state = None
        complete_state = None

        count = 0
                
        for i in range(len(activity)):
            
            if i == 0:
                if activity[i] == 1:
                    count += 1
                    state = 'Active'
                    complete_state = 'left-cutoff'
                    burst_start.append(start[i])
                elif activity[i] == 0:
                    count += 1
                    state = 'Sedentary'
                    complete_state = 'left-cutoff'
                    burst_start.append(start[i])
                elif np.isnan(activity[i]):
                    state = 'Missing'
                    
            if i > 0 and i < len(activity) - 1:
                
                if state == 'Active':
                    if activity[i] == 1:
                        count += 1
                    elif activity[i] == 0:
                        bursts.append(count)
                        classification.append('Active')
                        complete.append(complete_state)
                        burst_end.append(end[i-1])
                        burst_start.append(start[i])
                        count = 1
                        state = 'Sedentary'                        
                        complete_state = 'complete'
                    elif np.isnan(activity[i]):
                        bursts.append(count)
                        classification.append('Active')
                        if complete_state == 'left-cutoff':   
                            complete.append('both-cutoff')
                        elif complete_state == 'complete':
                            complete.append('right-cutoff')
                        burst_end.append(end[i-1])
                        count = 0
                        state = 'Missing'                        
                        complete_state = None
                        
                elif state == 'Sedentary':
                    if activity[i] == 1:
                        bursts.append(count)
                        classification.append('Sedentary')
                        complete.append(complete_state)
                        burst_end.append(end[i-1])
                        burst_start.append(start[i])
                        count = 1
                        state = 'Active'
                        complete_state = 'complete'
                    elif activity[i] == 0:
                        count += 1
                    elif np.isnan(activity[i]):
                        bursts.append(count)
                        classification.append('Sedentary')
                        if complete_state == 'left-cutoff':
                            complete.append('both-cutoff')
                        elif complete_state == 'complete':
                            complete.append('right-cutoff')
                        burst_end.append(end[i-1])
                        count = 0
                        state = 'Missing'
                        complete_state = None
                                                            
                elif state == 'Missing':
                    if activity[i] == 1:
                        count = 1
                        state = 'Active'
                        complete_state = 'left-cutoff'
                        burst_start.append(start[i])
                    elif activity[i] == 0:
                        count = 1
                        state = 'Sedentary'
                        complete_state = 'left-cutoff'                    
                        burst_start.append(start[i])
                        
            if i == len(activity) - 1:
                    
                if state == 'Active':
                    if activity[i] == 1:
                        bursts.append(count + 1)
                        classification.append('Active')
                        if complete_state == 'complete':
                            complete.append('right-cutoff')
                        if complete_state == 'left-cutoff':
                            complete.append('both-cutoff')
                        burst_end.append(end[i])
                    elif activity[i] == 0:
                        bursts.append(count)
                        classification.append('Active')
                        complete.append(complete_state)
                        burst_end.append(end[i-1])
                        bursts.append(1)
                        classification.append('Sedentary')
                        complete.append('right-cutoff')
                        burst_start.append(start[i])
                        burst_end.append(end[i])
                    elif np.isnan(activity[i]):
                        bursts.append(count)
                        classification.append('Active')
                        if complete_state == 'complete':
                            complete.append('right-cutoff')
                        if complete_state == 'left-cutoff':
                            complete.append('both-cutoff')
                        burst_end.append(end[i-1])
                                                                
                elif state == 'Sedentary':
                    if activity[i] == 1:
                        bursts.append(count)
                        classification.append('Sedentary')
                        complete.append(complete_state)
                        burst_end.append(end[i-1])
                        bursts.append(1)
                        classification.append('Active')
                        complete.append('right-cutoff')
                        burst_start.append(start[i])
                        burst_end.append(end[i])
                    elif activity[i] == 0:
                        bursts.append(count + 1)
                        classification.append('Sedentary')
                        if complete_state == 'complete':
                            complete.append('right-cutoff')
                        if complete_state == 'left-cutoff':
                            complete.append('both-cutoff')
                        burst_end.append(end[i])
                    elif np.isnan(activity[i]):
                        bursts.append(count)
                        classification.append('Sedentary')
                        if complete_state == 'complete':
                            complete.append('right-cutoff')
                        if complete_state == 'left-cutoff':
                            complete.append('both-cutoff')
                        burst_end.append(end[i-1])
  
                elif state == 'Missing':
                    if activity[i] == 1:
                        bursts.append(1)
                        classification.append('Active')
                        complete.append('both-cutoff')
                        burst_start.append(start[i])
                        burst_end.append(end[i])
                    elif activity[i] == 0:
                        bursts.append(1)
                        classification.append('Sedentary')
                        complete.append('both-cutoff')
                        burst_start.append(start[i])
                        burst_end.append(end[i])

        d = {'bursts'  : bursts, 
             'class'   : classification, 
             'complete': complete, 
             'start'   : burst_start, 
             'end'     : burst_end}
    
        result = pd.DataFrame.from_dict(d)

        
    else:
        result = helpers_parallel.get_times_Cython(activity, start, end)
        
    return result

    
#################################################
# Get samples from active and sedentary times
#################################################

def get_samples(times, complete = True):
    
    if complete:
        active    = times.loc[(times['complete'] == 'complete') & (times['class'] == 'Active')]['bursts']
        sedentary = times.loc[(times['complete'] == 'complete') & (times['class'] == 'Sedentary')]['bursts']        
        
    if not complete:
        active    = times.loc[times['class'] == 'Active']['bursts']
        sedentary = times.loc[times['class'] == 'Sedentary']['bursts']

    return [active, sedentary]
        

#################################################
# Estimate parameters for distributions of active and sedentary times
#################################################

def get_estimates(times, complete = True):

    samples = get_samples(times, complete)

    estimates = [0] * 8
     
    # Active bursts
    estimates[0] = len(samples[0])
    estimates[1] = powerlaw.Fit(samples[0], xmin = 1, discrete = True).alpha
    estimates[2] = powerlaw.Fit(samples[0], xmin = 1, discrete = True).lognormal_positive.mu
    estimates[3] = powerlaw.Fit(samples[0], xmin = 1, discrete = True).lognormal_positive.sigma

    # Sedentary periods 
    estimates[4] = len(samples[1])
    estimates[5] = powerlaw.Fit(samples[1], xmin = 1, discrete = True).alpha
    estimates[6] = powerlaw.Fit(samples[1], xmin = 1, discrete = True).lognormal_positive.mu
    estimates[7] = powerlaw.Fit(samples[1], xmin = 1, discrete = True).lognormal_positive.sigma
    
    return estimates    
    
    
#################################################
# Setup output file,
# Append row to output file,
# Add column to output file.
#################################################

def update_output(filename, action, update):

    if action == 'create':
        temp = pd.DataFrame(columns = update)
        temp.to_csv(filename, sep = ',', header = True, index = False)
        
    if action == 'row':
        temp = open(filename, 'a')
        temp.write(update + '\n')
        temp.close() 
        
    if action == 'column':    
        pass
    

#################################################
# Analyze estimates from downsampled data
#################################################

def get_downsample_plot(filename, generate = 'partial', title = None):

    ds = pd.read_csv(filename)
    cols = ds.columns

    if generate == 'partial':
        plot_cols = cols[9:17]
        tag = 'all periods'        
    if generate == 'complete':
        plot_cols = cols[1:9]
        tag = 'only complete periods'

    cycle = ds['downsample_cycle']
    x = range(len(cycle))

    idd  = filename.split('/')[-1].split('_')[1]
    date = filename.split('/')[-1].split('_')[3]
    
    titles = ['n', 'powerlaw alpha', 'lognorm. mu', 'lognorm. sigma']
    rows = ['Active Bursts', 'Sedentary Times']
    xticks = [0, 5, 10]
    xlabels = ['15s', '2m', 'None']

    plt.close('all')    
    f, ax = plt.subplots(2, 4, figsize = (22, 10))
    
    for i in range(2):        
        if i == 0: color = 'r'
        else: color = 'b'

        for j in range(4):            
            y = ds[plot_cols[4*i + j]]
            ax[i, j].plot(x, y, 'o', color = color, zorder = 1)
            ax[i, j].axhline(y = list(y)[-1], color = '0.5')
            ax[i, j].axis([-1, len(x), min(y) * 0.8, max(y) * 1.15])        
            if i == 0:
                ax[i, j].set_title(titles[j])
            ax[i, j].set_xticks(xticks)
            ax[i, j].set_xticklabels(xlabels)
            if j == 0:
                ax[i, j].set_ylabel(rows[i])
    ax[1, 0].set_xlabel('On-Off Cycle')
    plt.suptitle('Distribution estimates for ' + idd + ' on ' + date + ' using ' + tag)
    
    path = filename.split('downsample')[0]
    name = '_' + filename.split('/')[-1][0:-4] + '.png'
    f.savefig(path + tag + name, bbox_inches = 'tight')
    plt.close('all')    
                

#################################################
# Fit imputation model
#################################################

def fit(activity_ds, on_seconds, epoch_length = 5, depth = 0.5, run = 'OpenMP'):

    if run == 'Python':
        
        on_epochs = on_seconds / epoch_length
        n_predictors = int(np.floor(on_epochs * depth))
        n_epochs = len(activity_ds)

        n_0 = len([i for i in activity_ds if i == 0])        
        n_1 = len([i for i in activity_ds if i == 1])
        prop_ones = (n_1 + 0.0) / (n_0 + n_1)
        
        # stack data
        data = None        
        
        for i in range(n_predictors + 1):
            temp = list(activity_ds[i : n_epochs])
            end = len(temp) - len(temp)%(n_predictors + 1)
            d = np.array(temp[0:end]).reshape((end/(n_predictors + 1), n_predictors + 1))       
            if i == 0:
                data = d
            else:
                data = np.vstack( (data, d) )

        # drop rows with missing data
        find = np.repeat(1, len(data))
        
        for i in range(len(data)):            
            if any(list(np.isnan(data[i]))):
                find[i] = 0
        keep = [i for i in range(len(data)) if find[i] == 1]
        data = data[keep, :]

        # build dictionary        
        keys = range(int('1'*n_predictors, 2) + 1)
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
                zeros = temp[0]
                ones  = temp[1]
                model[k] = (ones + 0.0) / (ones + zeros)
            elif 0 in temp and not 1 in temp:
                model[k] = 0
            elif 1 in temp and not 0 in temp:
                model[k] = 1
            else:
                model[k] = prop_ones

    else:
        model = helpers_parallel.fit_Cython(activity_ds, on_seconds, epoch_length, depth)
        
    return model  
    
#################################################
# Run imputation model on downsampled data set
#################################################

def impute(activity_ds, model, on_seconds, epoch_length = 5, run = 'OpenMP'):
    
    if run == 'Python':

        imputed = np.copy(activity_ds)        
        depth = int(np.log2(len(model)))
        on_epochs = on_seconds / epoch_length
        cycle_length = on_epochs * 2
        starts = np.arange(on_epochs, len(activity_ds), cycle_length)
        
        for t in starts:            
            for i in range(on_epochs):
                if t + i < len(imputed):                 
                    bin_key = str(list(imputed[t + i - depth : t + i])).replace('.0', '').replace('[', '').replace(']', '').replace('.', '').replace(' ', '').replace(',', '')
                    key = int(bin_key, 2)
                    imputed[t + i] = np.random.binomial(1, model[key])
                
    else:
        imputed = helpers_parallel.impute_Cython(activity_ds, model, on_seconds, epoch_length)
                
    return imputed

    
#################################################
# Plot results from multiple imputation
#################################################

def plot_imputation(filename):
    
    results = pd.read_csv(filename)
    
    cycle = filename.split('/')[-1].split('_')[1]
    idd   = filename.split('/')[-1].split('_')[2]
    date  = filename.split('/')[-1].split('_')[4]
    
    cols = results.columns
    true = results.iloc[0]
    ds   = results.iloc[1]
    
    titles = ['n', 'powerlaw alpha', 'lognorm. mu', 'lognorm. sigma']
    rows = ['Active Bursts', 'Sedentary Times']

    plt.close('all')    
    f, ax = plt.subplots(2, 4, figsize = (22, 10))
    
    for i in range(2):        
        if i == 0: color = 'r'
        else: color = 'b'

        for j in range(4):            
            sample = results[cols[1 + 4*i + j]][2:-1]
            ax[i, j].hist(sample, bins = 15, orientation = 'horizontal', color = color, zorder = 1)            
            ax[i, j].axhline(y = true[1 + 4*i + j], color = 'k', linewidth = 2, zorder = 2)
            ax[i, j].axhline(y =   ds[1 + 4*i + j], color = 'k', linestyle = 'dashed', linewidth = 2, zorder = 3)
            if i == 0:
                ax[i, j].set_title(titles[j])
            ax[i, j].set_xticks([])
            if j == 0:
                ax[i, j].set_ylabel(rows[i])

    plt.suptitle('Multiple imputation estimates for ' + idd + ' on ' + date + ' with ' + cycle + ' second on/off cycle')
    
    path = filename.split('imputations')[0]
    name = 'plot_' + filename.split('/')[-1][0:-4] + '.png'
    f.savefig(path + name, bbox_inches = 'tight')
    plt.close('all')    
  

#################################################
# Generate plots for all imputation results
#################################################

def get_all_plots(output_dir):

    # do visualization of downsampled estimates
    ds_files = [f for f in os.listdir(output_dir) if f.find('downsample') > -1]
    for f in ds_files:
        get_downsample_plot(output_dir + f, generate = 'complete')
        get_downsample_plot(output_dir + f, generate = 'partial')

    # generate plots of imputation results
    impute_results = [f for f in os.listdir(output_dir) if f.find('imputations') > -1]
    for f in impute_results:
        plot_imputation(output_dir + f)


#################################################
# Run imputation on a directory of files
#################################################

def proc(data_dir, output_dir, seconds, reps = 1000, run = 'OpenMP'):
    
    # start timer
    t0 = time.time()
    
    # set up list of data files
    filelist = [f for f in os.listdir(data_dir) if f.find('epochs') > -1]
        
    # collect estimates from downsampled files
    for f in filelist:
        outfile = output_dir + 'downsample_' + f
        header  = ['downsample_cycle',
                   'complete_active_n','complete_active_alpha','complete_active_mu','complete_active_sigma',
                   'complete_sedent_n','complete_sedent_alpha','complete_sedent_mu','complete_sedent_sigma',
                   'partial_active_n','partial_active_alpha','partial_active_mu','partial_active_sigma',
                   'partial_sedent_n','partial_sedent_alpha','partial_sedent_mu','partial_sedent_sigma']
        update_output(filename = outfile, action = 'create', update = header)
        
        for s in seconds:
            activity, start, end = read_data(data_dir + f)
            activity_ds = downsample(activity, epoch_length = 5, 
                                            on_seconds = s, off_seconds = s, 
                                            run = run)
            
            times = get_times(activity_ds, start, end, run = run)
            
            estimates_complete = get_estimates(times, complete = True)
            estimates_partial  = get_estimates(times, complete = False)
            
            # save to file
            summary = str([s] + estimates_complete + estimates_partial).replace('[', '').replace(']', '').replace(' ', '')
            update_output(filename = outfile, action = 'row', update = summary)
        
        print 'Downsample summary complete for', f
    
    # do imputation
    for f in filelist:
        
        for s in seconds:
            # read data and initialize csv
            activity, start, end = read_data(data_dir + f)
            
            outfile = output_dir + 'imputations_' + str(s) + '_' + f
            header  = ['imputation',
                       'active_n','active_alpha','active_mu','active_sigma',
                       'sedent_n','sedent_alpha','sedent_mu','sedent_sigma']
            update_output(filename = outfile, action = 'create', update = header)
    
            times = get_times(activity, start, end, run = run)
            estimates = str([0] + get_estimates(times, complete = True)).replace('[', '').replace(']', '').replace(' ', '')
            update_output(filename = outfile, action = 'row', update = estimates)
            
            # downsample and record estimates 
            activity_ds = downsample(activity, epoch_length = 5, 
                                    on_seconds = s, off_seconds = s, 
                                    run = run)
            times = get_times(activity_ds, start, end, run = run)
            estimates = str([0] + get_estimates(times, complete = True)).replace('[', '').replace(']', '').replace(' ', '')
            update_output(filename = outfile, action = 'row', update = estimates)
            
            # fit imputation model
            model = fit(activity_ds, on_seconds = s, epoch_length = 5, depth = 0.5, run = run)
            print 'Imputation model fit for', f, 'with cycle of', s, 'seconds.'
                    
            # run the model multiple times and record results
            for r in range(reps):
                imputed = impute(activity_ds, model, on_seconds = s, epoch_length = 5, run = run)    
                times = get_times(imputed, start, end, run = run)
                estimates = str([r + 1] + get_estimates(times, complete = True)).replace('[', '').replace(']', '').replace(' ', '')
                update_output(filename = outfile, action = 'row', update = estimates)
            print 'Completed', reps, 'imputations for', f, 'with cycle of', s, 'seconds.'
                
    # record runtime
    t = time.time() - t0 # seconds
    
    if not os.path.exists(data_dir + 'time.txt'):
        os.mknod(data_dir + 'time.txt')
    
    temp = open(data_dir + 'time.txt', 'a')
    temp.write(' ' + str(t) + ',')
    temp.close()    


#################################################
# Run imputation on a directory of files
# Use MPI functionality
#################################################

def proc_mpi(data_dir, output_dir, seconds, reps = 1000, run = 'OpenMP'):
    
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # start timer
    if rank == 0:
        t0 = time.time()
    
    # set up list of data files
    if rank == 0:
        filelist = [f for f in os.listdir(data_dir) if f.find('epochs') > -1]
    else: 
        filelist = None

    filelist = comm.bcast(filelist, root = 0)                               
    
    comm.Barrier()             

    # collect estimates from downsampled files
    #for f in filelist:
    for i in range(len(filelist)):
        if i%size == rank:
            f = filelist[i]
            outfile = output_dir + 'downsample_' + f
            header  = ['downsample_cycle',
                       'complete_active_n','complete_active_alpha','complete_active_mu','complete_active_sigma',
                       'complete_sedent_n','complete_sedent_alpha','complete_sedent_mu','complete_sedent_sigma',
                       'partial_active_n','partial_active_alpha','partial_active_mu','partial_active_sigma',
                       'partial_sedent_n','partial_sedent_alpha','partial_sedent_mu','partial_sedent_sigma']
            update_output(filename = outfile, action = 'create', update = header)
            
            for s in seconds:
                activity, start, end = read_data(data_dir + f)
                activity_ds = downsample(activity, epoch_length = 5, 
                                                on_seconds = s, off_seconds = s, 
                                                run = run)
                
                times = get_times(activity_ds, start, end, run = run)
                
                estimates_complete = get_estimates(times, complete = True)
                estimates_partial  = get_estimates(times, complete = False)
                
                # save to file
                summary = str([s] + estimates_complete + estimates_partial).replace('[', '').replace(']', '').replace(' ', '')
                update_output(filename = outfile, action = 'row', update = summary)
            
            print 'Downsample summary complete for', f

    comm.Barrier()
            
    # do imputation
    #for f in filelist:
    for i in range(len(filelist)):
        if i%size == rank:
            f = filelist[i]
        
            for s in seconds:
                # read data and initialize csv
                activity, start, end = read_data(data_dir + f)
                
                outfile = output_dir + 'imputations_' + str(s) + '_' + f
                header  = ['imputation',
                           'active_n','active_alpha','active_mu','active_sigma',
                           'sedent_n','sedent_alpha','sedent_mu','sedent_sigma']
                update_output(filename = outfile, action = 'create', update = header)
        
                times = get_times(activity, start, end, run = run)
                estimates = str([0] + get_estimates(times, complete = True)).replace('[', '').replace(']', '').replace(' ', '')
                update_output(filename = outfile, action = 'row', update = estimates)
                
                # downsample and record estimates 
                activity_ds = downsample(activity, epoch_length = 5, 
                                        on_seconds = s, off_seconds = s, 
                                        run = run)
                times = get_times(activity_ds, start, end, run = run)
                estimates = str([0] + get_estimates(times, complete = True)).replace('[', '').replace(']', '').replace(' ', '')
                update_output(filename = outfile, action = 'row', update = estimates)
                
                # fit imputation model
                model = fit(activity_ds, on_seconds = s, epoch_length = 5, depth = 0.5, run = run)
                print 'Imputation model fit for', f, 'with cycle of', s, 'seconds.'
                        
                # run the model multiple times and record results
                for r in range(reps):
                    imputed = impute(activity_ds, model, on_seconds = s, epoch_length = 5, run = run)    
                    times = get_times(imputed, start, end, run = run)
                    estimates = str([r + 1] + get_estimates(times, complete = True)).replace('[', '').replace(']', '').replace(' ', '')
                    update_output(filename = outfile, action = 'row', update = estimates)
                print 'Completed', reps, 'imputations for', f, 'with cycle of', s, 'seconds.'

    comm.Barrier()
                
    # record runtime
    if rank == 0:
        t = time.time() - t0 # seconds
        
        if not os.path.exists(data_directory + 'time.txt'):
            os.mknod(data_directory + 'time.txt')
        
        temp = open(data_directory + 'time.txt', 'a')
        temp.write(' ' + str(t) + ',')
        temp.close()    
    












