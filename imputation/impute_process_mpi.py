
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# suppress RutimeWarnings:
import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

# process data
import os

func_dir = '.../imputation/' # set path to the directory that contains impute_functions.py
data_dir = '.../impute_test/'# set path to the directory that contains test cases
out_dir  = '.../imputation/' # for collection of output, set path to any directory

os.chdir(func_dir)
import impute_functions as impute

# set up directory and csvs to collect results
if rank == 0:
    import time
    folder_format = 'impute_output_%m-%d-%Y_%H:%M:%S/'
    output_dir = out_dir + time.strftime(folder_format, time.localtime())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)    

else:
    output_dir = None
    
output_dir = comm.bcast(output_dir, root = 0)
        
# set up parameters
seconds = [30, 45, 60, 75, 90, 105, 120]
reps = 1000

comm.Barrier()

# run downsample and imputation functions
#impute.proc(data_dir, output_dir, seconds = seconds, reps = reps, run = run)
impute.proc_mpi(data_dir, output_dir, seconds = seconds, reps = reps, run = 'Cython')

        






