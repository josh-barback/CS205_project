
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# suppress RutimeWarnings:
import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

# process data
import os

#func_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/impute_code/'
#data_dir = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/impute_data/proc_data/u19el13q/'
#out_dir  = '/home/josh/Desktop/Dropbox/_research 2017/work/CS_205 project/impute_data/'

func_dir = '/n/home07/cs205u1703/CS_205/Project/imputation/impute_code/'
data_dir = '/n/home07/cs205u1703/CS_205/Project/imputation/impute_data/proc_data/'
out_dir  = '/n/home07/cs205u1703/CS_205/Project/imputation/impute_data/' 

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
#seconds = [30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600]
#seconds = [30, 45, 60, 75, 90, 105, 120]
seconds = [120, 105, 90, 75, 60, 45, 30]
#reps = 1000
reps = 500
#run = 'Python'
#run = 'Cython'

comm.Barrier()

# run downsample and imputation functions
#impute.proc(data_dir, output_dir, seconds = seconds, reps = reps, run = run)
impute.proc_mpi(data_dir, output_dir, seconds = seconds, reps = reps, run = 'Cython')

        






