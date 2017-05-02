# Running Data Processing and Imputation Code

The hybrid SPMD code hsa been tested on up to five 32-core nodes on Odyssey's `seas_iacs` partition.  The OpenMP and serial versions have been tested on individual nodes with any number of cores (up to 32).  

Plotting functionality does not appear to work on the cluster.  Therefore, `plot = True` should be used only when processing data on a PC.

## Data Processing Code

1.  All openMP and hybrid SPMD code for data processing are found in the corresponding folders.  For either, begin by compiling the `.pyx` file with the corresponding `setup.py` script.  This can be done by running: 

`python setup.py build_ext --inplace`

2.  Test cases are in the folder `data_test`.

3.  The two versions of the code can by run with `acc_process.py` and `acc_hybrid_process.py`.  Before running either script, set the corresponding paths for `func_dir` and `data_dir` on lines 16-17.  Output will be saved to `data_dir`.


## Imputation Code

1.  Compile `impute_helpers_parallel.pyx` by running `setup.py`.  This can be done with the following command: 

`python setup.py build_ext --inplace`

2.  Test cases are in the folder `impute_test`.

3.  Create a folder to collect output, or just use the test case folder.

4.  The serial version of the code can be run with `impute_process.py` and the hybrid version can be run with `impute_process_mpi.py`.  Before running either script, set the corresponding paths for `func_dir`, `data_dir`, and `out_dir`.  These are on lines 9-11 of the sequential script, and 15-17 of the hybrid script.