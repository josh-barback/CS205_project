
# Processing and Imputation for Smartphone Accelerometer Data

## Background

At the Chan School's Onnela Lab, the *digital phenotyping* project is a framework for leveraging data from digital devices to obtain *moment-by-moment quantification of the individual-level human phenotype.*

Accelerometers have been used in the study of human activity for several decades, and these sensors have become increasingly ubiquitous in modern electronic devices.  Today, virtually every smartphone and tablet is equipped with a piezo-electric triaxial accelerometer.

In the context of digital phenotyping, smartphone accelerometer data can not only offer insight into individual physical activity levels, but can also inform analyses of other passively-collected data (e.g. GPS-based measures of mobility) and self-reported information (e.g. daily survey responses).


## Challenging Features of Smartphone Accelerometer Data

Smartphone accelerometry presents some special challenges due to the size of the corresponding data sets, and also due to specific missing data issues.

For example, a participant with an iPhone may generate more than half a million individual accelerometer observations over the course of a day, each observation consisting of a UTC timestamp and one measurement for each axis of the accelerometer.  Current studies involve observation of dozens of participants for multiple weeks; in the near future, studies with hundreds of participants are anticipated.

In addition, typical accelerometer data collection incorporates cyclic periods during which observation is suspended in order to conserve battery.  Android phones are subject to additional missing data arising from highly variable sampling rates.

Therefore, even basic research tasks (such as assessment of data quality and data exploration) can require time-consuming calculations.  And incomplete data hinders more sophisticated analyses of accelerometer data, such as estimation of parameters associated with active and sedentary time periods.


## Project Goals

The goals of the present project are to:

1. Assemble optimized code for processing of accelerometer data.

2. Develop an efficient framework for implementing and evaluating imputation strategies.

It is assumed that the resulting code may be run in a cloud computing environment (for bulk processing of raw data), or in a desktop environment (for convenient data exploration and quality monitoring).  Therefore, several optimization strategies will be pursued, including:

1.  Convert Python code for data processing into C extensions using Cython,

2.  Optimize time-consuming calculations with OpenMP,

3.  Implement concurrent processing of data from multiple days and/or multiple participants using Spark.

Lastly, efficient imputation routines will be implemented with hybrid parallel programming.  It is anticipated that this will allow rapid evaluation of several imputation methods, with the goal of selecting an optimal method for future use.


## Stage 1

After implementing the accelerometer data processing code in Python, several key segments of code were written as C extensions with Cython.  Further optimzation was accomplished with OpenMP functionality from the cython.parallel module.

!["runtime plots"](https://raw.githubusercontent.com/josh-barback/CS205_project/master/runtime_plots.png)

Each version of the code was benchmarked on a small data set containing 62 hours of accelerometer observations.  Computation took place on the Odyssey Cluster's general partition nodes (each equipped with four AMD Optern 6300-series CPU), and also on a quad-core laptop (equipped with an Intel Core i5-2540M).  Estimates were obtained by averaging runtimes for five repetitions of the data processing task.  The above plots show estimated processing speeds, in seconds per hour of accelerometer data.  

On a single node with four processors, the OpenMP version of the code improves the speed of the Python version by over 60%; loss of speed is evident with eight proccessors, possibly due to communication between nodes.  On a laptop, the OpenMP code completes the processing task in nearly a quarter of the time required by the Python version.  For both computing environments, the bulk of improvement is due to the use of C extensions, with smaller speed gains arising from parallelization.






