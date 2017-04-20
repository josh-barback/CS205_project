
# Processing and Imputation for Smartphone Accelerometer Data

## Background

At the Chan School's Onnela Lab, the *digital phenotyping* project is a framework for leveraging data from digital devices to obtain *moment-by-moment quantification of the individual-level human phenotype.*

Accelerometers have been used in the study of human activity for several decades, and these sensors have become increasingly ubiquitous in modern electronic devices.  Today, virtually every smartphone and tablet is equipped with a piezo-electric triaxial accelerometer.

In the context of digital phenotyping, smartphone accelerometer data can not only offer insight into individual physical activity levels, but can also inform analyses of other passively-collected data (e.g. GPS-based measures of mobility) and self-reported information (e.g. daily survey responses).


## Challenging Features of Smartphone Accelerometer Data

Smartphone accelerometry presents some special challenges due to the size of the data sets, and also due to specific missing data issues.

For example, a participant with an iPhone may generate more than half a million individual accelerometer observations over the course of a day, each observation consisting of a UTC timestamp and one measurement for each axis of the accelerometer.  Current studies involve observation of dozens of participants for multiple weeks; in the near future, studies with hundreds of participants are anticipated.

In addition, typical accelerometer data collection incorporates cyclic periods during which observation is suspended in order to conserve battery.  Android phones are subject to additional missing data arising from highly variable sampling rates.

Therefore, even basic research tasks (such as assessment of data quality and data exploration) can require time-consuming calculations.  And incomplete data hinders more sophisticated analyses of accelerometer data, such as estimation of parameters associated with active and sedentary time periods.


## Project Goals

The goals of the present project are to:

1. Assemble optimized code for processing of accelerometer data.

2. Develop an efficient framework for implementing and evaluating imputation strategies.

It is assumed that the resulting code may be run in a cluster or cloud computing environment (for bulk processing of raw data), or in a desktop environment (for convenient data exploration and quality monitoring).  Therefore, several optimization strategies will be pursued:

1.  Convert Python code for data processing into C extensions using Cython,

2.  Optimize time-consuming calculations with OpenMP,

3.  Implement concurrent processing of data from multiple days and/or multiple participants using MPI functionality.

Lastly, efficient imputation routines will be implemented with hybrid parallel programming.  It is anticipated that this will allow rapid evaluation of several imputation methods, with the goal of selecting an optimal method for future use.


## Stage 1:  Hybrid Code for Accelerometer Data Processing

The Onnela Lab's Beiwe software collects smartphone sensor data and delivers it in the form of zipped csv files.  A participant using an iPhone will generate 40-80 MB of uncompressed accelerometer data per day, sampled at a rate of about 9 Hz.  The raw data take the form of three time series (one for each accelerometer axis), divided into hours.  Before extracting useful information about active and sedentary periods, the researcher must transform the data in the following way.

First, raw hourly data is collected into periods of interest identified by the researcher.  The three time series are then collapsed to a single time series by calculating a *signal magnitude* for each observation; this is simply the length of the observed 3-dimensional acceleration vector.  The signal magnitudes are then compared to the expected resting accelerometer signal magnitude of 1 g; absolute deviations from resting acceleration are averaged over epochs of five seconds.  At this point, all available data from a given participant are examined in order to identify an individual cutoff point that distingushes *active* epochs from *sedentary* epochs.  After this classification is applied, *active bursts* and *sedentary periods* are located by tracking behavior across consecutive epochs; finally, summary statistics for the corresponding distributions are generated.

The following plots illustrate several steps in this pipeline, beginning with the signal magnitude time series at top.  The center plot shows mean absolute deviations from resting acceleration, with the individual cutoff represented by a dashed red line (just above the x-axis).  At bottom, green vertical lines represent epochs that have been classified as *active*.

!["summary plots"](https://raw.githubusercontent.com/josh-barback/CS205_project/master/summary_plots.png)

After implementing the accelerometer data processing code in Python, several key segments of code were written as C extensions with Cython.  Further optimzation was accomplished with OpenMP and MPI functionality from the `cython.parallel` module and the `mpi4py` package.

!["runtime plots"](https://raw.githubusercontent.com/josh-barback/CS205_project/master/runtime_plots.png)

Each version of the code was benchmarked on a small data set containing 62 hours of accelerometer observations from two participants.  Computation took place on the Odyssey Cluster's `seas_iacs` partition.  Preliminary versions of the code were run on individual nodes equipped with 32 CPUs; hybrid code was run on four such nodes.  Estimates were obtained by averaging runtimes for five repetitions of the data processing task.  The above plots show estimated processing speeds, in seconds per hour of accelerometer data, as well as speedups relative to the baseline of unaccelerated Python code.

On a single node with 32 processors, the use of C extensions improves runtime by a factor of almost 2.5, and OpenMP functionality yields a speedup factor of over 3.5.  On four contiguous nodes, the hybrid code is over 4.5 times as fast as unaccelerated Python code.  Prior to optimization, each hour of accelerometer data took about 2.5 seconds to process; the optimizations bring this time to under 0.6 seconds.


## Stage 2:  Imputation for Missing Accelerometer Data



