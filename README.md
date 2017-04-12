
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

2.  Optimize time-consuming calculations with parallel computation (e.g. finding quantiles of long lists of numbers),

3.  Implement concurrent processing of data from multiple days and/or multiple participants,

4.  Efficiently implement imputation routines by concurrently processing separate regions of missing data.

Regarding item 4, it is anticipated that efficient imputation will allow rapid evaluation of several imputation methods, with the goal of selecting an optimal method for future use.






