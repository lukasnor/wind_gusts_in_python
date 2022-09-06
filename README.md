# Read Me

## Description

This project is a Python implementation of a variety of network types found in the paper _"Machine learning methods for
postprocessing ensemble forecasts of wind gusts: A systematic comparison" by B. Schulz and S. Lerch. The use case for
these networks is postprocessing of ensemble forecasts or generating probabilistic forecasts from ensemble data.

## Structure of  the project

There are two main directories:

1. the source code `src`
2. the results `results`

The `src` directory consists of both standalone Python modules, helper modules used in multiple other modules and dummy
test modules of no particular interest.

### `src`

#### Helpler modules

1. `preprocessing.py` - this module contains many functions helpful for converting the "raw" .csv files into pandas
   DataFrames of the correct format, scaling variables, categorizing them, etc. The exact functionality is best
   understood by looking into standalone Python modules such as `bqn.py` and looking at the data structure of the
   DataFrames
2. `tuner_helper.py` - this module contains some plotting functionality, useful to analyze the multiple runs of a model
   type, and is used in the tuner_analysis modules

#### Standalone modules
1. `bqn.py`, `bqn_tuner.py`, `bqn_tuner_analysis.py` - implementation of the Bernstein Quantile Network, its
   hyperparameter tuning and the analysis of it
2. `data_exploration.py` - a small script to generate a plot of the wind_power data, used in the HackMD
3. `drn.py` - a first implementation of the Distributional Regression Network, not furthered into a big analysis because
   it seamed to be too inflexible, as can be seen in the forecast plots it generates
4. `emos.py`, `EmosModel.py` - emos.py contains the first and failed attempt to work with the pywatts framework
5. `hen.py`, `hen_tuner.py`, `hen_tuner_analysis.py` - implementation of the Histogram Estimation Network, its
   hyperparameter tuning and the analysis of it
6. `scoringRules.R` - an import file, needed for EMOS

#### Dummy modules
These files only helped the author to gain an understanding of what he was doing.
- `crps_test.py`
- `HyperParameterTest.py`, `HyperParameterTest2.py`
- `mean_model.py`
- `swedenExploration.ipynb`, `windDataExploration.ipynb`
- `xarray_tests`

### `results`
This folder is pretty self-explainatory