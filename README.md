# Experiment on SR model
Experiment effectivity of different patch dimension and batch size in a SR model (EDSR model)

# Table of Contents

* [Features] (#features)
* [Installation] (#installation)
* [Getting Started] (#getting-started)
* [Binary Search] (#binary-search)
* [Linear Search] (#linear-search)
* [Dimension And Shave Checker] (#dimension-and-shave-checker)
* [Iterative Forward Chop] (#iterative-forward-chop)
* [Batch Iterative Forward Chop] (#batch-iterative-forward-chop)
* [Maximum Batch Size For A Given Patch Dimension Range] (#maximum-batch-size-for-a-given-dimension-range)
* [All Possible Batch Details For A Specific Patch Dimension] (#all-possible-batch-details-for-a-specific-patch-dimension)

# Features

* Maximum acceptable patch dimension finder for a given SR model
* Statistics for every patch dimension in given dimension range
* Statistics for different shave size and patch dimension
* High resolution image creator with iterative forward chopping
* High resolution image creator with batch iterative forward chopping
* Maximum batch size finder for a given patch dimension range
* Statistics for all possible batches for a specific patch dimension

# Installation
```bash
git clone https://github.com/chashi-mahiul-islam-bd/experimentSRPatch.git

python3 -m pip install poetry

cd experimentSRPatch
```
poetry install

# Getting Started:

Inside experimentSRPatch folder:

```bash
poetry shell

cd experimentsrpatch
```

Two ways to run:

# Binary Search
Finds out the maximum acceptable patch dimension for a SR model in given system. Saves the maximum dimension size inside a temp_max_dim.txt inside experimentsrpatch folder.

```bash
python3 binary_search.py
```

# Linear Search
Calculates timings for different patch dimensions till the maximum acceptable dimension. Binary search dhould be run first before running this. Plots and statistics can be found inside the results folder.

```bash
python3 linear_search.py
```

# Dimension And Shave Checker
Finds out statistics for different dimension and shave in a given range. Plots and statistics can be found inside the results folder.
```bash
    python3 different_shave_checker.py # here you can change the config.toml file before running this command
    
    python3 different_patch_checker.py # here you can change the config.toml file before running this command
    
    # config.toml file contains arguments for different patch and shave checker.
```

# Iterative Forward Chop

Generates a HR image in iterative forward chopping. 

Inside experimentsrpatch folder: 

```bash
python3 forward_chop.py <file_path> <patch_dimension> <shave_value> 
```
Example: 

```bash
python3 forward_chop.py data/test2.jpg 32 12 
```
    # output will be available in the experimentsrpatch folder named result_imagex4.png

Experimenting EDSR iterative forward chop within a patch dimension range: 

Inside experimentsrpatch folder: 

```bash
python3 check_forward_chop.py <start_dimension> <end_dimension> <shave> <image_path> <total_run> <device_type>
```

Example:
 
```bash
python3 check_forward_chop.py 40 1023 10 data/t2.jpg 5 cuda 
```
    # plots and stats will be available inside results/forward_chop_experiment folder

# Batch Iterative Forward Chop
BATCH ITERATIVE FORWARD CHOP

# Maximum Batch Size For A Given Patch Dimension Range

Inside experimentsrpatch folder: 

```bash
sh batch_experiment.sh
```

For increasing the experiment speed we can increase the diff_gap amount inside the batch_processing.toml file. 

# All Possible Batch Details For A Specific Patch Dimension
ALL POSSIBLE BATCH DETAILS FOR A SPECIFIC PATCH DIMENSION


