# experimentSRPatch
This repository contains sourcecode for experimenting different patch dimension for EDSR model.

# How to run EDSR patch experiment:

git clone https://github.com/chashi-mahiul-islam-bd/experimentSRPatch.git

python3 -m pip install poetry

cd experimentSRPatch

poetry install

poetry shell

cd experimentsrpatch

Two ways to run:


Way 1 (manually):

    python3 main.py
    
    python3 different_shave_checker.py # here you can change the config.toml file before running this command
    
    python3 different_patch_checker.py # here you can change the config.toml file before running this command
    
    # config.toml file contains arguments for different patch and shave checker.
Way 2 ( with a script):

    sh demo.sh 
    
# How to run EDSR iterative forward chop: 

Inside experimentsrpatch folder: 

python3 forward_chop.py <file_path> <patch_dimension> <shave_value> 

example: python3 forward_chop.py data/test2.jpg 32 12 # output will be available in the experimentsrpatch folder named result_imagex4.png

# Experimenting EDSR iterative forward chop with patch dimension range: 

Inside experimentsrpatch folder: 

python3 check_forward_chop.py <start_dimension> <end_dimension> <shave> <image_path> <total_run> <device_type>

example: python3 check_forward_chop.py 40 1023 10 data/t2.jpg 5 cuda # plots and stats will be available inside results/forward_chop_experiment folder

# Running batch experiment:
Inside experimentsrpatch folder: 

sh batch_experiment.sh

For increasing the experiment speed we can increase the diff_gap amount inside the batch_processing.toml file. 


