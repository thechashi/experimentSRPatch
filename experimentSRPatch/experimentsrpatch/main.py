"""
Experiment on SR models
"""
import os
import time
import linearsearch as ls
import binarysearch as bs
import modelloader as md
import utilities as ut
import subprocess

def main():
    """
    Main function

    Returns
    -------
    None.

    """
    subprocess.run('gpustat', shell=True)
    
    
    # binary search
    print('\n\nStarting binary search...\n')
    subprocess.run('python3 binarysearch.py', shell=True)
    #subprocess.run('ls -la', shell=True)
    
    print('Binary search done...')
    subprocess.run('gpustat', shell=True)

    # linear search 
    print('\n\nStarting linear search...\n')
    subprocess.run('python3 linearsearch.py', shell=True)
    print('Linear search done...')
    
    subprocess.run('gpustat', shell=True)
    # get detailed result
    


if __name__ == "__main__":
    main()
