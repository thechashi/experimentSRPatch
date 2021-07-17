"""
Experiment on SR models
"""
import subprocess

def main():
    """
    Main function

    Returns
    -------
    None.

    """
    
    subprocess.run("gpustat", shell=True)

    # binary search
    print("\n\nStarting binary search...\n")
    subprocess.run("python3 binarysearch.py", shell=True, check=True)
    print("Binary search done...")
    
    subprocess.run("gpustat", shell=True)

    # linear search
    print("\n\nStarting linear search...\n")
    subprocess.run("python3 linearsearch.py", shell=True)
    print("Linear search done...")

    subprocess.run("gpustat", shell=True)

if __name__ == "__main__":
    main()
