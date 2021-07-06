import subprocess
import toml
if __name__ == "__main__":
    loader_config = toml.load('../loader_config.toml')
    if loader_config["batch_range_experiment"]["status"] == "ongoing":
        subprocess.run("gpustat", shell=True)
        subprocess.run("time python3 batch_forward_chop_experiment.py batch_range", shell= True)
    else:
        subprocess.run("gpustat", shell=True)
        p = subprocess.run("time python3 binarysearch.py", shell=True)
        print(p.stdout.decode())
        subprocess.run("gpustat", shell=True)
        p = subprocess.run("time python3 batch_forward_chop_experiment.py batch_range", shell= True)
        print(p.stdout.decode())