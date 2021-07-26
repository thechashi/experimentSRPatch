import numpy as np
import experimentsrpatch.forward_chop_recursive as fcr
import experimentsrpatch.utilities as ut
import experimentsrpatch.modelloader as md

def test_forward_chop():
    img_path = "../experimentsrpatch/data/test2.jpg"
    model_name = "EDSR"
    patch_dimension = 40
    img = None
    model = None
    print("\nLoading model and image... \n")
    if model_name in ["EDSR"]:
        img = ut.load_image(img_path)
        img = img.unsqueeze(0)
        model = md.load_edsr(device="cuda")
    else:
        img = ut.npz_loader(img_path)
        img = img.unsqueeze(0)
        model = md.load_rrdb(device="cuda")

    # Timers for saving stats
    timer = [0, 0, 0, 0, 0]

    print("Processing...")

    # Shfiting input image to CUDA
    total_time = ut.timer()
    cpu2gpu_time = ut.timer()
    img = img.to("cuda")
    cpu2gpu_time = cpu2gpu_time.toc()

    # Forward chopping and upsampling
    output = fcr.forward_chop(
        img, model, timer=timer, min_size=patch_dimension * patch_dimension
    )

    # Shifting output image to CPU
    gpu2cpu_time = ut.timer()
    output = output.to("cpu")
    gpu2cpu_time = gpu2cpu_time.toc()

    if model_name in ["EDSR"]:
        output = output.int()
    total_time = total_time.toc()

    timer[0] = cpu2gpu_time
    timer[-2] = gpu2cpu_time
    timer[-1] = total_time

    # Saving output
    np.savez("results/recursive_outputx4.npz", output)
    np.save("results/recursive_outputx4", output)

    # Printing processing times
    print("\nCPU 2 GPU time: ", timer[0])
    print("\nUpsampling time: ", timer[1])
    print("\nMerging time: ", timer[2])
    print("\nGPU 2 CPU time", timer[3])
    print("\nTotal execution time: ", timer[4])
    
    print(output.shape)