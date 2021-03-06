import math
import numpy as np
import utilities as ut
import modelloader as md
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from PIL import Image


def forward_chop_iterative(x, model=None, shave=10, min_size=1024):
    dim = round(math.sqrt(min_size))
    b, c, h, w = x.size()
    device = "cuda"
    count = 0
    output = torch.tensor(np.zeros((b, c, h * 4, w * 4)))
    total_time = 0
    new_i_s = 0
    x = x.to(device)
    for i in tqdm(range(0, h, dim - 2 * shave)):
        new_j_s = 0
        new_j_e = 0
        # =============================================================================
        #             subprocess.run("gpustat", shell=True)
        # =============================================================================
        for j in range(0, w, dim - 2 * shave):
            # =============================================================================
            #                 print(i,j)
            #                 subprocess.run("gpustat", shell=True)
            # =============================================================================
            count += 1
            h_s = i
            h_e = min(h, i + dim)
            w_s = j
            w_e = min(w, j + dim)
            lr = x[:, :, h_s:h_e, w_s:w_e]
            # =============================================================================
            #                 print('h: {}x{} w: {}x{}'.format(h_s, h_e, w_s, w_e))
            #                 print('current dim: {}x{}'.format(h_e-h_s,w_e-w_s))
            # =============================================================================
            with torch.no_grad():
                # lr = lr.to(device)
                # =============================================================================
                #                     subprocess.run("gpustat", shell=True)
                # =============================================================================
                start = time.time()
                sr = model(lr)
                end = time.time()
                processing_time = end - start
                total_time += processing_time
            # =============================================================================
            #                     subprocess.run("gpustat", shell=True)
            # =============================================================================
            # sr = sr.detach().numpy()
            n_h = (h_e - h_s) * 4
            n_w = (w_e - w_s) * 4

            # =============================================================================
            #                 print('next_dimension: {}x{}'.format(n_h, n_w))
            # =============================================================================
            n_h_s, n_h_e, n_w_s, n_w_e = 0, 0, 0, 0

            if h_s == 0:
                n_h_s = 0
            else:
                n_h_s = shave * 4
            if h_e == h:
                n_h_e = (h_e - h_s) * 4

            else:
                n_h_e = ((h_e - h_s) - shave) * 4

            new_i_e = new_i_s + n_h_e - n_h_s

            if w_s == 0:
                n_w_s = 0
            else:
                n_w_s = shave * 4
            if w_e == w:
                n_w_e = (w_e - w_s) * 4
            else:
                n_w_e = ((w_e - w_s) - shave) * 4
            new_j_e = new_j_e + n_w_e - n_w_s
            sr_small = sr[:, :, n_h_s:n_h_e, n_w_s:n_w_e]
            sr_small = sr_small.to("cpu")
            output[:, :, new_i_s:new_i_e, new_j_s:new_j_e] = sr_small
            del sr_small
            # =============================================================================
            #                 print('new -> h: {}x{} w: {}x{}'.format(n_h_s, n_h_e, n_w_s, n_w_e))
            #                 print('h-> {}:{}, w-> {}:{}'.format(new_i_s, new_i_e, new_j_s, new_j_e))
            #                 print()
            # =============================================================================
            if w_e == w:
                break
            new_j_s = new_j_e
            ut.clear_cuda(lr, sr)
        # =============================================================================
        #                 subprocess.run("gpustat", shell=True)
        #                 print('-----------------------------------------------------')
        # =============================================================================
        # =============================================================================
        #             print('After:')
        #             subprocess.run("gpustat", shell=True)
        #             print()
        # =============================================================================
        new_i_s = new_i_e
        if h_e == h:
            break
    # =============================================================================
    #         print(count)
    #         print(output.shape)
    # =============================================================================
    print("Patch dimension: {}x{}".format(dim, dim))
    print("Total pacthes: ", count)
    print("Total EDSR Processing time: ", total_time)
    return output


if __name__ == "__main__":

    img_path = sys.argv[1] if len(sys.argv) > 1 else "test2.jpg"
    dimension = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    shave = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    device = "cuda"
    img = torchvision.io.read_image(img_path)

    c, h, w = img.shape
    img = img.reshape((1, c, h, w))
    plt.imshow(img[0].permute((1, 2, 0)))
    input_image = img.float()

    model = md.load_edsr(device=device)
    model.eval()
    st = time.time()
    out = forward_chop_iterative(
        input_image, shave=shave, min_size=dimension * dimension, model=model
    )
    et = time.time()
    tt = et - st
    print("Total forward chopping time: ", tt)
    out = out.int()
    print(out.shape)
    # =============================================================================
    #     from PIL import Image
    #     im = Image.fromarray(np.array(out[0].permute(1,2,0)))
    #     im.save("4xinput.jpg")
    # =============================================================================
    b, c, h, w = out.size()
    output = out[0].reshape(w, h, c).numpy()
    print(output.shape)
    output = Image.fromarray(output, "RGB")
    output.save("result_imagex4.png")
    # print(output.shape)
# =============================================================================
#     plt.imshow(output)
#     plt.axis('off')
#     plt.savefig('result_imagex4.png', bbox_inches='tight',pad_inches = 0)
# =============================================================================
