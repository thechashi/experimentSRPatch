"""
Decides the best patch size and batch size from a given csv file
"""
import sys
import pandas as pd
import toml
import math
import numpy as np
import torch
import modelloader as md
import batch_patch_forward_chop as bpfc
import utilities as ut


def helper_rrdb_iterative(stat_path, img_dimension, shave, scale):
    total_patches = "Total Patches"
    total_batches_ = "Total Batches"
    maximum_batch_size = "Maximum Batch Size"
    total_batch_processing_time = "Total batch processing time"
    per_batch_processing_time_ = "Per Batch Processing Time"
    patch_dimension = "Patch Dimension"
    total_time = "Total time"
    stat_df = pd.read_csv(stat_path)
    # =============================================================================
    #     print(stat_df.columns)
    # =============================================================================
    total_batches = stat_df[total_patches] / stat_df[maximum_batch_size]
    stat_df[total_batches_] = total_batches
    per_batch_processing_time = (
        stat_df[total_batch_processing_time] / stat_df[total_batches_]
    )
    stat_df[per_batch_processing_time_] = per_batch_processing_time

    # =============================================================================
    #     print(stat_df.columns)
    #     print(stat_df)
    # =============================================================================
    maximum_patch_size = stat_df[patch_dimension].max()
    min_total_processing_time = stat_df[total_time].min()
    idx_min_total_processing_time = stat_df[total_time].idxmin()
    min_per_batch_processing_time = stat_df[per_batch_processing_time_].min()
    idx_min_per_batch_processing_time = stat_df[per_batch_processing_time_].idxmin()
    # =============================================================================
    #     print(min_total_processing_time)
    #     print(idx_min_total_processing_time)
    #     print(min_per_batch_processing_time)
    #     print(idx_min_per_batch_processing_time)
    #     print('here')
    # =============================================================================
    # =============================================================================
    #     patch_dimension = 0
    #     batch_size = 0
    # =============================================================================
    # =============================================================================
    #     print(stat_df.loc[idx_min_total_processing_time, patch_dimension])
    # =============================================================================
    img = np.load("data/slices/0.npz")
    img = img.f.arr_0
    img = np.resize(img, (img_dimension, img_dimension))
    img = img[np.newaxis, :, :]
    img = torch.from_numpy(img)
    # =============================================================================
    #     img = img.unsqueeze(0)
    # =============================================================================
    model = md.load_rrdb(device="cuda")
    c, h, w = img.shape
    if h < w:
        temp = h
        h = w
        w = temp
    dimensions = stat_df.loc[
        :, [patch_dimension, maximum_batch_size, per_batch_processing_time_]
    ].values
    total_patches_from_img = []
    for d in range(len(dimensions)):
        img_patch_count = ut.patch_count(h, w, dimensions[d][0], shave)
        img_batch_count = img_patch_count / dimensions[d][1]
        img_processing_time = img_batch_count * dimensions[d][2]
        total_patches_from_img.append(
            [
                dimensions[d][0],
                dimensions[d][1],
                img_patch_count,
                math.ceil(img_batch_count),
                img_processing_time,
            ]
        )
    # print(total_patches_from_img[269])

    img_df = pd.DataFrame(total_patches_from_img)
    best_index = img_df[4].idxmin()
    # =============================================================================
    #     print(best_index)
    #     print(img_df.iloc[best_index, 0])
    # =============================================================================
    if h < maximum_patch_size and w < maximum_patch_size:
        patch_dimension = int(img_df.iloc[idx_min_total_processing_time, 0])
        batch_size = int(img_df.iloc[idx_min_total_processing_time, 1])
    else:
        patch_dimension = int(img_df.iloc[best_index, 0])
        batch_size = int(img_df.iloc[best_index, 1])

    # =============================================================================
    #     print('Patch dimension: {}, batch size: {}'.format(patch_dimension, batch_size))
    # =============================================================================
    # =============================================================================
    #     bpfc.upsample(
    #         model_name,
    #         img_path,
    #         patch_dimension,
    #         shave,
    #         batch_size,
    #         scale,
    #         device="cuda",
    #     )
    # =============================================================================

    # =============================================================================
    #     if model_name == "EDSR":
    #         input_image = ut.load_image(img_path)
    #     elif model_name == "RRDB":
    #         # input_image = ut.load_grayscale_image(img_path)
    #         input_image = ut.npz_loader(img_path)
    #         # print(input_image.shape)
    #     else:
    #         raise Exception("{} : Unknown model...".format(model_name))
    # =============================================================================
    print(patch_dimension)
    print(batch_size)
    bpfc.patch_batch_forward_chop(
        input_image=img,
        patch_dimension=patch_dimension,
        patch_shave=shave,
        scale=scale,
        batch_size=batch_size,
        model_type="RRDB",
        device="cuda",
        print_timer=True,
    )


def main(stat_path, model_name, img_path, shave, scale):
    total_patches = "Total Patches"
    total_batches_ = "Total Batches"
    maximum_batch_size = "Maximum Batch Size"
    total_batch_processing_time = "Total batch processing time"
    per_batch_processing_time_ = "Per Batch Processing Time"
    patch_dimension = "Patch Dimension"
    total_time = "Total time"

    stat_df = pd.read_csv(stat_path)
    # =============================================================================
    #     print(stat_df.columns)
    # =============================================================================
    total_batches = stat_df[total_patches] / stat_df[maximum_batch_size]
    stat_df[total_batches_] = total_batches
    per_batch_processing_time = (
        stat_df[total_batch_processing_time] / stat_df[total_batches_]
    )
    stat_df[per_batch_processing_time_] = per_batch_processing_time
    # =============================================================================
    #     print(stat_df.columns)
    #     print(stat_df)
    # =============================================================================

    maximum_patch_size = stat_df[patch_dimension].max()
    min_total_processing_time = stat_df[total_time].min()
    idx_min_total_processing_time = stat_df[total_time].idxmin()
    min_per_batch_processing_time = stat_df[per_batch_processing_time_].min()
    idx_min_per_batch_processing_time = stat_df[per_batch_processing_time_].idxmin()

    print(min_total_processing_time)
    print(idx_min_total_processing_time)
    print(min_per_batch_processing_time)
    print(idx_min_per_batch_processing_time)
    print("here")
    # =============================================================================
    #     patch_dimension = 0
    #     batch_size = 0
    # =============================================================================
    # =============================================================================
    #     print(stat_df.loc[idx_min_total_processing_time, patch_dimension])
    # =============================================================================
    img = ut.npz_loader(img_path)
    c, h, w = img.shape
    if h < w:
        temp = h
        h = w
        w = temp
    dimensions = stat_df.loc[
        :, [patch_dimension, maximum_batch_size, per_batch_processing_time_]
    ].values
    total_patches_from_img = []
    for d in range(len(dimensions)):
        img_patch_count = ut.patch_count(h, w, dimensions[d][0], shave)
        img_batch_count = img_patch_count / dimensions[d][1]
        img_processing_time = img_batch_count * dimensions[d][2]
        total_patches_from_img.append(
            [
                dimensions[d][0],
                dimensions[d][1],
                img_patch_count,
                math.ceil(img_batch_count),
                img_processing_time,
            ]
        )
    # print(total_patches_from_img[269])

    img_df = pd.DataFrame(total_patches_from_img)
    best_index = img_df[4].idxmin()
    # =============================================================================
    #     print(best_index)
    #     print(img_df.iloc[best_index, 0])
    # =============================================================================
    if h < maximum_patch_size and w < maximum_patch_size:
        patch_dimension = int(img_df.iloc[idx_min_total_processing_time, 0])
        batch_size = int(img_df.iloc[idx_min_total_processing_time, 1])
    else:
        patch_dimension = int(img_df.iloc[best_index, 0])
        batch_size = int(img_df.iloc[best_index, 1])

    print("Patch dimension: {}, batch size: {}".format(patch_dimension, batch_size))


# =============================================================================
#     bpfc.upsample(
#         model_name,
#         img_path,
#         patch_dimension,
#         shave,
#         batch_size,
#         scale,
#         device="cuda",
#     )
# =============================================================================

# =============================================================================
#     if model_name == "EDSR":
#         input_image = ut.load_image(img_path)
#     elif model_name == "RRDB":
#         # input_image = ut.load_grayscale_image(img_path)
#         input_image = ut.npz_loader(img_path)
#         # print(input_image.shape)
#     else:
#         raise Exception("{} : Unknown model...".format(model_name))
#     bpfc.patch_batch_forward_chop(
#     input_image=input_image,
#     patch_dimension=patch_dimension,
#     patch_shave=shave,
#     scale=scale,
#     batch_size=batch_size,
#     model_type=model_name,
#     device="cuda",
#     print_timer=True)
# =============================================================================

if __name__ == "__main__":
    custom_upsampler_config = toml.load("../custom_upsampler_config.toml")
    main(
        custom_upsampler_config["stat_csv_path"],
        custom_upsampler_config["model_name"],
        custom_upsampler_config["img_path"],
        custom_upsampler_config["shave"],
        custom_upsampler_config["scale"],
    )
# =============================================================================
#     print(sys.argv)
# =============================================================================
# =============================================================================
#     helper_rrdb_iterative(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
#          int(sys.argv[4]))
# =============================================================================
