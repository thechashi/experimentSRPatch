"""
Decides the best patch size and batch size from a given csv file
"""
import pandas as pd 
import toml
import batch_patch_forward_chop as bpfc
import utilities as ut
def main(stat_path, model_name, img_path, shave, scale):
    stat_df = pd.read_csv(stat_path)
    print(stat_df.columns)
    total_batches = stat_df['Total Patches'] / stat_df['Maximum Batch Size']
    stat_df['Total Batches'] = total_batches
    per_batch_processing_time = stat_df['Total batch processing time'] / stat_df['Total Batches']
    stat_df['Per Batch Processing Time'] = per_batch_processing_time
    print(stat_df.columns)
    print(stat_df)
    maximum_patch_size = stat_df['Patch dimnesion'].max()
    min_total_processing_time = stat_df['Total time'].min()
    idx_min_total_processing_time = stat_df['Total time'].idxmin()
    min_per_batch_processing_time = stat_df['Per Batch Processing Time'].min()
    idx_min_per_batch_processing_time = stat_df['Per Batch Processing Time'].idxmin()
    
    print(min_total_processing_time)
    print(idx_min_total_processing_time)
    print(min_per_batch_processing_time)
    print(idx_min_per_batch_processing_time)
    patch_dimension = 0
    batch_size = 0
    print(stat_df.loc[idx_min_total_processing_time, 'Patch dimnesion'])
    img = ut.npz_loader(img_path)
    c, h, w = img.shape
    if h < w: 
        temp = h
        h = w
        w = temp
        
    if h < maximum_patch_size and w < maximum_patch_size:
        print('Height and width both are smaller than maximum valid patch size...')
    elif h >= maximum_patch_size and w < maximum_patch_size:
        print('Height is bigger than the maximum valid patch size but width is smaller...')
    elif h >= maximum_patch_size and w >= maximum_patch_size:
        print('Height and width both are larger than maximum valid patch size...')
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
#     
# =============================================================================
if __name__ == "__main__":
    custom_upsampler_config = toml.load('../custom_upsampler_config.toml')
    main(custom_upsampler_config['stat_csv_path'], custom_upsampler_config['model_name'], 
         custom_upsampler_config['img_path'], custom_upsampler_config['shave'],
         custom_upsampler_config['scale'])