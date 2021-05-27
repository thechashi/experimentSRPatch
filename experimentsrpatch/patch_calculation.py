def patch_calculation(dimension, height, width, shave, scale, h_tracker, w_tracker):
    min_size = dimension*dimension
    size = height * width 
    h_tracker = []
    w_trakcer = []
    h_size = 0
    w_size = 0
    while(size > min_size):
        h_half, w_half = height // 2, width // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += h_size % scale
        w_size += w_size%scale
        size = h_size * w_size
        height, width = h_size, w_size
    print(h_size, w_size)


patch_calculation(40, 100, 160, 10, 4, [], [])
        
        
    
    