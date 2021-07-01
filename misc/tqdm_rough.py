from tqdm import trange
from time import sleep
from tqdm import tqdm
# =============================================================================
# max_dim  = 1024
# min_dim = 40
# dim_gap = 4
# t = trange(max_dim, min_dim - 1, -dim_gap, desc='Bar desc', leave=True)
# for i in t:
#     t.set_description("Patch Dimension: {:04}x{:04}, Batch_size: {:03d}, Used memory: {:09.3f}, Leaked Memory: {:09.3f}".format(i*4, i*4, int(i*0.11), i* 90, i* 0.1))
#     t.refresh() # to show immediately the update
#     sleep(1)
# =============================================================================
runs = 10
pbar = tqdm(total = runs+1)
i = 0
j = 100
while i <= runs:

    ### ROLLING THE DICES PROCESS ###
    print('roll dice')
    i += 6  # Updating the current tile

    ### SURPASSING THE NUMBER OF TILES ONBOARD ###
    if j > 37:   # If more than a table turn is achieved,
        i += 1   # One more turn is registered
        j -= 38  # Update the tile to one coresponding to a board tile.
        pbar.update(1)
    else:
        pass
...
pbar.close()