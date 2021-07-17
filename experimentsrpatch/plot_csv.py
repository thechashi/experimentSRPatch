import time
import pandas as pd
import numpy as np
import click
import matplotlib.pyplot as plt

def plot_csv(filepath, x_column_index, y_column_index):
    """
    Plots data from a given csv

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    x_column_index : TYPE
        DESCRIPTION.
    y_column_index : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Timestamp
    date = "_".join(str(time.ctime()).split())
    date = "_".join(date.split(":"))
    
    full_result = pd.read_csv(filepath)

    # Plotting result message
    print("\nPlotting result...\n")
    
    # Creating plot title
    plt_title = "Maximum batch size experiment"

    # X axis - Patch dimension, Y axis - Maximum batch size
    x_data, y_data = (
        np.array(full_result.iloc[::8, x_column_index].values).flatten(),
        np.array(full_result.iloc[::8, y_column_index].values).flatten(),
    )

    # X and y labels
    x_label = "Dimension n of patch(nxn)"
    y_label = list(full_result.columns)[y_column_index]

    # Plotting
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data, linestyle="--", marker="o", color="r")

    # Saving plot
    plt.savefig("results/{0}.png".format("_".join(y_label.split()) + "_" + date))

    # Showing plot
    plt.show()
    

@click.command()
@click.option("--csv_path", default="results/2080ti.csv", help="Path of the csv file")
@click.option("--x_column_index", default=1, help="Data for X axis")  
@click.option("--y_column_index", default=12, help="Data for y axis")    
def main(csv_path, x_column_index, y_column_index):
    """
    

    Parameters
    ----------
    csv_path : str
        csv file path.
    x_column_index : int
        Dataframe column index for x axis.
    y_column_index : int
        Dataframe column index for y axis.

    Returns
    -------
    None.

    """
    plot_csv(csv_path, x_column_index, y_column_index)

# =============================================================================
# maximum_batch_csv
#     0 - index
#     1 - patch dim
#     2 - maximum batch size
#     3 - batch error
#     4 - total patches
#     5 - patch creation
#     6 - upsampling 
#     7 - c 2 g
#     8 - g 2 c
#     9 - batch creation 
#     10 - cuda clear time
#     11 - mjerging time
#     12 - total batch processing time
#     13 - total time
# =============================================================================
if __name__ == "__main__":
    main()
    
    
    
