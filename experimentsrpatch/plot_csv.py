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
    plt_title = "Iterative forward chop (GeForce GTX 860M)"

    # X axis - Patch dimension, Y axis - Maximum batch size
    x_data, y_data = (
        np.round(np.array(full_result.iloc[::2, x_column_index].values).flatten(), 2),
        np.round(np.array(full_result.iloc[::2, y_column_index].values).flatten(), 2),
    )
# =============================================================================
#     x_data, y_data = (
#         np.round(np.array(full_result.iloc[::4, x_column_index].values).flatten(), 2),
#         np.round(np.array(full_result.iloc[::4, y_column_index].values).flatten(), 2),
#     )
# =============================================================================

    # X and y labels
    x_label = list(full_result.columns)[x_column_index]
    y_label = list(full_result.columns)[y_column_index]

    # Plotting

    fig, ax = plt.subplots()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x_data, y_data, linestyle="--", marker="o", color="r")
    #plt.plot(x_data, y_data, linestyle="--", marker="o", color="r")
    #ax.plot(x, y, 'bo-')
    
    for X, Y, Z in zip(x_data, y_data, y_data):
        # Annotate the points 5 _points_ above and to the left of the vertex
        plt.annotate('{}'.format(Z), xy=(X,Y), xytext=(-5, 5), ha='right',
                    textcoords='offset points')
    # Saving plot
    plt.savefig("results/{0}.png".format("_".join(y_label.split()) + "_" + date))

    # Showing plot
    plt.show()
    
def plot_two_lines(file1, file2, x1, y1, x2, y2):
    
    fr1 = pd.read_csv(file1)
    fr2 = pd.read_csv(file2)

    # Plotting result message
    print("\nPlotting result...\n")
    
    # Creating plot title
    plt_title = "Image 1 vs Image 2 forward chop (GeForce GTX 860M)"

    # X axis - Patch dimension, Y axis - Maximum batch size
    x1d, y1d = (
        np.round(np.array(fr1.iloc[::2, x1].values).flatten(), 2),
        np.round(np.array(fr1.iloc[::2, y1].values).flatten(), 2),
    )
    x2d, y2d = (
        np.round(np.array(fr2.iloc[::2, x2].values).flatten(), 2),
        np.round(np.array(fr2.iloc[::2, y2].values).flatten(), 2),
    )
# =============================================================================
#     x_data, y_data = (
#         np.round(np.array(full_result.iloc[::4, x_column_index].values).flatten(), 2),
#         np.round(np.array(full_result.iloc[::4, y_column_index].values).flatten(), 2),
#     )
# =============================================================================

    # X and y labels
# =============================================================================
#     x_label = list(fr1.columns)[x1]
#     y_label = list(fr1.columns)[y1]
# =============================================================================
    x_label = "Patch Dimension"
    y_label = "Per Batch Processing Time"
    # Plotting

    fig, ax = plt.subplots()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.plot(x1d, y1d, linestyle="--", marker="o", color="r", label="Image 1")
    plt.plot(x2d, y2d, linestyle="--", marker="o", color="b", label="Image 2")
    #plt.plot(x_data, y_data, linestyle="--", marker="o", color="r")
    #ax.plot(x, y, 'bo-')
    
# =============================================================================
#     for X, Y, Z in zip(x_data, y_data, y_data):
#         # Annotate the points 5 _points_ above and to the left of the vertex
#         plt.annotate('{}'.format(Z), xy=(X,Y), xytext=(-5, 5), ha='right',
#                     textcoords='offset points')
# =============================================================================
    # Saving plot
# =============================================================================
#     plt.savefig("results/{0}.png".format("_".join(y_label.split()) + "_" + date))
# =============================================================================
    plt.legend()
    # Showing plot
    plt.show()
@click.command()
@click.option("--csv_path", default="results/2080ti.csv", help="Path of the csv file")
@click.option("--x_column_index", default=1, help="Data for X axis")  
@click.option("--y_column_index", default=6, help="Data for y axis")    
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
    #main()
    plot_two_lines("results/file1.csv", "results/file2.csv", 0, 14, 0, 14)
    
    
    
