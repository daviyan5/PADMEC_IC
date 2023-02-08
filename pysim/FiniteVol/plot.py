import numpy as np
import matplotlib.pyplot as plt




def plot_time():
    # Load the data
    time = np.load("./important/times.npy")
    vols = np.load("./important/vols.npy")

    # Polynomial fit
    estimator = np.load("./important/estimator.npy")
    low_estimator = np.load("./important/low_estimator.npy")
    up_estimator = np.load("./important/up_estimator.npy")

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle('Time x Number of cells')

    # Plot time x number of cells
    ax.plot(vols, time, label = "")
    ax.set_xlabel('Number of volumes')
    ax.set_ylabel('Time (s)')
    ax.grid()
    
    # Plot polynomial fit
    p = np.poly1d(estimator)
    estimated = p(vols)
    ax.plot(vols, estimated, label = "Linear fit")

    # Plot confidence interval
    p_low = np.poly1d(low_estimator)
    p_up = np.poly1d(up_estimator)
    low = p_low(vols)
    up = p_up(vols)

   

    ax.fill_between(vols, low, up, alpha=0.2, label = "Estimated range from linear fit of valleys and peaks")
    
    fig.legend()
    
    plt.tight_layout()
    fig.savefig("./tmp/time_x_cells.png")
    plt.show()


if __name__ == "__main__":
    plot_time()